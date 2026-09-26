"""Error budget, step 2b: rank error sources across the RTK bags (val + train) from rtk_epochs/*.parquet.

Every matched epoch (judge frame, base_link from both RTK antennas) is put in exactly one category, in this
priority order:
  ref_clock    reference GNSS header stamp off its bag median by > 0.3 s (+-1 s clock steps of the recorder):
               the reference itself is displaced by v * offset
  ref_jump     excursion of the along or cross error that our continuous estimate cannot make: > 1.5 m from its
               20 s rolling median for < 10 s, or a > 2 m step to a level > 2 m away from both neighbouring levels
               and back to the level it left within 120 s (false RTK fix, reference running 17-30 m off), not at one of our place fixes
  start_off    before our position reaches the main cycle (s_map < 0: start branch / unknown track)
  west_terminal |cross| > 1.5 m at s_map 5380-5750: dead-end stub / terminal fan tracks missing from the map
  parallel_track 3 m < |cross| < 6 m, lateral-dominated, at s_map 1100-1745: the tram on the other track of the
               double-track section (one map track only)
  off_track_other lateral-dominated |cross| > 1.5 m elsewhere
  drift_long   along error while > 1000 m travelled since the last accepted place fix (or the start)
  drift_short  the remaining along/cross/z error (<= 1000 m since the last fix)
For each category: share of the summed squared 3-D error, share of the mean per-bag MSE, and the mean / median
per-bag 3-D RMSE if its epochs are excluded. Also: along RMSE vs distance since the last fix, place-fix
outcomes (accepted fixes that made the error worse, rejections of a right candidate, stops without candidate).

    python analysis/ideas_check/error_budget/analyze_rtk.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EP = HERE / 'rtk_epochs'
CATS = ['ref_clock', 'ref_jump', 'start_off', 'west_terminal', 'parallel_track', 'off_track_other', 'drift_long',
        'drift_short']
REF = ('ref_clock', 'ref_jump')


def rmse(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float('nan')


def rolling_median(t, x, half):
    out = np.empty_like(x)
    j0 = j1 = 0
    n = len(t)
    # simple O(n * w) with a moving window on sorted t (RTK epochs ~ 10 Hz, windows of 200 samples)
    for i in range(n):
        while t[j0] < t[i] - half:
            j0 += 1
        while j1 < n and t[j1] <= t[i] + half:
            j1 += 1
        out[i] = np.median(x[j0:j1])
    return out


def ref_fault_segments(t, al, cr, v, after_fix, max_len=120.0):
    """Longer reference faults: the error steps by > 2 m between two epochs (not at one of our fixes), stays at a
    level > 2 m away from both neighbouring levels, and steps back within max_len seconds."""
    dt = np.diff(t)
    de = np.hypot(np.diff(al), np.diff(cr))
    thr = 2.0 + 0.01 * v[1:] * np.maximum(dt - 1.5, 0.0)
    jumps = np.flatnonzero((de > thr) & ~after_fix[1:] & ~after_fix[:-1]) + 1
    out = np.zeros(len(t), bool)
    if len(jumps) == 0:
        return out
    edges = np.r_[0, jumps, len(t)]
    med = [(np.median(al[a:b]), np.median(cr[a:b])) for a, b in zip(edges[:-1], edges[1:])]
    for k in range(1, len(med) - 1):
        a, b = edges[k], edges[k + 1]
        if t[b - 1] - t[a] > max_len:
            continue
        m0, m1 = med[k - 1], med[k + 1]
        # an excursion of > 3 m away from both neighbours that returns to the level it left (within 1.5 m)
        if (np.hypot(med[k][0] - m0[0], med[k][1] - m0[1]) > 3.0 and np.hypot(med[k][0] - m1[0], med[k][1] - m1[1]) > 3.0
                and np.hypot(m0[0] - m1[0], m0[1] - m1[1]) < 1.5):
            out[a:b] = True
    return out


# reviewed by hand (timelines in the report): the reference moves while our estimate is continuous
MANUAL_REF = {
    '30639_9c362687': [(950.0, 1017.5)],   # cross -9 m at standstill, the reference jumps back to 0.6 m while stopped
    '30618_28538acf': [(700.0, 767.5)],    # along +31 m while stopped at the 3433 m platform (accepted fix there)
}


def classify(E, lm):
    t = E.t.to_numpy()
    acc = lm[(lm.p_known >= 0.6) & (lm.n > 0)].t.to_numpy() if len(lm) else np.array([])
    # distance since the last accepted fix (our own odometer s)
    s = E.s.to_numpy()
    last = np.searchsorted(acc, t) - 1
    s_fix = np.where(last >= 0, np.interp(acc[np.clip(last, 0, None)] if len(acc) else t, t, s), s[0])
    E['d_fix'] = s - s_fix
    after_fix = np.zeros(len(t), bool)
    for ta in acc:
        after_fix |= (t >= ta - 0.5) & (t <= ta + 2.0)
    al, cr, sm = E.along.to_numpy(), E.cross.to_numpy(), E.s_map.to_numpy()
    cat = np.full(len(t), 'drift_short', dtype=object)
    cat[E.d_fix.to_numpy() > 1000.0] = 'drift_long'
    lateral = (np.abs(cr) > 1.5) & (np.abs(cr) > np.abs(al)) & (sm >= 0)
    cat[lateral] = 'off_track_other'
    cat[lateral & (sm >= 1100) & (sm <= 1745) & (np.abs(cr) > 3.0) & (np.abs(cr) < 6.0)] = 'parallel_track'
    cat[(np.abs(cr) > 1.5) & (sm >= 5380) & (sm <= 5750)] = 'west_terminal'
    cat[sm < 0] = 'start_off'
    # reference jumps: short excursions of the error that our continuous estimate cannot make
    jump = np.zeros(len(t), bool)
    for x in (al, cr):
        dev = np.abs(x - rolling_median(t, x, 10.0)) > 1.5
        m = np.r_[False, dev, False]
        dm = np.diff(m.astype(int))
        for a, b in zip(np.flatnonzero(dm == 1), np.flatnonzero(dm == -1) - 1):
            if t[b] - t[a] < 10.0 and not after_fix[a:b + 1].any():
                jump[a:b + 1] = True
    jump |= ref_fault_segments(t, al, cr, E.v.to_numpy(), after_fix)
    cat[jump] = 'ref_jump'
    cat[(np.abs(cr) > 1.5) & (sm >= 5380) & (sm <= 5750)] = 'west_terminal'  # known cause first
    for bag, spans in MANUAL_REF.items():
        if E.attrs.get('bag') == bag:
            for a, b in spans:
                cat[(t >= a) & (t <= b)] = 'ref_jump'
    cat[np.abs(E.ref_lat.to_numpy()) > 0.3] = 'ref_clock'
    E['cat'] = cat
    return E


def fix_outcomes(bag, E, lm):
    """Every place-fix attempt with the along error right before and right after it."""
    rows = []
    t = E.t.to_numpy()
    for q in lm.itertuples(index=False):
        b = (t >= q.t - 1.5) & (t < q.t)
        a_ = (t > q.t + 0.3) & (t <= q.t + 2.0)
        if not b.any() or not a_.any():
            continue
        before = float(E.along[b].mean())
        after = float(E.along[a_].mean())
        v = float(E.v[b].mean())
        rows.append({'bag': bag, 't': q.t, 'kind': 'cutoff' if v > 1.0 else 'stop', 'n': int(q.n), 'sd': q.sd,
                     'd0': q.d0, 'p_known': q.p_known, 'accepted': bool(q.p_known >= 0.6 and q.n > 0),
                     'along_before': before, 'along_after': after,
                     # if the nearest candidate were the true place: along_before = -d0
                     'cand_err': before + q.d0 if q.n > 0 else np.nan})
    return rows


def main():
    summ = pd.read_csv(HERE / 'rtk_summary.csv')
    summ = summ[summ.get('error').isna()] if 'error' in summ else summ
    allE, fixes = [], []
    for bag, split in zip(summ.bag, summ.split):
        E = pd.read_parquet(EP / f'{bag}.parquet')
        lm = pd.read_csv(EP / f'{bag}_lm.csv')
        E.attrs['bag'] = bag
        E = classify(E, lm)
        E['bag'], E['split'] = bag, split
        E['e2'] = E.along ** 2 + E.cross ** 2 + E.dz ** 2
        allE.append(E)
        fixes += fix_outcomes(bag, E, lm)
    A = pd.concat(allE, ignore_index=True)
    F = pd.DataFrame(fixes)
    F.to_csv(HERE / 'rtk_fix_outcomes.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    for split in ('val', 'train'):
        S = A[A.split == split]
        print('=' * 110)
        print(f'{split.upper()}: {S.bag.nunique()} bags with RTK, {len(S)} epochs')
        bag_mse = S.groupby('bag').e2.mean()
        bag_rmse = np.sqrt(bag_mse)
        print(f'  per-bag 3-D RMSE: mean {bag_rmse.mean():.3f}, median {bag_rmse.median():.3f}; '
              f'mean of per-bag MSE {bag_mse.mean():.3f} m^2')
        # per-bag component shares
        comp = S.groupby('bag').agg(along2=('along', lambda x: np.mean(x ** 2)), cross2=('cross', lambda x: np.mean(x ** 2)),
                                    z2=('dz', lambda x: np.mean(x ** 2)))
        tot = comp.sum(axis=1)
        print(f'  components of the mean per-bag MSE: along {comp.along2.mean() / tot.mean() * 100:.1f} %, '
              f'cross {comp.cross2.mean() / tot.mean() * 100:.1f} %, z {comp.z2.mean() / tot.mean() * 100:.1f} %')
        rows = []
        for c in CATS:
            k = S.cat == c
            per_bag = S[k].groupby('bag').e2.sum() / S.groupby('bag').e2.count()
            keep = S[~k].groupby('bag').e2.mean().reindex(bag_mse.index).fillna(0.0)
            r_ex = np.sqrt(keep)
            rows.append({'category': c, 'epochs_pct': k.mean() * 100, 'bags': int((per_bag > 0.01).sum()),
                         'share_sum_sq': S.e2[k].sum() / S.e2.sum() * 100,
                         'share_mean_bag_mse': per_bag.reindex(bag_mse.index).fillna(0).mean() / bag_mse.mean() * 100,
                         'rmse_in_cat': rmse(np.sqrt(S.e2[k])),
                         'mean_rmse_without': r_ex.mean(), 'median_rmse_without': r_ex.median()})
        C = pd.DataFrame(rows)
        print(C.round(3).to_string(index=False))
        C.to_csv(HERE / f'rtk_categories_{split}.csv', index=False)
        # judge-like view: the judge's reference has no outliers -> drop our reference faults first
        S2 = S[~S.cat.isin(REF)]
        m2 = S2.groupby('bag').e2.mean()
        print(f'  without reference faults: per-bag 3-D RMSE mean {np.sqrt(m2).mean():.3f}, median {np.sqrt(m2).median():.3f}; '
              f'mean per-bag MSE {m2.mean():.3f}')
        rows2 = []
        for c in CATS:
            if c in REF:
                continue
            k = S2.cat == c
            per_bag = (S2[k].groupby('bag').e2.sum() / S2.groupby('bag').e2.count()).reindex(m2.index).fillna(0)
            keep = np.sqrt(S2[~k].groupby('bag').e2.mean().reindex(m2.index).fillna(0.0))
            al_bag = (S2[k].groupby('bag').along.apply(lambda x: np.sum(x ** 2)) / S2.groupby('bag').e2.count()).reindex(m2.index).fillna(0)
            rows2.append({'category': c, 'share_mean_bag_mse': per_bag.mean() / m2.mean() * 100,
                          'along_part_of_it': al_bag.mean() / max(per_bag.mean(), 1e-12) * 100,
                          'mean_rmse_without': keep.mean(), 'median_rmse_without': keep.median()})
        C2 = pd.DataFrame(rows2)
        print(C2.round(3).to_string(index=False))
        C2.to_csv(HERE / f'rtk_categories_noref_{split}.csv', index=False)
        # per-bag table: RMSE and the dominant category
        B = S.groupby(['bag', 'cat']).e2.sum().unstack(fill_value=0.0)
        B = B.div(B.sum(axis=1), axis=0) * 100
        B['p3_rmse'] = bag_rmse
        B['along_rmse'] = np.sqrt(comp.along2)
        B['cross_rmse'] = np.sqrt(comp.cross2)
        B['z_rmse'] = np.sqrt(comp.z2)
        B['share_of_split_mse'] = bag_mse / bag_mse.sum() * 100
        B = B.sort_values('p3_rmse', ascending=False)
        B.to_csv(HERE / f'rtk_bags_{split}.csv')
        print('  per bag (category columns = % of the bag squared error):')
        print(B.round(1).head(20).to_string())
        # along RMSE vs distance since the last fix (main cycle, no reference faults)
        k = S.cat.isin(['drift_long', 'drift_short'])
        bins = [0, 250, 500, 1000, 2000, 4000, 1e9]
        S2 = S[k].assign(db=pd.cut(S.d_fix[k], bins))
        g = S2.groupby('db', observed=True).agg(n=('along', 'size'), along_rmse=('along', rmse),
                                                along_p95=('along', lambda x: np.percentile(np.abs(x), 95)))
        print('  along RMSE by distance since the last accepted fix:')
        print(g.round(3).to_string())
    # place-fix outcomes
    print('=' * 110)
    print('PLACE-FIX ATTEMPTS (all RTK bags; along error of the published base_link 1.5 s before / 0.3-2 s after)')
    F['gain'] = F.along_before.abs() - F.along_after.abs()
    for kind in ('stop', 'cutoff'):
        G = F[F.kind == kind]
        acc = G[G.accepted]
        rej = G[~G.accepted & (G.n > 0)]
        none = G[G.n == 0]
        print(f'  {kind}: attempts {len(G)}, accepted {len(acc)} (mean |along| {acc.along_before.abs().mean():.3f} -> '
              f'{acc.along_after.abs().mean():.3f}; worse by > 0.5 m: {int((acc.gain < -0.5).sum())}, '
              f'worse by > 1 m: {int((acc.gain < -1.0).sum())}); rejected with a candidate {len(rej)} '
              f'(candidate right within 1 m: {int((rej.cand_err.abs() < 1.0).sum())}, mean |along| there '
              f'{rej.along_before.abs().mean():.3f}); no candidate {len(none)} (|along| > 3 m there: '
              f'{int((none.along_before.abs() > 3).sum())})')
    bad = F[F.accepted & (F.gain < -0.5)].sort_values('gain')
    print('  accepted fixes that made the along error worse by > 0.5 m:')
    print(bad[['bag', 't', 'kind', 'sd', 'd0', 'p_known', 'along_before', 'along_after', 'cand_err']].round(2).head(25).to_string(index=False))
    miss = F[~F.accepted & (F.n > 0) & (F.cand_err.abs() < 1.0) & (F.along_before.abs() > 1.0)]
    print('  rejected right candidates while |along| > 1 m:')
    print(miss[['bag', 't', 'kind', 'n', 'sd', 'd0', 'p_known', 'along_before', 'cand_err']].round(2).head(25).to_string(index=False))
    far = F[(F.n == 0) & (F.along_before.abs() > 2.0)]
    print('  attempts without a candidate while |along| > 2 m (gate too narrow or no place):')
    print(far[['bag', 't', 'kind', 'sd', 'along_before']].round(2).head(25).to_string(index=False))
    A[['bag', 'split', 't', 'along', 'cross', 'dz', 'v', 's_map', 'd_fix', 'cat', 'ref_lat']].to_parquet(HERE / 'rtk_epochs_all.parquet')


if __name__ == '__main__':
    main()
