"""Error budget, step 1b: decompose the judge error on the check bag 30618_88aea4d9 (run run_check.py first).

Prints the report to stdout and writes tables: check_stops.csv (every stop / cut-off with its place-fix
attempt), check_segments.csv (along drift between accepted place fixes), check_speed_regimes.csv.

    python analysis/ideas_check/error_budget/analyze_check.py [--tag ''] [--end-s 1270]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge as CB  # noqa: E402
import eval_base_link as EB  # noqa: E402

BAG = '30618_88aea4d9'
LMS = CB.PKG / 'maps' / 'landmarks.csv'
SLIP_BITS = 0xF          # bits 0-3: front/rear slip/slide
MODEL_ONLY = 1 << 11
STILL = 1 << 12
MANEUVER = 1 << 17


def rmse(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float('nan')


def intervals(mask, t, min_len=0.0):
    """[start, end) times of runs of True in a boolean series sampled at t."""
    m = np.r_[False, np.asarray(mask, bool), False]
    dm = np.diff(m.astype(int))
    a, b = np.flatnonzero(dm == 1), np.flatnonzero(dm == -1) - 1
    out = [(t[i], t[j]) for i, j in zip(a, b) if t[j] - t[i] >= min_len]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    ap.add_argument('--end-s', type=float, default=1270.0)
    a = ap.parse_args()
    P_all = pd.read_parquet(HERE / f'pairs_pos{a.tag}.parquet')
    V_all = pd.read_parquet(HERE / f'pairs_vel{a.tag}.parquet')
    O = pd.read_parquet(HERE / f'out_check{a.tag}.parquet')
    lm = pd.read_csv(HERE / f'lm_check{a.tag}.csv')
    ref = pd.read_parquet(HERE / 'ref_check.parquet')
    L = pd.read_csv(LMS, comment='#')
    P = P_all[P_all.t < a.end_s].reset_index(drop=True)
    V = V_all[V_all.t < a.end_s].reset_index(drop=True)
    n = len(P)
    pr = lambda *s: print(*s)

    # ------------------------------------------------------------------ 1. decomposition
    pr('=' * 100)
    pr(f'1. POSITION DECOMPOSITION ({BAG})')
    tot_full = np.mean(P_all.dist ** 2)
    tail = P_all.t >= a.end_s
    pr(f'  full bag: 3-D RMSE {rmse(P_all.dist):.3f} m, MSE {tot_full:.3f} m^2; t >= {a.end_s:.0f} s '
       f'({tail.mean() * 100:.1f} % of pairs) carries {np.sum(P_all.dist[tail] ** 2) / np.sum(P_all.dist ** 2) * 100:.1f} % '
       f'of the squared error')
    mse = np.mean(P.dist ** 2)
    pr(f'  t < {a.end_s:.0f} s: 3-D RMSE {np.sqrt(mse):.3f} m, MSE {mse:.4f} m^2, n={n}')
    for c in ('along', 'cross', 'dz'):
        x = P[c].to_numpy()
        pr(f'    {c:6s}: RMSE {rmse(x):.3f}  bias {x.mean():+.3f}  std {x.std():.3f}  MSE {np.mean(x**2):.4f} '
           f'({np.mean(x**2) / mse * 100:5.1f} %)  of which bias^2 {x.mean()**2:.4f}')

    # ------------------------------------------------------------------ 2. timing
    pr('=' * 100)
    pr('2. TIMING')
    pr(f'  pair stamp offset ours-ref: mean {P.dt_ms.mean():+.1f} ms, p5 {np.percentile(P.dt_ms, 5):+.1f}, '
       f'p95 {np.percentile(P.dt_ms, 95):+.1f}; v_ref*dt term: RMSE {rmse(P.v_ref * P.dt_ms / 1e3):.3f} m, '
       f'mean {np.mean(P.v_ref * P.dt_ms / 1e3):+.3f} m')
    acc_fix = lm[(lm.p_known >= 0.6) & (lm.n > 0)].t.to_numpy()
    seg = np.searchsorted(acc_fix, P.t.to_numpy())
    y = P.along.to_numpy()
    pth = P.path.to_numpy()
    for knot in (100.0, 200.0, 400.0):
        cols = []
        for sg in np.unique(seg):
            msk = seg == sg
            p0 = pth[msk].min()
            cols.append(msk.astype(float))
            for kk in np.arange(p0, pth[msk].max() + knot, knot):
                cols.append(np.where(msk, np.maximum(pth - kk, 0.0), 0.0))
        N = np.column_stack(cols)
        N = N[:, np.abs(N).sum(0) > 0]
        X = np.column_stack([P.v_ref.to_numpy(), N])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        c0, *_ = np.linalg.lstsq(N, y, rcond=None)
        r1, r0 = y - X @ c, y - N @ c0
        pr(f'  along = tau*v + drift(path; knots {knot:.0f} m, reset at fixes): tau = {c[0]:+.4f} s; '
           f'removable MSE {np.mean(r0**2) - np.mean(r1**2):.4f} m^2 (= {(np.mean(r0**2) - np.mean(r1**2)) / mse * 100:.1f} % '
           f'of 3-D MSE); tau-term alone mean((tau v)^2) {np.mean((c[0] * P.v_ref)**2):.4f}')
    # reference timing against the RTK master antenna (GNSS stamp)
    d = np.load(CB.NPZ / f'{BAG}.npz')
    t0 = d['localization__kinematic_state'][0, 1]
    m = d['sensing__gnss__master__fix']
    m = m[m[:, 5] == 2]
    e, nn = EB.TR.transform(m[:, 3], m[:, 2])
    gt = m[:, 1] - t0
    yaw = np.interp(gt, ref.t, np.unwrap(ref.yaw))
    ax = np.interp(gt, ref.t, ref.x) - EB.ALONG * np.cos(yaw)
    ay = np.interp(gt, ref.t, ref.y) - EB.ALONG * np.sin(yaw)
    rv = np.interp(gt, ref.t, ref.vx)
    dx, dy = e - EB.E0 - ax, nn - EB.N0 - ay
    al = dx * np.cos(yaw) + dy * np.sin(yaw)
    dzg = m[:, 4] - (np.interp(gt, ref.t, ref.z) + EB.HEIGHT)
    cg, *_ = np.linalg.lstsq(np.c_[np.ones(len(al)), rv], al, rcond=None)
    pr(f'  RTK master antenna vs reference (antenna = base_link - 9.873 m along yaw), {len(al)} RTK fixes: '
       f'along GNSS-ref = {cg[0]:+.3f} + {cg[1]:+.4f}*v  -> reference position lags the GNSS stamp by {cg[1]:.3f} s; '
       f'GNSS alt - (ref z + 3.0) median {np.median(dzg):+.3f} m')
    # our speed vs reference speed lag
    ts = O.t.to_numpy()
    vo = O.v.to_numpy()
    tv = V.t.to_numpy() + V.dt_ms.to_numpy() / 1e3   # our stamp
    for dl in (0.0, 0.05, 0.08, 0.1, 0.12, 0.15):
        ev = np.interp(tv - dl, ts, vo) - V.v_ref.to_numpy()
        pr(f'    speed delayed by {dl:.2f} s: RMSE {rmse(ev):.4f}')

    # ------------------------------------------------------------------ 3. stops and place fixes
    pr('=' * 100)
    pr('3. STOPS AND PLACE FIXES')
    ot = O.t.to_numpy()
    fl = O['flags'].to_numpy().astype(np.int64)
    stops = intervals((fl & STILL) > 0, ot, min_len=0.0)
    Pt = P_all.t.to_numpy()

    def along_at(t_a, t_b):
        k = (Pt >= t_a) & (Pt < t_b)
        return float(P_all.along[k].mean()) if k.any() else np.nan

    rows = []
    used = set()
    for (sa, sb) in stops:
        att = lm[(lm.t >= sa - 0.05) & (lm.t <= sb + 0.05)]
        r = {'kind': 'stop', 't0': sa, 't1': sb, 'dur': sb - sa}
        i0 = np.searchsorted(ot, sa)
        r['s_map'] = float(O.s_map.iloc[i0])
        r['along_stop'] = along_at(sa, min(sa + 1.5, sb))
        r['s_true'] = r['s_map'] - r['along_stop'] if r['s_map'] >= 0 else np.nan
        if np.isfinite(r['s_true']):
            dd = L.s.to_numpy() - r['s_true']
            j = int(np.argmin(np.abs(dd)))
            r['lm_near_d'] = float(dd[j])
            r['lm_near_pstop'] = float(L.p_stop.iloc[j])
            r['lm_near_cls'] = L.cls.iloc[j]
        if len(att):
            q = att.iloc[0]
            used.add(q.name)
            r.update(attempt=True, t_att=q.t, n=int(q.n), sd=q.sd, d0=q.d0, p_known=q.p_known, k=q.k,
                     accepted=bool(q.p_known >= 0.6 and q.n > 0),
                     along_before=along_at(q.t - 1.0, q.t), along_after=along_at(q.t + 0.5, q.t + 1.5))
        else:
            r.update(attempt=False)
        rows.append(r)
    for idx, q in lm.iterrows():
        if idx in used:
            continue
        i0 = np.searchsorted(ot, q.t)
        rows.append({'kind': 'cutoff' if O.v.iloc[min(i0, len(O) - 1)] > 1.0 else 'other', 't0': q.t, 't1': q.t,
                     'dur': 0.0, 's_map': q.s_map, 'attempt': True, 't_att': q.t, 'n': int(q.n), 'sd': q.sd,
                     'd0': q.d0, 'p_known': q.p_known, 'k': q.k, 'accepted': bool(q.p_known >= 0.6 and q.n > 0),
                     'along_before': along_at(q.t - 0.3, q.t), 'along_after': along_at(q.t + 0.2, q.t + 1.0)})
    S = pd.DataFrame(rows).sort_values('t0').reset_index(drop=True)
    S.to_csv(HERE / f'check_stops{a.tag}.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    cols = ['kind', 't0', 'dur', 's_map', 's_true', 'lm_near_d', 'lm_near_pstop', 'lm_near_cls', 'attempt', 'n', 'sd', 'd0',
            'p_known', 'accepted', 'along_before', 'along_after']
    pr(S[[c for c in cols if c in S]].round(2).to_string(index=False))
    st = S[S.kind == 'stop']
    pr(f'  stops: {len(st)}, with an attempt {int(st.attempt.sum())}, accepted {int(st.accepted.fillna(False).sum())}; '
       f'no candidate in the gate (n=0): {int((st.n == 0).sum())}; cut-off attempts {int((S.kind == "cutoff").sum())}, '
       f'accepted {int(S[S.kind == "cutoff"].accepted.sum())}')

    # ------------------------------------------------------------------ 4. drift segments between accepted fixes
    pr('=' * 100)
    pr('4. ALONG-TRACK DRIFT BETWEEN ACCEPTED PLACE FIXES (t < end)')
    acc = S[S.accepted.fillna(False).astype(bool)]
    edges = [P.t.min()] + list(acc.t_att) + [a.end_s]
    seg_rows = []
    sum_sq = np.sum(P.along ** 2)
    for i in range(len(edges) - 1):
        ta, tb = edges[i], edges[i + 1]
        k = (P.t >= ta) & (P.t < tb)
        if k.sum() < 5:
            continue
        Q = P[k]
        # path-based drift rate: along at the end minus along at the start over the reference path length
        st0 = Q.iloc[:50]
        st1 = Q.iloc[-50:]
        dpath = float(st1.path.mean() - st0.path.mean())
        dal = float(st1.along.mean() - st0.along.mean())
        n_nofix = int(((S.kind == 'stop') & (S.t0 > ta) & (S.t0 < tb) & ~S.accepted.fillna(False).astype(bool)).sum())
        seg_rows.append({'t_start': ta, 't_end': tb, 'path_m': dpath, 'along_start': float(st0.along.mean()),
                         'along_end': float(st1.along.mean()), 'along_max': float(Q.along.abs().max()),
                         'drift_pct': dal / dpath * 100 if dpath > 20 else np.nan, 'k_mean': float(Q.k.mean()),
                         'stops_without_fix': n_nofix, 'along_rmse': rmse(Q.along),
                         'share_along_sq': float(np.sum(Q.along ** 2) / sum_sq * 100)})
    G = pd.DataFrame(seg_rows)
    G.to_csv(HERE / f'check_segments{a.tag}.csv', index=False)
    pr(G.round(3).to_string(index=False))
    # the "k error" view: drift_pct ~ (k_true - k_est) * 100
    long = G[G.path_m > 300]
    if len(long):
        w = long.path_m / long.path_m.sum()
        pr(f'  path-weighted drift over segments > 300 m: {np.sum(w * long.drift_pct):+.3f} %  '
           f'(k_est mean {np.sum(w * long.k_mean):+.4f}) -> implied k_true ~ {np.sum(w * (long.k_mean + long.drift_pct / 100)):+.4f}')
    # k trajectory
    kk = O.set_index('t').k
    pr('  k estimate: ' + ', '.join(f't={t:.0f}: {kk.iloc[np.searchsorted(kk.index, t)]:+.4f}'
                                  for t in (60, 90, 150, 200, 270, 400, 580, 590, 720, 850, 1000, 1200)))

    # ------------------------------------------------------------------ 5. slip / slide / model-only
    pr('=' * 100)
    pr('5. SLIP / SLIDE / MODEL-ONLY EVENTS')
    fo = O['flags'].to_numpy().astype(np.int64)
    for name, bits in (('slip/slide bits0-3', SLIP_BITS), ('model-only bit11', MODEL_ONLY), ('maneuver bit17', MANEUVER)):
        ints = intervals((fo & bits) > 0, ot)
        ints = [(x, y) for x, y in ints if x < a.end_s]
        dur = sum(y - x for x, y in ints)
        fp = (P['flags'].astype(np.int64) & bits) > 0
        fv = (V['flags'].astype(np.int64) & bits) > 0
        dal = []
        for x, y in ints:
            dal.append(along_at(y + 0.2, y + 1.2) - along_at(x - 1.0, x))
        dal = np.array([q for q in dal if np.isfinite(q)])
        pr(f'  {name}: {len(ints)} intervals, {dur:.1f} s total; speed MSE share of flagged pairs '
           f'{np.sum(V.ev[fv] ** 2) / np.sum(V.ev ** 2) * 100:.1f} % ({fv.mean() * 100:.2f} % of pairs); '
           f'along jump across intervals: sum {dal.sum() if len(dal) else 0:+.3f} m, max |.| {np.abs(dal).max() if len(dal) else 0:.3f} m')
        for x, y in ints[:12]:
            pr(f'      {x:8.2f}-{y:8.2f} s ({y - x:5.2f} s)  v {np.interp(x, ot, O.v):5.2f}  '
               f'along {along_at(x - 1.0, x):+.3f} -> {along_at(y + 0.2, y + 1.2):+.3f}')

    # ------------------------------------------------------------------ 6. z
    pr('=' * 100)
    pr('6. HEIGHT')
    pr(f'  dz bias {P.dz.mean():+.3f} m, std {P.dz.std():.3f}, RMSE {rmse(P.dz):.3f}; removing a constant bias -> '
       f'z RMSE {P.dz.std():.3f}, 3-D RMSE {np.sqrt(mse - P.dz.mean() ** 2):.3f}')
    P['sbin'] = (P.s_map // 500) * 500
    zz = P[P.s_map >= 0].groupby('sbin').dz.agg(['mean', 'std', 'count'])
    pr('  dz by s_map (500 m bins): ' + ', '.join(f'{int(i)}: {r["mean"]:+.2f}' for i, r in zz.iterrows()))

    # ------------------------------------------------------------------ 7. speed
    pr('=' * 100)
    pr('7. SPEED')
    ev = V.ev.to_numpy()
    tot = np.sum(ev ** 2)
    acc_r = V.acc_ref.to_numpy()
    vr = V.v_ref.to_numpy()
    reg = np.where((vr < 0.05) & (np.abs(acc_r) < 0.05), 'standstill',
                   np.where(acc_r > 0.15, 'accel', np.where(acc_r < -0.15, 'brake', 'cruise')))
    V['regime'] = reg
    rows = []
    for dl in (0.0, 0.1):
        e2 = np.interp(tv - dl, ts, vo) - vr
        for g in ('standstill', 'accel', 'cruise', 'brake'):
            k = reg == g
            rows.append({'delay_s': dl, 'regime': g, 'share_pairs': k.mean() * 100, 'rmse': rmse(e2[k]),
                         'bias': float(e2[k].mean()), 'share_mse': np.sum(e2[k] ** 2) / np.sum(e2 ** 2) * 100,
                         'mse_contrib': np.sum(e2[k] ** 2) / len(e2)})
    R = pd.DataFrame(rows)
    R.to_csv(HERE / f'check_speed_regimes{a.tag}.csv', index=False)
    pr(R.round(4).to_string(index=False))
    # biggest 5-s windows of speed error
    V['w'] = (V.t // 5) * 5
    W = V.groupby('w').agg(sse=('ev', lambda x: np.sum(x ** 2)), rmse=('ev', rmse), v=('v_ref', 'mean'),
                           acc=('acc_ref', lambda x: x.abs().max()),
                           fl=('flags', lambda f: int(np.bitwise_or.reduce(f.astype(np.int64)))))
    W['share'] = W.sse / tot * 100
    W = W.sort_values('sse', ascending=False)
    pr('  top 12 5-s windows by speed squared error (share of total, flags OR):')
    pr(W.head(12).round(3).to_string())
    pr(f'  top 5 % of 5-s windows carry {W.share.head(int(len(W) * 0.05)).sum():.1f} % of the speed MSE')
    # speed error vs acceleration: residual after the 0.1 s delay
    e2 = np.interp(tv - 0.1, ts, vo) - vr
    for lo, hi in ((-9, -0.8), (-0.8, -0.3), (-0.3, -0.05), (-0.05, 0.05), (0.05, 0.3), (0.3, 0.8), (0.8, 9)):
        k = (acc_r >= lo) & (acc_r < hi) & (vr > 0.05)
        if k.sum():
            pr(f'    acc_ref {lo:+5.2f}..{hi:+5.2f}: n={k.sum():6d} RMSE raw {rmse(ev[k]):.4f} bias {ev[k].mean():+.4f} | '
               f'delayed 0.1 s RMSE {rmse(e2[k]):.4f} bias {e2[k].mean():+.4f}')
    kst = reg == 'standstill'
    pr(f'  standstill pairs: our |v| max {np.abs(V.v[kst]).max():.4f}, ref |v| max {np.abs(vr[kst]).max():.4f}, '
       f'RMSE {rmse(ev[kst]):.4f}')


if __name__ == '__main__':
    main()
