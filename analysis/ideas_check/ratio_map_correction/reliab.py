"""Step 4: leave-one-out reliability map over TRAIN, then the gated correction evaluated on VAL.

TRAIN: for every run a ratio map from the OTHER train runs; the matcher every STEP m on the run's own estimator arc.
A trigger FAILS when an accepted correction would move the estimate the wrong way by more than FAIL m
(|e_corr| > |e_filt| + FAIL) while the truth is plausible (|e_filt| < 3 m: larger ones are mostly reference errors).
Unreliable = CELL-m cells of the trigger arc with a failure, widened by +-WIDEN cells.
VAL: train map + that reliability map + a Mahalanobis gate |d| < GATE_SD * sqrt(s_var + sig_c^2) + |d| < DMAX.
Output: reliab_cells.csv (cell, n, fails, unreliable), reliab_val.csv, reliab.txt
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]
import rm_common as M  # noqa: E402

STEP, W = 50.0, 200.0
GRID = np.arange(-8.0, 8.0001, 0.1)
MARGIN, SIGMAX = 3.0, 0.8
FAIL = 0.5
CELL, WIDEN = 25.0, 1
GATE_SD, DMAX = 3.0, 3.0


def parts():
    z = np.load(M.HERE / 'ratio_map_parts.npz')
    return z, list(z['bags']), (z['sv_v'], z['sv_s']), float(z['L'])


def make_map(z, sel, sv, L):
    return M.RatioMap(z['n'][sel].sum(0), z['s1'][sel].sum(0), z['s2'][sel].sum(0), L, sv)


def triggers(rm, df):
    ok = (df.est_ok & np.isfinite(df.s_est) & np.isfinite(df.y) & (df.v > M.VMIN)
          & ((df['flags'].astype(np.int64) & M.BAD_FLAGS) == 0) & (df.y.abs() < 0.05)).to_numpy()
    s, y, v, st, sv_ = df.s_est.to_numpy(), df.y.to_numpy(), df.v.to_numpy(), df.s_true.to_numpy(), df.s_var.to_numpy()
    idx = np.flatnonzero(ok)
    out, last = [], -np.inf
    for i in idx:
        if s[i] - last < STEP:
            continue
        last = s[i]
        w = idx[(s[idx] >= s[i] - W) & (idx <= i)]
        if len(w) < 100 or not np.isfinite(st[i]):
            continue
        ll, _ = M.match(rm, y[w], M.sigma_v(v[w], rm.sv_tab), s[w] - s[i], s[i], GRID)
        d_hat, sig_c, sig_p, margin, margin_lm, edge = M.peak_stats(GRID, ll)
        out.append({'s': s[i], 's_mod': np.mod(s[i], rm.L), 'e_filt': s[i] - st[i], 'e_corr': s[i] + d_hat - st[i],
                    'd_hat': d_hat, 'sig_c': sig_c, 'margin_lm': margin_lm, 'edge': bool(edge), 's_var': sv_[i],
                    'base': bool(margin_lm > MARGIN and sig_c < SIGMAX and not edge)})
    return out


def main():
    z, bags, sv, L = parts()
    split = np.array(z['split'])
    train = [b for b, sp in zip(bags, split) if sp == 'train']
    lines = []
    # ---- leave-one-out over train
    rows = []
    for b in train:
        p = M.HERE / 'prep_train' / f'{b}.pkl'
        if not p.exists():
            continue
        sel = (split == 'train') & (np.array(bags) != b)
        for r in triggers(make_map(z, sel, sv, L), pd.read_pickle(p)):
            r['bag'] = b
            rows.append(r)
    T = pd.DataFrame(rows)
    plaus = T.e_filt.abs() < 3.0
    T['fail'] = T.base & plaus & (T.e_corr.abs() > T.e_filt.abs() + FAIL)
    nc = int(np.ceil(L / CELL))
    cell = (T.s_mod // CELL).astype(int) % nc
    n = np.bincount(cell, minlength=nc)
    f = np.bincount(cell, weights=T.fail.astype(float), minlength=nc)
    bad = f > 0
    unrel = bad.copy()
    for k in range(1, WIDEN + 1):
        unrel |= np.roll(bad, k) | np.roll(bad, -k)
    pd.DataFrame({'cell_start_m': np.arange(nc) * CELL, 'n': n, 'fails': f.astype(int), 'unreliable': unrel}).to_csv(
        M.HERE / 'reliab_cells.csv', index=False)
    lines.append(f'TRAIN leave-one-out: {len(T)} triggers in {T.bag.nunique()} runs, accepted {T.base.mean() * 100:.0f} %, '
                 f'fails {int(T.fail.sum())}; unreliable cells {int(unrel.sum())} of {nc} ({unrel.mean() * 100:.0f} % of the cycle); '
                 f'cells with failures {int(bad.sum())}')

    def evaluate(D, name):
        cellv = (D.s_mod // CELL).astype(int) % nc
        rel = ~unrel[cellv]
        maha = D.d_hat.abs() < GATE_SD * np.sqrt(np.maximum(D.s_var, 0) + D.sig_c ** 2)
        for label, acc in (('base gate', D.base), ('+ reliability', D.base & rel),
                           ('+ reliability + Mahalanobis + |d|<3', D.base & rel & maha & (D.d_hat.abs() < DMAX))):
            after = np.where(acc, D.e_corr, D.e_filt)
            pl = D.e_filt.abs() < 3.0
            worse = int((acc & pl & (D.e_corr.abs() > D.e_filt.abs() + FAIL)).sum())
            badc = int((acc & pl & (D.e_corr.abs() > 1.5)).sum())
            lines.append(f'{name} {label:36s}: accepted {acc.mean() * 100:5.1f} %, made worse > {FAIL} m {worse:3d}, '
                         f'> 1.5 m {badc:3d} | RMS all {np.sqrt(np.mean(D.e_filt ** 2)):.3f} -> {np.sqrt(np.mean(after ** 2)):.3f}, '
                         f'plausible-truth {np.sqrt(np.mean(D.e_filt[pl] ** 2)):.3f} -> {np.sqrt(np.mean(after[pl] ** 2)):.3f}')
        return D.base & rel & maha & (D.d_hat.abs() < DMAX)

    evaluate(T, 'TRAIN(LOO)')
    # ---- val with the full train map
    rm = make_map(z, split == 'train', sv, L)
    rows = []
    for p in sorted((M.HERE / 'prep_so').glob('*.pkl')):
        if p.stem.endswith('_fix'):
            continue
        for r in triggers(rm, pd.read_pickle(p)):
            r['bag'] = p.stem
            rows.append(r)
    V = pd.DataFrame(rows)
    acc = evaluate(V, 'VAL')
    V['accepted'] = acc
    V.to_csv(M.HERE / 'reliab_val.csv', index=False)
    for b, g in V.groupby('bag'):
        after = np.where(g.accepted, g.e_corr, g.e_filt)
        lines.append(f'  {b}: triggers {len(g)}, accepted {int(g.accepted.sum())}, RMS {np.sqrt(np.mean(g.e_filt ** 2)):.2f} -> '
                     f'{np.sqrt(np.mean(after ** 2)):.2f}')
    txt = '\n'.join(lines)
    print(txt)
    (M.HERE / 'reliab.txt').write_text(txt, encoding='utf-8')


if __name__ == '__main__':
    main()
