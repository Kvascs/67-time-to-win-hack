"""Error budget: along-track error picked up while passing tight curves (|curvature| > 0.012 1/m).

For every pass through a tight-curve zone of the map (moving forward on the main cycle, no accepted place fix
and no reference fault inside), the change of the along error between 10 m before the zone and 10 m after it.
A systematic negative change = our distance falls behind in the curve (wheel under-reading not fully corrected).
Uses rtk_epochs/*.parquet (+ their _lm.csv) for the RTK bags (train-only map) and pairs_pos.parquet for the
check bag (package map).

    python analysis/ideas_check/error_budget/curve_passes.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def zones(map_csv, thr=0.012, min_len=5.0):
    M = pd.read_csv(map_csv, comment='#')
    k = np.abs(M.curvature.to_numpy()) > thr
    m = np.r_[False, k, False]
    d = np.diff(m.astype(int))
    out = []
    for i, j in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
        if M.s[j] - M.s[i] > min_len:
            seg = M.iloc[i:j + 1]
            out.append((float(M.s[i]), float(M.s[j]), float(np.abs(seg.curvature).max()),
                        float(np.trapezoid(np.abs(seg.curvature), seg.s)),
                        float(np.trapezoid(seg.curvature ** 2, seg.s))))
    return out


def passes(bag, t, sm, along, fix_t, bad, Z, margin=10.0):
    rows = []
    for (a, b, kmax, ik, ik2) in Z:
        lo, hi = a - margin, b + margin
        inside = (sm >= lo) & (sm <= hi)
        if not inside.any():
            continue
        m = np.r_[False, inside, False]
        d = np.diff(m.astype(int))
        for i, j in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
            if sm[i] > lo + 3.0 or sm[j] < hi - 3.0 or sm[j] <= sm[i]:
                continue  # started or ended inside the zone / not a forward pass
            if t[j] - t[i] > 300.0:
                continue
            if np.any((fix_t >= t[i] - 1.0) & (fix_t <= t[j] + 2.0)) or bad[i:j + 1].any():
                continue
            a0 = float(np.median(along[max(i - 5, 0):i + 5]))
            a1 = float(np.median(along[max(j - 5, 0):j + 5]))
            rows.append({'bag': bag, 'zone': f'{a:.0f}-{b:.0f}', 'len': b - a, 'kmax': kmax, 'int_abs_k': ik,
                         'int_k2': ik2, 't0': t[i], 'dt': t[j] - t[i], 'd_along': a1 - a0})
    return rows


def main():
    rows = []
    Zv = zones(ROOT / 'analysis' / 'validation_maps' / 'track_map.csv')
    summ = pd.read_csv(HERE / 'rtk_summary.csv')
    summ = summ[summ['error'].isna()]
    for bag in summ.bag:
        E = pd.read_parquet(HERE / 'rtk_epochs' / f'{bag}.parquet').sort_values('t')
        lm = pd.read_csv(HERE / 'rtk_epochs' / f'{bag}_lm.csv')
        fix_t = lm[(lm.p_known >= 0.6) & (lm.n > 0)].t.to_numpy()
        bad = (np.abs(E.ref_lat.to_numpy()) > 0.3) | (np.abs(E.cross.to_numpy()) > 1.5)
        rows += passes(bag, E.t.to_numpy(), E.s_map.to_numpy(), E.along.to_numpy(), fix_t, bad, Zv)
    R = pd.DataFrame(rows)
    # check bag (package map, judge reference)
    Zp = zones(ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps' / 'track_map.csv')
    P = pd.read_parquet(HERE / 'pairs_pos.parquet')
    P = P[P.t < 1270]
    lm = pd.read_csv(HERE / 'lm_check.csv')
    fix_t = lm[(lm.p_known >= 0.6) & (lm.n > 0)].t.to_numpy()
    C = pd.DataFrame(passes('CHECK_88aea4d9', P.t.to_numpy(), P.s_map.to_numpy(), P.along.to_numpy(), fix_t,
                            np.zeros(len(P), bool), Zp))
    R = pd.concat([R, C], ignore_index=True)
    R.to_csv(HERE / 'curve_passes.csv', index=False)
    pd.set_option('display.width', 200)
    g = R[R.bag != 'CHECK_88aea4d9'].groupby('zone').agg(n=('d_along', 'size'), len=('len', 'first'), kmax=('kmax', 'first'),
                                                        int_k2=('int_k2', 'first'), med=('d_along', 'median'),
                                                        mean=('d_along', 'mean'), p25=('d_along', lambda x: np.percentile(x, 25)),
                                                        p75=('d_along', lambda x: np.percentile(x, 75)))
    g = g.sort_values('kmax', ascending=False)
    print('RTK bags (train-only map): change of the along error across each tight-curve zone (m)')
    print(g.round(3).to_string())
    x = R[R.bag != 'CHECK_88aea4d9']
    c = np.polyfit(x.int_k2, x.d_along, 1)
    print(f'  pooled fit d_along = {c[1]:+.3f} + {c[0]:+.2f} * int(k^2 ds)   (n={len(x)})')
    print('CHECK bag (package map, judge reference):')
    print(C.round(3).to_string(index=False) if len(C) else '  none')


if __name__ == '__main__':
    main()
