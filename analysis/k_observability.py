"""Is the wheel scale k observable from wheels + controller alone (teammate idea: force -> power regime)?

Traction model: constant force at low speed, constant power at high speed. An estimator that only sees the
wheel speed v_w = (1+k) v fits a gain G to the wheel acceleration a_w = G * a*(n, v_w):
  constant force  (a* independent of v):  G_low  = (1+k) g
  constant power  (a* = C / v):            G_high = (1+k)^2 g     ->  G_high / G_low = 1 + k
so in theory k follows from the two regimes. This script measures that ratio per run (steady traction
notch >= 8 held >= 1.5 s, grade removed with the map term a_ext from the replay) and compares it with the
GNSS truth k (analysis/cross_check/scale_lag_per_bag.csv: k_mean = km/h per m/s; 1+k = k_mean / 3.6 / 1.00037).
Result 26.09: see docs/TEAMMATE_FEEDBACK.md (k not practically observable: ratio scatter >> +-1.6 % range).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge  # noqa: E402

MAPS = ROOT / 'analysis' / 'validation_maps'
EXE = ROOT / 'build_core' / 'tbo_replay_fix5.exe'  # frozen snapshot


def lut():
    t = pd.read_csv(cpp_bridge.PKG / 'config' / 'traction_lut.csv', comment='#')
    vs = np.array([float(c) for c in t.columns[1:]])
    return t.notch.to_numpy().astype(int), vs, t.iloc[:, 1:].to_numpy()


NOTCH, VS, TAB = lut()


def a_star(n, v):
    row = TAB[np.searchsorted(NOTCH, n)]
    return np.interp(v, VS, row)


def per_bag(bag):
    cpp_bridge.REPLAY_EXE = EXE
    tmp = ROOT / 'build_core' / 'replay_tmp' / 'kobs'
    tmp.mkdir(parents=True, exist_ok=True)
    ev, out = tmp / f'{bag}_ev.csv', tmp / f'{bag}_out.csv'
    cpp_bridge.export_events(bag, ev)
    br = ','.join(str(b) for b in sorted(MAPS.glob('branch_*.csv')))
    o = cpp_bridge.run_replay(ev, out, map_csv=MAPS / 'track_map.csv',
                              traction_csv=cpp_bridge.PKG / 'config' / 'traction_lut.csv',
                              sets={'output_frame': 'enu'}, branches=br).drop_duplicates('stamp_ns')
    d = np.load(cpp_bridge.NPZ / f'{bag}.npz')
    fr, rr, cm = (d[k] for k in ('vehicle__front_bogie_velocity', 'vehicle__rear_bogie_velocity',
                                  'vehicle__driver_position_cmd'))
    t0 = max(fr[0, 1], rr[0, 1], cm[0, 1]) + 8.0
    grid = np.arange(t0, min(fr[-1, 1], rr[-1, 1], cm[-1, 1]), 0.1)
    vw = 0.5 * (np.interp(grid, fr[:, 1], fr[:, 2]) + np.interp(grid, rr[:, 1], rr[:, 2])) * 1.00037 / 3.6
    n = cm[np.clip(np.searchsorted(cm[:, 1], grid, side='right') - 1, 0, len(cm) - 1), 2].astype(int)
    ot = o.stamp_ns.to_numpy() * 1e-9
    aext = np.interp(grid, ot, o.a_ext.to_numpy())
    aw = np.gradient(vw, grid)
    rows = []
    # steady traction segments: same notch >= 8 for >= 1.5 s, use the part after 1 s (drive lag settled)
    i = 0
    while i < len(grid):
        j = i
        while j + 1 < len(grid) and n[j + 1] == n[i]:
            j += 1
        if n[i] >= 8 and grid[j] - grid[i] >= 1.5:
            m = slice(i + 10, j + 1)
            v = vw[m]
            if v.min() > 0.5:
                rows.append({'v': float(v.mean()), 'aw': float(np.mean(aw[m] - aext[m])),
                             'astar': float(np.mean(a_star(n[i], v))), 'dur': float(grid[j] - grid[i])})
        i = j + 1
    seg = pd.DataFrame(rows)
    if seg.empty:
        return None
    lo, hi = seg[seg.v < 5.0], seg[seg.v > 8.0]
    if len(lo) < 3 or len(hi) < 3:
        return None
    g_lo = float(np.sum(lo.aw * lo.astar) / np.sum(lo.astar ** 2))  # least squares gain per regime
    g_hi = float(np.sum(hi.aw * hi.astar) / np.sum(hi.astar ** 2))
    return {'bag': bag, 'n_lo': len(lo), 'n_hi': len(hi), 'G_low': g_lo, 'G_high': g_hi, 'ratio': g_hi / g_lo}


def main():
    splits = json.load(open(ROOT / 'data' / 'splits.json'))
    bags = splits['train'] + splits['val']
    with ProcessPoolExecutor(2) as ex:
        res = [r for r in ex.map(per_bag, bags) if r]
    df = pd.DataFrame(res)
    truth = pd.read_csv(ROOT / 'analysis' / 'cross_check' / 'scale_lag_per_bag.csv').set_index('bag')
    df['k_true'] = df.bag.map(truth.k_mean) / 3.6 / 1.00037 - 1.0
    df['k_est'] = df.ratio - 1.0
    df = df.dropna()
    c = np.corrcoef(df.k_true, df.k_est)[0, 1]
    err = df.k_est - df.k_true
    out = ROOT / 'analysis' / 'k_observability.csv'
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f'\n{len(df)} runs: k_true std {df.k_true.std():.4f} (range {df.k_true.min():.4f}..{df.k_true.max():.4f}); '
          f'k_est std {df.k_est.std():.4f}; corr {c:.2f}; error median |.| {np.median(np.abs(err)):.4f}, '
          f'rmse {np.sqrt(np.mean(err ** 2)):.4f}')


if __name__ == '__main__':
    main()
