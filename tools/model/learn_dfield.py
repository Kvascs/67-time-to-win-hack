"""Learn the disturbance field d(s): the mean acceleration the traction + grade model does not explain,
as a function of the place on the main map cycle.

For every run (C++ replay, GNSS-anchored, nominal mode) the residual
    r(t) = dv/dt - (g * a_drive + a_ext)          (a_drive: notch model, a_ext: grade/curve map terms)
is binned by the main-cycle arc length s_map. The field is the per-bin median over runs (bins seen in
fewer than --min-runs runs are 0), lightly smoothed. It captures map-grade errors, curve resistance and
place-typical behaviour; the filter's own random-walk d then only tracks what is left.

    python tools/model/learn_dfield.py --split train --maps analysis/validation_maps --out analysis/validation_maps/dfield.csv
    python tools/model/learn_dfield.py --split all --maps ros2_ws/src/tram_backup_odometry/maps --out ros2_ws/src/tram_backup_odometry/maps/dfield.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge  # noqa: E402

FLAG_SLIP_ANY = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)  # front/rear slip/slide bits (types.hpp)


def residuals(bag: str, maps: Path, exe: str, tmp: Path, extra_sets: dict) -> pd.DataFrame:
    cpp_bridge.REPLAY_EXE = Path(exe)
    tmp.mkdir(parents=True, exist_ok=True)
    ev, out = tmp / f'{bag}_ev.csv', tmp / f'{bag}_out.csv'
    cpp_bridge.export_events(bag, ev)
    branches = ','.join(str(b) for b in sorted(maps.glob('branch_*.csv')))
    sets = {'output_frame': 'enu', 'landmark_file': str(maps / 'landmarks.csv'),
            'cutoff_file': str(maps / 'cutoffs.csv'), **extra_sets}
    o = cpp_bridge.run_replay(ev, out, map_csv=maps / 'track_map.csv',
                              traction_csv=cpp_bridge.PKG / 'config' / 'traction_lut.csv', sets=sets,
                              branches=branches)
    o = o.drop_duplicates('stamp_ns').sort_values('stamp_ns')
    t = o.stamp_ns.to_numpy() * 1e-9
    grid = np.arange(t[0], t[-1], 0.1)
    v = np.interp(grid, t, o.v.to_numpy())
    dvdt = np.full_like(v, np.nan)
    k = 5  # +-0.5 s central difference
    dvdt[k:-k] = (v[2 * k:] - v[:-2 * k]) / (grid[2 * k:] - grid[:-2 * k])
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    g = o.iloc[idx]
    model = (g.accel.to_numpy() - g.d.to_numpy()) + g.a_ext.to_numpy()  # g*a_drive + map terms
    r = dvdt - model
    keep = ((v > 1.0) & (g.mu0.to_numpy() > 0.9) & (g.s_map.to_numpy() >= 0) & np.isfinite(r) &
            ((g['flags'].to_numpy().astype(np.int64) & FLAG_SLIP_ANY) == 0) & (np.abs(r) < 2.0))
    return pd.DataFrame({'bag': bag, 's_map': g.s_map.to_numpy()[keep], 'r': r[keep], 'v': v[keep]})


def _job(a):
    try:
        return residuals(*a)
    except Exception as e:  # report, keep going
        print('FAILED', a[0], str(e)[:200], flush=True)
        return None


def build_field(df: pd.DataFrame, length: float, bin_m: float, min_runs: int) -> pd.DataFrame:
    nb = int(np.ceil(length / bin_m))
    b = np.clip((df.s_map.to_numpy() / bin_m).astype(int), 0, nb - 1)
    df = df.assign(b=b)
    per_run = df.groupby(['b', 'bag']).r.median().reset_index()  # one vote per run and bin
    agg = per_run.groupby('b').r.agg(['median', 'count'])
    field = np.zeros(nb)
    ok = agg['count'] >= min_runs
    field[agg.index[ok]] = agg['median'][ok]
    sm = (np.roll(field, 1) + 2 * field + np.roll(field, -1)) / 4.0  # light smoothing on the cycle
    runs = np.zeros(nb, dtype=int)
    runs[agg.index] = agg['count'].to_numpy()
    return pd.DataFrame({'s': (np.arange(nb) + 0.5) * bin_m, 'd': sm, 'runs': runs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--maps', default=str(ROOT / 'analysis' / 'validation_maps'))
    ap.add_argument('--exe', default=str(cpp_bridge.REPLAY_EXE))
    ap.add_argument('--out', required=True)
    ap.add_argument('--bin', type=float, default=10.0)
    ap.add_argument('--min-runs', type=int, default=3)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--check-split', default='', help='also report explained variance on this split')
    a = ap.parse_args()
    maps = Path(a.maps)
    splits = json.load(open(ROOT / 'data' / 'splits.json'))
    bags = splits['train'] + splits['val'] if a.split == 'all' else splits[a.split]
    tmp = ROOT / 'build_core' / 'replay_tmp' / 'dfield'
    with ProcessPoolExecutor(a.jobs) as ex:
        parts = [p for p in ex.map(_job, [(b, maps, a.exe, tmp, {}) for b in bags]) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    length = float(pd.read_csv(maps / 'track_map.csv', comment='#').s.iloc[-1])
    field = build_field(df, length, a.bin, a.min_runs)
    with open(a.out, 'w', newline='\n') as fh:
        fh.write(f'# learned disturbance field on the main cycle: median unexplained accel (m/s^2) per '
                 f'{a.bin:g} m bin over {len(parts)} runs ({a.split}); min {a.min_runs} runs per bin\n')
        fh.write('s,d,runs\n')
        for r in field.itertuples(index=False):
            fh.write(f'{r.s:.1f},{r.d:.5f},{r.runs}\n')
    fd = np.interp(df.s_map, field.s, field.d, period=length)
    print(f'{a.split}: {len(parts)} runs, {len(df)} samples; residual std {df.r.std():.4f} -> '
          f'{(df.r - fd).std():.4f} m/s^2 (in-sample); field |d| p50/p95 '
          f'{np.percentile(np.abs(field.d), 50):.3f}/{np.percentile(np.abs(field.d), 95):.3f}; '
          f'bins with data {(field.runs >= a.min_runs).mean():.2%}')
    if a.check_split:
        with ProcessPoolExecutor(a.jobs) as ex:
            chk = [p for p in ex.map(_job, [(b, maps, a.exe, tmp, {}) for b in splits[a.check_split]])
                   if p is not None]
        dc = pd.concat(chk, ignore_index=True)
        fc = np.interp(dc.s_map, field.s, field.d, period=length)
        print(f'{a.check_split} (out-of-sample): residual std {dc.r.std():.4f} -> {(dc.r - fc).std():.4f} m/s^2')


if __name__ == '__main__':
    main()
