"""Where does the along-track error grow? Rate of change of the along-track error per metre of map arc,
binned by place on the main cycle, from GNSS-anchored replays against RTK master fixes.

A place-dependent odometry scale error (wheel distance vs map arc: curves, loops, rail geometry) shows up
as a consistent slope in the same bins across runs. Learned on one split it can correct the odometry of
another (field c(s): map metres per wheel metre - 1).

    python analysis/odometry_scale_field.py --split train --out analysis/validation_maps/cfield.csv
    python analysis/odometry_scale_field.py --split val --check analysis/validation_maps/cfield.csv
"""
from __future__ import annotations

import argparse
import json

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge  # noqa: E402
import quick_eval  # noqa: E402

VAL = ROOT / 'analysis' / 'validation_maps'
BIN = 25.0


def per_bag(args):
    bag, exe, sets = args
    cpp_bridge.REPLAY_EXE = Path(exe)
    tmp = ROOT / 'build_core' / 'replay_tmp' / 'cfield'
    tmp.mkdir(parents=True, exist_ok=True)
    ev, out = tmp / f'{bag}_ev.csv', tmp / f'{bag}_out.csv'
    cpp_bridge.export_events(bag, ev)
    br = ','.join(str(b) for b in sorted(VAL.glob('branch_*.csv')))
    s = {'output_frame': 'enu', 'base_link_along_m': 0, 'base_link_height_m': 0,
         'landmark_file': str(VAL / 'landmarks.csv'), 'cutoff_file': str(VAL / 'cutoffs.csv'),
         'dfield_file': str(VAL / 'dfield.csv'), **sets}
    o = cpp_bridge.run_replay(ev, out, map_csv=VAL / 'track_map.csv',
                              traction_csv=cpp_bridge.PKG / 'config' / 'traction_lut.csv', sets=s, branches=br)
    o = o[o.pos_valid == 1].drop_duplicates('stamp_ns').sort_values('stamp_ns')
    d = np.load(cpp_bridge.NPZ / f'{bag}.npz')
    mf = d['sensing__gnss__master__fix']
    mf = mf[np.isfinite(mf[:, 2])]
    la, lo, h0 = mf[0, 2], mf[0, 3], mf[0, 4]  # the replay's ENU origin: first master fix of the bag
    mf = mf[mf[:, 5] == 2]
    if len(mf) < 500 or o.empty:
        return None
    p = quick_eval.enu(mf[:, 2], mf[:, 3], mf[:, 4], la, lo, h0)
    ot = o.stamp_ns.to_numpy() * 1e-9
    i = np.clip(np.searchsorted(ot, mf[:, 1]), 1, len(ot) - 1)
    ok = np.abs(ot[i] - mf[:, 1]) < 0.05
    x, y, yaw, sm, v = (o[c].to_numpy()[i[ok]] for c in ('x', 'y', 'yaw', 's_map', 'v'))
    e = np.c_[x - p[ok, 0], y - p[ok, 1]]
    along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
    fl = o['flags'].to_numpy()[i[ok]]
    rows = []
    # consecutive moving epochs on the main cycle without a landmark correction in between
    for k in range(1, len(along)):
        if sm[k] < 0 or sm[k - 1] < 0 or v[k] < 2.0:
            continue
        ds = sm[k] - sm[k - 1]
        if not (0.2 < ds < 5.0):
            continue
        if (fl[k] >> 19) & 1 or (fl[k - 1] >> 19) & 1:
            continue
        rows.append((0.5 * (sm[k] + sm[k - 1]), ds, along[k] - along[k - 1]))
    return pd.DataFrame(rows, columns=['s', 'ds', 'de']).assign(bag=bag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--exe', default=str(ROOT / 'build_core' / 'tbo_replay.exe'))
    ap.add_argument('--out', default='')
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    sets = dict(s.split('=', 1) for s in a.set)
    bags = json.load(open(ROOT / 'data' / 'splits.json'))[a.split]
    with ProcessPoolExecutor(a.jobs) as ex:
        parts = [r for r in ex.map(per_bag, [(b, a.exe, sets) for b in bags]) if r is not None and len(r)]
    df = pd.concat(parts, ignore_index=True)
    df['b'] = (df.s // BIN).astype(int)
    # per bin: sum of error growth / sum of distance = odometry scale error there (estimate - truth)
    g = df.groupby('b').agg(ds=('ds', 'sum'), de=('de', 'sum'), runs=('bag', 'nunique'))
    g['rate'] = g.de / g.ds
    g = g[g.runs >= 3]
    tot = df.de.sum() / df.ds.sum()
    print(f'{a.split}: {len(parts)} runs, overall error growth {100 * tot:+.3f} % of distance; '
          f'bins with |rate| > 1 %: {(g.rate.abs() > 0.01).sum()} of {len(g)}')
    worst = g.reindex(g.rate.abs().sort_values(ascending=False).index).head(12)
    print((worst.assign(s0=worst.index * BIN, rate_pct=100 * worst.rate)[['s0', 'runs', 'ds', 'rate_pct']]).round(2).to_string())
    if a.out:
        with open(a.out, 'w', newline='\n') as fh:
            fh.write(f'# odometry scale error by place (estimate - truth per metre), {BIN:g} m bins, {a.split} runs\n')
            fh.write('s,rate,runs\n')
            for b, r in g.iterrows():
                fh.write(f'{(b + 0.5) * BIN:.1f},{r.rate:.5f},{int(r.runs)}\n')
        print('wrote', a.out)


if __name__ == '__main__':
    main()
