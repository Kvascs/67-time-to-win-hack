"""Export the dead-end arrival track west_arrival_2 as a "stub" map for the estimator.

The stub leaves the main cycle at the western terminal and ends at a buffer ~120 m later; some runs end
on it (the organisers' check bag 30618_88aea4d9 does). Its first point lies on the main line; the header
carries join_s = main-cycle arc of that point (for a stub it is where it starts, not where it merges).

    python tools/map/export_stub.py --map-dir analysis/map_build/map        # package (all data)
    python tools/map/export_stub.py --map-dir analysis/map_build/map_train --out analysis/validation_maps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_core_map import write_edge  # noqa: E402

PKG_MAPS = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map-dir', default=str(ROOT / 'analysis' / 'map_build' / 'map'))
    ap.add_argument('--out', default=str(PKG_MAPS))
    ap.add_argument('--main', default='', help='main-cycle csv of the same map (default: <out>/track_map.csv)')
    ap.add_argument('--step', type=float, default=1.0)
    a = ap.parse_args()
    stub = pd.read_csv(Path(a.map_dir) / 'edge_west_arrival_2.csv', comment='#')
    main_csv = Path(a.main) if a.main else Path(a.out) / 'track_map.csv'
    main = pd.read_csv(main_csv, comment='#')
    # main-cycle arc of the stub's first point (projection onto the main polyline)
    p = stub[['x', 'y']].to_numpy()[0]
    xy = main[['x', 'y']].to_numpy()
    seg = xy[1:] - xy[:-1]
    t = np.clip(((p - xy[:-1]) * seg).sum(1) / np.maximum((seg ** 2).sum(1), 1e-9), 0, 1)
    proj = xy[:-1] + seg * t[:, None]
    i = int(np.argmin(np.hypot(*(proj - p).T)))
    join_s = main.s.to_numpy()[i] + t[i] * (main.s.to_numpy()[i + 1] - main.s.to_numpy()[i])
    off = float(np.hypot(*(proj[i] - p)))
    print(f'stub start is {off:.2f} m from the main line at main s = {join_s:.2f}')
    write_edge(stub, Path(a.out) / 'stub_west_arrival_2.csv', a.step, f'cyclic=0 join_s={join_s:.3f}',
               'dead-end stub west_arrival_2 leaving the main cycle at join_s')


if __name__ == '__main__':
    main()
