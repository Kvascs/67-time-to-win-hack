"""Export the track map built by analysis/map_build into the estimator's CSV format.

Input : <map_dir>/edge_*.csv + track_map.json (map ENU frame anchored at data_io.MAP_ORIGIN).
Output: ros2_ws/src/tram_backup_odometry/maps/track_map.csv          main closed cycle
        ros2_ws/src/tram_backup_odometry/maps/branch_<id>.csv        branches that merge into main
Header comments carry origin_lat/lon/h, cyclic and (branches) join_s = main s at the merge.
Columns: s,x,y,z,grade,curvature
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis' / 'map_build'))
import data_io as D  # noqa: E402  (MAP_ORIGIN lives there)

OUT_DIR = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps'


def write_edge(df: pd.DataFrame, path: Path, step: float, meta: str, title: str):
    stride = max(1, int(round(step / float(df.s.iloc[1] - df.s.iloc[0]))))
    df = df.iloc[::stride]
    lat0, lon0, h0 = D.MAP_ORIGIN
    with open(path, 'w', newline='\n') as fh:
        fh.write(f'# {title}, map ENU frame\n')
        fh.write(f'# origin_lat={lat0:.9f} origin_lon={lon0:.9f} origin_h={h0:.4f} {meta}\n')
        fh.write('s,x,y,z,grade,curvature\n')
        for r in df.itertuples(index=False):
            fh.write(f'{r.s:.3f},{r.x:.4f},{r.y:.4f},{r.z:.4f},{r.grade:.6f},{r.curvature:.6f}\n')
    print(f'wrote {path.name}: {len(df)} points, length {df.s.iloc[-1]:.1f} m  [{meta}]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map-dir', default=str(ROOT / 'analysis' / 'map_build' / 'map_train'))
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--step', type=float, default=1.0, help='output resampling step, m')
    a = ap.parse_args()
    mdir, out = Path(a.map_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((mdir / 'track_map.json').read_text(encoding='utf-8'))
    for old in out.glob('branch_*.csv'):
        old.unlink()
    for e in meta['edges']:
        df = pd.read_csv(mdir / e['file'])
        if e['id'] == 'main':
            write_edge(df, out / 'track_map.csv', a.step, 'cyclic=1', 'tram track map: main cycle')
        elif e.get('to') and e['to'].get('edge') == 'main':
            write_edge(df, out / f"branch_{e['id']}.csv", a.step,
                       f"cyclic=0 join_s={e['to']['s']:.3f}", f"branch {e['id']} merging into main")
        else:
            print(f"skip {e['id']}: does not merge into main (route choice unobservable without GNSS)")
    # stop landmarks on the main cycle (repeatable stops: platforms, signals, terminal layovers)
    st = pd.read_csv(mdir / 'stops.csv')
    lm = st[(st.edge == 'main') & st.cls.isin(['platform', 'signal', 'terminal']) &
            (st.n_runs >= 3) & (st.s_robust_std <= 1.0)].sort_values('s_median')
    with open(out / 'landmarks.csv', 'w', newline='\n') as fh:
        fh.write('# stop landmarks on the main cycle: s = median master-antenna arc length at standstill\n')
        fh.write('s,sigma,p_stop,cls\n')
        for r in lm.itertuples(index=False):
            p_stop = r.p_stop if pd.notna(r.p_stop) else 0.3
            fh.write(f'{r.s_median:.3f},{max(r.s_robust_std, 0.2):.3f},{p_stop:.3f},{r.cls}\n')
    print(f'wrote landmarks.csv: {len(lm)} landmarks ({lm.cls.value_counts().to_dict()})')


if __name__ == '__main__':
    main()
