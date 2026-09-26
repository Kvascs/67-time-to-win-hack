"""Judge-like position check in the jury frame: published base_link (MGRS 37U CB) vs a reference base_link
built from RTK fixes of both antennas with the organisers' TF (master x = -9.873, rover x = +2.563,
both z = +3.0 in base_link): base_link = master + 9.873 * unit(rover - master) (3-D, body axis),
minus 3.0 m along the body z (~vertical). Only epochs with both antennas RTK within 20 ms.

    python tools/replay/eval_base_link.py --tag cur --exe build_core/tbo_replay.exe --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpp_bridge  # noqa: E402

VAL = cpp_bridge.ROOT / 'analysis' / 'validation_maps'
ALONG, HEIGHT, E0, N0 = 9.873, 3.0, 300000.0, 6100000.0
TR = Transformer.from_crs('EPSG:4326', 'EPSG:32637', always_xy=True)


def reference(bag):
    d = np.load(cpp_bridge.NPZ / f'{bag}.npz')
    m, r = d['sensing__gnss__master__fix'], d['sensing__gnss__rover__fix']
    m, r = m[m[:, 5] == 2], r[r[:, 5] == 2]
    if len(m) < 50 or len(r) < 50:
        return None
    j = np.clip(np.searchsorted(r[:, 1], m[:, 1]), 1, len(r) - 1)
    j = np.where(np.abs(r[j - 1, 1] - m[:, 1]) < np.abs(r[j, 1] - m[:, 1]), j - 1, j)
    ok = np.abs(r[j, 1] - m[:, 1]) < 0.02
    m, r = m[ok], r[j[ok]]
    me, mn = TR.transform(m[:, 3], m[:, 2])
    re_, rn = TR.transform(r[:, 3], r[:, 2])
    A = np.c_[me, mn, m[:, 4]]
    R = np.c_[re_, rn, r[:, 4]]
    # Horizontal body axis from both antennas; the rover's RTK altitude is often metres off, so the
    # vertical comes from the master alone: its altitude trend along its own track (+-15 m) gives
    # the pitch, and base_link lies 9.873 m ahead and 3.0 m below along the body.
    uh = R[:, :2] - A[:, :2]
    Lh = np.linalg.norm(uh, axis=1)
    good = (np.abs(Lh - 12.436) < 0.15) & (np.abs(R[:, 2] - A[:, 2]) < 0.6)  # consistent fixes only
    uh = uh / np.maximum(Lh, 1e-9)[:, None]
    dist = np.r_[0.0, np.cumsum(np.hypot(np.diff(A[:, 0]), np.diff(A[:, 1])))]
    keep = np.r_[True, np.diff(dist) > 1e-3]
    alt_d = lambda q: np.interp(q, dist[keep], A[keep, 2])
    slope = np.clip((alt_d(dist + 15.0) - alt_d(dist - 15.0)) / 30.0, -0.06, 0.06)
    u = np.c_[uh, np.zeros(len(uh))]
    B = A + ALONG * u
    B[:, 2] = A[:, 2] + ALONG * slope - HEIGHT
    B[:, 0] -= E0
    B[:, 1] -= N0
    return pd.DataFrame({'t': m[good, 1], 'x': B[good, 0], 'y': B[good, 1], 'z': B[good, 2],
                         'yaw': np.arctan2(u[good, 1], u[good, 0])})


def one(args):
    bag, exe, tag, sets, no_gnss = args
    ref = reference(bag)
    if ref is None:
        return {'bag': bag, 'error': 'no RTK'}
    cpp_bridge.REPLAY_EXE = Path(exe)
    tmp = cpp_bridge.ROOT / 'build_core' / 'replay_tmp' / f'bl_{tag}'
    tmp.mkdir(parents=True, exist_ok=True)
    ev, out = tmp / f'{bag}_ev.csv', tmp / f'{bag}_out.csv'
    cpp_bridge.export_events(bag, ev)
    if no_gnss:  # drop every GNSS fix: the position must come from the GNSS-free localisation
        lines = [ln for ln in ev.read_text().splitlines() if not ln.startswith('GF')]
        with open(ev, 'w', newline='\n') as fh:
            fh.write('\n'.join(lines) + '\n')
    br = ','.join(str(b) for b in sorted(VAL.glob('branch_*.csv')))
    s = {'output_frame': 'mgrs', 'landmark_file': str(VAL / 'landmarks.csv'), 'cutoff_file': str(VAL / 'cutoffs.csv'),
         'dfield_file': str(VAL / 'dfield.csv'), 'gl_stops_file': str(VAL / 'gl_stops.csv'),
         'gl_cutoffs_file': str(VAL / 'gl_cutoffs.csv'), 'speed_envelope_file': str(VAL / 'speed_envelope.csv'), **sets}
    o = cpp_bridge.run_replay(ev, out, map_csv=VAL / 'track_map.csv',
                              traction_csv=cpp_bridge.PKG / 'config' / 'traction_lut.csv', sets=s, branches=br)
    t_first = o.stamp_ns.min() * 1e-9
    o = o[o.pos_valid == 1].drop_duplicates('stamp_ns').sort_values('stamp_ns')
    if o.empty:
        return {'bag': bag, 'n': 0, 't_valid': float('nan'), 'error': 'no valid position'}
    ot = o.stamp_ns.to_numpy() * 1e-9
    t_valid = float(ot[0] - t_first)
    i = np.clip(np.searchsorted(ot, ref.t.to_numpy()), 1, len(ot) - 1)
    i = np.where(np.abs(ot[i - 1] - ref.t.to_numpy()) < np.abs(ot[i] - ref.t.to_numpy()), i - 1, i)
    ok = np.abs(ot[i] - ref.t.to_numpy()) <= 0.05  # the judge's matching tolerance
    e = o[['x', 'y', 'z']].to_numpy()[i[ok]] - ref[['x', 'y', 'z']].to_numpy()[ok]
    yaw = ref.yaw.to_numpy()[ok]
    along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
    cross = -e[:, 0] * np.sin(yaw) + e[:, 1] * np.cos(yaw)
    e3 = np.linalg.norm(e, axis=1)
    return {'bag': bag, 'n': int(ok.sum()), 't_valid': t_valid, 'p3_rmse': float(np.sqrt(np.mean(e3 ** 2))),
            'p3_med': float(np.median(e3)), 'p2_rmse': float(np.sqrt(np.mean(np.sum(e[:, :2] ** 2, axis=1)))),
            'along_rmse': float(np.sqrt(np.mean(along ** 2))), 'cross_rmse': float(np.sqrt(np.mean(cross ** 2))),
            'cross_p99': float(np.percentile(np.abs(cross), 99)), 'z_rmse': float(np.sqrt(np.mean(e[:, 2] ** 2))),
            'z_bias': float(np.mean(e[:, 2]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--exe', default=str(cpp_bridge.REPLAY_EXE))
    ap.add_argument('--split', default='val')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--no-gnss', action='store_true', help='remove every GNSS fix from the input')
    a = ap.parse_args()
    sets = dict(s.split('=', 1) for s in a.set)
    splits = json.load(open(cpp_bridge.ROOT / 'data' / 'splits.json'))
    bags = splits[a.split]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(one, [(b, a.exe, a.tag, sets, a.no_gnss) for b in bags]))
    df = pd.DataFrame(rows)
    out = cpp_bridge.ROOT / 'build_core' / 'eval'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f'bl_{a.tag}.csv', index=False)
    df = df[df.get('error', pd.Series(index=df.index, dtype=object)).isna()] if 'error' in df else df
    keys = ['t_valid', 'p3_rmse', 'p3_med', 'p2_rmse', 'along_rmse', 'cross_rmse', 'cross_p99', 'z_rmse', 'z_bias']
    print(f'[{a.tag}] bags={len(df)} sets={sets}')
    print('  mean  ', df[keys].mean().round(3).to_dict())
    print('  median', df[keys].median().round(3).to_dict())


if __name__ == '__main__':
    main()
