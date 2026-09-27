"""Score a replay exactly like the organisers' checker (check-code/src/checker_ros/.../metrics.py).

The checker compares /result/velocity with /localization/kinematic_state.twist.twist.linear.x and
/result/position with kinematic_state.pose.pose.position (|dx|, |dy|, |dz| and the 3-D distance). Pairs
come from message_filters.ApproximateTimeSynchronizer (queue 100, slop 0.05 s), one per stream: on every
arrival the newest message is paired with the closest queued counterpart within the slop, both are
consumed, and the oldest entries fall out of a full queue. That greedy pairing is replayed here in
arrival order: reference messages arrive at their bag time, ours at the arrival of the input that
triggered them plus the processing time.

    python tools/replay/eval_checker.py --bag 30618_88aea4d9
    python tools/replay/eval_checker.py --bag 30618_88aea4d9 --set position_lead_s=0.03
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpp_bridge  # noqa: E402

MAPS = cpp_bridge.PKG / 'maps'
SLOP_NS = 50_000_000
QUEUE = 100


def package_sets() -> dict:
    """The node's own map and calibration files (config/params.yaml), with absolute paths."""
    return {'output_frame': 'mgrs', 'landmark_file': str(MAPS / 'landmarks.csv'), 'cutoff_file': str(MAPS / 'cutoffs.csv'),
            'dfield_file': str(MAPS / 'dfield.csv'), 'gl_stops_file': str(MAPS / 'gl_stops.csv'),
            'gl_cutoffs_file': str(MAPS / 'gl_cutoffs.csv'), 'speed_envelope_file': str(MAPS / 'speed_envelope.csv'),
            'stub_file': str(MAPS / 'stub_west_arrival_2.csv'), 'wheel_epochs_file': str(MAPS / 'wheel_epochs.csv')}


def approximate_sync(ref_arrival, ref_stamp, out_arrival, out_stamp):
    """Replay of message_filters.ApproximateTimeSynchronizer for two inputs. Returns (ref_idx, out_idx) pairs."""
    ev = np.concatenate([np.c_[ref_arrival, np.zeros(len(ref_arrival)), np.arange(len(ref_arrival))],
                         np.c_[out_arrival, np.ones(len(out_arrival)), np.arange(len(out_arrival))]])
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    stamps = (ref_stamp, out_stamp)
    queues = ({}, {})  # stamp_ns -> index
    pairs = []
    for _, src, idx in ev:
        src, idx = int(src), int(idx)
        st = int(stamps[src][idx])
        mine, other = queues[src], queues[1 - src]
        mine[st] = idx
        while len(mine) > QUEUE:
            del mine[min(mine)]
        best, best_d = None, None
        for s in other:
            dd = abs(s - st)
            if dd <= SLOP_NS and (best_d is None or dd < best_d):
                best, best_d = s, dd
        if best is None or not best_d < SLOP_NS:
            continue
        pair = (idx, other[best]) if src == 0 else (other[best], idx)
        pairs.append(pair)
        del mine[st]
        del other[best]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def rmse(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', default='30618_88aea4d9')
    ap.add_argument('--exe', default=str(cpp_bridge.REPLAY_EXE))
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--proc-ms', type=float, default=0.5, help='node processing time added to input arrival')
    ap.add_argument('--tag', default='')
    ap.add_argument('--post-v-delay', type=float, default=0.0, help='what-if: publish the speed of (stamp - delay)')
    ap.add_argument('--post-pos-delay', type=float, default=0.0, help='what-if: publish the position of (stamp - delay)')
    ap.add_argument('--end-s', type=float, default=0.0, help='also report metrics for reference t < end-s')
    ap.add_argument('--no-gnss', action='store_true', help='remove every GNSS fix from the input')
    a = ap.parse_args()
    cpp_bridge.REPLAY_EXE = Path(a.exe)
    tmp = cpp_bridge.ROOT / 'build_core' / 'replay_tmp' / 'checker'
    tmp.mkdir(parents=True, exist_ok=True)
    ev, out = tmp / f'{a.bag}_ev.csv', tmp / f'{a.bag}_out{a.tag}.csv'
    cpp_bridge.export_events(a.bag, ev)
    if a.no_gnss:  # drop every GNSS fix: the position must come from the GNSS-free localisation
        lines = [ln for ln in ev.read_text().splitlines() if not ln.startswith('GF')]
        with open(ev, 'w', newline='\n') as fh:
            fh.write('\n'.join(lines) + '\n')
    sets = {**package_sets(), **dict(s.split('=', 1) for s in a.set)}
    br = ','.join(str(MAPS / f) for f in ('branch_fan_F2.csv', 'branch_fan_F3.csv', 'branch_wb_detour.csv'))
    o = cpp_bridge.run_replay(ev, out, map_csv=MAPS / 'track_map.csv',
                              traction_csv=cpp_bridge.PKG / 'config' / 'traction_lut.csv', sets=sets, branches=br)
    o = o.drop_duplicates('stamp_ns').sort_values('stamp_ns').reset_index(drop=True)
    if a.post_v_delay or a.post_pos_delay:  # what-if on the published stream: values of (stamp - delay)
        ts = o.stamp_ns.to_numpy() * 1e-9
        if a.post_v_delay:
            o['v'] = np.interp(ts - a.post_v_delay, ts, o.v.to_numpy())
        if a.post_pos_delay:
            for c in ('x', 'y', 'z'):
                o[c] = np.interp(ts - a.post_pos_delay, ts, o[c].to_numpy())
    d = np.load(cpp_bridge.NPZ / f'{a.bag}.npz')
    ks = d['localization__kinematic_state']
    r_arr = np.round(ks[:, 0] * 1e9).astype(np.int64)
    r_st = np.round(ks[:, 1] * 1e9).astype(np.int64)
    o_arr = o.recv_ns.to_numpy(np.int64) + int(a.proc_ms * 1e6)
    o_st = o.stamp_ns.to_numpy(np.int64)

    # velocity: every published output; position: only published positions
    pv = approximate_sync(r_arr, r_st, o_arr, o_st)
    ev_v = o.v.to_numpy()[pv[:, 1]] - ks[pv[:, 0], 9]
    pos = o.pos_valid.to_numpy() == 1
    op = np.flatnonzero(pos)
    pp = approximate_sync(r_arr, r_st, o_arr[op], o_st[op])
    oi, ri = op[pp[:, 1]], pp[:, 0]
    dx = o.x.to_numpy()[oi] - ks[ri, 2]
    dy = o.y.to_numpy()[oi] - ks[ri, 3]
    dz = o.z.to_numpy()[oi] - ks[ri, 4]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    print(f'[{a.bag}{a.tag}] sets={dict(s.split("=", 1) for s in a.set)}')
    print(f'  velocity: RMSE={rmse(ev_v):.4f}  max={np.nanmax(np.abs(ev_v)):.3f}  n={len(ev_v)}  (reference {len(ks)})')
    print(f'  position: x RMSE={rmse(dx):.3f}  y RMSE={rmse(dy):.3f}  z RMSE={rmse(dz):.3f}  '
          f'distance RMSE={rmse(dist):.3f} max={dist.max():.2f}  n={len(dist)}')
    # diagnostics: stamp offsets of the pairs, along/cross split by the reference heading
    qx, qy, qz, qw = (ks[ri, c] for c in (5, 6, 7, 8))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    along = dx * np.cos(yaw) + dy * np.sin(yaw)
    cross = -dx * np.sin(yaw) + dy * np.cos(yaw)
    dt_ms = (o_st[oi] - r_st[ri]) / 1e6
    vref = ks[ri, 9]
    print(f'  along RMSE={rmse(along):.3f} bias={np.mean(along):+.3f}  cross RMSE={rmse(cross):.3f} bias={np.mean(cross):+.3f}  '
          f'z bias={np.mean(dz):+.3f}')
    print(f'  pair stamp offset (ours - ref) ms: mean {dt_ms.mean():+.1f}, p5 {np.percentile(dt_ms, 5):+.1f}, '
          f'p95 {np.percentile(dt_ms, 95):+.1f}; along explained by offset: mean {np.mean(vref * dt_ms / 1e3):+.3f} m')
    print(f'  velocity bias={np.mean(ev_v):+.4f}; position published from t={(o_st[op[0]] - r_st[0]) / 1e9:.1f} s of the reference')
    if a.end_s:
        k = (r_st[ri] - r_st[0]) / 1e9 < a.end_s
        kv = (r_st[pv[:, 0]] - r_st[0]) / 1e9 < a.end_s
        print(f'  [t < {a.end_s:.0f} s] velocity RMSE={rmse(ev_v[kv]):.4f} | distance RMSE={rmse(dist[k]):.3f} '
              f'x={rmse(dx[k]):.3f} y={rmse(dy[k]):.3f} z={rmse(dz[k]):.3f} along={rmse(along[k]):.3f} bias {np.mean(along[k]):+.3f}')
    pd.DataFrame({'t': (r_st[ri] - r_st[0]) / 1e9, 'dt_ms': dt_ms, 'v_ref': vref, 'along': along, 'cross': cross,
                  'dz': dz, 'dist': dist}).to_csv(tmp / f'{a.bag}_pairs{a.tag}.csv', index=False)


if __name__ == '__main__':
    main()
