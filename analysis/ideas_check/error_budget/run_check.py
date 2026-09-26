"""Error budget, step 1: replay the organisers' check bag (30618_88aea4d9) with the submitted build and
keep everything needed for the decomposition.

Same configuration and pairing as tools/replay/eval_checker.py (package maps, branches, GNSS only in the
first 5 s, message_filters.ApproximateTimeSynchronizer replay with slop 0.05 s), but
  * TBO_DEBUG_LM=1: every place-fix attempt (landmark at a stop or traction cut-off) is logged,
  * all outputs stay in this directory (the shared build_core/replay_tmp/checker is not touched),
  * the pair tables carry our full state at the paired output (flags, k, s, s_map, v ...).

    python analysis/ideas_check/error_budget/run_check.py [--exe build_core/tbo_replay_final2.exe]

Writes: out_check.parquet (all published outputs), lm_check.csv (place-fix attempts),
pairs_pos.parquet / pairs_vel.parquet (judge pairs), ref_check.parquet (reference with derived columns).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge as CB  # noqa: E402
import eval_checker as EC  # noqa: E402

BAG = '30618_88aea4d9'
LM_RE = re.compile(r'LM t=([\d.]+) s_map=([-\d.]+) sd=([\d.]+) n=(\d+) d0=([-\d.]+) p_known=([\d.]+) k=([-\d.]+)')


def yaw_of(q):
    qx, qy, qz, qw = q.T
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=str(ROOT / 'build_core' / 'tbo_replay_final2.exe'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    CB.REPLAY_EXE = Path(a.exe)
    tmp = HERE / 'tmp'
    tmp.mkdir(exist_ok=True)
    ev, out = tmp / f'{BAG}_ev{a.tag}.csv', tmp / f'{BAG}_out{a.tag}.csv'
    CB.export_events(BAG, ev)
    # the submitted build (final2) knows no stub_file: keep the package set of commit c1f3512 only
    base = {k: v for k, v in EC.package_sets().items() if not k.startswith('stub')}
    sets = {**base, **dict(s.split('=', 1) for s in a.set)}
    br = ','.join(str(EC.MAPS / f) for f in ('branch_fan_F2.csv', 'branch_fan_F3.csv', 'branch_wb_detour.csv'))
    os.environ['TBO_DEBUG_LM'] = '1'
    o = CB.run_replay(ev, out, map_csv=EC.MAPS / 'track_map.csv',
                      traction_csv=CB.PKG / 'config' / 'traction_lut.csv', sets=sets, branches=br)
    err = o.attrs.get('stderr', '')
    lm = sorted({m.groups() for m in LM_RE.finditer(err)}, key=lambda g: float(g[0]))
    lm = pd.DataFrame([[float(v) for v in g] for g in lm], columns=['t_abs', 's_map', 'sd', 'n', 'd0', 'p_known', 'k'])
    o = o.drop_duplicates('stamp_ns').sort_values('stamp_ns').reset_index(drop=True)
    out.unlink()
    ev.unlink()

    d = np.load(CB.NPZ / f'{BAG}.npz')
    ks = d['localization__kinematic_state']
    t0 = ks[0, 1]
    r_arr = np.round(ks[:, 0] * 1e9).astype(np.int64)
    r_st = np.round(ks[:, 1] * 1e9).astype(np.int64)
    o_arr = o.recv_ns.to_numpy(np.int64) + int(0.5e6)
    o_st = o.stamp_ns.to_numpy(np.int64)
    # reference with derived columns (heading, acceleration, path length)
    ref = pd.DataFrame({'t': ks[:, 1] - t0, 'recv': ks[:, 0] - t0, 'x': ks[:, 2], 'y': ks[:, 3], 'z': ks[:, 4],
                        'yaw': yaw_of(ks[:, 5:9]), 'vx': ks[:, 9], 'vy': ks[:, 10], 'vz': ks[:, 11], 'wz': ks[:, 12]})
    ref['path'] = np.r_[0.0, np.cumsum(np.hypot(np.diff(ref.x), np.diff(ref.y)))]
    # acceleration: central difference over +-0.5 s of the reference speed
    tt, vv = ref.t.to_numpy(), ref.vx.to_numpy()
    ref['acc'] = (np.interp(tt + 0.5, tt, vv) - np.interp(tt - 0.5, tt, vv)) / 1.0
    if not a.tag:
        ref.to_parquet(HERE / 'ref_check.parquet')

    o['t'] = o.stamp_ns * 1e-9 - t0
    lm['t'] = lm.t_abs - t0
    lm.to_csv(HERE / f'lm_check{a.tag}.csv', index=False)
    o.to_parquet(HERE / f'out_check{a.tag}.parquet')

    keep = ['v', 'v_var', 's', 's_var', 's_map', 'k', 'd', 'g', 'flags', 'yaw', 'a_model', 'accel', 'mu3', 'slip_f', 'slip_r']
    # velocity pairs (every output)
    pv = EC.approximate_sync(r_arr, r_st, o_arr, o_st)
    V = pd.DataFrame({'t': ref.t.to_numpy()[pv[:, 0]], 'ri': pv[:, 0], 'oi': pv[:, 1],
                      'dt_ms': (o_st[pv[:, 1]] - r_st[pv[:, 0]]) / 1e6,
                      'v_ref': ks[pv[:, 0], 9], 'acc_ref': ref.acc.to_numpy()[pv[:, 0]]})
    for c in keep:
        V[c] = o[c].to_numpy()[pv[:, 1]]
    V['ev'] = V.v - V.v_ref
    V.to_parquet(HERE / f'pairs_vel{a.tag}.parquet')
    # position pairs (published positions only)
    op = np.flatnonzero(o.pos_valid.to_numpy() == 1)
    pp = EC.approximate_sync(r_arr, r_st, o_arr[op], o_st[op])
    oi, ri = op[pp[:, 1]], pp[:, 0]
    dx = o.x.to_numpy()[oi] - ks[ri, 2]
    dy = o.y.to_numpy()[oi] - ks[ri, 3]
    dz = o.z.to_numpy()[oi] - ks[ri, 4]
    yaw = ref.yaw.to_numpy()[ri]
    P = pd.DataFrame({'t': ref.t.to_numpy()[ri], 'ri': ri, 'oi': oi, 'dt_ms': (o_st[oi] - r_st[ri]) / 1e6,
                      'v_ref': ks[ri, 9], 'acc_ref': ref.acc.to_numpy()[ri], 'path': ref.path.to_numpy()[ri],
                      'dx': dx, 'dy': dy, 'dz': dz,
                      'along': dx * np.cos(yaw) + dy * np.sin(yaw), 'cross': -dx * np.sin(yaw) + dy * np.cos(yaw),
                      'dist': np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)})
    for c in keep:
        P[c] = o[c].to_numpy()[oi]
    P.to_parquet(HERE / f'pairs_pos{a.tag}.parquet')
    r = EC.rmse
    k = P.t < 1270
    print(f'[{BAG}{a.tag}] outputs {len(o)}, LM attempts {len(lm)}')
    print(f'  velocity RMSE {r(V.ev):.4f} (t<1270: {r(V.ev[V.t < 1270]):.4f}), n={len(V)}')
    print(f'  position 3-D RMSE {r(P.dist):.3f} (t<1270: {r(P.dist[k]):.3f}), along {r(P.along[k]):.3f} '
          f'cross {r(P.cross[k]):.3f} z {r(P.dz[k]):.3f}, n={len(P)}')


if __name__ == '__main__':
    main()
