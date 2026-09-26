"""Quality / noise floor of the GNSS reference itself (per bag) - what no estimator can beat.

    python -m harness.refstats --split val --out harness/results/refstats_val.json

Per bag:
  v_master_vs_rover_rmse   speed disagreement between the two GNSS receivers [m/s]
  v_2d_vs_3d_rmse          |v_xy| vs |v_xyz| (judge could use either) [m/s]
  wheel_best_rmse_clean    min over bogies of RMSE(wheel/3.6*k - v_ref) on clean moving samples,
                           k fitted per bag (the achievable speed floor with calibrated odometry)
  wheel_ratio_k            fitted k (v_ref = k * wheel_kmh/3.6)
  frame_enu_vs_utm_max_m   max horizontal distance between ENU and UTM-relative coordinates
  antenna_baseline_m       median master-rover distance (rover is ahead of master)
  alt_range_m              altitude span of the run (a z=const estimator eats this in 3D error)
  moving_at_start          reference speed > 0.3 m/s during the GNSS window (first 5 s)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from harness.loader import load_bag, resolve_bags, T_BAG, T_HDR, V_COL, LAT, LON, ALT, VX, VY, VZ
from harness.reference import (LocalFrame, RefConfig, build_reference, frame_discrepancy, repair_header_glitches,
                               select_origin)


def bag_stats(name: str, gnss_seconds: float = 5.0) -> dict:
    bag = load_bag(name)
    ref = build_reference(bag, RefConfig(time_base='header_fixed'))
    out = {'bag': name, 'vehicle': bag.vehicle, 'duration_s': bag.duration}
    out.update({k: ref.diag[k] for k in ('frac_status2', 'frac_outlier', 'hdr_glitch_fix', 'hdr_glitch_vel',
                                         'distance_path_m', 'distance_vint_m', 'alt_range_m')})
    vm, vr = bag['vel_master'], bag['vel_rover']
    tm = repair_header_glitches(vm[:, T_BAG], vm[:, T_HDR])[0]
    tr = repair_header_glitches(vr[:, T_BAG], vr[:, T_HDR])[0]
    sm = np.hypot(vm[:, VX], vm[:, VY])
    sr = np.hypot(vr[:, VX], vr[:, VY])
    o = np.argsort(tr)
    sri = np.interp(tm, tr[o], sr[o])
    out['v_master_vs_rover_rmse'] = float(np.sqrt(np.mean((sm - sri) ** 2)))
    s3 = np.sqrt(sm ** 2 + vm[:, VZ] ** 2)
    out['v_2d_vs_3d_rmse'] = float(np.sqrt(np.mean((s3 - sm) ** 2)))
    # wheel floor on clean moving samples
    best = np.inf
    for w_name in ('front', 'rear'):
        w = bag[w_name]
        th = repair_header_glitches(w[:, T_BAG], w[:, T_HDR])[0]
        ow = np.argsort(th)
        vw = np.interp(tm, th[ow], w[ow, V_COL]) / 3.6
        clean = (sm > 1.0) & (np.abs(vw - sm) < 0.5)
        if clean.sum() < 100:
            continue
        k = float(np.sum(vw[clean] * sm[clean]) / np.sum(vw[clean] ** 2))
        rm = float(np.sqrt(np.mean((k * vw[clean] - sm[clean]) ** 2)))
        out[f'wheel_ratio_k_{w_name}'] = k
        out[f'wheel_rmse_clean_{w_name}'] = rm
        best = min(best, rm)
    out['wheel_best_rmse_clean'] = float(best)
    fd = frame_discrepancy(bag)
    out['frame_enu_vs_utm_max_m'] = fd['max_m']
    # antenna baseline
    mf, rf = bag['fix_master'], bag['fix_rover']
    oi = select_origin(mf)
    fr = LocalFrame('enu', *mf[oi, [LAT, LON, ALT]])
    pm = np.stack(fr.forward(mf[:, LAT], mf[:, LON], mf[:, ALT]), 1)
    pr = np.stack(fr.forward(rf[:, LAT], rf[:, LON], rf[:, ALT]), 1)
    pri = np.stack([np.interp(mf[:, T_BAG], rf[:, T_BAG], pr[:, c]) for c in range(3)], 1)
    out['antenna_baseline_m'] = float(np.median(np.hypot(*(pm[:, :2] - pri[:, :2]).T)))
    g = vm[:, T_BAG] - bag.t_start <= gnss_seconds
    out['moving_at_start'] = bool(np.any(sm[g] > 0.3)) if g.any() else None
    out['n_fix_in_gnss_window'] = int(np.sum(mf[:, T_BAG] - bag.t_start <= gnss_seconds))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    rows = [bag_stats(b) for b in resolve_bags(a.split)]
    keys = ['bag', 'frac_status2', 'frac_outlier', 'hdr_glitch_fix', 'v_master_vs_rover_rmse', 'v_2d_vs_3d_rmse',
            'wheel_best_rmse_clean', 'frame_enu_vs_utm_max_m', 'antenna_baseline_m', 'alt_range_m',
            'distance_path_m', 'distance_vint_m', 'moving_at_start']
    print('| ' + ' | '.join(keys) + ' |')
    print('|' + '---|' * len(keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            cells.append(f'{v:.3f}' if isinstance(v, float) and abs(v) < 100 else (f'{v:.0f}' if isinstance(v, float) else str(v)))
        print('| ' + ' | '.join(cells) + ' |')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=1)
    return rows


if __name__ == '__main__':
    main()
