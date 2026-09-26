"""Usage demo + sanity checks of timing.judge_metrics on validation bags.

Estimators (all stamped with cmd header stamps, 20 Hz, as recommended):
  oracle_stamp : reference itself interpolated at the output stamp          -> residual = pairing only
  oracle_snap  : reference at the nearest 0.1-s GNSS epoch                  -> must give ~0 error
  wheel_map    : raw wheel odometry (mean of bogies / 3.6, no calibration) moved along the *reference
                 polyline of the same run* (stand-in for a perfect map; along-track error is genuine),
                 position lead +48 ms, z from the polyline
  wheel_map_nolead / wheel_map_z0 / wheel_map_utm : the same with one design choice changed
Numbers are printed as a table and saved to demo_baseline.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing as T  # noqa: E402

OUT = Path(__file__).resolve().parent


def cmd_stamps(bag):
    c = T.cmd(bag)
    return np.unique(c.t_hdr)


def wheel_series(bag):
    f, r = T.wheel(bag, 'front'), T.wheel(bag, 'rear')
    t = np.sort(f.t_hdr)
    vf = f.v[np.argsort(f.t_hdr)]
    vr = np.interp(t, np.sort(r.t_hdr), r.v[np.argsort(r.t_hdr)])
    keep = np.r_[True, np.diff(t) > 1e-6]
    return t[keep], 0.5 * (vf + vr)[keep]


def run(name, gnss_s=2.0):
    bag = T.load_bag(name)
    ref = T.reference_trajectory(bag)
    mpos, mvel = T.reference_quality_mask(bag, ref)
    ts = cmd_stamps(bag)
    rows = []

    def score(label, v, x, y, z):
        for masked in (False, True):
            m = T.judge_metrics(ref, ts, v, x, y, z, mask_pos=mpos if masked else None, mask_vel=mvel if masked else None)
            rows.append(dict(bag=name, estimator=label, ref_masked=masked, **m))

    # oracles
    v_or = np.interp(ts, ref.vel_t, ref.vel_speed)
    score('oracle_stamp', v_or, np.interp(ts, ref.t, ref.x), np.interp(ts, ref.t, ref.y), np.interp(ts, ref.t, ref.z))
    tsn = np.round(ts * 10) / 10
    score('oracle_snap', np.interp(tsn, ref.vel_t, ref.vel_speed), np.interp(tsn, ref.t, ref.x),
          np.interp(tsn, ref.t, ref.y), np.interp(tsn, ref.t, ref.z))
    # wheel odometry along the reference polyline
    init = T.init_alignment(bag, gnss_s)
    tw, vw = wheel_series(bag)
    S = np.r_[0.0, np.cumsum(0.5 * (vw[1:] + vw[:-1]) * np.diff(tw))]
    # starting arc length: project the robust initial position on the polyline (first 30 s of reference)
    x0, y0, _ = init['p0_enu']
    n0 = np.searchsorted(ref.t, ref.t[0] + 30)
    j = int(np.argmin(np.hypot(ref.x[:n0] - x0, ref.y[:n0] - y0)))
    s0 = ref.s[j]
    t_start = init['t_first_fix_hdr']
    s_uniq, iu = np.unique(ref.s, return_index=True)

    def along_path(s):
        return (np.interp(s, s_uniq, ref.x[iu]), np.interp(s, s_uniq, ref.y[iu]), np.interp(s, s_uniq, ref.z[iu]))

    v_out = np.interp(ts, tw, vw)                              # no extra lead for speed (hdr-time judge)
    for label, lead, zmode, frame in (('wheel_map', T.WHEEL_LAG_VS_POS, 'poly', 'enu'),
                                      ('wheel_map_nolead', 0.0, 'poly', 'enu'),
                                      ('wheel_map_z0', T.WHEEL_LAG_VS_POS, 'zero', 'enu'),
                                      ('wheel_map_utm', T.WHEEL_LAG_VS_POS, 'poly', 'utm')):
        s = s0 + np.interp(ts + lead, tw, S) - np.interp(t_start, tw, S)
        x, y, z = along_path(s)
        if zmode == 'zero':
            z = np.zeros_like(z)
        if frame == 'utm':
            # what a solution working in 'UTM minus origin' would publish for the same physical positions
            lat, lon, h = T.enu_to_llh(x, y, z, *ref.origin)
            X, Y = T.llh_to_utm(lat, lon)
            X0, Y0 = T.llh_to_utm(ref.origin[0], ref.origin[1])
            x, y, z = X - X0, Y - Y0, h - ref.origin[2]
        score(label, v_out, x, y, z)
    return rows


def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    d = pd.read_csv(OUT / 'delays_per_bag.csv')
    good = set(d[(d['front_pos_hh_rms'] < 0.06)]['bag'])
    val = [b for b in T.list_bags('val') if b in good][:8]
    rows = []
    for b in val:
        rows += run(b)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'demo_baseline.csv', index=False, float_format='%.4g')
    cols = ['speed_match_frac', 'speed_rmse', 'speed_bias_accel', 'speed_bias_brake', 'pos_match_frac', 'rmse_x', 'rmse_y',
            'rmse_z', 'rmse_3d', 'along_mean', 'along_rmse', 'along_max_abs', 'cross_poly_rmse', 'end_drift_pct']
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(f'validation bags (good fixes): {val}')
    print(df.groupby(['estimator', 'ref_masked'])[cols].median().round(4).to_string())


if __name__ == '__main__':
    main()
