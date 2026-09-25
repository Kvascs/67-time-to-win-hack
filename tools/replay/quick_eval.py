"""Quick judge-like check of the C++ estimator on real bags (before the full harness).

Speed reference: horizontal |v| of /sensing/gnss/master/vel at its header stamps.
Position reference: /sensing/gnss/master/fix in ENU at the first master fix of the bag.
Matching: nearest output stamp within 0.05 s (like the jury). Along/cross-track errors use
the estimator's own heading at the matched output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpp_bridge import NPZ, PKG, ROOT, export_events, run_replay  # noqa: E402

TMP = ROOT / 'build_core' / 'replay_tmp'
MAP = PKG / 'maps' / 'track_map.csv'
LANDMARKS = PKG / 'maps' / 'landmarks.csv'
CUTOFFS = PKG / 'maps' / 'cutoffs.csv'
BRANCHES = ','.join(str(b) for b in sorted((PKG / 'maps').glob('branch_*.csv')))

A, F = 6378137.0, 1 / 298.257223563
E2 = F * (2 - F)


def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


def enu(lat, lon, h, lat0, lon0, h0):
    d = ecef(lat, lon, h) - ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))
    la, lo = np.radians(lat0), np.radians(lon0)
    r = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return d @ r.T


def match_nearest(ref_t, out_t, tol=0.05):
    idx = np.searchsorted(out_t, ref_t)
    idx = np.clip(idx, 1, len(out_t) - 1)
    left, right = out_t[idx - 1], out_t[idx]
    pick = np.where(np.abs(ref_t - left) <= np.abs(right - ref_t), idx - 1, idx)
    ok = np.abs(out_t[pick] - ref_t) <= tol
    return pick, ok


def eval_bag(bag, sets=None, map_csv=MAP, traction_csv=None):
    TMP.mkdir(parents=True, exist_ok=True)
    ev = TMP / f'{bag}_events.csv'
    out = TMP / f'{bag}_out.csv'
    export_events(bag, ev)
    traction_csv = traction_csv or (PKG / 'config' / 'traction_lut.csv')
    o = run_replay(ev, out, map_csv=map_csv if map_csv and Path(map_csv).exists() else None,
                   traction_csv=traction_csv, sets={'output_frame': 'enu', 'base_link_along_m': 0, 'base_link_height_m': 0, **({'landmark_file': str(LANDMARKS)} if LANDMARKS.exists() else {}), **({'cutoff_file': str(CUTOFFS)} if CUTOFFS.exists() else {}), **(sets or {})}, branches=BRANCHES)
    d = np.load(NPZ / f'{bag}.npz')
    out_t = o.stamp_ns.to_numpy() * 1e-9
    res = {'bag': bag}

    # ---- speed ----
    mv = d['sensing__gnss__master__vel']
    ref_t = mv[:, 1]
    ref_v = np.hypot(mv[:, 2], mv[:, 3])
    pick, ok = match_nearest(ref_t, out_t)
    err = o.v.to_numpy()[pick][ok] - ref_v[ok]
    moving = ref_v[ok] > 0.5
    res.update(v_rmse=float(np.sqrt(np.mean(err ** 2))), v_mae=float(np.mean(np.abs(err))),
               v_bias=float(np.mean(err)), v_rmse_mov=float(np.sqrt(np.mean(err[moving] ** 2))),
               v_p99=float(np.percentile(np.abs(err), 99)), match=float(ok.mean()))

    # ---- position ----
    mf = d['sensing__gnss__master__fix']
    mf = mf[np.isfinite(mf[:, 2])]
    p_ref = enu(mf[:, 2], mf[:, 3], mf[:, 4], mf[0, 2], mf[0, 3], mf[0, 4])
    op = o[o.pos_valid == 1] if 'pos_valid' in o.columns else o   # only published positions
    out_tp = op.stamp_ns.to_numpy() * 1e-9
    pick, ok = match_nearest(mf[:, 1], out_tp)
    po = op[['x', 'y', 'z']].to_numpy()[pick][ok]
    yaw = op.yaw.to_numpy()[pick][ok]
    e = po - p_ref[ok]
    along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
    cross = -e[:, 0] * np.sin(yaw) + e[:, 1] * np.cos(yaw)
    dist = float(np.sum(np.hypot(np.diff(p_ref[:, 0]), np.diff(p_ref[:, 1]))))
    e3 = np.linalg.norm(e, axis=1)
    res.update(p3_rmse=float(np.sqrt(np.mean(e3 ** 2))), p3_max=float(e3.max()),
               along_rmse=float(np.sqrt(np.mean(along ** 2))), along_max=float(np.abs(along).max()),
               cross_rmse=float(np.sqrt(np.mean(cross ** 2))), z_rmse=float(np.sqrt(np.mean(e[:, 2] ** 2))),
               end_err=float(e3[-1]), drift_pct=float(100 * e3[-1] / max(dist, 1.0)), dist_m=dist,
               map_matched=float(((o['flags'].to_numpy() & (1 << 14)) == 0).mean()))
    res['ref_rtk'] = float((mf[:, 5] == 2).mean())
    try:
        ck = pd.read_csv(ROOT / 'analysis' / 'timing_reference' / 'clocks_per_bag.csv').set_index('bag')
        res['ref_clock_anom'] = float(ck.loc[bag, 'gnss_vs_veh_anom_frac']) if bag in ck.index else 0.0
    except Exception:
        res['ref_clock_anom'] = float('nan')
    res['ref_good'] = bool(res['ref_rtk'] > 0.8 and not (res['ref_clock_anom'] > 0.01))
    res.update(model_only=float((o.mu3 > 0.5).mean()), rate_hz=float(len(o) / (out_t[-1] - out_t[0])),
               proc_us_p99=float(np.percentile(o.proc_ns, 99) / 1e3))
    return res, o


if __name__ == '__main__':
    splits = json.load(open(ROOT / 'data' / 'splits.json'))
    args = sys.argv[1:]
    sets = dict(a.split('=', 1) for a in args if '=' in a)
    bags = [a for a in args if '=' not in a] or splits['val']
    rows = [eval_bag(b, sets=sets)[0] for b in bags]
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    cols = ['bag', 'v_rmse', 'v_mae', 'v_bias', 'v_p99', 'p3_rmse', 'p3_max', 'along_rmse', 'along_max',
            'cross_rmse', 'z_rmse', 'end_err', 'drift_pct', 'dist_m', 'model_only']
    print(df[cols].round(3).to_string(index=False))
    num = df.drop(columns=['bag']).astype(float)
    keys = ['v_rmse', 'v_mae', 'v_bias', 'along_rmse', 'cross_rmse', 'z_rmse', 'end_err', 'drift_pct',
            'model_only']
    print('\nALL   mean  :', num[keys].mean().round(4).to_dict())
    print('ALL   median:', num[keys].median().round(4).to_dict())
    good = num[num.ref_good > 0.5]
    print(f'GOOD-REF ({len(good)} bags) mean  :', good[keys].mean().round(4).to_dict())
    print(f'GOOD-REF ({len(good)} bags) median:', good[keys].median().round(4).to_dict())
    print(df[['bag', 'ref_rtk', 'ref_clock_anom', 'ref_good']].round(3).to_string(index=False))
