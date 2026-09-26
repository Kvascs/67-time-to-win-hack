"""Cross-check harness metrics against tools/replay/quick_eval.py on the same C++ outputs.

For each bag:
  A. quick_eval end-to-end (its own export_events + run_replay + metrics), redirected to the SNAPSHOT binary
     and snapshot map files, temp files in a scratch dir (build_core is not touched);
  B. the harness adapter (cpp_estimator) on the same bag -> binary outputs + harness metrics;
  C. quick_eval's metric formulas applied to the adapter's outputs (isolates metric definitions);
  D. harness variants that mimic quick_eval choices (drift denominator / end sample, heading-based along/cross).
Writes harness/results/tbo_quickeval_check.json and .csv.

    python -m harness.tbo_quickeval_check 30618_e3d94878 30618_defd0170
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from harness.cpp_estimator import CppConfig, SNAP_DIR, SNAP_EXE, evaluate_bag_cpp
from harness.loader import load_bag
from harness.replay import EvalConfig
from harness.run_eval import sanitize
from harness import metrics as M

TOOLS = Path(__file__).resolve().parents[1]
R = Path(__file__).resolve().parent / 'results'


def quick_eval_module(tmpdir: Path):
    sys.path.insert(0, str(TOOLS / 'replay'))
    import cpp_bridge
    import quick_eval
    cpp_bridge.REPLAY_EXE = SNAP_EXE                     # snapshot binary, not build_core
    quick_eval.TMP = tmpdir                              # temp files outside build_core
    quick_eval.LANDMARKS = SNAP_DIR / 'maps' / 'landmarks.csv'
    quick_eval.BRANCHES = ','.join(str(b) for b in sorted((SNAP_DIR / 'maps').glob('branch_*.csv')))
    quick_eval.CUTOFFS = tmpdir / 'no_cutoffs.csv'           # newer input, unknown to the snapshot binary
    return quick_eval


def qe_metrics_on(o: pd.DataFrame, bag: str, qe) -> dict:
    """quick_eval.eval_bag's metric block, verbatim, applied to an outputs DataFrame ``o``."""
    from cpp_bridge import NPZ
    d = np.load(NPZ / f'{bag}.npz')
    out_t = o.stamp_ns.to_numpy() * 1e-9
    res = {}
    mv = d['sensing__gnss__master__vel']
    ref_t = mv[:, 1]
    ref_v = np.hypot(mv[:, 2], mv[:, 3])
    pick, ok = qe.match_nearest(ref_t, out_t)
    err = o.v.to_numpy()[pick][ok] - ref_v[ok]
    res.update(v_rmse=float(np.sqrt(np.mean(err ** 2))), v_mae=float(np.mean(np.abs(err))),
               v_bias=float(np.mean(err)), match=float(ok.mean()), n_v=int(ok.sum()))
    mf = d['sensing__gnss__master__fix']
    mf = mf[np.isfinite(mf[:, 2])]
    p_ref = qe.enu(mf[:, 2], mf[:, 3], mf[:, 4], mf[0, 2], mf[0, 3], mf[0, 4])
    pick, ok = qe.match_nearest(mf[:, 1], out_t)
    po = o[['x', 'y', 'z']].to_numpy()[pick][ok]
    yaw = o.yaw.to_numpy()[pick][ok]
    e = po - p_ref[ok]
    along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
    cross = -e[:, 0] * np.sin(yaw) + e[:, 1] * np.cos(yaw)
    dist = float(np.sum(np.hypot(np.diff(p_ref[:, 0]), np.diff(p_ref[:, 1]))))
    e3 = np.linalg.norm(e, axis=1)
    res.update(p3_rmse=float(np.sqrt(np.mean(e3 ** 2))), p3_max=float(e3.max()),
               along_rmse=float(np.sqrt(np.mean(along ** 2))), along_max=float(np.abs(along).max()),
               cross_rmse=float(np.sqrt(np.mean(cross ** 2))), z_rmse=float(np.sqrt(np.mean(e[:, 2] ** 2))),
               end_err=float(e3[-1]), drift_pct=float(100 * e3[-1] / max(dist, 1.0)), dist_m=dist, n_pos=int(ok.sum()),
               n_fix_total=int(len(mf)), first_fix_status=int(mf[0, 5]))
    return res


def harness_view(r: dict) -> dict:
    s, p, sp = r['summary'], r['pos'], r['speed']
    return {'v_rmse': s['v_rmse'], 'v_mae': s['v_mae'], 'v_bias': s['v_bias'], 'match': s['match_v'],
            'n_v': sp['all']['n'], 'p3_rmse': s['pos_rmse3d'], 'p3_max': s['pos_max3d'],
            'along_rmse(arc)': s['along_rmse'], 'along_max(arc)': s['along_max'],
            'along_rmse(tan)': s['along_tan_rmse'], 'cross_rmse(map)': s['cross_map_rmse'],
            'cross_rmse(tan)': p['cross_tan']['rmse'], 'z_rmse': s['z_rmse'], 'end_err': s['final_err3d'],
            'drift_pct': s['drift_pct_3d'], 'dist_m': s['dist_m'], 'n_pos': p['n_matched'],
            'n_ref_pos': p['n_ref'], 'frac_outlier': r['ref_diag']['frac_outlier'],
            'end_t_rel': p['final']['t_rel']}


def harness_mimic(r: dict) -> dict:
    """Harness data re-scored with quick_eval's definitions: heading-based along/cross (estimator yaw),
    raw polyline distance of all fixes, end error at the last matched fix even if it is an outlier."""
    ref, log = r['_ref'], r['_log']
    valid = np.all(np.isfinite(log.xyz), axis=1) & np.isfinite(log.stamp)
    iref, iout, _ = M.pair(ref.t_pos, log.stamp, valid, 0.05, 'ref2out')
    e = log.xyz[iout] - ref.xyz[iref]
    yaw = log.yaw[iout]
    along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
    cross = -e[:, 0] * np.sin(yaw) + e[:, 1] * np.cos(yaw)
    e3 = np.linalg.norm(e, axis=1)
    o = np.argsort(ref.tbag_pos, kind='stable')          # bag (file) order like quick_eval's polyline
    dist_raw = float(np.sum(np.hypot(np.diff(ref.xyz[o, 0]), np.diff(ref.xyz[o, 1]))))
    last = int(np.argmax(ref.t_pos[iref]))
    return {'along_rmse(yaw)': float(np.sqrt(np.mean(along ** 2))), 'cross_rmse(yaw)': float(np.sqrt(np.mean(cross ** 2))),
            'dist_raw_m': dist_raw, 'end_err_last_any': float(e3[last]),
            'drift_pct_raw_dist_last_any': float(100 * e3[last] / dist_raw)}


def main(argv=None):
    bags = (argv if argv is not None else sys.argv[1:]) or ['30618_e3d94878', '30618_defd0170']
    tmp = Path(tempfile.mkdtemp(prefix='tbo_qe_'))
    qe = quick_eval_module(tmp)
    out, rows = {}, []
    for b in bags:
        qres, qo = qe.eval_bag(b, map_csv=SNAP_DIR / 'maps' / 'track_map.csv',
                               traction_csv=SNAP_DIR / 'config' / 'traction_lut.csv')
        r = evaluate_bag_cpp(b, EvalConfig(), CppConfig(), keep_log=True)
        df = r['_df']
        same_len = len(df) == len(qo)
        cols = ['stamp_ns', 'v', 'x', 'y', 'z', 'yaw', 's', 'flags']
        identical = bool(same_len and all(np.array_equal(df[c].to_numpy(), qo[c].to_numpy()) for c in cols))
        maxdiff = {c: float(np.max(np.abs(df[c].to_numpy(float) - qo[c].to_numpy(float)))) for c in cols} if same_len else None
        qe_on_adapter = qe_metrics_on(df, b, qe)
        hv = harness_view(r)
        hm = harness_mimic(r)
        out[b] = {'quick_eval_end_to_end': {k: v for k, v in qres.items() if k != 'bag'},
                  'quick_eval_on_adapter_outputs': qe_on_adapter, 'harness': hv, 'harness_mimic_quick_eval': hm,
                  'binary_outputs_identical': identical, 'n_out_adapter': len(df), 'n_out_quick_eval': len(qo),
                  'max_abs_diff': maxdiff}
        for name, dct in (('quick_eval_end_to_end', qres), ('quick_eval_on_adapter_outputs', qe_on_adapter),
                          ('harness', hv), ('harness_mimic_quick_eval', hm)):
            for k, v in dct.items():
                if k != 'bag' and isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                    rows.append({'bag': b, 'source': name, 'metric': k, 'value': float(v)})
        print(f'{b}: outputs identical={identical} (n={len(df)} vs {len(qo)})')
        for k in ('v_rmse', 'v_mae', 'v_bias', 'match', 'p3_rmse', 'z_rmse', 'end_err', 'drift_pct', 'dist_m'):
            print(f'   {k:10s} quick_eval={qres.get(k, float("nan")):10.4f}  qe_on_adapter={qe_on_adapter.get(k, float("nan")):10.4f}'
                  f'  harness={hv.get(k, float("nan")):10.4f}')
        print(f"   along: qe(yaw)={qres['along_rmse']:.3f} harness_mimic(yaw)={hm['along_rmse(yaw)']:.3f} "
              f"harness(arc)={hv['along_rmse(arc)']:.3f} harness(tan)={hv['along_rmse(tan)']:.3f}")
        print(f"   cross: qe(yaw)={qres['cross_rmse']:.3f} harness_mimic(yaw)={hm['cross_rmse(yaw)']:.3f} "
              f"harness(map)={hv['cross_rmse(map)']:.3f} harness(tan)={hv['cross_rmse(tan)']:.3f}")
        print(f"   drift: qe={qres['drift_pct']:.4f}% (dist {qres['dist_m']:.0f} m)  harness={hv['drift_pct']:.4f}% "
              f"(dist {hv['dist_m']:.0f} m)  harness_mimic={hm['drift_pct_raw_dist_last_any']:.4f}% (dist {hm['dist_raw_m']:.0f} m)")
    with open(R / 'tbo_quickeval_check.json', 'w', encoding='utf-8') as f:
        json.dump(sanitize(out), f, indent=1)
    pd.DataFrame(rows).to_csv(R / 'tbo_quickeval_check.csv', index=False, float_format='%.6g')
    return out


if __name__ == '__main__':
    main()
