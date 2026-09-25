"""Quick judge-like speed check of the C++ estimator on real bags (before the full harness).

Reference speed: horizontal |v| of /sensing/gnss/master/vel at its header stamps.
Matching: nearest output stamp within 0.05 s (like the jury). Prints RMSE/MAE/bias
overall and for moving samples, plus IMM mode statistics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpp_bridge import NPZ, ROOT, export_events, run_replay  # noqa: E402

TMP = ROOT / 'build_core' / 'replay_tmp'


def match_nearest(ref_t, out_t, tol=0.05):
    idx = np.searchsorted(out_t, ref_t)
    idx = np.clip(idx, 1, len(out_t) - 1)
    left, right = out_t[idx - 1], out_t[idx]
    pick = np.where(np.abs(ref_t - left) <= np.abs(right - ref_t), idx - 1, idx)
    ok = np.abs(out_t[pick] - ref_t) <= tol
    return pick, ok


def eval_bag(bag, sets=None, map_csv=None, traction_csv=None):
    TMP.mkdir(parents=True, exist_ok=True)
    ev = TMP / f'{bag}_events.csv'
    out = TMP / f'{bag}_out.csv'
    export_events(bag, ev)
    o = run_replay(ev, out, map_csv=map_csv, traction_csv=traction_csv, sets=sets)
    d = np.load(NPZ / f'{bag}.npz')
    mv = d['sensing__gnss__master__vel']
    ref_t = mv[:, 1]
    ref_v = np.hypot(mv[:, 2], mv[:, 3])
    out_t = o.stamp_ns.to_numpy() * 1e-9
    pick, ok = match_nearest(ref_t, out_t)
    err = o.v.to_numpy()[pick][ok] - ref_v[ok]
    moving = ref_v[ok] > 0.5
    # naive baseline: mean of the two bogies (latest sample at or before the reference stamp)
    fr, rr = d['vehicle__front_bogie_velocity'], d['vehicle__rear_bogie_velocity']
    fi = np.clip(np.searchsorted(fr[:, 1], ref_t[ok], side='right') - 1, 0, len(fr) - 1)
    ri = np.clip(np.searchsorted(rr[:, 1], ref_t[ok], side='right') - 1, 0, len(rr) - 1)
    base = 0.5 * (fr[fi, 2] + rr[ri, 2]) / 3.5965
    err_b = base - ref_v[ok]
    res = {
        'bag': bag,
        'match_rate': float(ok.mean()),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'bias': float(np.mean(err)),
        'rmse_moving': float(np.sqrt(np.mean(err[moving] ** 2))),
        'rmse_raw': float(np.sqrt(np.mean(err_b ** 2))),
        'rmse_raw_moving': float(np.sqrt(np.mean(err_b[moving] ** 2))),
        'p99_abs': float(np.percentile(np.abs(err), 99)),
        'max_abs': float(np.max(np.abs(err))),
        'frac_model_only': float((o.mu3 > 0.5).mean()),
        'frac_front_bad': float((o.mu1 > 0.5).mean()),
        'frac_rear_bad': float((o.mu2 > 0.5).mean()),
        'out_rate_hz': float(len(o) / (out_t[-1] - out_t[0])),
        'proc_us_p99': float(np.percentile(o.proc_ns, 99) / 1e3),
    }
    return res, o


if __name__ == '__main__':
    splits = json.load(open(ROOT / 'data' / 'splits.json'))
    bags = sys.argv[1:] or splits['val'][:6]
    rows = [eval_bag(b)[0] for b in bags]
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(df.round(4).to_string(index=False))
    print('\nmean:', df.drop(columns=['bag']).mean().round(4).to_dict())
