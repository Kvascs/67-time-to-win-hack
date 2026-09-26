"""Experiment (b), in-filter check: does the L-BFGS-refined traction table help the estimator in model-only windows?

Both bogies are removed for 10 s every 60 s (tools/replay/eval_par.py --dropwin 60:10 logic), VAL bags,
frozen build final2, train-only maps (quick_eval defaults). Arm A: shipped LUT and parameters. Arm B: the
table of b_oe_lbfgs_warm_best.pt with its tau and kg (drive_tau_s, kg_brake/coast/traction).
Metrics per window (eval_par._window_metrics): speed RMSE and |distance error| inside the windows.

    python b_filter_ab.py [state_dict.pt]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(r'C:\MosTransHack')
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
sys.path.insert(0, str(HERE))
import cpp_bridge  # noqa: E402
import eval_par  # noqa: E402
import quick_eval  # noqa: E402
import qn_harness as H  # noqa: E402

DROP = (60.0, 10.0)
V_KNOTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.5, 18.0, 20.0]


def export_lut(sd_path: Path, out: Path) -> dict:
    import torch
    sd = torch.load(sd_path)
    T = sd['table'].numpy().astype(float)
    tau = float(np.exp(sd['log_tau'].numpy()))
    kg = [float(v) for v in sd['kg'].numpy()]
    with open(out, 'w', newline='\n') as fh:
        fh.write(f'# L-BFGS refinement of the OE table (analysis/ideas_check/quasi_newton), tau={tau:.4f}, kg={kg}\n')
        fh.write('notch,' + ','.join(f'{v:g}' for v in V_KNOTS) + '\n')
        for i, n in enumerate(range(-15, 16)):
            fh.write(f'{n},' + ','.join(f'{a:.5f}' for a in T[i]) + '\n')
    return {'drive_tau_s': tau, 'kg_brake': kg[0], 'kg_coast': kg[1], 'kg_traction': kg[2]}


def _one(args):
    bag, arm, sets, lut = args
    cpp_bridge.REPLAY_EXE = H.EXE
    quick_eval.TMP = H.SCR / 'dropwin' / arm
    quick_eval.export_events = eval_par._dropwin_export(*DROP)
    res, o = quick_eval.eval_bag(bag, sets=sets, traction_csv=lut)
    res.update(eval_par._window_metrics(bag, o, *DROP))
    res['arm'] = arm
    for f in (quick_eval.TMP / f'{bag}_out.csv', quick_eval.TMP / f'{bag}_events.csv'):
        try:
            f.unlink()
        except OSError:
            pass
    return res


def main():
    sd = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'b_oe_lbfgs_warm_best.pt')
    lut = HERE / f'b_lut_{sd.stem}.csv'
    sets_b = export_lut(sd, lut)
    print('arm B sets', sets_b, flush=True)
    bags = H.splits()['val']
    jobs = [(b, 'A', {}, None) for b in bags] + [(b, 'B', {k: f'{v:.6f}' for k, v in sets_b.items()}, lut) for b in bags]
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(_one, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f'b_filter_ab_{sd.stem}.csv', index=False)
    p = df.pivot(index='bag', columns='arm', values=['win_v_rmse', 'win_ds_abs', 'v_rmse', 'along_rmse'])
    rng = np.random.default_rng(0)
    for m in ('win_v_rmse', 'win_ds_abs', 'v_rmse', 'along_rmse'):
        a, b = p[(m, 'A')].to_numpy(float), p[(m, 'B')].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        bs = [b[i].mean() / a[i].mean() - 1 for i in (rng.integers(0, len(a), len(a)) for _ in range(4000))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'{m:11s} A {a.mean():.4f} B {b.mean():.4f} ({100 * (b.mean() / a.mean() - 1):+.2f} %, 95 % CI '
              f'{100 * lo:+.2f}..{100 * hi:+.2f} %), better {int((b < a).sum())}/{len(a)}', flush=True)


if __name__ == '__main__':
    main()
