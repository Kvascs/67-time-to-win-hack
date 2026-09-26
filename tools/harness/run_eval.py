"""CLI: replay bags into an estimator and score it like the judge.

Examples (from C:\\MosTransHack\\tools):
    python -m harness.run_eval --est harness.baselines:B0 --split val --out harness/results/b0_val.json
    python -m harness.run_eval --est harness.baselines:B1 --split val --time-base bag --params "{\"stamp_mode\": \"bag\"}"
    python -m harness.run_eval --est my_pkg.est:MyEstimator --bags 30618_e3d94878 --plots harness/figures/dbg
    python -m harness.run_eval --est harness.baselines:B1 --split val --faults suite:basic
    python harness/run_eval.py ...          (also works as a script)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ''):          # executed as a script: make 'harness' importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from harness.loader import resolve_bags
from harness.reference import RefConfig
from harness.replay import EvalConfig, evaluate_bag, evaluate_many
from harness import metrics as M

TABLE_COLS = [
    ('v_rmse', '{:.3f}'), ('v_mae', '{:.3f}'), ('v_bias', '{:+.3f}'), ('v_rmse_accel', '{:.3f}'),
    ('v_rmse_brake', '{:.3f}'), ('v_rmse_trans', '{:.3f}'), ('pos_rmse3d', '{:.1f}'), ('along_rmse', '{:.1f}'),
    ('along_max', '{:.1f}'), ('cross_map_rmse', '{:.2f}'), ('z_rmse', '{:.2f}'), ('drift_pct_3d', '{:.3f}'),
    ('match_v', '{:.3f}'), ('match_pos', '{:.3f}'), ('rate_hz', '{:.1f}'),
]


def sanitize(o):
    """NaN/inf -> None, numpy scalars -> python, drop private keys (e.g. '_log')."""
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    return o


def fmt_table(results, agg) -> str:
    head = f"{'bag':16s} {'dur':>6s} {'dist':>6s} " + ' '.join(f'{c:>{max(7, len(c))}s}' for c, _ in TABLE_COLS)
    lines = [head, '-' * len(head)]
    for r in results:
        if 'error' in r:
            lines.append(f"{r['bag']:16s} ERROR {r['error']}")
            continue
        s = r['summary']
        cells = []
        for c, f in TABLE_COLS:
            v = s.get(c)
            cells.append(f'{(f.format(v) if v is not None and np.isfinite(v) else "nan"):>{max(7, len(c))}s}')
        lines.append(f"{r['bag']:16s} {r['duration_s']:6.0f} {s['dist_m']:6.0f} " + ' '.join(cells))
    lines.append('-' * len(head))
    for stat in ('mean', 'median', 'max'):
        cells = []
        for c, f in TABLE_COLS:
            v = agg.get(c, {}).get(stat) if isinstance(agg.get(c), dict) else None
            cells.append(f'{(f.format(v) if v is not None and np.isfinite(v) else "nan"):>{max(7, len(c))}s}')
        lines.append(f"{stat.upper():16s} {'':6s} {'':6s} " + ' '.join(cells))
    p = agg.get('pooled', {})
    lines.append('POOLED: ' + '  '.join(f'{k}={v:.4g}' for k, v in p.items() if v is not None and np.isfinite(v)))
    return '\n'.join(lines)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--est', default='harness.baselines:B0', help="estimator 'module:Class' or 'file.py:Class'")
    ap.add_argument('--params', default='{}', help='JSON dict of constructor kwargs')
    ap.add_argument('--split', default='val', help="split name(s): val, train, train+val, ... ")
    ap.add_argument('--bags', nargs='*', help='explicit bag names (override --split)')
    ap.add_argument('--gnss-seconds', type=float, default=5.0)
    ap.add_argument('--gnss-mode', default='first_n', choices=['first_n', 'none', 'all'])
    ap.add_argument('--frame', default='enu', choices=['enu', 'utm'])
    ap.add_argument('--time-base', default='header', choices=['header', 'bag', 'header_fixed'])
    ap.add_argument('--antenna', default='master', choices=['master', 'rover'])
    ap.add_argument('--speed-dims', type=int, default=2, choices=[2, 3])
    ap.add_argument('--tol', type=float, default=0.05)
    ap.add_argument('--direction', default='ref2out', choices=['ref2out', 'out2ref'])
    ap.add_argument('--clean-ref', action='store_true', help='score position only on non-outlier GNSS fixes')
    ap.add_argument('--tick-hz', type=float, default=0.0)
    ap.add_argument('--faults', nargs='*', default=None, help="e.g. suite:basic  'slip:front@40%%+6:peak=0.4'")
    ap.add_argument('--fault-seed', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--out', default=None, help='write JSON results here')
    ap.add_argument('--plots', default=None, help='directory for per-bag PNGs')
    ap.add_argument('--quiet', action='store_true')
    return ap


def config_from_args(a) -> EvalConfig:
    ref = RefConfig(antenna=a.antenna, frame=a.frame, time_base=a.time_base, speed_dims=a.speed_dims,
                    clean=a.clean_ref)
    return EvalConfig(gnss_seconds=a.gnss_seconds, gnss_mode=a.gnss_mode, tick_hz=a.tick_hz, tol=a.tol,
                      direction=a.direction, ref=ref, faults=a.faults, fault_seed=a.fault_seed)


def main(argv=None):
    a = build_parser().parse_args(argv)
    params = json.loads(a.params)
    cfg = config_from_args(a)
    bags = resolve_bags(a.bags if a.bags else a.split)
    t0 = time.time()
    if a.plots:
        from harness.plots import plot_bag
        results = []
        for b in bags:
            r = evaluate_bag(b, a.est, params, cfg, keep_log=True)
            if 'error' not in r:
                plot_bag(r, str(Path(a.plots) / f'{b}.png'), tol=cfg.tol, direction=cfg.direction)
            results.append(r)
        out = {'config': cfg.to_dict(), 'estimator': a.est, 'params': params, 'bags': results,
               'aggregate': M.aggregate(results)}
    else:
        out = evaluate_many(bags, a.est, params, cfg, jobs=a.jobs)
    out['wall_s'] = time.time() - t0
    if not a.quiet:
        print(f"estimator {a.est} params {params}  bags={len(bags)}  frame={a.frame} time_base={a.time_base} "
              f"gnss={a.gnss_mode}:{a.gnss_seconds}s faults={a.faults}  wall={out['wall_s']:.1f}s")
        print(fmt_table(out['bags'], out['aggregate']))
        for r in out['bags']:
            if 'error' in r:
                print(r['traceback'])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(sanitize(out), f, indent=1)
    return out


if __name__ == '__main__':
    main()
