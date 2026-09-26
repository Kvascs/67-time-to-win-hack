"""Step 0: replay every train + val bag with the frozen binary and cache the outputs.

    python analysis/consistency/run_replays.py [--jobs 2] [--force]

Writes cache/replay/base/<bag>.parquet|json and results/replay_metrics.csv (quick_eval metrics,
including ref_rtk / ref_clock_anom / ref_good used to flag bad-reference bags).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

EXPECTED = {'sigma_wheel': 0.05, 'sigma_wheel_rel': 0.004, 'sigma_accel': 0.12, 'q_disturbance': 0.004,
            'cusum_slip_accel': 0.6, 'cusum_slide_accel': 0.8, 'cusum_h': 0.3, 'cmd_fault_accel': 0.5,
            'cmd_fault_h': 0.4, 'disturbance_max': 0.6, 'standstill_kmh': 0.15, 'map_sigma_cross': 0.3,
            'position_lead_s': 0.045}


def check_binary():
    txt = subprocess.run([str(C.EXE), '--dump-params'], capture_output=True, text=True).stdout
    vals = {}
    for line in txt.splitlines():
        line = line.strip()
        if ':' in line and not line.startswith('#'):
            k, rest = line.split(':', 1)
            try:
                vals[k.strip()] = float(rest.split('#')[0].strip())
            except ValueError:
                pass
    bad = {k: (vals.get(k), v) for k, v in EXPECTED.items() if vals.get(k) is None or abs(vals[k] - v) > 1e-12}
    if bad:
        raise SystemExit(f'binary defaults differ from the analysis constants: {bad}')
    print('binary defaults OK:', C.EXE.name)


def _one(args):
    bag, force = args
    try:
        return C.run_bag(bag, force=force)
    except Exception as e:  # keep going, report
        return {'bag': bag, 'error': str(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    C.ensure_dirs()
    check_binary()
    sp = C.splits()
    bags = [(b, 'train') for b in sp['train']] + [(b, 'val') for b in sp['val']]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(_one, [(b, a.force) for b, _ in bags]))
    df = pd.DataFrame(rows)
    df.insert(1, 'split', [s for _, s in bags])
    df.drop(columns=[c for c in ('stderr',) if c in df.columns]).to_csv(C.RESULTS / 'replay_metrics.csv', index=False)
    if 'error' in df.columns and df['error'].notna().any():
        print(df[df['error'].notna()][['bag', 'error']].to_string(index=False))
    print(df.groupby('split')[['v_rmse', 'along_rmse', 'ref_good']].agg(['mean', 'median', 'count']).round(3))


if __name__ == '__main__':
    main()
