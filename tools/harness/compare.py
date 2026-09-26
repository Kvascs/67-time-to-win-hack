"""Compare result JSONs written by run_eval (markdown tables for reports / PR descriptions).

    python -m harness.compare harness/results/b0_val.json harness/results/b1_val.json
    python -m harness.compare --per-bag harness/results/b1_val.json
    python -m harness.compare --stat median --keys v_rmse,along_rmse,drift_pct_3d a.json b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_KEYS = ['v_rmse', 'v_mae', 'v_bias', 'v_rmse_accel', 'v_bias_accel', 'v_rmse_brake', 'v_bias_brake',
                'v_rmse_trans', 'v_rmse_stop', 'v_max', 'v_rmse_anom', 'v_naive_rmse_anom',
                'pos_rmse3d', 'pos_rmse2d', 'z_rmse', 'along_mean', 'along_mean_abs', 'along_rmse', 'along_max',
                'cross_map_rmse', 'final_err3d', 'drift_pct_3d', 'drift_pct_along', 'match_v', 'match_pos',
                'rate_hz', 'max_gap_s', 'proc_p99_ms', 'stamp_age_p95_ms']


def _fmt(v, key=''):
    if v is None:
        return 'n/a'
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        v = float(v)
        if 'ratio' in key:
            return f'{v:.4f}'
        if abs(v) >= 100:
            return f'{v:.0f}'
        if abs(v) >= 10:
            return f'{v:.1f}'
        return f'{v:.3f}'
    return str(v)


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def label_of(path, res):
    return Path(path).stem


def table(paths, keys=DEFAULT_KEYS, stat='mean', labels=None) -> str:
    runs = [load(p) for p in paths]
    labels = labels or [label_of(p, r) for p, r in zip(paths, runs)]
    lines = ['| metric (' + stat + ' over bags) | ' + ' | '.join(labels) + ' |',
             '|---|' + '---|' * len(labels)]
    for k in keys:
        row = []
        for r in runs:
            a = r['aggregate'].get(k)
            row.append(_fmt(a.get(stat) if isinstance(a, dict) else None, k))
        lines.append(f'| {k} | ' + ' | '.join(row) + ' |')
    pooled = sorted({k for r in runs for k in r['aggregate'].get('pooled', {})})
    for k in pooled:
        lines.append(f'| pooled {k} | ' + ' | '.join(_fmt(r['aggregate'].get('pooled', {}).get(k), k) for r in runs) + ' |')
    return '\n'.join(lines)


def per_bag(path, keys=('v_rmse', 'v_bias', 'v_rmse_accel', 'v_rmse_brake', 'along_rmse', 'along_max',
                        'cross_map_rmse', 'z_rmse', 'drift_pct_3d', 'match_v', 'match_pos')) -> str:
    r = load(path)
    lines = ['| bag | dist m | ' + ' | '.join(keys) + ' |', '|---|---|' + '---|' * len(keys)]
    for b in r['bags']:
        if 'error' in b:
            lines.append(f"| {b['bag']} | ERROR {b['error']} |")
            continue
        s = b['summary']
        lines.append(f"| {b['bag']} | {_fmt(s.get('dist_m'))} | " + ' | '.join(_fmt(s.get(k), k) for k in keys) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--stat', default='mean', choices=['mean', 'median', 'max', 'min'])
    ap.add_argument('--keys', default=None, help='comma-separated headline keys')
    ap.add_argument('--labels', default=None, help='comma-separated column labels')
    ap.add_argument('--per-bag', action='store_true')
    ap.add_argument('--summary-json', default=None, help='write {run: {config, aggregate}} for all files')
    a = ap.parse_args(argv)
    keys = a.keys.split(',') if a.keys else None
    if a.summary_json:
        summary = {}
        for f in a.files:
            if Path(f).name == Path(a.summary_json).name:
                continue
            r = load(f)
            if 'aggregate' not in r:
                continue
            summary[Path(f).stem] = {'estimator': r.get('estimator'), 'params': r.get('params'),
                                     'config': r.get('config'), 'aggregate': r['aggregate']}
        with open(a.summary_json, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=1)
        return
    labels = a.labels.split(',') if a.labels else None
    if a.per_bag:
        for f in a.files:
            print(f'### {Path(f).stem}\n')
            print(per_bag(f, keys) if keys else per_bag(f))
            print()
    else:
        print(table(a.files, keys or DEFAULT_KEYS, a.stat, labels))


if __name__ == '__main__':
    main()
