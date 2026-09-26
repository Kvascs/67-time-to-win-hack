"""Print markdown comparison tables of the standard study runs (harness/results/*.json) to stdout.

    python -m harness.report_tables
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.compare import table, per_bag, load, _fmt

R = Path(__file__).resolve().parent / 'results'


def have(*names):
    return [str(R / f'{n}.json') for n in names if (R / f'{n}.json').exists()]


def labels(*names):
    return [n for n in names if (R / f'{n}.json').exists()]


def section(title, names, keys, stat='median', labs=None):
    files = have(*names)
    if not files:
        return ''
    labs = labs or labels(*names)
    return f'#### {title} ({stat} over val bags)\n\n' + table(files, keys, stat, labs) + '\n\n'


def main():
    out = []
    base = ['b0_val', 'b1_val', 'b2_val']
    k_main = ['v_rmse', 'v_mae', 'v_bias', 'v_max', 'v_rmse_accel', 'v_bias_accel', 'v_rmse_brake', 'v_bias_brake',
              'v_rmse_trans', 'v_rmse_stop', 'pos_rmse3d', 'pos_rmse2d', 'z_rmse', 'along_mean', 'along_mean_abs',
              'along_rmse', 'along_max', 'cross_map_rmse', 'final_err3d', 'drift_pct_3d', 'match_v', 'match_pos',
              'rate_hz', 'max_gap_s', 'proc_p99_ms', 'stamp_age_p95_ms']
    for stat in ('median', 'mean'):
        out.append(section('Baselines', base, k_main, stat))
    for n in base:
        if (R / f'{n}.json').exists():
            out.append(f'#### per-bag {n}\n\n' + per_bag(str(R / f'{n}.json')) + '\n\n')
    k_t = ['v_rmse', 'v_bias_accel', 'v_bias_brake', 'v_rmse_accel', 'v_rmse_brake', 'v_rmse_trans', 'along_mean',
           'along_rmse', 'match_v', 'match_pos', 'rate_hz', 'stamp_age_p95_ms']
    out.append(section('Stamp convention x reference time base, outputs on wheel msgs (B1)',
                       ['b1_val', 'b1_refhdr_stampbag', 'b1_refbag_stamphdr', 'b1_refbag_stampbag', 'b1_reffix_stamphdr'],
                       k_t, 'median',
                       ['refHDR/stHDR', 'refHDR/stBAG', 'refBAG/stHDR', 'refBAG/stBAG', 'refHDRfix/stHDR']))
    names40 = ['b2_val', 'b2_40hz_hdrx_refhdr', 'b2_40hz_hdrx_refhdr_pred', 'b2_40hz_hdrx_refhdr_pred20',
               'b2_40hz_hdrx_reffix_pred20', 'b2_40hz_bag_refbag', 'b2_40hz_bag_refbag_pred',
               'b2_40hz_hdrx_refbag_pred20', 'b2_40hz_bag_refhdr', 'b2_40hz_hdrx_refhdr_trap_pl45',
               'b2_40hz_bag_refbag_trap_pl45']
    out.append(section('Output rate / stamping / state prediction (B2)', names40, k_t, 'median'))
    out.append(section('Output rate / stamping / state prediction (B2)', names40, k_t, 'mean'))
    k_j = ['v_rmse', 'v_bias', 'pos_rmse3d', 'along_rmse', 'along_max', 'drift_pct_3d', 'match_v', 'match_pos']
    out.append(section('Judge-definition knobs (B1)', ['b1_val', 'b1_out2ref', 'b1_cleanref', 'b1_utm', 'b1_speed3d'],
                       k_j, 'median'))
    out.append(section('Judge-definition knobs (B1)', ['b1_val', 'b1_out2ref', 'b1_cleanref', 'b1_utm', 'b1_speed3d'],
                       k_j, 'mean'))
    # robustness
    rows = ['| run | v_rmse | anomaly frac | v_rmse in anomaly | naive in anomaly | slip_any | naive slip_any | '
            'dropout | naive dropout | recovery | spike>1m/s frac | drift % |', '|---|' + '---|' * 11]
    for n in ['b1_val', 'b2_val', 'b1_faults_realistic', 'b2_faults_realistic', 'b2_faults_basic', 'b2_faults_garbage']:
        p = R / f'{n}.json'
        if not p.exists():
            continue
        r = load(p)
        import numpy as np

        def med(path):
            vals = []
            for b in r['bags']:
                if 'error' in b:
                    continue
                cur = b
                for k in path.split('.'):
                    cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is not None:
                    vals.append(cur)
            return float(np.median(vals)) if vals else None
        rows.append(f"| {n} | {_fmt(med('summary.v_rmse'))} | {_fmt(med('robust.anomaly.frac_time'))} | "
                    f"{_fmt(med('robust.anomaly.rmse'))} | {_fmt(med('robust.anomaly.naive_rmse'))} | "
                    f"{_fmt(med('robust.slip_any.rmse'))} | {_fmt(med('robust.slip_any.naive_rmse'))} | "
                    f"{_fmt(med('robust.dropout.rmse'))} | {_fmt(med('robust.dropout.naive_rmse'))} | "
                    f"{_fmt(med('robust.recovery.rmse'))} | {_fmt(med('robust.spike_frac_1mps'))} | "
                    f"{_fmt(med('summary.drift_pct_3d'))} |")
    out.append('#### Robustness (median over val bags; anomaly windows found from wheels vs GNSS truth)\n\n'
               + '\n'.join(rows) + '\n\n')
    # reference stats
    p = R / 'refstats_val.json'
    if p.exists():
        rs = json.loads(p.read_text(encoding='utf-8'))
        keys = ['bag', 'frac_status2', 'frac_outlier', 'hdr_glitch_fix', 'v_master_vs_rover_rmse',
                'wheel_best_rmse_clean', 'wheel_ratio_k_front', 'frame_enu_vs_utm_max_m', 'antenna_baseline_m',
                'alt_range_m', 'distance_path_m', 'distance_vint_m', 'moving_at_start']
        lines = ['| ' + ' | '.join(keys) + ' |', '|' + '---|' * len(keys)]
        for row in rs:
            lines.append('| ' + ' | '.join(_fmt(row.get(k), k) if not isinstance(row.get(k), bool) else str(row.get(k))
                                             for k in keys) + ' |')
        out.append('#### Reference (GNSS truth) quality per val bag\n\n' + '\n'.join(lines) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
