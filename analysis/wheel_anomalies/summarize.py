"""Merge all per-bag findings into bag_difficulty.csv and the master episode list episodes_all.csv.

episodes_all.csv (one row per anomaly/event, all sources):
  source  = gnss_label (slip/slide/lock/stuck_zero vs GNSS truth) | stream (dropout gaps, stamp issues,
            negative/frozen values) | monitor_nognss (GNSS-free detections in bags without GNSS) |
            controller (accelerating with non-traction handle) | reference (GNSS reference faults)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import anomalies as A  # noqa: E402

SP = A.splits()
INFO = {x['bag']: x for x in SP['info']}


def split_of(b):
    for k in ('train', 'val', 'no_gnss_long', 'short'):
        if b in SP[k]:
            return k
    return '?'


def main():
    sc = pd.read_csv(HERE / 'scale_factors.csv').set_index('bag')
    eg = pd.read_csv(HERE / 'episodes_gnss.csv')
    se = pd.read_csv(HERE / 'stream_events.csv')
    it = pd.read_csv(HERE / 'integrity_topics.csv')
    gi = pd.read_csv(HERE / 'gnss_integrity.csv')
    ni = pd.read_csv(HERE / 'notch_inconsistency.csv').set_index('bag') if (HERE / 'notch_inconsistency.csv').exists() else None
    mf = pd.read_csv(HERE / 'monitor_flags.csv') if (HERE / 'monitor_flags.csv').exists() else None
    rows = []
    for b in sorted(INFO, key=lambda x: INFO[x]['t0']):
        r = dict(bag=b, split=split_of(b), vehicle=b[:5], dur_s=round(INFO[b]['dur'], 1))
        if b in sc.index:
            r.update(k_front=round(sc.loc[b, 'k_speed_front'], 4), k_dist_all=round(sc.loc[b, 'k_dist_all_front'], 4),
                     scale_dev_pct=round((sc.loc[b, 'k_dist_all_front'] / 3.597 - 1) * 100, 3))
        e = eg[eg.bag == b]
        for kind in ('slip', 'slide', 'lock', 'stuck_zero'):
            r[f'n_{kind}'] = int((e.kind == kind).sum())
        r['max_abs_err_ms'] = round(float(e.e_peak.abs().max()), 2) if len(e) else 0.0
        s = se[se.bag == b]
        g = s[(s.kind == 'gap') & (s.topic.isin(['front', 'rear']))]
        r['n_wheel_gaps_gt1s'] = int((g.dur > 1.0).sum())
        r['max_wheel_gap_s'] = round(float(g.dur.max()), 1) if len(g) else 0.0
        both = 0
        gf = g[g.topic == 'front']; gr = g[g.topic == 'rear']
        for _, x in gf.iterrows():
            if ((gr.t_rel - x.t_rel).abs() < 0.3).any() and x.dur > 0.5:
                both += 1
        r['n_both_sensor_gaps_gt0p5'] = both
        r['n_stamp_nonmono'] = int((s.kind == 'stamp_nonmonotonic').sum())
        ex = s[(s.kind == 'stamp_excursion') & (s.topic == 'cmd')]
        offs = ex.detail.str.extract(r'latency offset ([+-][0-9.]+)s')[0].astype(float) if len(ex) else pd.Series(dtype=float)
        # >= 0.5 s latency steps outside the start-up phase (t > 3 s) = recorder-clock steps
        r['n_recorder_clock_steps'] = int(((offs.abs() >= 0.5).values & (ex.t_rel > 3.0).values).sum()) if len(ex) else 0
        r['n_negative'] = int((s.kind == 'negative').sum())
        r['n_frozen'] = int((s.kind == 'frozen').sum())
        gg = gi[gi.bag == b]
        r['gnss_hdr_clock_excursion_s'] = round(float(gg.hdr_clock_excursion_s.max()), 1) if len(gg) and gg.hdr_clock_excursion_s.notna().any() else np.nan
        r['gnss_exact_zero_msgs'] = int(gg.n_exact_zero.fillna(0).sum()) if len(gg) else 0
        r['gnss_gap_time_s'] = round(float(gg.total_gap_time_s.max()), 1) if len(gg) and gg.total_gap_time_s.notna().any() else np.nan
        if ni is not None and b in ni.index:
            r['frac_acc_without_traction'] = round(float(ni.loc[b, 'frac_acc_without_trac']), 4)
            r['n_acc_notrac_segments'] = int(ni.loc[b, 'n_segs_acc_notrac_3s'])
        if mf is not None:
            m = mf[mf.bag == b]
            r['monitor_flag_runs'] = int(len(m))
            r['monitor_flag_runs_gt0p5s'] = int((m.dur > 0.5).sum())
        tags = []
        if r.get('n_slip', 0) + r.get('n_slide', 0) + r.get('n_lock', 0) > 0 and r['max_abs_err_ms'] >= 1.0:
            tags.append('SLIP/SLIDE')
        if r.get('n_stuck_zero', 0) > 0 or r['max_wheel_gap_s'] > 5:
            tags.append('SENSOR_DROPOUT')
        if r['n_both_sensor_gaps_gt0p5'] > 0:
            tags.append('BOTH_SENSOR_GAP')
        if r['n_stamp_nonmono'] > 0 or r['n_recorder_clock_steps'] > 0:
            tags.append('STAMP_ISSUES')
        if r.get('gnss_hdr_clock_excursion_s', 0) and r['gnss_hdr_clock_excursion_s'] > 5:
            tags.append('GNSS_CLOCK')
        if r['gnss_exact_zero_msgs'] > 30:
            tags.append('GNSS_ZEROS')
        if abs(r.get('scale_dev_pct', 0)) > 0.4:
            tags.append('SCALE_OFF')
        if r.get('n_acc_notrac_segments', 0) >= 2:
            tags.append('HANDLE_INCONSISTENT')
        if r['n_negative'] > 0:
            tags.append('ROLLBACK')
        if r['split'] in ('no_gnss_long', 'short') and r.get('monitor_flag_runs_gt0p5s', 0) > 0:
            tags.append('NOGNSS_CANDIDATES')
        r['tags'] = ' '.join(tags)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'bag_difficulty.csv', index=False)

    # master episode list
    out = []
    for _, e in eg.iterrows():
        out.append(dict(bag=e.bag, source='gnss_label', topic=e.sensor, kind=e.kind, regime=e.regime,
                        t_start=round(e.t_start, 2), dur=round(e.dur, 2), magnitude=round(e.e_peak, 3),
                        magnitude_rel=round(e.e_rel_peak, 3) if pd.notna(e.e_rel_peak) else np.nan,
                        v_ref=round(e.v_ref_peak, 2), notch_mode=e.notch_mode, notch_min=e.notch_min, notch_max=e.notch_max,
                        fr_max_abs=round(e.fr_max_abs, 3), a_wheel_max=round(e.a_wheel_max, 2), a_wheel_min=round(e.a_wheel_min, 2),
                        both_bogies=e.both, dist_err_m=round(e.dist_err_m, 2), confidence=e.confidence,
                        detail=e.note if isinstance(e.note, str) else ''))
    for _, s in se.iterrows():
        if s.kind in ('startup_burst',):
            continue
        if s.kind == 'gap' and s.dur < 0.5:
            continue
        if s.kind == 'arrival_gap' and s.dur < 0.75:
            continue
        out.append(dict(bag=s.bag, source='stream', topic=s.topic, kind=s.kind, t_start=round(s.t_rel, 2),
                        dur=round(s.dur, 3) if pd.notna(s.dur) else np.nan, detail=s.detail))
    if mf is not None:
        for _, m in mf[~mf.bag.isin(sc.index)].iterrows():
            if m.dur < 0.3:
                continue
            out.append(dict(bag=m.bag, source='monitor_nognss', topic=m.sensor, kind=m.kinds, t_start=round(m.t_start, 2),
                            dur=round(m.dur, 2), notch_mode=m.notch, magnitude=round(m.v_max - m.v_hat, 2) if m.v_max > m.v_hat else round(m.v_min - m.v_hat, 2),
                            detail=f'classes={m.classes} v_range=[{m.v_min:.2f},{m.v_max:.2f}] v_model={m.v_hat:.2f}'))
    ea = pd.DataFrame(out).sort_values(['bag', 't_start'])
    ea.to_csv(HERE / 'episodes_all.csv', index=False)
    print(df[df.tags != ''][['bag', 'split', 'tags']].to_string())
    print('episodes_all rows', len(ea), ea.groupby(['source', 'kind']).size().to_string())


if __name__ == '__main__':
    main()
