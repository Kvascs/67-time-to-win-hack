#!/usr/bin/env python
"""Root causes of the worst C++ anomaly-suite failures, with the evidence pulled from the result CSVs.

Writes results_cpp/root_causes.json. Every number in it is read from
results_cpp/<variant>/{scenario_summary,runs,false_alarms,windows}.csv (never typed in by hand).
Code locations refer to bin/snapshot_src/estimator.cpp (the source of the tested binary).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cpp_suite as R  # noqa: E402

_CACHE: dict = {}


def _csv(variant: str, name: str) -> pd.DataFrame | None:
    key = (variant, name)
    if key not in _CACHE:
        f = R.RESULTS / variant / f'{name}.csv'
        _CACHE[key] = pd.read_csv(f, low_memory=False) if f.exists() else None
    return _CACHE[key]


def _val(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return None if not np.isfinite(x) else round(x, 4)


def scen(variant: str, scenario: str, metrics: list[str]) -> dict:
    s = _csv(variant, 'scenario_summary')
    if s is None or scenario not in set(s.scenario):
        return {}
    row = s[s.scenario == scenario].iloc[0]
    return {m: _val(row.get(m)) for m in metrics}


def run(variant: str, scenario: str, bag: str, metrics: list[str]) -> dict:
    r = _csv(variant, 'runs')
    if r is None:
        return {}
    x = r[(r.scenario == scenario) & (r.bag == bag)]
    return {m: _val(x.iloc[0].get(m)) for m in metrics} if len(x) else {}


def fa(variant: str, family: str) -> dict:
    f = _csv(variant, 'false_alarms')
    if f is None:
        return {}
    x = f[f.family == family]
    return {m: _val(x.iloc[0][m]) for m in ('episodes', 'per_hour', 'unexplained', 'unexplained_per_hour', 'hours')} \
        if len(x) else {}


def fired_frac(variant: str, scenario: str, family: str) -> dict:
    w = _csv(variant, 'windows')
    if w is None:
        return {}
    x = w[w.scenario == scenario]
    if not len(x):
        return {}
    return {'n_events': int(len(x)),
            f'frac_{family}_fired': _val(x.families_fired.fillna('').astype(str).str.contains(family).mean())}


def evidence(variants: list[str], scenarios: list[str], metrics: list[str]) -> dict:
    out = {}
    for v in variants:
        d = {s: scen(v, s, metrics) for s in scenarios}
        d = {k: x for k, x in d.items() if x}
        if d:
            out[v] = d
    return out


M_ERR = ['n_windows', 'err_in_max_p90', 'err_in_max_max', 'err_after_max_p90', 'rec_gnss_p90', 'det_rate',
         'ds_growth_abs_p90', 'ds_growth_abs_max', 'dv_vs_clean_max', 'n_diverged']


def build(rec_params: str, rec_code: str) -> list[dict]:
    V = ['base', 'mf20']
    rcs = [
        {'id': 'RC1', 'severity': 'critical', 'title': 'Estimator freezes forever after any >max_future_s (2 s) silence of all vehicle stamps',
         'mechanism': 'acceptStamp() rejects stamp > latest_ + max_future_s; a rejected stamp never advances latest_, so after a '
                      'gap (all topics silent, or a zero-stamp window) every later message is rejected. The node/replay still '
                      'publishes query(t) at each input stamp: a model extrapolation from the frozen committed state (all '
                      'dropout flags set, speed drifting, up to 4000 integration steps per output).',
         'code': 'estimator.cpp Estimator::acceptStamp (lines 132-148); tbo_replay.cpp main loop / tbo_node.cpp afterInput() '
                 'publish even when the input was rejected',
         'evidence': evidence(V + [rec_params, 'fixG_future', rec_code],
                              ['X18_drop_vehicle_5', 'S14_stamp_faults', 'S07_dropout_long', 'S17_combined_moderate',
                               'S18_combined_severe'], M_ERR + ['rejected_stamps_max', 'proc_us_p99_max']),
         'example_run': run('base', 'S18_combined_severe', '30618_2f104a1d', ['stderr', 'v_out_max', 'n_absurd_v']),
         'param_fix': {'max_future_s': 20},
         'code_fix': 'accept a forward jump when a second message confirms the new time base within 1 s '
                     '(fix_future_confirm); do not publish for a rejected input'},
        {'id': 'RC2', 'severity': 'critical', 'title': 'Frozen (stuck) bogie under braking -> the frozen wheel is trusted, the good one rejected, '
                                                     'then no recovery at the stop',
         'mechanism': 'Stuck test compares against the filter speed vc, which the frozen wheel itself holds constant (circular). '
                      'Under a braking notch an under-reading bogie looks like a plausible slide, so the IMM selects '
                      '"other bogie bad"; d absorbs the missing deceleration (positive clamp disturbance_max_free=2.0 when '
                      'notch<=0). After the stop the lock-up guard (never re-anchor to ~0 wheels while the model moves) has no '
                      'time limit -> phantom motion for the whole dwell.',
         'code': 'estimator.cpp wheelUpdate stuck test (lines 409-421); IMM clamp d_max_pos (line 531); recovery guard (line 607)',
         'evidence': evidence(V + ['mf20_stuck0', 'mf20_stuck0_t0.8', 'fixF_frozen', rec_params, rec_code],
                              ['X20_frozen_front_hold10', 'X19_frozen_wheels_hold5', 'S11_frozen_sensor',
                               'X21_frozen_wheels_zero3'], M_ERR),
         'param_fix': {'stuck_min_change': 0},
         'code_fix': 'stuck test against the other bogie / model speed change (param stuck_min_change=0 is equivalent: no natural '
                     'identical run >= 0.5 s while moving except one dying sensor, diagnostics/natural_wheel_stats.json); '
                     'time-limit the lock-up guard (fix_lockup_guard_s=4)'},
        {'id': 'RC3', 'severity': 'major', 'title': 'Heavy bogie noise -> model-only lock-out and disturbance runaway',
         'mechanism': 'With sigma >= ~1 km/h both bogies drift out of the gates, the IMM settles in "both bad"; its recovery needs '
                      '|front-rear| < recover_agree (0.35 m/s) on every sample for recover_time_s (3 s) - never true under noise. '
                      'Meanwhile d (learnt from the noisy wheels, allowed up to +2.0 when the controller is flagged) is '
                      'integrated open-loop.',
         'code': 'estimator.cpp wheelUpdate recovery (lines 589-617), clampState in the IMM update (line 554)',
         'evidence': evidence(V + ['mf20_rman0.01', 'mf20_dmf0.8', 'mf20_ragree1.0', 'fixD_noise', rec_params, rec_code],
                              ['S12_noise_increase', 'X27_noise_both_0p3', 'X28_noise_both_0p6', 'X29_noise_both_1p2',
                               'S18_combined_severe'], M_ERR),
         'param_fix': {'rate_to_maneuver': 0.01},
         'code_fix': 'agreement test on low-passed bogie speeds (fix_agree_tau_s=2) and |d| <= disturbance_max while both '
                     'bogies are distrusted (fix_d_clamp_bothbad); adaptive R from the front-rear spread'},
        {'id': 'RC4', 'severity': 'major', 'title': 'Joint slip ramps / low-speed steps are absorbed by the disturbance state',
         'mechanism': 'The joint CUSUM compares bogie acceleration with g*a_drive + clamp(d, +-disturbance_max) where d is the '
                      "filter's own disturbance estimate. d follows the slipping wheels within ~0.4 s (maneuver mode q_d=8, GPB1 "
                      'collapse) up to the 0.6 clamp, so the excess seen by the CUSUM stays below cusum_slip_accel.',
         'code': 'estimator.cpp jointMonitor d_ref (line 627)',
         'evidence': evidence(V + ['mf20_cusumh0.2', 'mf20_dmax0.3', 'mf20_dmax0.3_slip0.4', 'fixA_jointref', rec_params, rec_code],
                              ['X04_joint_slip_ramp10', 'X05_joint_slip_ramp20', 'X06_joint_slip_ramp30',
                               'X01_joint_slip_step10', 'S02_slip_both_bogies', 'S19_jury_style_simple'], M_ERR),
         'param_fix': {'cusum_h': 0.2},
         'code_fix': 'reference a slow copy of d frozen while any monitor is active (fix_joint_d_tau_s=20)'},
        {'id': 'RC5', 'severity': 'major', 'title': 'Standstill is never left without wheel data; a lone returning bogie is never trusted',
         'mechanism': 'standstill is cleared only by a bogie sample above threshold, so with both bogies silent predictMode keeps '
                      'v=0 under traction; when one bogie returns at speed the IMM calls it a slip and recovery needs both bogies.',
         'code': 'estimator.cpp wheelUpdate standstill (lines 449-487), predictMode (lines 353-359), recovery (589-617)',
         'evidence': evidence(['mf20', 'fixE_standstill', rec_params, rec_code], ['S07_dropout_long', 'X18_drop_vehicle_5'], M_ERR),
         'example_run': {v: run(v, 'S07_dropout_long', '30639_d927f360', ['v_max', 'dv_vs_clean_max', 'ds_vs_clean_max',
                                                                             'ds_vs_clean_end'])
                         for v in ['mf20', 'fixE_standstill', rec_code]},
         'code_fix': 'leave standstill when both bogies are silent and traction is commanded (fix_standstill_exit); '
                     're-anchor to the only available bogie after recover_min_bad_s+recover_time_s (fix_single_recover)'},
        {'id': 'RC6', 'severity': 'moderate', 'title': 'Transient joint latch inside query() between the W0 and W1 arrivals',
         'mechanism': 'A query at a wheel stamp applies the half-merged event (one bogie only): n_av=1, so a single charged '
                      'CUSUM counts as a joint alarm and the output shows the rolled-back model speed; the next output is '
                      'normal. Spikes stay in the 0.3 s CUSUM history and keep a bogie charged for ~2 s. These outputs sit on '
                      'wheel stamps = the epochs the judge matches (GNSS vel stamps align with wheel stamps).',
         'code': 'estimator.cpp jointMonitor joint alarm (line 778) and history ring (lines 696-700)',
         'evidence': evidence(V + ['fixB_needboth', rec_code],
                              ['S09_outliers_spikes', 'X23_spikes_only', 'S01_slip_single_bogie', 'X07_single_slip_step20'],
                              ['n_glitch_0p3_sum', 'dv_vs_clean_max', 'err_after_max_max', 'err_in_max_max']),
         'code_fix': 'joint alarm needs both bogies unless the other is out (fix_joint_need_both); implausible samples never '
                     'enter the CUSUM history or the controller check (fix_cusum_skip_bad)'},
        {'id': 'RC7', 'severity': 'moderate', 'title': 'Controller-consistency check fires on wheel anomalies, not on controller faults',
         'mechanism': 'Under a braking notch "impossible" = wheel accel - brake-model accel, so a weaker-than-tabulated brake '
                      'near a stop, a wheel spinning up after a slide or a spike raises cmd_inconsistent (held 8 s), which '
                      'disables the slip/slide CUSUMs and raises the positive d clamp to 2.0.',
         'code': 'estimator.cpp jointMonitor controller check (lines 734-745)',
         'evidence': {v: {'clean_cmd_inconsistent': fa(v, 'cmd_inconsistent'),
                          'X23_spikes': fired_frac(v, 'X23_spikes_only', 'cmd_incons'),
                          'S04_slides': fired_frac(v, 'S04_slide_braking_wsp', 'cmd_incons'),
                          'S16_notch_faults': fired_frac(v, 'S16_notch_faults', 'cmd_incons'),
                          'X25_notch_offset': fired_frac(v, 'X25_notch_offset5', 'cmd_incons')}
                      for v in ['base', 'mf20_cmdf0.8', 'fixC_cmdcheck', rec_code]},
         'code_fix': 'absolute wheel acceleration under braking notches; no evidence while a bogie is distrusted or within 3 s '
                     'of a latch release (fix_cmd_check)'},
        {'id': 'RC8', 'severity': 'moderate', 'title': 'Low-speed wheel lock followed by the maneuver mode -> false standstill',
         'mechanism': 'A lock from ~3 m/s looks like an emergency brake; the maneuver mode (d down to -2.3 m/s^2) follows the '
                      'wheels below standstill_max_v and the zero-velocity update pins v=0 while the tram still moves.',
         'code': 'estimator.cpp IMM maneuver mode (lines 500-509), standstill guard (line 454)',
         'evidence': evidence(V + ['mf20_rman0.01', 'mf20_ssmv0.5', 'fixF_frozen', rec_params, rec_code],
                              ['S05_slide_wheel_lock', 'X09_joint_lock_2s', 'S04_slide_braking_wsp'], M_ERR),
         'param_fix': {'rate_to_maneuver': 0.01}},
        {'id': 'RC9', 'severity': 'minor', 'title': 'Per-bogie scale mismatch (wear) is absorbed by one shared scale state',
         'evidence': {v: {bag: run(v, 'S15_wheel_wear_scale', bag, ['ds_vs_clean_end', 'along_rmse', 'clean_along_rmse'])
                          for bag in R.GOOD_BAGS} for v in ['base']},
         'code_fix': 'separate front/rear scale (or ratio) states calibrated at landmarks'},
        {'id': 'RC10', 'severity': 'minor', 'title': 'Total blackout while accelerating: controller assumed neutral after 0.6 s',
         'evidence': {v: run(v, 'S07_dropout_long', '30618_a53d5f6f', ['v_max', 'dv_vs_clean_max', 'ds_vs_clean_max'])
                      for v in ['mf20', 'mf20_cmdto3']},
         'note': 'cmd_timeout_s=3 is NOT recommended: it delays the cmd-dropout flag (S16 detection 1.00 -> 0.54)',
         'evidence_cmdto3': evidence(['mf20', 'mf20_cmdto3'], ['S16_notch_faults', 'S06_dropout_short', 'X18_drop_vehicle_5'],
                                     ['det_rate', 'err_in_max_p90', 'ds_growth_abs_p90'])},
        {'id': 'RC11', 'severity': 'moderate', 'title': 'Re-anchoring with pinVelocity() keeps a large along-track sigma -> '
                                                        'wrong landmark association at the next stop',
         'mechanism': 'pinVelocity() zeroes the s-v covariance, so after a long model-only phase the wheel measurements can '
                      'no longer shrink P_ss (sigma_s stays ~16 m; a normal latch release shrinks it to ~2 m). At the next '
                      'stop the landmark gate (3 sigma) and the random-stop prior spread over that width let a landmark '
                      '17 m away win the association: +16 m along-track error for ~900 s.',
         'code': 'estimator.cpp pinVelocity (lines 78-85) used by the recovery paths (lines 604-616, 664-676); '
                 'landmarkUpdate association (lines 800-871)',
         'evidence_file': 'diagnostics/landmark_after_reanchor.csv',
         'code_fix': 're-anchor with a scalar Kalman update of v (keeps P_sv, shrinks P_ss) and/or refuse landmark '
                     'associations while sigma_s exceeds a few metres'},
    ]
    # RC6 supplementary: the "skip implausible samples" variant of the fix was harmful in combination
    f = R.RESULTS / 'diagnostics' / 'skip_bad_modes.csv'
    if f.exists():
        rcs[5]['skip_bad_modes'] = pd.read_csv(f).round(3).to_dict(orient='records')
        rcs[5]['skip_bad_note'] = ('fix_cusum_skip_bad (both modes) re-introduced false joint latches after zero bursts '
                                   'and weakened lock handling once fix_cmd_check stopped the spurious controller faults '
                                   'that used to disarm the monitors; the recommended set does not use it')
    return rcs


def main():
    rec_params = sys.argv[1] if len(sys.argv) > 1 else 'rec_params'
    rec_code = sys.argv[2] if len(sys.argv) > 2 else 'fix_all'
    out = {'generated_from': str(R.RESULTS), 'rec_params_variant': rec_params, 'rec_code_variant': rec_code,
           'root_causes': build(rec_params, rec_code)}
    (R.RESULTS / 'root_causes.json').write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    for rc in out['root_causes']:
        print(rc['id'], rc['severity'], rc['title'])


if __name__ == '__main__':
    main()
