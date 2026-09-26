"""Sensitivity (+-20 % on the main thresholds) and ablation (one feature off at a time) of the estimator.

    python tools/replay/sensitivity.py --what sens --split val --exe build_core/tbo_replay.exe
    python tools/replay/sensitivity.py --what ablation --split val --exe build_core/tbo_replay.exe
    python tools/replay/sensitivity.py --what ablation --split val --dropwin 60:10   # model-only windows

Each variant is one eval_par run (same bags, same binary); results go to build_core/eval/<tag>.csv and a
summary table to docs/tables/<what>_<split>[_dropwin].csv. A flat response to +-20 % means the defaults are
not a fragile optimum; the ablation table shows what each feature buys.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
from eval_par import summarize  # noqa: E402

# main thresholds and noise levels (defaults read from the binary's --dump-params)
SENS_KEYS = ['sigma_accel', 'q_disturbance', 'sigma_wheel', 'rate_to_bad', 'cusum_slip_accel', 'cusum_slide_accel',
             'cusum_h', 'max_wheel_accel', 'landmark_min_prob', 'landmark_gate_sigma', 'init_sigma_scale',
             'drive_tau_s', 'standstill_kmh', 'disturbance_max']

# one feature off at a time (name -> overrides)
ABLATIONS = {
    'full': {},
    'no_grade': {'map_grade_gain': 0},
    'no_curve_corr': {'wheel_curv_abs': 0, 'wheel_curv_signed': 0},
    'no_landmarks': {'landmark_enable': 0},
    'no_cutoffs': {'cutoff_enable': 0},
    'no_landmarks_cutoffs': {'landmark_enable': 0, 'cutoff_enable': 0},
    'no_maneuver_mode': {'rate_to_maneuver': 0},
    'no_joint_cusum': {'cusum_h': 1e6},
    'no_cmd_check': {'cmd_fault_h': 1e6},
    'no_wrong_sign': {'wrong_sign_penalty': 0},
    'no_lead': {'position_lead_s': 0},
    'no_dfield': {'dfield_gain': 0},
    'narrow_k_prior': {'init_sigma_scale': 0.006},
    'no_robust_fixes': {'joint_need_both': 0, 'joint_d_tau_s': 0, 'cmd_check_absolute': 0,
                        'standstill_exit_no_wheels': 0, 'single_bogie_recover': 0, 'agree_tau_s': 0,
                        'd_clamp_model_only': 0},
}


def defaults(exe: str) -> dict:
    out = subprocess.run([exe, '--dump-params'], capture_output=True, text=True).stdout
    d = {}
    for ln in out.splitlines():
        ln = ln.split('#')[0].strip()
        if ':' in ln:
            k, v = [x.strip() for x in ln.split(':', 1)]
            try:
                d[k] = float(v)
            except ValueError:
                pass
    return d


def run(tag, exe, split, sets, dropwin, jobs, extra_sets):
    cmd = [sys.executable, str(ROOT / 'tools' / 'replay' / 'eval_par.py'), '--tag', tag, '--exe', exe,
           '--split', split, '--jobs', str(jobs)]
    for k, v in {**extra_sets, **sets}.items():
        cmd += ['--set', f'{k}={v}']
    if dropwin:
        cmd += ['--dropwin', dropwin]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    df = pd.read_csv(ROOT / 'build_core' / 'eval' / f'{tag}.csv')
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    s = summarize(df)
    row = {'variant': tag}
    for k in ('v_rmse', 'v_p99', 'along_rmse', 'end_err', 'drift_pct', 'win_v_rmse', 'win_ds_abs'):
        if k in s['all_mean']:
            row[f'{k}_mean'] = s['all_mean'][k]
            row[f'{k}_med'] = s['all_median'][k]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', choices=['sens', 'ablation'], required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--exe', default=str(ROOT / 'build_core' / 'tbo_replay.exe'))
    ap.add_argument('--dropwin', default='')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--set', action='append', default=[], help='common overrides for every variant')
    a = ap.parse_args()
    extra = dict(s.split('=', 1) for s in a.set)
    suffix = f"_{a.split}" + ('_dropwin' if a.dropwin else '')
    rows = []
    if a.what == 'sens':
        d = defaults(a.exe)
        rows.append(run(f'sens_base{suffix}', a.exe, a.split, {}, a.dropwin, a.jobs, extra) | {'param': '-', 'factor': 1.0})
        for k in SENS_KEYS:
            for f in (0.8, 1.2):
                v = d[k] * f
                r = run(f'sens_{k}_{f:g}{suffix}', a.exe, a.split, {k: f'{v:.6g}'}, a.dropwin, a.jobs, extra)
                rows.append(r | {'param': k, 'factor': f, 'value': v})
                print(k, f, {kk: round(vv, 4) for kk, vv in r.items() if kk.endswith('_mean')}, flush=True)
    else:
        for name, sets in ABLATIONS.items():
            r = run(f'abl_{name}{suffix}', a.exe, a.split, sets, a.dropwin, a.jobs, extra)
            rows.append(r | {'ablation': name})
            print(name, {kk: round(vv, 4) for kk, vv in r.items() if kk.endswith('_mean')}, flush=True)
    out = ROOT / 'docs' / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f'{a.what}{suffix}.csv', index=False)
    print('wrote', out / f'{a.what}{suffix}.csv')


if __name__ == '__main__':
    main()
