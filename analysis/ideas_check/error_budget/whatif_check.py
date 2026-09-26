"""Error budget, step 3: counterfactual replays of the check bag to size the along-track error sources.

Each variant re-runs run_check.py with one change (--set) and reports the judge metrics for t < 1270 s:
  oracle_k*     wheel calibration divided by (1 + k_true): the wheel scale error removed (upper bound)
  max_dk_1e-3   landmark_max_dk 0.004 -> 0.001: one place fix may move the wheel scale less
  sig_scale_5e-3 init_sigma_scale 0.015 -> 0.005: tighter wheel-scale prior
  lead_0.02     position_lead_s 0.045 -> 0.02 (timing term)
  p_rand_0.05   landmark_p_random 0.15 -> 0.05 (less of the prior kept after a fix)
  sig_signal1   landmark file with sigma >= 1.0 m for the 'signal' class (queue positions at signals)
  oracle_stops  landmark file + the two unmapped stops of the check bag at their true s (upper bound)
  frozen_k*     wheel calibration divided by (1 + k_true) and init_sigma_scale 1e-5: k frozen at the truth
  --summary     only re-tabulate the variants already run (the replays are run one by one with run_check.py)

    python analysis/ideas_check/error_budget/whatif_check.py [variant ...]
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LMS = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps' / 'landmarks.csv'
CAL = 0.2778805556


def lm_files():
    L = pd.read_csv(LMS, comment='#')
    f1 = HERE / 'tmp' / 'lm_signal_sigma1.csv'
    L1 = L.copy()
    sig = L1.cls == 'signal'
    L1.loc[sig, 'sigma'] = np.maximum(L1.loc[sig, 'sigma'], 1.0)
    L1.to_csv(f1, index=False, float_format='%.3f')
    f2 = HERE / 'tmp' / 'lm_oracle_stops.csv'
    L2 = pd.concat([L, pd.DataFrame({'s': [1508.70, 1720.92], 'sigma': [0.3, 0.3], 'p_stop': [0.5, 0.5],
                                     'cls': ['oracle', 'oracle']})]).sort_values('s')
    L2.to_csv(f2, index=False, float_format='%.3f')
    return f1, f2


def variants():
    f1, f2 = lm_files()
    return {
        'oracle_k0.0010': [f'wheel_kmh_to_ms={CAL / 1.0010:.10f}', 'init_sigma_scale=0.002'],
        'oracle_k0.0012': [f'wheel_kmh_to_ms={CAL / 1.0012:.10f}', 'init_sigma_scale=0.002'],
        'max_dk_1e-3': ['landmark_max_dk=0.001'],
        'sig_scale_5e-3': ['init_sigma_scale=0.005'],
        'lead_0.02': ['position_lead_s=0.02'],
        'p_rand_0.05': ['landmark_p_random=0.05'],
        'sig_signal1': [f'landmark_file={f1}'],
        'oracle_stops': [f'landmark_file={f2}'],
        'frozen_k0.0012': [f'wheel_kmh_to_ms={CAL / 1.0012:.10f}', 'init_sigma_scale=0.00001'],
        'frozen_k0.0015': [f'wheel_kmh_to_ms={CAL / 1.0015:.10f}', 'init_sigma_scale=0.00001'],
    }


def run(name, sets):
    cmd = [sys.executable, str(HERE / 'run_check.py'), '--tag', f'_{name}']
    for s in sets:
        cmd += ['--set', s]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return name, r.stdout + r.stderr


def summarize(tag):
    P = pd.read_parquet(HERE / f'pairs_pos{tag}.parquet')
    V = pd.read_parquet(HERE / f'pairs_vel{tag}.parquet')
    k = P.t < 1270
    r = lambda x: float(np.sqrt(np.mean(np.square(x))))
    return {'v_rmse': r(V.ev), 'p3_rmse_1270': r(P.dist[k]), 'along_1270': r(P.along[k]), 'along_bias': float(P.along[k].mean()),
            'along_max_1270': float(P.along[k].abs().max()), 'p3_rmse_full': r(P.dist),
            'k_end': float(P.k.iloc[-1])}


def main():
    V = variants()
    names = sys.argv[1:] or list(V)
    (HERE / 'tmp').mkdir(exist_ok=True)
    if names == ['--summary']:
        names = []
    with ThreadPoolExecutor(2) as ex:
        for name, log in ex.map(lambda n: run(n, V[n]), names):
            print(f'--- {name}: {V[name]}\n{log.strip()}', flush=True)
    done = [n for n in V if (HERE / f'pairs_pos_{n}.parquet').exists()]
    rows = [{'variant': 'base', **summarize('')}] + [{'variant': n, **summarize(f'_{n}')} for n in done]
    df = pd.DataFrame(rows)
    f = HERE / 'whatif_check.csv'
    if f.exists():
        old = pd.read_csv(f)
        df = pd.concat([old[~old.variant.isin(df.variant)], df], ignore_index=True)
    df.to_csv(f, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
