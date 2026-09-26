"""Parameter sweep of the C++ estimator (snapshot binary) with --set overrides, scored by the harness.

Tune on 'train' (never on 'val'); confirm the chosen setting once on 'val'.

    python -m harness.tbo_sweep --split train --variants default isc10 lmp05 ...
    python -m harness.tbo_sweep --split train --table            # comparison table of finished variants

Writes harness/results/tbo_sweep_<split>_<variant>.json/.csv and tbo_sweep_<split>_table.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from harness.cpp_estimator import CppConfig, evaluate_many_cpp, write_csv
from harness.loader import resolve_bags
from harness.replay import EvalConfig
from harness.reference import RefConfig
from harness.run_eval import sanitize

R = Path(__file__).resolve().parent / 'results'

VARIANTS = {
    'default': {},
    'lm_off': {'landmark_enable': 0},
    'isc10': {'init_sigma_scale': 0.010},
    'isc15': {'init_sigma_scale': 0.015},
    'lmp05': {'landmark_min_prob': 0.5},
    'lmp06': {'landmark_min_prob': 0.6},
    'lmpr05': {'landmark_p_random': 0.05},
    'lmpr30': {'landmark_p_random': 0.30},
    'dk10': {'landmark_max_dk': 0.010},
    'qs1e8': {'q_scale': 1e-8},
    'gate4': {'landmark_gate_sigma': 4.0},
    'gate2': {'landmark_gate_sigma': 2.0},
    'isc10_dk10': {'init_sigma_scale': 0.010, 'landmark_max_dk': 0.010},
    'isc10_lmp06': {'init_sigma_scale': 0.010, 'landmark_min_prob': 0.6},
    'isc10_dk10_lmp06': {'init_sigma_scale': 0.010, 'landmark_max_dk': 0.010, 'landmark_min_prob': 0.6},
    'isc15_dk15': {'init_sigma_scale': 0.015, 'landmark_max_dk': 0.015},
    # timing (judge time base sensitivity)
    'pl0': {'position_lead_s': 0.0},
    'plm03': {'position_lead_s': -0.03},
    'pl09': {'position_lead_s': 0.09},
    # fault handling
    'wsp0': {'wrong_sign_penalty': 0.0},
    'smc015': {'stuck_min_change': 0.15},
    'cslide08': {'cusum_slide_accel': 0.8},
    'cslip035': {'cusum_slip_accel': 0.35},
    'ch02': {'cusum_h': 0.2},
    'mf30': {'max_future_s': 30},
    # publish the state L seconds "late" (hedge for a bag-receive-time judge): wheels/cmd events are
    # placed L later, position lead reduced by L
    'shift25': {'wheel_delay_s': -0.025, 'cmd_delay_s': 0.025, 'position_lead_s': 0.020},
    'shift50': {'wheel_delay_s': -0.050, 'cmd_delay_s': 0.050, 'position_lead_s': -0.005},
    # combined candidates
    'isc15_mf30': {'init_sigma_scale': 0.015, 'max_future_s': 30},
    'isc20': {'init_sigma_scale': 0.020},
    'isc15_lmp06': {'init_sigma_scale': 0.015, 'landmark_min_prob': 0.6},
    'isc15_lmp05': {'init_sigma_scale': 0.015, 'landmark_min_prob': 0.5},
    'isc15_dk15_lmp06': {'init_sigma_scale': 0.015, 'landmark_max_dk': 0.015, 'landmark_min_prob': 0.6},
    # recommended combination (train-tuned): landmarks/scale + fault handling + lock-up fix
    'rec1': {'init_sigma_scale': 0.015, 'landmark_min_prob': 0.6, 'wrong_sign_penalty': 0.0, 'max_future_s': 30},
    'rec2': {'init_sigma_scale': 0.015, 'landmark_min_prob': 0.6, 'cusum_slide_accel': 0.8, 'max_future_s': 30},
    'rec2_noslide': {'init_sigma_scale': 0.015, 'landmark_min_prob': 0.6, 'max_future_s': 30},
    'rec3': {'init_sigma_scale': 0.015, 'landmark_min_prob': 0.6, 'cusum_slide_accel': 0.8, 'max_future_s': 30,
             'stuck_min_change': 0.15, 'cusum_slip_accel': 0.35},
}


def tag(split: str, faults=None, time_base: str = 'header') -> str:
    t = split.replace('+', '_')
    if time_base != 'header':
        t += f'_ref{time_base}'
    if faults:
        t += '_' + '_'.join(f.replace('suite:', '') for f in faults)
    return t


def bags_of(split: str):
    """split name, or '<split>_even' = every other bag of <split> (cheaper sweeps)."""
    if split.endswith('_even'):
        return resolve_bags(split[:-5])[::2]
    return resolve_bags(split)


def run_variant(name: str, sets: dict, split: str, jobs: int, time_base: str = 'header', gnss_seconds: float = 5.0,
                faults=None):
    bags = bags_of(split)
    cfg = EvalConfig(gnss_seconds=gnss_seconds, ref=RefConfig(time_base=time_base), faults=faults)
    cc = CppConfig(sets={k: str(v) for k, v in sets.items()})
    out = evaluate_many_cpp(bags, cfg, cc, jobs=jobs, label=f'sweep:{name}')
    stem = f'tbo_sweep_{tag(split, faults, time_base)}_{name}'
    with open(R / f'{stem}.json', 'w', encoding='utf-8') as f:
        json.dump(sanitize(out), f, indent=1)
    write_csv(out, R / f'{stem}.csv')
    return out


def good_ref(b: dict) -> bool:
    rd = b.get('ref_diag', {})
    return rd.get('frac_status2', 0) > 0.8 and rd.get('frac_outlier', 1) < 0.02


def table(split: str, faults=None, time_base: str = 'header') -> pd.DataFrame:
    rows = []
    t = tag(split, faults, time_base)
    for p in sorted(R.glob(f'tbo_sweep_{t}_*.json')):
        res = json.loads(p.read_text(encoding='utf-8'))
        if 'bags' not in res:
            continue
        name = p.stem.split(f'tbo_sweep_{t}_', 1)[1]
        if not faults and (res.get('config') or {}).get('faults'):
            continue
        if ((res.get('config') or {}).get('ref') or {}).get('time_base', 'header') != time_base:
            continue
        bags = [b for b in res['bags'] if 'error' not in b and 'summary' in b and b['summary'].get('along_rmse') is not None]
        for subset, sel in (('all', bags), ('goodref', [b for b in bags if good_ref(b)])):
            if not sel:
                continue
            s = pd.DataFrame([b['summary'] for b in sel])
            n_al = np.array([b['pos']['along_arc']['n'] for b in sel], float)
            n_v = np.array([b['speed']['all']['n'] for b in sel], float)
            rows.append({
                'variant': name, 'subset': subset, 'n_bags': len(sel), 'sets': json.dumps(res.get('params', {})),
                'along_rmse_mean': s.along_rmse.mean(), 'along_rmse_median': s.along_rmse.median(),
                'along_rmse_pooled': float(np.sqrt(np.sum(n_al * s.along_rmse ** 2) / n_al.sum())),
                'along_max_mean': s.along_max.mean(), 'along_max_max': s.along_max.max(),
                'n_along_rmse_gt5': int((s.along_rmse > 5).sum()), 'n_along_rmse_gt10': int((s.along_rmse > 10).sum()),
                'pos_rmse3d_mean': s.pos_rmse3d.mean(), 'pos_rmse3d_median': s.pos_rmse3d.median(),
                'drift_pct_mean': s.drift_pct_3d.mean(), 'drift_pct_median': s.drift_pct_3d.median(),
                'v_rmse_mean': s.v_rmse.mean(), 'v_rmse_median': s.v_rmse.median(),
                'v_rmse_pooled': float(np.sqrt(np.sum(n_v * s.v_rmse ** 2) / n_v.sum())),
                'v_bias_mean': s.v_bias.mean(), 'v_mae_mean': s.v_mae.mean(),
                'n_landmark_fixes_mean': float(np.mean([b['cpp']['est']['n_landmark_fixes'] for b in sel])),
            })
            # per-fault-window medians (fault runs)
            per_spec = {}
            for b in sel:
                for w in (b.get('windows') or {}).get('applied', []):
                    per_spec.setdefault(w['spec'], []).append(w)
            for spec, lst in per_spec.items():
                v = [w['v_rmse_in'] for w in lst if w.get('v_rmse_in') is not None]
                da = [abs(w['d_along']) for w in lst if w.get('d_along') is not None]
                rows[-1][f'win_vrmse_med[{spec}]'] = float(np.median(v)) if v else None
                rows[-1][f'win_absdalong_med[{spec}]'] = float(np.median(da)) if da else None
    d = pd.DataFrame(rows)
    if len(d):
        d.to_csv(R / f'tbo_sweep_{t}_table.csv', index=False, float_format='%.6g')
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='train')
    ap.add_argument('--variants', nargs='*', default=[])
    ap.add_argument('--set', dest='sets', action='append', default=[], help='ad-hoc variant: name:key=val,key=val')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--faults', nargs='*', default=None, help='e.g. suite:basic (window metrics are added)')
    ap.add_argument('--time-base', default='header', choices=['header', 'bag', 'header_fixed'])
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--table', action='store_true')
    a = ap.parse_args(argv)
    todo = [(v, VARIANTS[v]) for v in a.variants]
    for s in a.sets:
        name, _, kvs = s.partition(':')
        todo.append((name, dict(kv.split('=', 1) for kv in kvs.split(',') if kv)))
    for name, sets in todo:
        stem = R / f'tbo_sweep_{tag(a.split, a.faults, a.time_base)}_{name}.json'
        if stem.exists() and not a.force:
            print(f'skip {name}')
            continue
        out = run_variant(name, sets, a.split, a.jobs, time_base=a.time_base, faults=a.faults)
        agg = out['aggregate']
        print(f"{name:20s} {sets}  along_rmse mean={agg['along_rmse']['mean']:.3f} med={agg['along_rmse']['median']:.3f}"
              f"  v_rmse mean={agg['v_rmse']['mean']:.4f}  wall={out['wall_s']:.0f}s", flush=True)
    if a.table or todo:
        d = table(a.split, a.faults, a.time_base)
        if len(d):
            pd.set_option('display.width', 250)
            cols = ['variant', 'subset', 'n_bags', 'along_rmse_mean', 'along_rmse_median', 'along_rmse_pooled',
                    'along_max_max', 'n_along_rmse_gt5', 'drift_pct_mean', 'v_rmse_mean', 'v_rmse_pooled', 'v_bias_mean',
                    'n_landmark_fixes_mean']
            print(d[cols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
