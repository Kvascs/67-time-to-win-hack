"""Study driver: C++ estimator (snapshot) under the judge-replica harness, all configurations of the
evaluation plan, plus Python baselines re-run with the same per-fault-window metrics.

    python -m harness.tbo_study --runs task2            # judge-definition / frame / GNSS-window variants
    python -m harness.tbo_study --runs faults refs      # fault suites (C++) + baselines B1/B2 (same windows)
    python -m harness.tbo_study --runs val_gnss1 --force
    python -m harness.tbo_study --summary               # results/tbo_summary.{json,csv}, tbo_regimes.csv, tbo_faults.csv

Every run writes harness/results/tbo_<name>.json (+ .csv per bag). Baseline runs are named tbo_ref_*.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from harness.loader import resolve_bags
from harness.reference import RefConfig
from harness.replay import EvalConfig, evaluate_bag
from harness.faults import SUITES
from harness import metrics as M
from harness.cpp_estimator import (CppConfig, evaluate_many_cpp, fault_windows, flat_row, per_sample_errors,
                                   window_metrics, write_csv)
from harness.run_eval import sanitize

R = Path(__file__).resolve().parent / 'results'
SUITE_NAMES = ('realistic', 'basic', 'garbage', 'dropouts', 'slip')
ALL_WINDOWS = {s: SUITES[s] for s in SUITE_NAMES}
P_BASE = {'stamp_mode': 'hdr_extrap', 'emit_on_cmd': True, 'extrapolate': True, 'lead': 0.02}  # as run_all.sh

# name -> kwargs: judge options (EvalConfig/RefConfig) + 'sets' (binary overrides) + 'windows'
CPP_RUNS = {
    'val': dict(windows=ALL_WINDOWS),
    'val_refbag': dict(time_base='bag'),
    'val_reffix': dict(time_base='header_fixed'),
    'val_cleanref': dict(clean=True),
    'val_speed3d': dict(speed_dims=3),
    'val_out2ref': dict(direction='out2ref'),
    'val_utm': dict(frame='utm'),                                          # judge UTM, estimator UTM
    'val_utm_estenu': dict(frame='utm', sets={'output_frame': 'enu'}),     # judge UTM, estimator ENU (mismatch)
    'val_enu_estutm': dict(frame='enu', sets={'output_frame': 'utm'}),     # judge ENU, estimator UTM (mismatch)
    'val_gnss1': dict(gnss_seconds=1.0),
    'val_gnss3': dict(gnss_seconds=3.0),
    'val_gnss10': dict(gnss_seconds=10.0),
    # binary's own window relaxed to N+5 s (covers the start-up burst): what the harness delivers is used
    'val_gnss1_win6': dict(gnss_seconds=1.0, sets={'gnss_init_window_s': 6}),
    'val_gnss3_win8': dict(gnss_seconds=3.0, sets={'gnss_init_window_s': 8}),
    'val_gnss5_win10': dict(gnss_seconds=5.0, sets={'gnss_init_window_s': 10}),
    'val_gnss10_win15': dict(gnss_seconds=10.0, sets={'gnss_init_window_s': 15}),
}
for _s in SUITE_NAMES:
    CPP_RUNS[f'faults_{_s}'] = dict(faults=[f'suite:{_s}'])
# proposed-fix emulations: GNSS window anchored at the first GNSS fix ('fg'); lag_window_s=6 lets
# updateAnchor's query(t_med) rewind to the median fix time (emulates fixing the anchor-time bug;
# CPU-heavy, diagnostic only)
for _n in (1.0, 3.0, 5.0, 10.0):
    CPP_RUNS[f'val_gnss{_n:g}_fg'] = dict(gnss_seconds=_n, anchor='first_gnss')
    CPP_RUNS[f'val_gnss{_n:g}_fg_lag6'] = dict(gnss_seconds=_n, anchor='first_gnss', sets={'lag_window_s': 6})
CPP_RUNS['val_lag6'] = dict(sets={'lag_window_s': 6})
CPP_RUNS['val_gnss10_lag6'] = dict(gnss_seconds=10.0, sets={'lag_window_s': 6})
GNSS_FIX_RUNS = [f'val_gnss{_n:g}_fg' for _n in (1.0, 3.0, 5.0, 10.0)] + \
    [f'val_gnss{_n:g}_fg_lag6' for _n in (1.0, 3.0, 5.0, 10.0)] + ['val_lag6', 'val_gnss10_lag6']
# max_future_s=2 locks the estimator after any all-input gap > 2 s (acceptStamp never advances latest_)
CPP_RUNS['val_mf30'] = dict(sets={'max_future_s': 30}, windows=ALL_WINDOWS)
for _s in SUITE_NAMES:
    CPP_RUNS[f'faults_{_s}_mf30'] = dict(faults=[f'suite:{_s}'], sets={'max_future_s': 30})
MF30_RUNS = ['val_mf30'] + [f'faults_{_s}_mf30' for _s in SUITE_NAMES]
# what-if: full-data map geometry (adds branch fan_F3; NOT honest for val: val traces are in it),
# train-only landmarks. Shows the effect of map coverage on the off-map starts.
WHATIF_MAP = Path(__file__).resolve().parent / 'bin' / 'whatif_fullmap'
CPP_RUNS['whatif_val_fullmap'] = dict(map_dir=str(WHATIF_MAP), landmarks=str(WHATIF_MAP / 'landmarks_train.csv'))

PY_RUNS = {
    'ref_b1P_val': dict(est='harness.baselines:B1', params=P_BASE, windows=ALL_WINDOWS),
    'ref_b2P_val': dict(est='harness.baselines:B2', params=P_BASE, windows=ALL_WINDOWS),
}
for _s in SUITE_NAMES:
    PY_RUNS[f'ref_b1P_faults_{_s}'] = dict(est='harness.baselines:B1', params=P_BASE, faults=[f'suite:{_s}'])
    PY_RUNS[f'ref_b2P_faults_{_s}'] = dict(est='harness.baselines:B2', params=P_BASE, faults=[f'suite:{_s}'])

GROUPS = {
    'task2': ['val', 'val_refbag', 'val_reffix', 'val_cleanref', 'val_speed3d', 'val_out2ref', 'val_utm',
              'val_utm_estenu', 'val_enu_estutm', 'val_gnss1', 'val_gnss3', 'val_gnss10', 'val_gnss1_win6',
              'val_gnss3_win8', 'val_gnss5_win10', 'val_gnss10_win15'],
    'faults': [f'faults_{s}' for s in SUITE_NAMES],
    'refs': list(PY_RUNS),
    'gnssfix': GNSS_FIX_RUNS,
    'mf30': MF30_RUNS,
}


def make_cfg(kw: dict) -> EvalConfig:
    ref = RefConfig(frame=kw.get('frame', 'enu'), time_base=kw.get('time_base', 'header'),
                    speed_dims=kw.get('speed_dims', 2), clean=kw.get('clean', False),
                    antenna=kw.get('antenna', 'master'))
    return EvalConfig(gnss_seconds=kw.get('gnss_seconds', 5.0), gnss_mode=kw.get('gnss_mode', 'first_n'),
                      tol=kw.get('tol', 0.05), direction=kw.get('direction', 'ref2out'), ref=ref,
                      faults=kw.get('faults'), fault_seed=kw.get('fault_seed', 0),
                      tick_hz=kw.get('tick_hz', 0.0))


# ----------------------------------------------------------------------------------------------
# Python baselines with window metrics
# ----------------------------------------------------------------------------------------------
def _py_worker(args):
    bag, est, params, cfg, windows = args
    r = evaluate_bag(bag, est, params, cfg, keep_log=True)
    if 'error' in r:
        return r
    try:
        ref, log, run_bag = r['_ref'], r['_log'], r['_bag']
        if ref is not None and (cfg.faults or windows):
            pse = per_sample_errors(ref, log, run_bag, cfg.tol, cfg.direction)
            r['windows'] = {}
            if cfg.faults:
                r['faults_applied'] = run_bag.meta.get('faults', [])
                r['windows']['applied'] = window_metrics(pse, r['faults_applied'])
            for name, specs in (windows or {}).items():
                from harness.loader import load_bag
                r['windows'][name] = window_metrics(pse, fault_windows(load_bag(bag), specs, cfg.fault_seed))
    except Exception as ex:
        r['windows_error'] = f'{type(ex).__name__}: {ex}\n{traceback.format_exc()}'
    for k in ('_log', '_ref', '_bag'):
        r.pop(k, None)
    return r


def run_py(name: str, kw: dict, bags, jobs: int) -> dict:
    cfg = make_cfg(kw)
    t0 = time.time()
    tasks = [(b, kw['est'], kw.get('params', {}), cfg, kw.get('windows')) for b in bags]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        import importlib
        fn = importlib.import_module('harness.tbo_study')._py_worker
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(fn, tasks))
    else:
        results = [_py_worker(t) for t in tasks]
    return {'label': name, 'config': cfg.to_dict(), 'estimator': kw['est'], 'params': kw.get('params', {}),
            'bags': results, 'aggregate': M.aggregate(results), 'wall_s': time.time() - t0}


def run_cpp(name: str, kw: dict, bags, jobs: int) -> dict:
    cfg = make_cfg(kw)
    cc = CppConfig(sets={k: str(v) for k, v in kw.get('sets', {}).items()}, window_specs=kw.get('windows') or {},
                   gnss_window_anchor=kw.get('anchor', 'first_msg'))
    if kw.get('map_dir'):
        md = Path(kw['map_dir'])
        cc.map_file = str(md / 'track_map.csv')
        cc.branches = ','.join(str(p) for p in sorted(md.glob('branch_*.csv')))
    if kw.get('landmarks'):
        cc.landmarks = kw['landmarks']
    return evaluate_many_cpp(bags, cfg, cc, jobs=jobs, label=name)


def save(name: str, out: dict):
    p = R / f'tbo_{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(sanitize(out), f, indent=1)
    write_csv(out, R / f'tbo_{name}.csv')
    agg = out['aggregate']
    med = {k: agg.get(k, {}).get('median') for k in ('v_rmse', 'along_rmse', 'pos_rmse3d', 'drift_pct_3d')}
    errs = [b['bag'] for b in out['bags'] if 'error' in b]
    print(f"[{time.strftime('%H:%M:%S')}] {name}: wall={out.get('wall_s', 0):.0f}s median={med} errors={errs}",
          flush=True)


# ----------------------------------------------------------------------------------------------
# Summary tables
# ----------------------------------------------------------------------------------------------
KEYS = ['v_rmse', 'v_mae', 'v_bias', 'v_max', 'v_rmse_accel', 'v_bias_accel', 'v_rmse_brake', 'v_bias_brake',
        'v_rmse_trans', 'v_rmse_stop', 'v_rmse_anom', 'v_naive_rmse_anom', 'pos_rmse3d', 'pos_max3d', 'z_rmse',
        'along_mean', 'along_rmse', 'along_max', 'along_tan_rmse', 'cross_map_rmse', 'cross_map_max',
        'final_err3d', 'drift_pct_3d', 'drift_pct_along', 'match_v', 'match_pos', 'rate_hz', 'max_gap_s',
        'proc_p99_ms', 'stamp_age_p95_ms', 'yaw_err_mean_abs_deg']
REG = ('all', 'moving', 'stopped', 'accel', 'brake', 'cruise', 'low_speed', 'depart', 'arrive', 'transitions',
       'cmd_traction', 'cmd_brake', 'cmd_coast')


def pooled_regimes(res: dict) -> dict:
    out = {}
    for g in REG:
        n = rm2 = mae = bias = 0.0
        for b in res['bags']:
            st = b.get('speed', {}).get(g) if 'error' not in b else None
            if not st or not st.get('n'):
                continue
            k = st['n']
            n += k
            rm2 += k * st['rmse'] ** 2
            mae += k * st['mae']
            bias += k * st['bias']
        if n:
            out[g] = {'n': int(n), 'rmse': float(np.sqrt(rm2 / n)), 'mae': float(mae / n), 'bias': float(bias / n)}
    return out


def summarize():
    files = sorted(p for p in R.glob('tbo_*.json') if not p.stem.startswith('tbo_summary'))
    rows, reg_rows, fault_rows, summary = [], [], [], {}
    for p in files:
        res = json.loads(p.read_text(encoding='utf-8'))
        if 'aggregate' not in res:
            continue
        name = p.stem[4:]
        agg = res['aggregate']
        row = {'run': name, 'estimator': res.get('estimator'), 'n_bags': agg.get('n_bags'),
               'n_errors': sum(1 for b in res['bags'] if 'error' in b)}
        for k in KEYS:
            a = agg.get(k)
            if isinstance(a, dict):
                row[f'{k}_median'] = a.get('median')
                row[f'{k}_mean'] = a.get('mean')
                row[f'{k}_max'] = a.get('max')
        for k, v in agg.get('pooled', {}).items():
            row[f'pooled_{k}'] = v
        pr = pooled_regimes(res)
        for g, st in pr.items():
            row[f'pooled_v_{g}_rmse'] = st['rmse']
            reg_rows.append({'run': name, 'regime': g, **st})
        for g in REG:     # median over bags of the per-bag regime stats
            for stat in ('rmse', 'mae', 'bias'):
                vals = [b['speed'][g][stat] for b in res['bags'] if 'error' not in b
                        and (b.get('speed', {}).get(g) or {}).get('n')]
                row[f'v_{g}_{stat}_median'] = float(np.median(vals)) if vals else None
        # per-fault-spec window metrics (median over bags)
        per_spec = {}
        for b in res['bags']:
            for wname, lst in (b.get('windows') or {}).items():
                for w in lst:
                    per_spec.setdefault((wname, w['spec']), []).append(w)
        for (wname, spec), lst in per_spec.items():
            fr = {'run': name, 'windows': wname, 'spec': spec, 'n_bags': len(lst)}
            for k in ('v_rmse_in', 'naive_rmse_in', 'v_max_in', 'v_bias_in', 'v_rmse_after', 'naive_rmse_after',
                      'd_along', 'along_maxabs_in'):
                vals = np.array([w.get(k) for w in lst if w.get(k) is not None], float)
                if k == 'd_along':
                    vals = np.abs(vals)
                    k = 'abs_d_along'
                fr[f'{k}_median'] = float(np.median(vals)) if len(vals) else None
                fr[f'{k}_mean'] = float(np.mean(vals)) if len(vals) else None
                fr[f'{k}_max'] = float(np.max(vals)) if len(vals) else None
            fault_rows.append(fr)
        rows.append(row)
        summary[name] = {'estimator': res.get('estimator'), 'params': res.get('params'), 'config': res.get('config'),
                         'aggregate': agg, 'pooled_regimes': pr}
    pd.DataFrame(rows).to_csv(R / 'tbo_summary.csv', index=False, float_format='%.6g')
    pd.DataFrame(reg_rows).to_csv(R / 'tbo_summary_regimes.csv', index=False, float_format='%.6g')
    pd.DataFrame(fault_rows).to_csv(R / 'tbo_summary_fault_windows.csv', index=False, float_format='%.6g')
    with open(R / 'tbo_summary.json', 'w', encoding='utf-8') as f:
        json.dump(sanitize(summary), f, indent=1)
    degradation(pd.DataFrame(rows))
    windows_compare(pd.DataFrame(fault_rows))
    print(f'summary of {len(rows)} runs -> {R / "tbo_summary.csv"}')


def windows_compare(fw: pd.DataFrame):
    """Per injected fault window (median over val bags): speed RMSE inside the window and |along-track
    change| across it, naive wheel average vs B1P / B2P / C++ default / C++ recommended sets."""
    if fw.empty:
        return
    fw = fw[fw.windows == 'applied']
    runs = {'B1P': 'ref_b1P_faults_{}', 'B2P': 'ref_b2P_faults_{}', 'cpp': 'faults_{}',
            'cpp_mf30': 'faults_{}_mf30', 'cpp_rec2': 'rec_rec2_faults_{}', 'cpp_rec3': 'rec_rec3_faults_{}',
            'cpp_rec4': 'rec_rec4_faults_{}'}
    rows = []
    for suite in SUITE_NAMES:
        base = fw[fw.run == f'faults_{suite}']
        for spec in base.spec:
            r = {'suite': suite, 'spec': spec, 'v_rmse_in:naive': base[base.spec == spec].naive_rmse_in_median.iloc[0]}
            for k, pat in runs.items():
                x = fw[(fw.run == pat.format(suite)) & (fw.spec == spec)]
                if len(x):
                    r[f'v_rmse_in:{k}'] = x.v_rmse_in_median.iloc[0]
                    r[f'abs_d_along:{k}'] = x.abs_d_along_median.iloc[0]
                    r[f'v_rmse_after5s:{k}'] = x.v_rmse_after_median.iloc[0]
            rows.append(r)
    pd.DataFrame(rows).to_csv(R / 'tbo_fault_windows_compare.csv', index=False, float_format='%.6g')


def degradation(s: pd.DataFrame):
    """Fault suites vs the clean run of the same estimator (medians / means over val bags)."""
    s = s.set_index('run')
    pairs = [('cpp', 'val', 'faults_{}'), ('cpp_mf30', 'val_mf30', 'faults_{}_mf30'),
             ('cpp_rec2', 'rec_rec2_val', 'rec_rec2_faults_{}'), ('cpp_rec3', 'rec_rec3_val', 'rec_rec3_faults_{}'),
             ('cpp_rec4', 'rec_rec4_val', 'rec_rec4_faults_{}'),
             ('B1P', 'ref_b1P_val', 'ref_b1P_faults_{}'), ('B2P', 'ref_b2P_val', 'ref_b2P_faults_{}')]
    keys = ['v_rmse_median', 'v_rmse_mean', 'pooled_v_rmse', 'v_mae_median', 'along_rmse_median', 'along_rmse_mean',
            'along_max_median', 'drift_pct_3d_median', 'drift_pct_3d_mean', 'pos_rmse3d_median', 'v_rmse_anom_median',
            'v_naive_rmse_anom_median']
    rows = []
    for est, clean, pat in pairs:
        if clean not in s.index:
            continue
        for suite in SUITE_NAMES:
            f = pat.format(suite)
            if f not in s.index:
                continue
            r = {'estimator': est, 'suite': suite, 'clean_run': clean, 'fault_run': f}
            for k in keys:
                c, v = s.loc[clean].get(k), s.loc[f].get(k)
                r[f'{k}_clean'] = c
                r[f'{k}_fault'] = v
                r[f'{k}_delta'] = (v - c) if (c is not None and v is not None and np.isfinite(c) and np.isfinite(v)) else None
            rows.append(r)
    pd.DataFrame(rows).to_csv(R / 'tbo_fault_degradation.csv', index=False, float_format='%.6g')


def register_rec(tag: str, sets: dict) -> list:
    """Confirmation runs of a recommended override set on held-out val: clean (+window metrics), all fault
    suites, GNSS 1 s (binary window N+5 s), bag receive time base. Returns the run names."""
    base = {k: v for k, v in sets.items() if k != 'gnss_init_window_s'}
    win = sets.get('gnss_init_window_s')
    names = []

    def add(name, kw):
        CPP_RUNS[name] = kw
        names.append(name)
    add(f'rec_{tag}_val', dict(sets={**base, **({'gnss_init_window_s': win} if win else {})}, windows=ALL_WINDOWS))
    for s in SUITE_NAMES:
        add(f'rec_{tag}_faults_{s}', dict(faults=[f'suite:{s}'], sets={**base, **({'gnss_init_window_s': win} if win else {})}))
    add(f'rec_{tag}_val_gnss1', dict(gnss_seconds=1.0, sets={**base, 'gnss_init_window_s': 6}))
    add(f'rec_{tag}_val_refbag', dict(time_base='bag', sets={**base, **({'gnss_init_window_s': win} if win else {})}))
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--split', default='val')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--rec', default=None, help="tag:key=val,key=val -> rec_<tag>_* confirmation runs")
    a = ap.parse_args(argv)
    names = []
    if a.rec:
        tag, _, kvs = a.rec.partition(':')
        names += register_rec(tag, dict(kv.split('=', 1) for kv in kvs.split(',') if kv))
    for n in a.runs:
        names += GROUPS.get(n, [n])
    bags = resolve_bags(a.split)
    for n in names:
        if not a.force and (R / f'tbo_{n}.json').exists():
            print(f'skip {n} (exists)')
            continue
        if n in CPP_RUNS:
            out = run_cpp(n, CPP_RUNS[n], bags, a.jobs)
        elif n in PY_RUNS:
            out = run_py(n, PY_RUNS[n], bags, a.jobs)
        else:
            raise SystemExit(f'unknown run {n}')
        save(n, out)
    if a.summary:
        summarize()


if __name__ == '__main__':
    main()
