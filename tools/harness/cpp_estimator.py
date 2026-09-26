"""Evaluate the C++ estimator (offline replay binary ``tbo_replay``) with the judge-replica harness.

The harness normally streams a bag into a Python object (``replay.replay``). The C++ estimator is a
separate executable, so this adapter

  1. builds EXACTLY the event stream the harness would feed (``replay.build_events``: bag-time order,
     GNSS only while ``t_bag - t_start <= gnss_seconds``, fault injection via ``faults.apply_faults``),
  2. writes it as the CSV event format of ``tbo_replay`` (W0/W1/C/GF/GV lines, recv = bag time,
     stamp = header.stamp, integer ns),
  3. runs the SNAPSHOT binary once per bag (``tools/harness/bin/tbo_replay_snapshot.exe``) with the
     snapshot map / branches / landmarks / traction table (``tools/harness/bin/snapshot``),
  4. converts its published outputs into a :class:`metrics.OutputLog` (emit time = bag time of the
     triggering input, stamp = published header.stamp, yaw, covariances, speed variance) and scores it
     with the unmodified harness (``replay.score_log`` -> speed / position / robustness / real-time /
     validity metrics, ``metrics.aggregate``).

GNSS rule: the harness rule (bag time) decides which GNSS messages are written to the CSV; the binary
additionally applies its own window ``gnss_init_window_s`` (header time since its first message). By
default the adapter sets ``gnss_init_window_s = gnss_seconds`` (override with ``--set``).

CLI (from C:\\MosTransHack\\tools), same judge options as ``harness.run_eval``:
    python -m harness.cpp_estimator --split val --out harness/results/tbo_val.json --csv harness/results/tbo_val.csv
    python -m harness.cpp_estimator --split val --time-base bag
    python -m harness.cpp_estimator --split val --frame utm --set output_frame=utm
    python -m harness.cpp_estimator --split val --gnss-seconds 1 --set gnss_init_window_s=1
    python -m harness.cpp_estimator --split val --faults suite:realistic
    python -m harness.cpp_estimator --bags 30618_e3d94878 --set landmark_enable=0 --keep-tmp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ''):          # executed as a script: make 'harness' importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from harness.loader import BagData, load_bag, resolve_bags, T_BAG, T_HDR, V_COL, LAT, LON, ALT, STATUS, VX, VY, VZ
from harness.reference import RefConfig, build_reference
from harness.replay import (EvalConfig, K_CMD, K_FIXM, K_FIXR, K_FRONT, K_REAR, K_VELM, K_VELR, _KIND_TOPIC,
                            build_events, score_log)
from harness import metrics as M

HARNESS_DIR = Path(__file__).resolve().parent
BIN_DIR = HARNESS_DIR / 'bin'
SNAP_DIR = BIN_DIR / 'snapshot'
SNAP_EXE = BIN_DIR / ('tbo_replay_snapshot.exe' if os.name == 'nt' else 'tbo_replay_snapshot')

# HealthFlag bits of the estimator (core/include/tbo/types.hpp)
FLAG_MODEL_ONLY = 1 << 11
FLAG_STANDSTILL = 1 << 12
FLAG_NOT_INIT = 1 << 13
FLAG_NO_MAP = 1 << 14
FLAG_RECOVERED = 1 << 15
FLAG_LANDMARK = 1 << 19
FLAG_SLIP_ANY = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
FLAG_DROPOUT_ANY = (1 << 4) | (1 << 5)


@dataclass
class CppConfig:
    """How to run the binary. Paths default to the snapshot taken for the evaluation."""
    exe: str = str(SNAP_EXE)
    map_file: Optional[str] = str(SNAP_DIR / 'maps' / 'track_map.csv')
    branches: Optional[str] = ','.join(str(p) for p in sorted((SNAP_DIR / 'maps').glob('branch_*.csv')))
    landmarks: Optional[str] = str(SNAP_DIR / 'maps' / 'landmarks.csv')
    traction: Optional[str] = str(SNAP_DIR / 'config' / 'traction_lut.csv')
    params_file: Optional[str] = None          # None: compiled defaults of the binary (= its --dump-params)
    sets: Dict[str, str] = field(default_factory=dict)   # --set key=value overrides
    export_gnss_vel: bool = True               # GV lines (the estimator ignores them, like the node)
    tmp_root: Optional[str] = None
    keep_tmp: bool = False
    # per-window metrics: {name: fault specs}; windows are placed on the CLEAN bag exactly like
    # apply_faults does (so a clean run can be scored in the same windows as a faulted one)
    window_specs: Dict[str, list] = field(default_factory=dict)
    # 'first_msg': binary default (window starts at the header stamp of the first accepted message);
    # 'first_gnss': emulates a window anchored at the first GNSS fix (per-bag gnss_init_window_s =
    #               N + (first fix header - first message header)), i.e. the proposed code change
    gnss_window_anchor: str = 'first_msg'

    def to_dict(self):
        return asdict(self)


def file_md5(path) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def provenance(cc: CppConfig) -> dict:
    files = {'exe': cc.exe, 'map': cc.map_file, 'landmarks': cc.landmarks, 'traction': cc.traction}
    for i, b in enumerate([b for b in (cc.branches or '').split(',') if b]):
        files[f'branch{i}'] = b
    out = {}
    for k, p in files.items():
        if p:
            st = os.stat(p) if os.path.exists(p) else None
            out[k] = {'path': p, 'md5': file_md5(p),
                      'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)) if st else None}
    return out


# ----------------------------------------------------------------------------------------------
# Event export (identical stream to what the harness would feed a Python estimator)
# ----------------------------------------------------------------------------------------------
def _ns(x: float) -> int:
    return int(round(x * 1e9)) if math.isfinite(x) else 0


def export_harness_events(bag: BagData, cfg: EvalConfig, path: Path, gnss_vel: bool = True,
                          est_window_s: Optional[float] = None) -> dict:
    """Write the harness event stream of ``bag`` (already fault-injected if applicable) as tbo_replay CSV.

    Returns counts plus an estimate of how many master fixes the binary's own GNSS window
    (header time since its first accepted message) will accept."""
    t_ev, k_ev, r_ev = build_events(bag, cfg)
    lists = {kk: bag[tp] for kk, tp in _KIND_TOPIC.items()}
    lines = []
    n = {'W': 0, 'C': 0, 'GF_master': 0, 'GF_rover': 0, 'GV': 0}
    first_hdr = None               # header stamp of the first message the estimator accepts (W/C/GF)
    t_first_gf_bag = None
    t_last_gf_bag = None
    fix_hdr_master = []
    for t, k, r in zip(t_ev.tolist(), k_ev.tolist(), r_ev.tolist()):
        row = lists[k][r]
        recv = _ns(t)
        stamp = _ns(row[T_HDR])
        if k in (K_FRONT, K_REAR):
            v = row[V_COL]
            lines.append(f"{'W0' if k == K_FRONT else 'W1'},{recv},{stamp},{v:.6f}\n" if math.isfinite(v)
                         else f"{'W0' if k == K_FRONT else 'W1'},{recv},{stamp},nan\n")
            n['W'] += 1
            if first_hdr is None and stamp > 0:
                first_hdr = stamp
        elif k == K_CMD:
            notch = row[V_COL]
            notch_i = int(notch) if math.isfinite(notch) else 127
            lines.append(f'C,{recv},{stamp},{notch_i}\n')
            n['C'] += 1
            if first_hdr is None and stamp > 0 and -15 <= notch_i <= 15:
                first_hdr = stamp
        elif k in (K_FIXM, K_FIXR):
            src = 0 if k == K_FIXM else 1
            lines.append(f'GF,{recv},{stamp},{src},{row[LAT]:.10f},{row[LON]:.10f},{row[ALT]:.4f},{int(row[STATUS])}\n')
            n['GF_master' if src == 0 else 'GF_rover'] += 1
            ok = stamp > 0 and math.isfinite(row[LAT]) and row[STATUS] >= 0
            if first_hdr is None and ok:
                first_hdr = stamp
            if src == 0 and ok:
                fix_hdr_master.append(stamp)
            t_first_gf_bag = t if t_first_gf_bag is None else t_first_gf_bag
            t_last_gf_bag = t
        elif k in (K_VELM, K_VELR):
            if not gnss_vel:
                continue
            src = 0 if k == K_VELM else 1
            lines.append(f'GV,{recv},{stamp},{src},{row[VX]:.6f},{row[VY]:.6f},{row[VZ]:.6f}\n')
            n['GV'] += 1
    with open(path, 'w', newline='\n') as fh:
        fh.writelines(lines)
    info = {'n_events': len(lines), **n,
            'gnss_bag_span_s': (t_last_gf_bag - bag.t_start) if t_last_gf_bag is not None else None,
            'first_msg_hdr_age_s': (float(t_ev[0]) - first_hdr * 1e-9) if first_hdr is not None and len(t_ev) else None,
            'first_msg_hdr_ns': first_hdr,
            'first_master_fix_hdr_ns': min(fix_hdr_master) if fix_hdr_master else None,
            '_fix_hdr_master': fix_hdr_master}
    if est_window_s is not None and first_hdr is not None:
        lim = first_hdr + int(round(est_window_s * 1e9))
        info['gnss_master_accepted_est'] = int(sum(1 for s in fix_hdr_master if s <= lim))
    return info


# ----------------------------------------------------------------------------------------------
# Binary
# ----------------------------------------------------------------------------------------------
def build_command(cc: CppConfig, ev_csv: Path, out_csv: Path, sets: Dict[str, str]) -> List[str]:
    cmd = [cc.exe, '--in', str(ev_csv), '--out', str(out_csv)]
    if cc.params_file:
        cmd += ['--params', cc.params_file]
    if cc.map_file:
        cmd += ['--map', cc.map_file]
    if cc.traction:
        cmd += ['--traction', cc.traction]
    if cc.branches:
        cmd += ['--branches', cc.branches]
    if cc.landmarks:
        cmd += ['--landmarks', cc.landmarks]
    for k, v in sets.items():
        cmd += ['--set', f'{k}={v}']
    return cmd


def parse_stderr(text: str) -> dict:
    """'inputs=.. outputs=.. invalid_wheel=.. ...' -> dict of ints."""
    out = {}
    for line in text.strip().splitlines():
        for tok in line.split():
            if '=' in tok:
                k, _, v = tok.partition('=')
                try:
                    out[k] = int(v)
                except ValueError:
                    out[k] = v
    return out


def run_binary(cc: CppConfig, ev_csv: Path, out_csv: Path, sets: Dict[str, str]):
    cmd = build_command(cc, ev_csv, out_csv, sets)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f'tbo_replay failed (rc={res.returncode}): {res.stderr.strip()}  cmd={" ".join(cmd)}')
    df = pd.read_csv(out_csv)
    return df, parse_stderr(res.stderr), wall, cmd


def to_output_log(df: pd.DataFrame, n_inputs: int) -> M.OutputLog:
    stamp = df['stamp_ns'].to_numpy(np.int64) * 1e-9
    emit = df['recv_ns'].to_numpy(np.int64) * 1e-9
    proc = df['proc_ns'].to_numpy(float) * 1e-9
    # processing time per triggering input = max over the outputs it produced (cumulative timer)
    key = df['recv_ns'].astype(str) + df['trigger'].astype(str)
    grp = (key != key.shift()).cumsum()
    proc_cb = df.groupby(grp)['proc_ns'].max().to_numpy(float) * 1e-9
    return M.OutputLog(stamp=stamp, v=df['v'].to_numpy(float), xyz=df[['x', 'y', 'z']].to_numpy(float),
                       emit_tbag=emit, emit_proc=proc, frame_id=['map'] * len(df),
                       cov=df[['cov_xx', 'cov_yy', 'cov_zz']].to_numpy(float), v_var=df['v_var'].to_numpy(float),
                       slip=1.0 - df['mu0'].to_numpy(float), yaw=df['yaw'].to_numpy(float),
                       proc_all=proc_cb, n_callbacks=int(n_inputs))


def estimator_diag(df: pd.DataFrame) -> dict:
    """Estimator-internal diagnostics from the output stream (fractions of published outputs)."""
    fl = df['flags'].to_numpy(np.int64)
    lm = (fl & FLAG_LANDMARK) != 0
    lm_events = int(np.sum(lm[1:] & ~lm[:-1]) + (1 if len(lm) and lm[0] else 0))
    return {
        'n_out': int(len(df)),
        'frac_no_map': float(np.mean((fl & FLAG_NO_MAP) != 0)),
        'frac_not_init': float(np.mean((fl & FLAG_NOT_INIT) != 0)),
        'frac_model_only': float(np.mean((fl & FLAG_MODEL_ONLY) != 0)),
        'frac_mu3_gt_05': float(np.mean(df['mu3'].to_numpy() > 0.5)),
        'frac_slip_flag': float(np.mean((fl & FLAG_SLIP_ANY) != 0)),
        'frac_dropout_flag': float(np.mean((fl & FLAG_DROPOUT_ANY) != 0)),
        'n_recovered_events': int(np.sum(np.diff(((fl & FLAG_RECOVERED) != 0).astype(np.int8)) == 1)),
        'n_landmark_fixes': lm_events,
        'scale_final': float(df['k'].iloc[-1]) if len(df) else None,
        'gain_final': float(df['g'].iloc[-1]) if len(df) else None,
        's_final': float(df['s'].iloc[-1]) if len(df) else None,
        'sigma_s_final': float(np.sqrt(max(df['s_var'].iloc[-1], 0.0))) if len(df) else None,
        'trigger_counts': {str(k): int(v) for k, v in df['trigger'].value_counts().items()},
    }


# ----------------------------------------------------------------------------------------------
# Per-sample errors and per-fault-window metrics (shared with the Python baselines in tbo_study)
# ----------------------------------------------------------------------------------------------
def per_sample_errors(ref, log: M.OutputLog, bag_seen: BagData, tol: float = 0.05, direction: str = 'ref2out') -> dict:
    """Matched per-sample errors with the harness definitions (speed; along_arc / cross_map / 3D)."""
    valid = np.isfinite(log.v) & np.isfinite(log.stamp)
    iref, iout, _ = M.pair(ref.t_vel, log.stamp, valid, tol, direction)
    naive = M.naive_wheel_speed(ref, bag_seen)
    out = {'tv': ref.tbag_vel[iref], 'ev': log.v[iout] - ref.v[iref], 'env': naive[iref] - ref.v[iref],
           'vref': ref.v[iref]}
    valid = np.all(np.isfinite(log.xyz), axis=1) & np.isfinite(log.stamp)
    iref, iout, _ = M.pair(ref.t_pos, log.stamp, valid, tol, direction)
    keep = ref.pos_mask()[iref]
    iref, iout = iref[keep], iout[keep]
    e = log.xyz[iout] - ref.xyz[iref]
    e2 = np.hypot(e[:, 0], e[:, 1])
    win = np.clip(20.0 + 1.6 * e2, 20.0, 2000.0)
    s_est, lat, _ = ref.path.project(log.xyz[iout][:, :2], s_hint=ref.s_pos[iref], window=win)
    o = np.argsort(ref.tbag_pos[iref], kind='stable')
    out.update({'tp': ref.tbag_pos[iref][o], 'along': (s_est - ref.s_pos[iref])[o], 'cross': lat[o],
                'e3': np.sqrt(e2 ** 2 + e[:, 2] ** 2)[o]})
    return out


def fault_windows(bag: BagData, specs, seed: int = 0) -> List[dict]:
    """Windows [t_start, t_end] (bag clock) of the fault specs, exactly as apply_faults places them."""
    from harness.faults import apply_faults
    return apply_faults(bag, specs, seed=seed).meta.get('faults', [])


def window_metrics(pse: dict, windows: List[dict], after_s: float = 5.0) -> List[dict]:
    """Speed error inside each window (+ recovery ``after_s``) vs the naive wheel average, and the
    change of the along-track error across the window (d_along = error at end+after - error at start)."""
    rows = []
    for w in windows:
        a, b = float(w['t_start']), float(w['t_end'])
        spec = f"{w['kind']}:{w['topic']}@{w['t0']}+{w['dur']:g}"
        extra = ','.join(f'{k}={v:g}' for k, v in w.items()
                         if k not in ('kind', 'topic', 't0', 'dur', 't_start', 't_end') and isinstance(v, (int, float)))
        if extra:
            spec += ':' + extra
        m_in = (pse['tv'] >= a) & (pse['tv'] <= b)
        m_after = (pse['tv'] > b) & (pse['tv'] <= b + after_s)
        r = {'spec': spec, 't_start': a, 't_end': b, 'n_in': int(m_in.sum())}
        for tag, m in (('in', m_in), ('after', m_after)):
            e = pse['ev'][m]
            en = pse['env'][m]
            r[f'v_rmse_{tag}'] = float(np.sqrt(np.mean(e ** 2))) if len(e) else None
            r[f'v_max_{tag}'] = float(np.max(np.abs(e))) if len(e) else None
            r[f'v_bias_{tag}'] = float(np.mean(e)) if len(e) else None
            en = en[np.isfinite(en)]
            r[f'naive_rmse_{tag}'] = float(np.sqrt(np.mean(en ** 2))) if len(en) else None
        r['vref_mean_in'] = float(np.mean(pse['vref'][m_in])) if m_in.any() else None
        tp, al = pse['tp'], pse['along']
        if len(tp):
            j0 = np.searchsorted(tp, a) - 1
            j1 = np.searchsorted(tp, b + after_s, 'right') - 1
            if 0 <= j0 < len(tp) and 0 <= j1 < len(tp):
                r['along_start'] = float(al[j0])
                r['along_end'] = float(al[j1])
                r['d_along'] = float(al[j1] - al[j0])
            m = (tp >= a) & (tp <= b + after_s)
            r['along_maxabs_in'] = float(np.max(np.abs(al[m]))) if m.any() else None
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------------------------
# Evaluation of one bag / many bags
# ----------------------------------------------------------------------------------------------
def effective_sets(cfg: EvalConfig, cc: CppConfig) -> Dict[str, str]:
    """Binary overrides: output frame follows the judge frame and the binary's own GNSS window
    equals the harness GNSS window, unless explicitly overridden."""
    sets = {}
    sets['output_frame'] = cfg.ref.frame
    if cfg.gnss_mode == 'first_n':
        sets['gnss_init_window_s'] = f'{cfg.gnss_seconds:g}'
    elif cfg.gnss_mode == 'all':
        sets['gnss_init_window_s'] = '1e9'
    sets.update({k: str(v) for k, v in cc.sets.items()})
    return sets


def evaluate_bag_cpp(bag_name: str, cfg: EvalConfig = None, cc: CppConfig = None, keep_log: bool = False) -> dict:
    cfg = cfg or EvalConfig()
    cc = cc or CppConfig()
    tmp = Path(tempfile.mkdtemp(prefix=f'tbo_{bag_name}_', dir=cc.tmp_root))
    try:
        bag = load_bag(bag_name)
        try:
            ref = build_reference(bag, cfg.ref)
        except ValueError:
            ref = None
        run_bag = bag
        if cfg.faults:
            from harness.faults import apply_faults
            run_bag = apply_faults(bag, cfg.faults, seed=cfg.fault_seed)
        sets = effective_sets(cfg, cc)
        ev_csv, out_csv = tmp / 'events.csv', tmp / 'outputs.csv'
        win = float(sets['gnss_init_window_s']) if 'gnss_init_window_s' in sets else 5.0
        ev_info = export_harness_events(run_bag, cfg, ev_csv, cc.export_gnss_vel, est_window_s=win)
        if cc.gnss_window_anchor == 'first_gnss' and ev_info.get('first_master_fix_hdr_ns') and ev_info.get('first_msg_hdr_ns'):
            off = (ev_info['first_master_fix_hdr_ns'] - ev_info['first_msg_hdr_ns']) * 1e-9
            win = win + max(off, 0.0)
            sets['gnss_init_window_s'] = f'{win:.6f}'
            ev_info['gnss_window_anchor_offset_s'] = off
            lim = ev_info['first_msg_hdr_ns'] + int(round(win * 1e9))
            ev_info['gnss_master_accepted_est'] = int(sum(1 for s in ev_info['_fix_hdr_master'] if s <= lim))
        ev_info.pop('_fix_hdr_master', None)
        t0 = time.perf_counter()
        df, diag, wall_bin, cmd = run_binary(cc, ev_csv, out_csv, sets)
        log = to_output_log(df, ev_info['W'] + ev_info['C'])
        if ref is None:
            res = {'bag': bag.name, 'vehicle': bag.vehicle, 'duration_s': bag.duration, 'no_reference': True,
                   'rt': M.realtime_metrics(log, bag), 'valid': M.validity_metrics(log, bag)}
            res['summary'] = M.headline(res)
        else:
            res = score_log(bag, ref, log, cfg, run_bag)
        res['replay_wall_s'] = time.perf_counter() - t0
        res['cpp'] = {'sets': sets, 'events': ev_info, 'binary': diag, 'binary_wall_s': wall_bin,
                      'est': estimator_diag(df)}
        if cfg.faults:
            res['faults_applied'] = run_bag.meta.get('faults', [])
        if ref is not None and (cfg.faults or cc.window_specs):
            pse = per_sample_errors(ref, log, run_bag, cfg.tol, cfg.direction)
            res['windows'] = {}
            if cfg.faults:
                res['windows']['applied'] = window_metrics(pse, res['faults_applied'])
            for name, specs in cc.window_specs.items():
                res['windows'][name] = window_metrics(pse, fault_windows(bag, specs, cfg.fault_seed))
        if keep_log:
            res['_log'] = log
            res['_ref'] = ref
            res['_bag'] = run_bag
            res['_df'] = df
        return res
    except Exception as ex:
        return {'bag': bag_name, 'error': f'{type(ex).__name__}: {ex}', 'traceback': traceback.format_exc()}
    finally:
        if cc.keep_tmp:
            print(f'[cpp_estimator] kept {tmp}', file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def _worker(args):
    bag_name, cfg, cc = args
    return evaluate_bag_cpp(bag_name, cfg, cc)


def evaluate_many_cpp(bags: Sequence[str], cfg: EvalConfig = None, cc: CppConfig = None, jobs: int = 4,
                      label: str = '') -> dict:
    cfg = cfg or EvalConfig()
    cc = cc or CppConfig()
    t0 = time.time()
    tasks = [(b, cfg, cc) for b in bags]
    if jobs > 1 and len(bags) > 1:
        from concurrent.futures import ProcessPoolExecutor
        import importlib
        fn = importlib.import_module('harness.cpp_estimator')._worker     # picklable by module path
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(fn, tasks))
    else:
        results = [_worker(t) for t in tasks]
    return {'label': label, 'config': cfg.to_dict(), 'estimator': 'cpp:tbo_replay_snapshot',
            'params': effective_sets(cfg, cc), 'cpp_config': cc.to_dict(), 'provenance': provenance(cc),
            'bags': results, 'aggregate': M.aggregate(results), 'wall_s': time.time() - t0}


# ----------------------------------------------------------------------------------------------
# Flat per-bag table (CSV)
# ----------------------------------------------------------------------------------------------
REGIMES = ('all', 'moving', 'stopped', 'accel', 'brake', 'cruise', 'low_speed', 'depart', 'arrive', 'transitions',
           'cmd_traction', 'cmd_brake', 'cmd_coast')


def flat_row(r: dict) -> dict:
    """One flat dict per bag: headline + per-regime speed stats + position details + estimator diag."""
    if 'error' in r:
        return {'bag': r['bag'], 'error': r['error']}
    row = {'bag': r['bag'], 'vehicle': r.get('vehicle'), 'duration_s': r.get('duration_s')}
    row.update(r.get('summary', {}))
    sp = r.get('speed', {})
    for g in REGIMES:
        st = sp.get(g)
        if isinstance(st, dict):
            for s in ('n', 'rmse', 'mae', 'bias', 'max'):
                row[f'v_{g}_{s}'] = st.get(s)
    pos = r.get('pos', {})
    for comp in ('along_arc', 'along_tan', 'cross_map', 'cross_tan', 'x', 'y', 'z'):
        st = pos.get(comp)
        if isinstance(st, dict):
            for s in ('mean', 'mean_abs', 'rmse', 'max', 'p95'):
                row[f'pos_{comp}_{s}'] = st.get(s)
    for comp in ('err3d', 'err2d'):
        st = pos.get(comp)
        if isinstance(st, dict):
            for s in ('mean', 'rmse', 'max', 'p95', 'median'):
                row[f'pos_{comp}_{s}'] = st.get(s)
    fin = pos.get('final', {})
    for s in ('err3d', 'err2d', 'along_arc', 'cross_map', 'drift_pct_3d', 'drift_pct_2d', 'drift_pct_along', 't_rel'):
        row[f'final_{s}'] = fin.get(s)
    row['max_rel_err2d_pct'] = pos.get('max_rel_err2d_pct')
    rb = r.get('robust', {})
    for g in ('anomaly', 'slip_any', 'slip_both', 'dropout', 'recovery', 'clean'):
        st = rb.get(g)
        if isinstance(st, dict):
            row[f'rob_{g}_rmse'] = st.get('rmse')
            row[f'rob_{g}_naive_rmse'] = st.get('naive_rmse')
            row[f'rob_{g}_frac_time'] = st.get('frac_time')
    row['spike_frac_1mps'] = rb.get('spike_frac_1mps')
    row['spike_frac_2mps'] = rb.get('spike_frac_2mps')
    rt = r.get('rt', {})
    for s in ('n_out', 'rate_hz', 'max_gap_s', 'frac_1s_bins_ge_min_rate', 'stamp_age_med_ms', 'stamp_age_p95_ms',
              'stamp_age_max_ms', 'frac_stamp_age_gt_100ms', 'proc_p99_ms', 'proc_max_ms', 'cpu_load_pct'):
        row[f'rt_{s}'] = rt.get(s)
    va = r.get('valid', {})
    for s in ('ok', 'n_nan_v', 'n_nan_pos', 'n_stamp_backwards', 'n_stamp_repeat', 'n_pos_jump', 'max_pos_step_m'):
        row[f'valid_{s}'] = va.get(s)
    cov = r.get('cov', {})
    for s in ('nees2d_mean', 'cover95_2d', 'sigma2d_median', 'nees_v_mean', 'cover95_v'):
        row[f'cov_{s}'] = cov.get(s)
    hd = r.get('heading', {}).get('yaw_err_deg', {})
    row['yaw_err_mean_abs_deg'] = hd.get('mean_abs')
    rd = r.get('ref_diag', {})
    for s in ('frac_status2', 'frac_outlier', 'hdr_glitch_fix', 'hdr_glitch_vel', 'distance_path_m'):
        row[f'ref_{s}'] = rd.get(s)
    cp = r.get('cpp', {})
    for s, v in cp.get('est', {}).items():
        if not isinstance(v, dict):
            row[f'est_{s}'] = v
    for s, v in cp.get('events', {}).items():
        row[f'ev_{s}'] = v
    for s, v in cp.get('binary', {}).items():
        row[f'bin_{s}'] = v
    return row


def write_csv(results: dict, path) -> None:
    rows = [flat_row(r) for r in results['bags']]
    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.6g')


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='val')
    ap.add_argument('--bags', nargs='*')
    ap.add_argument('--gnss-seconds', type=float, default=5.0)
    ap.add_argument('--gnss-mode', default='first_n', choices=['first_n', 'none', 'all'])
    ap.add_argument('--frame', default='enu', choices=['enu', 'utm'], help='JUDGE frame (estimator follows unless --set output_frame=..)')
    ap.add_argument('--time-base', default='header', choices=['header', 'bag', 'header_fixed'])
    ap.add_argument('--antenna', default='master', choices=['master', 'rover'])
    ap.add_argument('--speed-dims', type=int, default=2, choices=[2, 3])
    ap.add_argument('--tol', type=float, default=0.05)
    ap.add_argument('--direction', default='ref2out', choices=['ref2out', 'out2ref'])
    ap.add_argument('--clean-ref', action='store_true')
    ap.add_argument('--faults', nargs='*', default=None)
    ap.add_argument('--fault-seed', type=int, default=0)
    ap.add_argument('--set', dest='sets', action='append', default=[], help='binary override key=value (repeatable)')
    ap.add_argument('--exe', default=str(SNAP_EXE))
    ap.add_argument('--params-file', default=None)
    ap.add_argument('--no-landmarks', action='store_true', help='run without the landmark file')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--out', default=None, help='JSON results')
    ap.add_argument('--csv', default=None, help='flat per-bag CSV')
    ap.add_argument('--label', default='')
    ap.add_argument('--keep-tmp', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    return ap


def config_from_args(a) -> EvalConfig:
    ref = RefConfig(antenna=a.antenna, frame=a.frame, time_base=a.time_base, speed_dims=a.speed_dims, clean=a.clean_ref)
    return EvalConfig(gnss_seconds=a.gnss_seconds, gnss_mode=a.gnss_mode, tick_hz=0.0, tol=a.tol,
                      direction=a.direction, ref=ref, faults=a.faults, fault_seed=a.fault_seed)


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    sets = dict(kv.split('=', 1) for kv in a.sets)
    cc = CppConfig(exe=a.exe, params_file=a.params_file, sets=sets, keep_tmp=a.keep_tmp)
    if a.no_landmarks:
        cc.landmarks = None
    bags = resolve_bags(a.bags if a.bags else a.split)
    out = evaluate_many_cpp(bags, cfg, cc, jobs=a.jobs, label=a.label)
    if not a.quiet:
        from harness.run_eval import fmt_table
        print(f"C++ estimator (snapshot)  sets={out['params']}  bags={len(bags)} frame={a.frame} "
              f"time_base={a.time_base} gnss={a.gnss_mode}:{a.gnss_seconds}s faults={a.faults}  wall={out['wall_s']:.1f}s")
        print(fmt_table(out['bags'], out['aggregate']))
        for r in out['bags']:
            if 'error' in r:
                print(r['traceback'])
    if a.out:
        from harness.run_eval import sanitize
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(sanitize(out), f, indent=1)
    if a.csv:
        write_csv(out, a.csv)
    return out


if __name__ == '__main__':
    main()
