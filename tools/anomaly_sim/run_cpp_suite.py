#!/usr/bin/env python
"""Anomaly suite for the C++ estimator (snapshot of tbo_replay) scored against the CLEAN GNSS reference.

For every (bag, scenario):
  1. the anomaly_sim injectors corrupt the clean npz (deterministic: scenario seed x bag name);
  2. the vehicle messages are exported as an arrival-ordered event stream exactly like
     tools/replay/cpp_bridge.export_events (W0/W1/C from the corrupted run, GF fixes of both antennas
     only while recv <= first vehicle recv + gnss_keep_s);
  3. bin/tbo_replay_snapshot.exe replays it with the snapshot map / traction table / landmarks
     (same arguments as tools/replay/quick_eval.py) plus the variant's --set overrides;
  4. the outputs are scored against the clean GNSS reference (master vel |v_h| at header stamps, nearest
     output stamp within 0.05 s, like quick_eval) and against the clean-input replay of the same variant.

Results (machine readable) go to results_cpp/<variant>/:
  clean_runs.csv            clean-input run per bag (accuracy vs GNSS, flags, timing)
  false_alarm_episodes.csv  every flag episode on the clean runs (natural anomalies / true gaps marked)
  false_alarms.csv          episodes per hour per flag family (all clean runs)
  runs.csv                  one row per (scenario, bag): errors, position, crash/non-finite, rate, stamps
  windows.csv               one row per injected event: error during/after, recovery, detection
  scenario_summary.csv      aggregates per scenario;  meta.json  variant settings + binary hash
Usage (from C:\\MosTransHack\\tools):
  python anomaly_sim/run_cpp_suite.py --variant base --keep-out
  python anomaly_sim/run_cpp_suite.py --variant cusum_h_0.2 --set cusum_h=0.2 --scenarios S02 S19 X0
  python anomaly_sim/run_cpp_suite.py --variant base --summarize-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from anomaly_sim import apply_scenario, load_run, load_suite  # noqa: E402
from anomaly_sim.constants import CMD, FRONT, NPZ_DIR, REAR, SPLITS_JSON  # noqa: E402

ROOT = HERE.parents[1]
BIN = HERE / 'bin'
SNAP_EXE = BIN / ('tbo_replay_snapshot.exe' if os.name == 'nt' else 'tbo_replay_snapshot')
if os.environ.get('TBO_REPLAY_EXE'):          # --exe (set by main, inherited by the worker processes)
    SNAP_EXE = Path(os.environ['TBO_REPLAY_EXE'])
PKG = BIN / 'snapshot_pkg'
MAP = PKG / 'maps' / 'track_map.csv'
LANDMARKS = PKG / 'maps' / 'landmarks.csv'
TRACTION = PKG / 'config' / 'traction_lut.csv'
BRANCHES = ','.join(str(p) for p in sorted((PKG / 'maps').glob('branch_*.csv')))
RESULTS = HERE / 'results_cpp'
CACHE = RESULTS / 'cache'
SUITE_FILES = [HERE / 'scenarios' / 'suite.yaml', HERE / 'scenarios' / 'cpp_extra.yaml']
NATURAL = HERE / 'out' / 'natural' / 'natural_events.json'
CLOCKS = ROOT / 'analysis' / 'timing_reference' / 'clocks_per_bag.csv'

GOOD_BAGS = ['30618_2f104a1d', '30618_22c1c589', '30618_0f120b35', '30618_a53d5f6f', '30618_6cb3280a',
             '30639_d927f360', '30618_01f73500']

# ------------------------------------------------------------------------------------------ flags
BIT = {'front_slip': 0, 'rear_slip': 1, 'front_slide': 2, 'rear_slide': 3, 'front_dropout': 4,
       'rear_dropout': 5, 'cmd_dropout': 6, 'front_stuck': 7, 'rear_stuck': 8, 'front_invalid': 9,
       'rear_invalid': 10, 'model_only': 11, 'standstill': 12, 'not_initialized': 13, 'no_map': 14,
       'recovered': 15, 'late_data': 16, 'unmodeled_accel': 17, 'cmd_inconsistent': 18, 'landmark': 19}
FAM = {
    'slip': (1 << 0) | (1 << 1),
    'slide': (1 << 2) | (1 << 3),
    'dropout': (1 << 4) | (1 << 5),
    'cmd_dropout': 1 << 6,
    'stuck': (1 << 7) | (1 << 8),
    'invalid': (1 << 9) | (1 << 10),
    'model_only': 1 << 11,
    'recovered': 1 << 15,
    'late_data': 1 << 16,
    'unmodeled_accel': 1 << 17,
    'cmd_inconsistent': 1 << 18,
}
# flags that must be clear (relative to the clean run) for "recovered"
RECOVERY_MASK = (FAM['slip'] | FAM['slide'] | FAM['dropout'] | FAM['cmd_dropout'] | FAM['stuck'] |
                 FAM['invalid'] | FAM['model_only'] | FAM['unmodeled_accel'] | FAM['cmd_inconsistent'])
FA_FAMILIES = ['slip', 'slide', 'dropout', 'cmd_dropout', 'stuck', 'invalid', 'model_only', 'unmodeled_accel',
               'cmd_inconsistent', 'late_data', 'recovered']
GLOBAL_TYPES = {'scale_drift', 'duplicates', 'reorder', 'stamp_jitter', 'gnss_cut'}

REC_TOL = 0.1       # m/s, recovery tolerance
REC_HOLD = 1.0      # s, the tolerance must hold this long
AFTER_S = 30.0      # s, "after" window
TAIL_S = 0.5        # s, the "during" window is [t0, t1 + TAIL_S]
DET_PRE = 0.5       # s, detection counted from t0 - DET_PRE ...
DET_GRACE = 1.0     # s, ... to t1 + DET_GRACE
STARTUP_S = 5.0     # s after the first vehicle message: start-up transients, not counted as false alarms


def _bits(front: bool, rear: bool, fb: int, rb: int) -> int:
    return (int(front) << fb) | (int(rear) << rb)


def expected_detection(ev: dict, dv_true: float) -> tuple[str, int, bool, set]:
    """(expected family, expected flag mask, detection required?, allowed families) of an injected event."""
    typ = ev['type']
    topics = set(ev.get('topics', []))
    p = ev.get('params', {}) or {}
    f, r, c = FRONT in topics, REAR in topics, CMD in topics
    dur = float(ev['t1'] - ev['t0'])
    base_ok = {'model_only', 'recovered'}
    if typ == 'slip':
        return 'slip', _bits(f, r, 0, 1), True, base_ok | {'slip'}
    if typ == 'slide':
        return 'slide', _bits(f, r, 2, 3), True, base_ok | {'slide'}
    if typ == 'dropout':
        if p.get('mode') == 'stall':
            return 'late_data', FAM['late_data'], False, base_ok | {'late_data', 'dropout', 'cmd_dropout'}
        m = _bits(f, r, 4, 5) | (int(c) << 6)
        required = dur > 0.8 and not (f and r and c)   # nothing is published while every input is silent
        return 'dropout', m, required, base_ok | {'dropout', 'cmd_dropout', 'late_data'}
    if typ == 'frozen':
        if topics == {CMD}:
            return 'cmd_inconsistent', FAM['cmd_inconsistent'], False, {'cmd_inconsistent', 'unmodeled_accel'}
        if p.get('mode') == 'zero':
            return 'slide|stuck', _bits(f, r, 2, 3) | _bits(f, r, 7, 8), True, base_ok | {'slide', 'stuck'}
        # "stuck" needs identical readings >= 1.2 s AND the vehicle speed to change by >= 0.4 m/s
        return 'stuck', _bits(f, r, 7, 8), dur > 2.0 and dv_true > 0.6, base_ok | {'stuck', 'slip', 'slide'}
    if typ == 'outliers':
        kind = p.get('kind')
        if c:
            return 'cmd_inconsistent', FAM['cmd_inconsistent'], False, {'cmd_inconsistent', 'unmodeled_accel'}
        if kind in ('nan', 'inf', 'absurd', 'negative'):
            # a burst of invalid samples is also a dropout of valid data (> wheel_timeout_s)
            return 'invalid', _bits(f, r, 9, 10), True, base_ok | {'invalid', 'slip', 'slide', 'dropout'}
        if kind == 'zero':
            return 'slide', _bits(f, r, 2, 3), False, base_ok | {'slide', 'slip'}
        return 'slip|slide', _bits(f, r, 0, 1) | _bits(f, r, 2, 3), False, base_ok | {'slip', 'slide'}
    if typ == 'notch_fault':
        return 'cmd_inconsistent', FAM['cmd_inconsistent'], False, {'cmd_inconsistent', 'unmodeled_accel'}
    if typ in ('stamp_glitch', 'clock_offset', 'zero_stamps'):
        return 'late_data', FAM['late_data'], False, set(FAM)
    return 'none', 0, False, set(FAM)


# ------------------------------------------------------------------------------------------ geometry
A_WGS, F_WGS = 6378137.0, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)


def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A_WGS / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


def enu(lat, lon, h, lat0, lon0, h0):
    d = ecef(lat, lon, h) - ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))
    la, lo = np.radians(lat0), np.radians(lon0)
    r = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return d @ r.T


def match_nearest(ref_t: np.ndarray, out_t: np.ndarray, tol: float = 0.05):
    """Index of the nearest output stamp per reference stamp (quick_eval convention) and a mask |dt| <= tol."""
    if len(out_t) < 2:
        return np.zeros(len(ref_t), int), np.zeros(len(ref_t), bool)
    idx = np.clip(np.searchsorted(out_t, ref_t), 1, len(out_t) - 1)
    left, right = out_t[idx - 1], out_t[idx]
    pick = np.where(np.abs(ref_t - left) <= np.abs(right - ref_t), idx - 1, idx)
    return pick, np.abs(out_t[pick] - ref_t) <= tol


# ------------------------------------------------------------------------------------------ scenarios
def load_scenarios() -> dict:
    out = {}
    for f in SUITE_FILES:
        for sc in load_suite(f):
            out[sc.name] = sc
    return out


SCENARIOS = load_scenarios()


def select_scenarios(specs: list[str] | None) -> list[str]:
    names = list(SCENARIOS)
    if not specs or specs == ['all']:
        return names
    sel = []
    for s in specs:
        for n in names:
            if (n == s or n.startswith(s)) and n not in sel:
                sel.append(n)
    return sel


def bag_list(spec: list[str]) -> list[str]:
    splits = json.loads(Path(SPLITS_JSON).read_text())
    out = []
    for s in spec:
        names = GOOD_BAGS if s == 'good' else (splits[s] if s in splits and isinstance(splits[s], list) else [s])
        out += [n for n in names if n not in out]
    return out


# ------------------------------------------------------------------------------------------ export
def export_run_events(run, bag: str, out_csv: Path, gnss_seconds: float | None) -> dict:
    """Arrival-ordered event CSV of a (corrupted) run, formatted exactly like cpp_bridge.export_events.

    Vehicle messages come from the run; GNSS fixes (both antennas) from the clean npz, kept only while
    recv <= first vehicle recv + gnss_seconds (cpp_bridge convention)."""
    typ, recv, stamp, vals = [], [], [], []
    for code, key in ((0, FRONT), (1, REAR), (2, CMD)):
        s = run.streams.get(key)
        if s is None or len(s) == 0:
            continue
        typ.append(np.full(len(s), code, np.int8))
        recv.append((s.t_bag * 1e9).round().astype(np.int64))
        stamp.append((s.t_hdr * 1e9).round().astype(np.int64))
        v = np.zeros((len(s), 4))
        v[:, 0] = s.val[:, 0]
        vals.append(v)
    t_first = min(int(r.min()) for r in recv)
    d = np.load(NPZ_DIR / f'{bag}.npz')
    n_gf = 0
    for code, key in ((3, 'sensing__gnss__master__fix'), (4, 'sensing__gnss__rover__fix')):
        a = d[key]
        if len(a) == 0:
            continue
        rv = (a[:, 0] * 1e9).round().astype(np.int64)
        keep = np.ones(len(a), bool) if gnss_seconds is None else rv <= t_first + int(gnss_seconds * 1e9)
        a, rv = a[keep], rv[keep]
        typ.append(np.full(len(a), code, np.int8))
        recv.append(rv)
        stamp.append((a[:, 1] * 1e9).round().astype(np.int64))
        vals.append(a[:, 2:6].copy())
        n_gf += len(a)
    typ, recv, stamp, vals = np.concatenate(typ), np.concatenate(recv), np.concatenate(stamp), np.vstack(vals)
    order = np.argsort(recv, kind='stable')
    n_skip = 0
    lines = []
    for i in order:
        t = typ[i]
        if t <= 1:
            lines.append(f'W{t},{recv[i]},{stamp[i]},{vals[i, 0]:.6f}\n')
        elif t == 2:
            if not np.isfinite(vals[i, 0]):
                n_skip += 1
                continue
            lines.append(f'C,{recv[i]},{stamp[i]},{int(vals[i, 0])}\n')
        else:
            lines.append(f'GF,{recv[i]},{stamp[i]},{t - 3},{vals[i, 0]:.10f},{vals[i, 1]:.10f},'
                         f'{vals[i, 2]:.4f},{int(vals[i, 3])}\n')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='\n') as fh:
        fh.writelines(lines)
    return {'events': len(lines), 't_first_ns': t_first, 'n_gnss_fix': n_gf, 'n_cmd_skipped_nonfinite': n_skip}


def ensure_events(scn: str, bag: str) -> tuple[Path, dict]:
    """Cached event CSV + anomaly event list for (scenario, bag)."""
    d = CACHE / 'events' / scn
    csv_p, js_p = d / f'{bag}.csv', d / f'{bag}.anom.json'
    if csv_p.exists() and js_p.exists():
        return csv_p, json.loads(js_p.read_text(encoding='utf-8'))
    sc = SCENARIOS[scn]
    t = time.perf_counter()
    clean = load_run(NPZ_DIR / f'{bag}.npz', name=bag)
    run = apply_scenario(clean, sc)
    tmp = d / f'{bag}.csv.tmp{os.getpid()}'
    info = export_run_events(run, bag, tmp, sc.gnss_keep_s)
    meta = {'scenario': scn, 'bag': bag, 'seed': sc.seed, 'gnss_keep_s': sc.gnss_keep_s,
            't_start': clean.t_start, 'export': info, 'gen_s': round(time.perf_counter() - t, 2),
            'events': [e for e in run.events if e['type'] != 'gnss_cut']}
    os.replace(tmp, csv_p)
    js_tmp = d / f'{bag}.anom.json.tmp{os.getpid()}'
    js_tmp.write_text(json.dumps(meta, default=_np_default), encoding='utf-8')
    os.replace(js_tmp, js_p)
    return csv_p, json.loads(js_p.read_text(encoding='utf-8'))


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------ replay
def run_replay(events_csv: Path, out_csv: Path, sets: dict) -> tuple[int, str, float]:
    cmd = [str(SNAP_EXE), '--in', str(events_csv), '--out', str(out_csv), '--map', str(MAP),
           '--traction', str(TRACTION), '--branches', BRANCHES, '--set', f'landmark_file={LANDMARKS}']
    for k, v in (sets or {}).items():
        cmd += ['--set', f'{k}={v}']
    t = time.perf_counter()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        rc, err = res.returncode, (res.stderr or '').strip()
    except subprocess.TimeoutExpired:
        rc, err = -999, 'timeout'
    return rc, err, time.perf_counter() - t


NUM_COLS = ['v', 'v_var', 'x', 'y', 'z', 'yaw', 's', 's_var', 'cov_xx', 'cov_xy', 'cov_yy', 'cov_zz', 'mu0', 'mu1',
            'mu2', 'mu3', 'mu4', 'slip_f', 'slip_r', 'd', 'k', 'g', 'a_model', 'accel']


def read_output(out_csv: Path) -> dict | None:
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return None
    df = pd.read_csv(out_csv, low_memory=False)
    if len(df) == 0:
        return {'n': 0}
    o = {'n': len(df)}
    o['stamp_ns'] = pd.to_numeric(df['stamp_ns'], errors='coerce').fillna(0).astype(np.int64).to_numpy()
    o['recv_ns'] = pd.to_numeric(df['recv_ns'], errors='coerce').fillna(0).astype(np.int64).to_numpy()
    for c in NUM_COLS:
        o[c] = pd.to_numeric(df[c], errors='coerce').to_numpy(np.float64)
    o['flags'] = pd.to_numeric(df['flags'], errors='coerce').fillna(0).astype(np.int64).to_numpy()
    o['proc_ns'] = pd.to_numeric(df['proc_ns'], errors='coerce').fillna(0).to_numpy(np.float64)
    o['trigger'] = df['trigger'].astype(str).to_numpy()
    return o


def sort_by_stamp(o: dict) -> dict:
    """Outputs ordered by stamp (the scheduler guarantees it; this guards the matching if it ever fails)."""
    st = o['stamp_ns']
    if len(st) < 2 or np.all(np.diff(st) > 0):
        return o
    order = np.argsort(st, kind='stable')
    return {k: (v[order] if isinstance(v, np.ndarray) and len(v) == len(st) else v) for k, v in o.items()}


def save_compact(o: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, stamp_ns=o['stamp_ns'], recv_ns=o['recv_ns'], v=o['v'], s=o['s'], x=o['x'], y=o['y'],
                        z=o['z'], yaw=o['yaw'], flags=o['flags'].astype(np.uint32),
                        mu=np.stack([o[f'mu{j}'] for j in range(5)], 1).astype(np.float32),
                        slip_f=o['slip_f'].astype(np.float32), slip_r=o['slip_r'].astype(np.float32),
                        a_model=o['a_model'].astype(np.float32), accel=o['accel'].astype(np.float32),
                        d=o['d'].astype(np.float32), k=o['k'].astype(np.float32), g=o['g'].astype(np.float32),
                        v_var=o['v_var'].astype(np.float32), s_var=o['s_var'].astype(np.float32),
                        proc_ns=o['proc_ns'].astype(np.float32))


def load_compact(path: Path) -> dict:
    with np.load(path) as z:
        o = {k: z[k] for k in z.files}
    o['n'] = len(o['stamp_ns'])
    o['mu3'] = o['mu'][:, 3].astype(np.float64)
    o['flags'] = o['flags'].astype(np.int64)
    return o


# ------------------------------------------------------------------------------------------ reference
_REF_CACHE: dict = {}


def load_ref(bag: str) -> dict:
    if bag in _REF_CACHE:
        return _REF_CACHE[bag]
    d = np.load(NPZ_DIR / f'{bag}.npz')
    ref: dict = {'bag': bag}
    mv = d['sensing__gnss__master__vel']
    mv = mv[np.isfinite(mv[:, 1:4]).all(1)] if len(mv) else mv
    if len(mv) > 50:
        mv = mv[np.argsort(mv[:, 1], kind='stable')]
        ref['t'] = mv[:, 1]
        ref['v'] = np.hypot(mv[:, 2], mv[:, 3])
        # master-vel glitches (e.g. master 9.3 m/s while rover and wheels say 2.4): epochs where the rover
        # velocity (nearest header stamp within 0.05 s) disagrees by > 0.3 m/s are removed from the reference
        rv = d['sensing__gnss__rover__vel']
        rv = rv[np.isfinite(rv[:, 1:4]).all(1)] if len(rv) else rv
        ref['n_ref_glitch'] = 0
        if len(rv) > 50:
            rv = rv[np.argsort(rv[:, 1], kind='stable')]
            j, okr = match_nearest(ref['t'], rv[:, 1])
            bad = okr & (np.abs(np.hypot(rv[j, 2], rv[j, 3]) - ref['v']) > 0.3)
            ref['v'] = np.where(bad, np.nan, ref['v'])
            ref['n_ref_glitch'] = int(bad.sum())
    mf = d['sensing__gnss__master__fix']
    mf = mf[np.isfinite(mf[:, 2])] if len(mf) else mf
    if len(mf) > 50:
        ref['t_fix'] = mf[:, 1]
        ref['p_fix'] = enu(mf[:, 2], mf[:, 3], mf[:, 4], mf[0, 2], mf[0, 3], mf[0, 4])
        ref['rtk'] = float((mf[:, 5] == 2).mean())
        ref['dist_m'] = float(np.sum(np.hypot(np.diff(ref['p_fix'][:, 0]), np.diff(ref['p_fix'][:, 1]))))
    t_all = [d[k][:, 0].min() for k in d.files if len(d[k])]
    ref['t_start'] = float(min(t_all))
    ref['t_first_veh'] = float(min(d[k][:, 0].min() for k in ('vehicle__front_bogie_velocity',
                                                               'vehicle__rear_bogie_velocity',
                                                               'vehicle__driver_position_cmd') if len(d[k])))
    for key, name in ((FRONT, 'front'), (REAR, 'rear'), (CMD, 'cmd')):
        a = d[key]
        ref[f'hdr_{name}'] = np.sort(a[:, 1]) if len(a) else np.zeros(0)
    try:
        ck = pd.read_csv(CLOCKS).set_index('bag')
        anom = float(ck.loc[bag, 'gnss_vs_veh_anom_frac']) if bag in ck.index else 0.0
    except Exception:
        anom = float('nan')
    ref['clock_anom'] = anom
    ref['good'] = bool('t' in ref and ref.get('rtk', 0) > 0.8 and not (anom > 0.01))
    _REF_CACHE[bag] = ref
    return ref


_NAT: dict | None = None


def natural_windows(bag: str, t_start: float) -> list[tuple[float, float, str]]:
    global _NAT
    if _NAT is None:
        _NAT = {}
        try:
            for e in json.loads(NATURAL.read_text(encoding='utf-8')):
                _NAT.setdefault(e['bag'], []).append(e)
        except Exception:
            pass
    out = []
    for e in _NAT.get(bag, []):
        a = t_start + float(e.get('t0_rel', 0.0))
        out.append((a - 3.0, a + float(e.get('duration', 0.0) or 0.0) + 3.0, e['type']))
    return out


# ------------------------------------------------------------------------------------------ metrics
def _rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float('nan')


def _max_abs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.max(np.abs(x))) if len(x) else float('nan')


def first_sustained(t: np.ndarray, cond: np.ndarray, t_from: float, t_to: float, hold: float = REC_HOLD):
    """Earliest time >= t_from at which `cond` holds on all samples for `hold` seconds (None if never)."""
    sel = np.flatnonzero((t >= t_from) & (t <= t_to + hold))
    start = None
    for i in sel:
        if cond[i]:
            if start is None:
                start = t[i]
            if t[i] - start >= hold - 1e-9:
                return start
        else:
            start = None
    return None


def episodes(t: np.ndarray, mask: np.ndarray, merge_gap: float = 1.0) -> list[tuple[float, float]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    ts = t[idx]
    br = np.flatnonzero(np.diff(ts) > merge_gap)
    st, en = np.r_[0, br + 1], np.r_[br, len(ts) - 1]
    return [(float(ts[a]), float(ts[b])) for a, b in zip(st, en)]


def timing_metrics(o: dict) -> dict:
    st = o['stamp_ns'].astype(np.int64)
    rv = o['recv_ns'].astype(np.int64)
    ds = np.diff(st)
    span = (st.max() - st.min()) * 1e-9 if len(st) > 1 else float('nan')
    lat = (rv - st) * 1e-9
    body = rv >= rv.min() + int(60e9)               # skip the start-up burst (history delivered at start)
    lat_b = lat[body] if body.sum() > 100 else lat
    med = float(np.median(lat_b)) if len(lat_b) else float('nan')
    return {'n_out': int(len(st)), 'rate_hz': float(len(st) / span) if span and span > 0 else float('nan'),
            'distinct_stamps_hz': float(len(np.unique(st)) / span) if span and span > 0 else float('nan'),
            'n_backward_stamps': int((ds <= 0).sum()), 'max_stamp_gap_s': float(ds.max() * 1e-9) if len(ds) else float('nan'),
            'n_stamp_gaps_gt_0p2s': int((ds > 2e8).sum()),
            'lat_med_s': med, 'lat_p99_s': float(np.percentile(lat_b, 99)) if len(lat_b) else float('nan'),
            'lat_max_s': float(lat_b.max()) if len(lat_b) else float('nan'),
            'n_stamp_off_gt_0p5s': int((np.abs(lat_b - med) > 0.5).sum()),
            'proc_us_p99': float(np.percentile(o['proc_ns'], 99) / 1e3) if len(o.get('proc_ns', [])) else float('nan'),
            'proc_us_max': float(o['proc_ns'].max() / 1e3) if len(o.get('proc_ns', [])) else float('nan')}


def sanity_metrics(o: dict) -> dict:
    cols = ['v', 'x', 'y', 'z', 's', 'v_var', 's_var', 'yaw']
    nonfinite = int(sum((~np.isfinite(np.asarray(o[c], float))).sum() for c in cols if c in o))
    v = np.asarray(o['v'], float)
    # isolated output glitches: one output off by > 0.3 m/s from BOTH neighbours in the same direction
    n_gl, gl_max = 0, 0.0
    if len(v) > 2:
        d1, d2 = v[1:-1] - v[:-2], v[1:-1] - v[2:]
        g = (np.abs(d1) > 0.3) & (np.abs(d2) > 0.3) & (np.sign(d1) == np.sign(d2))
        n_gl = int(g.sum())
        gl_max = float(np.minimum(np.abs(d1), np.abs(d2))[g].max()) if n_gl else 0.0
    return {'n_nonfinite': nonfinite, 'n_absurd_v': int(((v < 0) | (v > 40)).sum()),
            'v_out_max': float(np.nanmax(v)) if len(v) else float('nan'),
            'n_glitch_0p3': n_gl, 'glitch_max': gl_max}


def parse_counters(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostics counters printed by tbo_replay on stderr -> numeric columns."""
    if 'stderr' not in df:
        return df
    s = df['stderr'].fillna('').astype(str)
    for key, col in (('rejected', 'rejected_stamps'), ('late', 'late_dropped'), ('invalid_wheel', 'invalid_wheel'),
                     ('resets', 'resets')):
        df[col] = pd.to_numeric(s.str.extract(rf'\b{key}=(\d+)')[0], errors='coerce')
    return df


def accuracy(o: dict, ref: dict) -> dict:
    """Judge-like accuracy of one output stream vs the GNSS reference (quick_eval conventions)."""
    out_t = o['stamp_ns'] * 1e-9
    res: dict = {}
    if 't' in ref:
        pick, ok = match_nearest(ref['t'], out_t)
        err = o['v'][pick][ok] - ref['v'][ok]
        mov = ref['v'][ok] > 0.5
        res.update(v_rmse=_rms(err), v_mae=float(np.nanmean(np.abs(err))) if len(err) else float('nan'),
                   v_bias=float(np.nanmean(err)) if len(err) else float('nan'), v_rmse_mov=_rms(err[mov]),
                   v_p99=float(np.nanpercentile(np.abs(err), 99)) if len(err) else float('nan'),
                   v_max=_max_abs(err), match_rate=float(ok.mean()))
    if 't_fix' in ref:
        pick, ok = match_nearest(ref['t_fix'], out_t)
        e = np.column_stack([o['x'], o['y'], o['z']])[pick][ok] - ref['p_fix'][ok]
        yaw = o['yaw'][pick][ok]
        along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
        cross = -e[:, 0] * np.sin(yaw) + e[:, 1] * np.cos(yaw)
        e3 = np.linalg.norm(e, axis=1)
        if len(e3):
            res.update(p3_rmse=_rms(e3), p3_max=_max_abs(e3), along_rmse=_rms(along), along_max=_max_abs(along),
                       cross_rmse=_rms(cross), z_rmse=_rms(e[:, 2]), end_err=float(e3[-1]),
                       end_along=float(along[-1]), drift_pct=float(100 * e3[-1] / max(ref.get('dist_m', 1.0), 1.0)))
    fl = o['flags']
    res['model_only_frac'] = float((np.asarray(o['mu3']) > 0.5).mean()) if len(fl) else float('nan')
    res['model_only_flag_frac'] = float(((fl & FAM['model_only']) != 0).mean()) if len(fl) else float('nan')
    return res


def clean_flag_episodes(o: dict, ref: dict, bag: str) -> list[dict]:
    """Every flag episode of a clean-input run, annotated with natural anomalies / real input gaps."""
    t = o['stamp_ns'] * 1e-9
    fl = o['flags']
    nat = natural_windows(bag, ref['t_start'])
    rows = []
    for fam in FA_FAMILIES:
        for a, b in episodes(t, (fl & FAM[fam]) != 0):
            i = int(np.searchsorted(t, a))
            nat_hit = [typ for (x0, x1, typ) in nat if x0 <= b and x1 >= a]
            gap = float('nan')
            if fam in ('dropout', 'cmd_dropout'):
                keys = ['cmd'] if fam == 'cmd_dropout' else ['front', 'rear']
                g = 0.0
                for k in keys:
                    h = ref[f'hdr_{k}']
                    if len(h) < 2:
                        g = max(g, 999.0)
                        continue
                    j0, j1 = np.searchsorted(h, a - 2.0), np.searchsorted(h, b + 1.0)
                    hh = h[max(j0 - 1, 0):min(j1 + 1, len(h))]
                    if len(hh) >= 2:
                        g = max(g, float(np.diff(hh).max()))
                    if a < h[0] or b > h[-1]:
                        g = max(g, 999.0)
                gap = g
            rows.append({'bag': bag, 'family': fam, 't0_rel': round(a - ref['t_first_veh'], 3),
                         'duration_s': round(b - a, 3), 'v_est': float(o['v'][min(i, len(t) - 1)]),
                         'a_model': float(o['a_model'][min(i, len(t) - 1)]),
                         'natural': ';'.join(sorted(set(nat_hit))), 'input_gap_s': gap,
                         'startup': bool(a - ref['t_first_veh'] < STARTUP_S)})
    return rows


def run_seconds(o: dict) -> tuple[float, float]:
    t = o['stamp_ns'] * 1e-9
    if len(t) < 2:
        return 0.0, 0.0
    dt = np.clip(np.diff(t), 0, 1.0)
    mov = o['v'][1:] > 0.5
    return float(dt.sum()), float(dt[mov].sum())


def score_scenario(o: dict, oc: dict, ref: dict, anom: dict, scn: str, bag: str) -> tuple[dict, list[dict]]:
    """Run-level row and per-event rows for one scenario replay `o` vs the clean replay `oc`."""
    row: dict = {}
    row.update(timing_metrics(o))
    o = sort_by_stamp(o)
    row.update({f'{k}': v for k, v in accuracy(o, ref).items()})
    acc_c = accuracy(oc, ref)
    row.update({f'clean_{k}': v for k, v in acc_c.items() if k in ('v_rmse', 'v_max', 'p3_rmse', 'along_rmse',
                                                                     'along_max', 'end_err', 'end_along')})
    row.update(sanity_metrics(o))
    t_o = o['stamp_ns'] * 1e-9
    t_c = oc['stamp_ns'] * 1e-9
    # ---- scenario vs clean replay on the scenario's own output stamps ----
    pk, okc = match_nearest(t_o, t_c, tol=0.1)
    dv = np.where(okc, o['v'] - oc['v'][pk], np.nan)
    dsx = np.where(okc, o['s'] - oc['s'][pk], np.nan)
    dp = np.where(okc, np.sqrt((o['x'] - oc['x'][pk]) ** 2 + (o['y'] - oc['y'][pk]) ** 2 +
                               (o['z'] - oc['z'][pk]) ** 2), np.nan)
    flags_c_at_o = np.where(okc, oc['flags'][pk], 0)
    extra_o = o['flags'] & ~flags_c_at_o
    row.update(dv_vs_clean_rms=_rms(dv), dv_vs_clean_max=_max_abs(dv), ds_vs_clean_max=_max_abs(dsx),
               ds_vs_clean_end=float(dsx[np.isfinite(dsx)][-1]) if np.isfinite(dsx).any() else float('nan'),
               dp_vs_clean_end=float(dp[np.isfinite(dp)][-1]) if np.isfinite(dp).any() else float('nan'),
               dp_vs_clean_max=_max_abs(dp))
    row['extra_flag_frac'] = float(((extra_o & RECOVERY_MASK) != 0).mean()) if len(extra_o) else float('nan')
    row['diverged'] = bool((row.get('v_max', 0) > 5.0) or (row['ds_vs_clean_max'] > 100.0) or
                           (row['n_nonfinite'] > 0) or (row['n_absurd_v'] > 0))
    # ---- reference timeline (10 Hz GNSS vel header stamps) ----
    windows: list[dict] = []
    events = [e for e in anom['events'] if e['type'] not in GLOBAL_TYPES]
    row['n_events'] = len(anom['events'])
    row['event_types'] = ';'.join(sorted({e['type'] for e in anom['events']}))
    if 't' not in ref:
        return row, windows
    keep_ref = np.isfinite(ref['v'])                # reference epochs without a GNSS glitch
    tr, vr = ref['t'][keep_ref], ref['v'][keep_ref]
    ps, oks = match_nearest(tr, t_o)
    pc, okc2 = match_nearest(tr, t_c)
    e_s = np.where(oks, o['v'][ps] - vr, np.nan)
    e_c = np.where(okc2, oc['v'][pc] - vr, np.nan)
    d_sc = np.where(oks & okc2, o['v'][ps] - oc['v'][pc], np.nan)
    ds_sc = np.where(oks & okc2, o['s'][ps] - oc['s'][pc], np.nan)
    fl_s = np.where(oks, o['flags'][ps], 0)
    fl_c = np.where(okc2, oc['flags'][pc], 0)
    extra = fl_s & ~fl_c & RECOVERY_MASK
    mu3_s = np.where(oks, o['mu3'][ps], np.nan)
    cond_gnss = oks & (np.abs(np.nan_to_num(e_s, nan=9.0)) < REC_TOL) & (extra == 0)
    cond_clean = oks & okc2 & (np.abs(np.nan_to_num(d_sc, nan=9.0)) < REC_TOL) & (extra == 0)
    cond_flags = oks & (extra == 0)
    cond_speed = oks & (np.abs(np.nan_to_num(e_s, nan=9.0)) < REC_TOL)   # speed only, flags ignored
    t_end = float(tr[-1])
    starts = sorted(float(e['t0']) for e in events)
    # ---- fix epochs for along-track error vs GNSS ----
    along_s = along_c = None
    if 't_fix' in ref:
        fs, fok = match_nearest(ref['t_fix'], t_o)
        fc, fokc = match_nearest(ref['t_fix'], t_c)

        def _along(oo, pick, ok):
            e = np.column_stack([oo['x'], oo['y']])[pick] - ref['p_fix'][:, :2]
            yaw = oo['yaw'][pick]
            return np.where(ok, e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw), np.nan)
        along_s, along_c = _along(o, fs, fok), _along(oc, fc, fokc)

    def along_at(arr, t):
        if arr is None:
            return float('nan')
        j = int(np.clip(np.searchsorted(ref['t_fix'], t), 0, len(ref['t_fix']) - 1))
        lo, hi = max(j - 5, 0), min(j + 6, len(arr))
        seg, ts = arr[lo:hi], ref['t_fix'][lo:hi]
        okk = np.isfinite(seg)
        if not okk.any():
            return float('nan')
        return float(seg[okk][np.argmin(np.abs(ts[okk] - t))])

    def ds_at(t):
        j = int(np.clip(np.searchsorted(tr, t), 0, len(tr) - 1))
        lo, hi = max(j - 5, 0), min(j + 6, len(tr))
        seg, ts = ds_sc[lo:hi], tr[lo:hi]
        okk = np.isfinite(seg)
        return float(seg[okk][np.argmin(np.abs(ts[okk] - t))]) if okk.any() else float('nan')

    ev_all = sorted(events, key=lambda e: e['t0'])
    for k, ev in enumerate(ev_all):
        t0, t1 = float(ev['t0']), float(ev['t1'])
        nxt = [s for s in starts if s > t0 + 1e-6]
        t_next = min(nxt) if nxt else t_end + 1.0
        # other events overlapping this one (combined scenarios)
        overlap = any((o2 is not ev) and (float(o2['t0']) <= t1 + 2.0) and (float(o2['t1']) >= t0 - 2.0)
                      for o2 in ev_all)
        w_in = (tr >= t0) & (tr <= t1 + TAIL_S)
        a_end = min(t1 + AFTER_S, t_next)
        w_after = (tr > t1 + TAIL_S) & (tr <= a_end)
        vref_in = vr[w_in]
        dv_true = float(np.nanmax(vref_in) - np.nanmin(vref_in)) if w_in.any() else 0.0
        fam, mask, required, allowed = expected_detection(ev, dv_true)
        peak_dw = float('nan')
        st = ev.get('stats') or {}
        pk_vals = [abs(float(v2.get('peak_dw', 0.0))) for k2, v2 in st.items() if isinstance(v2, dict) and 'peak_dw' in v2]
        if pk_vals:
            peak_dw = max(pk_vals)
        w: dict = {'scenario': scn, 'bag': bag, 'event_id': ev.get('id', k), 'type': ev['type'],
                   'topics': ';'.join(x.split('__')[-1] for x in ev.get('topics', [])),
                   'kind': (ev.get('params') or {}).get('kind') or (ev.get('params') or {}).get('mode') or
                           (ev.get('params') or {}).get('model') or '',
                   'bogie': (ev.get('params') or {}).get('bogie', ''), 't0_rel': round(t0 - ref['t_first_veh'], 3),
                   'dur_s': round(t1 - t0, 3), 'v_ref_t0': float(np.interp(t0, tr, vr)), 'dv_true_in': dv_true,
                   'peak_dw_ms': peak_dw, 'overlap': overlap, 'after_s': round(max(a_end - t1, 0.0), 2),
                   'exp_family': fam, 'det_required': bool(required)}
        for key, val in (ev.get('params') or {}).items():
            if key in ('peak_rel', 'depth_rel', 'lock', 'lock_duration', 'both_mode', 'offset_s', 'sigma_kmh'):
                w[f'p_{key}'] = val
        # ---- errors during / after ----
        w.update(n_ref_in=int(w_in.sum()), unmatched_in=int((w_in & ~oks).sum()),
                 err_in_rms=_rms(e_s[w_in]), err_in_max=_max_abs(e_s[w_in]),
                 clean_err_in_rms=_rms(e_c[w_in]), clean_err_in_max=_max_abs(e_c[w_in]),
                 dv_in_rms=_rms(d_sc[w_in]), dv_in_max=_max_abs(d_sc[w_in]),
                 err_after_rms=_rms(e_s[w_after]), err_after_max=_max_abs(e_s[w_after]),
                 clean_err_after_rms=_rms(e_c[w_after]), clean_err_after_max=_max_abs(e_c[w_after]),
                 dv_after_max=_max_abs(d_sc[w_after]),
                 model_only_in=float(np.nanmean(mu3_s[w_in] > 0.5)) if w_in.any() else float('nan'))
        # ---- along-track: distance lost/gained through this event (vs clean replay and vs GNSS) ----
        ds0, ds1 = ds_at(t0 - 0.2), ds_at(a_end)
        w.update(ds_t0=ds0, ds_after=ds1, ds_growth=ds1 - ds0,
                 ds_max_in_after=_max_abs(ds_sc[(tr >= t0) & (tr <= a_end)] - (ds0 if np.isfinite(ds0) else 0.0)))
        if along_s is not None:
            a0s, a1s = along_at(along_s, t0 - 0.2), along_at(along_s, a_end)
            a0c, a1c = along_at(along_c, t0 - 0.2), along_at(along_c, a_end)
            w.update(along_growth=a1s - a0s, clean_along_growth=a1c - a0c,
                     along_growth_excess=(a1s - a0s) - (a1c - a0c))
        # ---- recovery (from the window end) ----
        horizon = min(t1 + 120.0, t_next, t_end)
        for name, cond in (('rec_gnss_s', cond_gnss), ('rec_clean_s', cond_clean), ('rec_flags_s', cond_flags),
                           ('rec_speed_s', cond_speed)):
            r = first_sustained(tr, cond, t1, horizon)
            w[name] = max(r - t1, 0.0) if r is not None else float('nan')
        w['rec_horizon_s'] = round(horizon - t1, 2)
        w['recovered'] = bool(np.isfinite(w['rec_gnss_s']))
        # ---- detection on the output timeline (flags beyond the clean run) ----
        sel = (t_o >= t0 - DET_PRE) & (t_o <= t1 + DET_GRACE)
        ex = extra_o[sel]
        fired = [f2 for f2 in FA_FAMILIES if ((ex & FAM[f2]) != 0).any()]
        w['families_fired'] = ';'.join(fired)
        # misdiagnosis: a family that should not appear, judged on the event core (transition artifacts at
        # the event edges - e.g. wheels returning below a latched model - are excluded)
        core = (t_o >= t0) & (t_o <= (t1 if t1 - t0 > 0.6 else t1 + 0.5))
        exc = extra_o[core]
        fired_core = [f2 for f2 in FA_FAMILIES if ((exc & FAM[f2]) != 0).any()]
        w['wrong_families'] = ';'.join(f2 for f2 in fired_core if f2 not in allowed)
        if mask:
            hit = np.flatnonzero((ex & mask) != 0)
            w['det_fired'] = bool(len(hit))
            w['det_delay_s'] = float(t_o[sel][hit[0]] - t0) if len(hit) else float('nan')
            fin = (t_o >= t0) & (t_o <= t1)
            w['det_frac_in'] = float(((extra_o[fin] & mask) != 0).mean()) if fin.any() else float('nan')
            fam_all = 0
            for f2 in fam.split('|'):
                fam_all |= FAM.get(f2, 0)
            w['det_other_bogie'] = bool(((ex & fam_all & ~mask) != 0).any())
        else:
            w['det_fired'] = False
            w['det_delay_s'] = float('nan')
            w['det_frac_in'] = float('nan')
            w['det_other_bogie'] = False
        windows.append(w)
    return row, windows


# ------------------------------------------------------------------------------------------ workers
def _paths(variant: str) -> dict:
    vd = RESULTS / variant
    return {'dir': vd, 'clean': vd / 'clean_out', 'out': vd / 'out', 'tmp': RESULTS / 'tmp'}


def task_clean(variant: str, sets: dict, bag: str, keep_out: bool) -> dict:
    p = _paths(variant)
    try:
        ev_csv, anom = ensure_events('S00_baseline_gnss_cut', bag)
        out_csv = p['tmp'] / f'{variant}__S00__{bag}.csv'
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rc, err, wall = run_replay(ev_csv, out_csv, sets)
        o = read_output(out_csv) if rc == 0 else None
        ref = load_ref(bag)
        row = {'bag': bag, 'rc': rc, 'stderr': err[-300:], 'wall_s': round(wall, 2), 'ref_good': ref['good'],
               'ref_rtk': ref.get('rtk'), 'ref_clock_anom': ref['clock_anom'], 'dist_m': ref.get('dist_m'),
               'n_ref_glitch': ref.get('n_ref_glitch')}
        if o is None or o.get('n', 0) == 0:
            row['crash'] = True
            return {'row': row, 'episodes': []}
        row['crash'] = False
        row.update(timing_metrics(o))
        o = sort_by_stamp(o)
        row.update(accuracy(o, ref))
        row.update(sanity_metrics(o))
        hrs, hrs_mov = run_seconds(o)
        row['hours'] = hrs / 3600.0
        row['hours_moving'] = hrs_mov / 3600.0
        save_compact(o, p['clean'] / f'{bag}.npz')
        if keep_out:
            save_compact(o, p['out'] / 'S00_baseline_gnss_cut' / f'{bag}.npz')
        eps = clean_flag_episodes(o, ref, bag)
        out_csv.unlink(missing_ok=True)
        return {'row': row, 'episodes': eps}
    except Exception:
        return {'row': {'bag': bag, 'crash': True, 'error': traceback.format_exc()[-800:]}, 'episodes': []}


def task_scenario(variant: str, sets: dict, bag: str, scn: str, keep_out: bool) -> dict:
    p = _paths(variant)
    base = {'scenario': scn, 'bag': bag}
    try:
        ev_csv, anom = ensure_events(scn, bag)
        out_csv = p['tmp'] / f'{variant}__{scn}__{bag}.csv'
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rc, err, wall = run_replay(ev_csv, out_csv, sets)
        o = read_output(out_csv) if rc == 0 else None
        row = {**base, 'rc': rc, 'stderr': err[-300:], 'wall_s': round(wall, 2), 'gen_s': anom.get('gen_s')}
        if o is None or o.get('n', 0) == 0:
            row['crash'] = True
            out_csv.unlink(missing_ok=True)
            return {'row': row, 'windows': []}
        row['crash'] = False
        oc = load_compact(p['clean'] / f'{bag}.npz')
        ref = load_ref(bag)
        r2, wins = score_scenario(o, oc, ref, anom, scn, bag)
        row.update(r2)
        if keep_out:
            save_compact(o, p['out'] / scn / f'{bag}.npz')
        out_csv.unlink(missing_ok=True)
        return {'row': row, 'windows': wins}
    except Exception:
        return {'row': {**base, 'crash': True, 'error': traceback.format_exc()[-800:]}, 'windows': []}


def task_rescore_clean(variant: str, bag: str) -> dict:
    """Re-score a saved clean replay (no binary run)."""
    p = _paths(variant)
    try:
        o = load_compact(p['clean'] / f'{bag}.npz')
        ref = load_ref(bag)
        row = {'bag': bag, 'ref_good': ref['good'], 'ref_rtk': ref.get('rtk'), 'ref_clock_anom': ref['clock_anom'],
               'dist_m': ref.get('dist_m'), 'n_ref_glitch': ref.get('n_ref_glitch'), 'crash': False}
        row.update(timing_metrics(o))
        o = sort_by_stamp(o)
        row.update(accuracy(o, ref))
        row.update(sanity_metrics(o))
        hrs, hrs_mov = run_seconds(o)
        row['hours'] = hrs / 3600.0
        row['hours_moving'] = hrs_mov / 3600.0
        return {'row': row, 'episodes': clean_flag_episodes(o, ref, bag)}
    except Exception:
        return {'row': {'bag': bag, 'crash': True, 'error': traceback.format_exc()[-800:]}, 'episodes': []}


def task_rescore(variant: str, bag: str, scn: str) -> dict:
    """Re-score a saved scenario replay (no binary run); None if the run has no saved output."""
    p = _paths(variant)
    f = p['out'] / scn / f'{bag}.npz'
    if not f.exists():
        return {'row': None, 'windows': []}
    try:
        o = load_compact(f)
        oc = load_compact(p['clean'] / f'{bag}.npz')
        anom = json.loads((CACHE / 'events' / scn / f'{bag}.anom.json').read_text(encoding='utf-8'))
        row = {'scenario': scn, 'bag': bag, 'crash': False, 'gen_s': anom.get('gen_s')}
        r2, wins = score_scenario(o, oc, load_ref(bag), anom, scn, bag)
        row.update(r2)
        return {'row': row, 'windows': wins}
    except Exception:
        return {'row': {'scenario': scn, 'bag': bag, 'crash': True, 'error': traceback.format_exc()[-800:]},
                'windows': []}


SLIP_LABELS = 0b111  # anomaly_sim Label.SLIP | SLIDE | LOCK


def task_slip_accuracy(variant: str, bag: str, scn: str) -> list[dict]:
    """Published slip ratio (slip_f / slip_r) vs the injected wheel error during slip/slide events.

    True ratio per bogie sample = (corrupted - clean wheel) [m/s] / max(v_ref, 1); the estimator publishes
    (wheel - v_est (1 + k)) / max(v_est, 1) at every output stamp (nearest output within 0.05 s)."""
    p = _paths(variant)
    f = p['out'] / scn / f'{bag}.npz'
    if not f.exists():
        return []
    o = load_compact(f)
    ref = load_ref(bag)
    run = apply_scenario(load_run(NPZ_DIR / f'{bag}.npz', name=bag), SCENARIOS[scn])
    t_o = o['stamp_ns'] * 1e-9
    keep = np.isfinite(ref['v'])
    tr, vr = ref['t'][keep], ref['v'][keep]
    rows = []
    for ev in run.events:
        if ev['type'] not in ('slip', 'slide'):
            continue
        for key, col, name in ((FRONT, 'slip_f', 'front'), (REAR, 'slip_r', 'rear')):
            if key not in ev['topics']:
                continue
            s = run.streams[key]
            sel = (s.t_hdr0 >= ev['t0']) & (s.t_hdr0 <= ev['t1']) & ((s.label & SLIP_LABELS) != 0) & \
                  np.isfinite(s.val[:, 0])
            if sel.sum() < 2:
                continue
            ts = s.t_hdr0[sel]
            v_true = np.interp(ts, tr, vr)
            true_ratio = (s.val[sel, 0] - s.clean[sel, 0]) / 3.6 / np.maximum(v_true, 1.0)
            pk, ok = match_nearest(ts, t_o)
            est = np.where(ok, o[col][pk], np.nan)
            err = est - true_ratio
            fin = np.isfinite(err)
            if fin.sum() < 2:
                continue
            rows.append({'scenario': scn, 'bag': bag, 'type': ev['type'], 'bogie': name,
                         'model': ev['params'].get('model'), 't0_rel': round(ev['t0'] - ref['t_first_veh'], 2),
                         'dur_s': round(ev['t1'] - ev['t0'], 2), 'n': int(fin.sum()),
                         'true_peak': float(true_ratio[np.argmax(np.abs(true_ratio))]),
                         'est_peak': float(est[fin][np.argmax(np.abs(est[fin]))]),
                         'true_mean': float(np.mean(true_ratio[fin])), 'est_mean': float(np.mean(est[fin])),
                         'rmse': _rms(err[fin]), 'bias': float(np.mean(err[fin])),
                         'corr': float(np.corrcoef(est[fin], true_ratio[fin])[0, 1]) if np.std(est[fin]) > 0 and
                         np.std(true_ratio[fin]) > 0 else float('nan')})
    return rows


def slip_accuracy(variant: str, workers: int, scns: list[str], bags: list[str]) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs = [(b, s) for s in scns for b in bags]
        res = list(ex.map(task_slip_accuracy, [variant] * len(jobs), [j[0] for j in jobs], [j[1] for j in jobs]))
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(RESULTS / variant / 'slip_estimate.csv', index=False)
    return df


def rescore(variant: str, workers: int):
    """Recompute every metric of a variant from its saved outputs (needs a run with --keep-out)."""
    p = _paths(variant)
    old_c = pd.read_csv(p['dir'] / 'clean_runs.csv')
    old_r = pd.read_csv(p['dir'] / 'runs.csv', low_memory=False)
    keep_c = ['bag', 'rc', 'stderr', 'wall_s']
    keep_r = ['scenario', 'bag', 'rc', 'stderr', 'wall_s', 'proc_us_p99', 'proc_us_max']
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(task_rescore_clean, [variant] * len(old_c), list(old_c.bag)))
        cr = pd.DataFrame([r['row'] for r in res])
        cr = cr.merge(old_c[[c for c in keep_c if c in old_c]], on='bag', how='left').sort_values('bag')
        cr.to_csv(p['dir'] / 'clean_runs.csv', index=False)
        eps = [e for r in res for e in r['episodes']]
        pd.DataFrame(eps, columns=['bag', 'family', 't0_rel', 'duration_s', 'v_est', 'a_model', 'natural',
                                   'input_gap_s', 'startup']).to_csv(p['dir'] / 'false_alarm_episodes.csv', index=False)
        false_alarm_summary(variant)
        keys = list(zip(old_r.scenario, old_r.bag))
        res = list(ex.map(task_rescore, [variant] * len(keys), [k[1] for k in keys], [k[0] for k in keys]))
    rows, wins = [], []
    for (scn, bag), r in zip(keys, res):
        if r['row'] is None:   # crashed / no output saved: keep the original row
            rows.append(old_r[(old_r.scenario == scn) & (old_r.bag == bag)].iloc[0].to_dict())
            continue
        rows.append(r['row'])
        wins += r['windows']
    new_r = pd.DataFrame(rows)
    drop = [c for c in keep_r[2:] if c in new_r.columns]
    new_r = new_r.drop(columns=drop).merge(old_r[[c for c in keep_r if c in old_r]], on=['scenario', 'bag'], how='left')
    new_r = parse_counters(new_r)
    new_r.to_csv(p['dir'] / 'runs.csv', index=False)
    pd.DataFrame(wins).to_csv(p['dir'] / 'windows.csv', index=False)
    return summarize(variant)


# ------------------------------------------------------------------------------------------ aggregation
def _merge_write(path: Path, new: pd.DataFrame, keys: list[str]):
    if path.exists() and len(new):
        old = pd.read_csv(path, low_memory=False)
        if all(k in old.columns for k in keys):
            idx = pd.MultiIndex.from_frame(new[keys].drop_duplicates())
            keep = ~pd.MultiIndex.from_frame(old[keys]).isin(idx)
            new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def _q(x, q):
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna()
    return float(np.percentile(x, q)) if len(x) else float('nan')


def summarize(variant: str) -> pd.DataFrame:
    vd = RESULTS / variant
    runs = pd.read_csv(vd / 'runs.csv', low_memory=False) if (vd / 'runs.csv').exists() else pd.DataFrame()
    wins = pd.read_csv(vd / 'windows.csv', low_memory=False) if (vd / 'windows.csv').exists() else pd.DataFrame()
    rows = []
    for scn in sorted(set(runs.get('scenario', pd.Series(dtype=str)).dropna())):
        r = runs[runs.scenario == scn]
        w = wins[wins.scenario == scn] if len(wins) else wins
        req = w[w.det_required.astype(bool)] if len(w) else w
        iso = w[~w.overlap.astype(bool)] if len(w) else w
        rr = {'scenario': scn, 'n_runs': len(r), 'n_crash': int(r.crash.astype(bool).sum()),
              'n_nonfinite_runs': int((r.get('n_nonfinite', 0) > 0).sum()),
              'n_absurd_runs': int((r.get('n_absurd_v', 0) > 0).sum()),
              'n_diverged': int(r.get('diverged', pd.Series(False)).astype(bool).sum()),
              'v_rmse_med': _q(r.get('v_rmse'), 50), 'clean_v_rmse_med': _q(r.get('clean_v_rmse'), 50),
              'v_rmse_max': _q(r.get('v_rmse'), 100), 'v_max_max': _q(r.get('v_max'), 100),
              'dv_vs_clean_max': _q(r.get('dv_vs_clean_max'), 100),
              'along_rmse_med': _q(r.get('along_rmse'), 50), 'clean_along_rmse_med': _q(r.get('clean_along_rmse'), 50),
              'ds_vs_clean_max_med': _q(r.get('ds_vs_clean_max'), 50), 'ds_vs_clean_max_max': _q(r.get('ds_vs_clean_max'), 100),
              'dp_end_med': _q(r.get('dp_vs_clean_end'), 50), 'dp_end_max': _q(r.get('dp_vs_clean_end'), 100),
              'end_err_med': _q(r.get('end_err'), 50), 'clean_end_err_med': _q(r.get('clean_end_err'), 50),
              'model_only_frac_med': _q(r.get('model_only_frac'), 50), 'rate_hz_min': _q(r.get('rate_hz'), 0),
              'backward_stamps': int(pd.to_numeric(r.get('n_backward_stamps'), errors='coerce').fillna(0).sum()),
              'max_stamp_gap_max': _q(r.get('max_stamp_gap_s'), 100),
              'n_windows': len(w), 'n_windows_isolated': len(iso),
              'n_glitch_0p3_sum': int(pd.to_numeric(r.get('n_glitch_0p3'), errors='coerce').fillna(0).sum())
              if 'n_glitch_0p3' in r else np.nan,
              'glitch_max_max': _q(r.get('glitch_max'), 100) if 'glitch_max' in r else np.nan,
              'rejected_stamps_max': _q(r.get('rejected_stamps'), 100) if 'rejected_stamps' in r else np.nan,
              'proc_us_p99_max': _q(r.get('proc_us_p99'), 100) if 'proc_us_p99' in r else np.nan}
        if len(w):
            rr.update(err_in_rms_med=_q(w.err_in_rms, 50), err_in_rms_p90=_q(w.err_in_rms, 90),
                      err_in_max_med=_q(w.err_in_max, 50), err_in_max_p90=_q(w.err_in_max, 90),
                      err_in_max_max=_q(w.err_in_max, 100), clean_err_in_max_med=_q(w.clean_err_in_max, 50),
                      dv_in_max_p90=_q(w.dv_in_max, 90), dv_in_max_max=_q(w.dv_in_max, 100),
                      err_after_rms_med=_q(w.err_after_rms, 50), err_after_max_p90=_q(w.err_after_max, 90),
                      err_after_max_max=_q(w.err_after_max, 100), clean_err_after_max_p90=_q(w.clean_err_after_max, 90),
                      rec_gnss_med=_q(w.rec_gnss_s, 50), rec_gnss_p90=_q(w.rec_gnss_s, 90),
                      rec_gnss_max=_q(w.rec_gnss_s, 100), rec_clean_med=_q(w.rec_clean_s, 50),
                      rec_clean_p90=_q(w.rec_clean_s, 90), rec_flags_med=_q(w.rec_flags_s, 50),
                      rec_flags_p90=_q(w.rec_flags_s, 90),
                      rec_speed_med=_q(w.get('rec_speed_s'), 50), rec_speed_p90=_q(w.get('rec_speed_s'), 90),
                      not_recovered=int((~w.recovered.astype(bool)).sum()),
                      ds_growth_abs_med=_q(w.ds_growth.abs(), 50), ds_growth_abs_p90=_q(w.ds_growth.abs(), 90),
                      ds_growth_abs_max=_q(w.ds_growth.abs(), 100),
                      model_only_in_med=_q(w.model_only_in, 50),
                      wrong_family_rate=float((w.wrong_families.fillna('') != '').mean()))
            if len(iso):
                rr.update(iso_err_in_max_p90=_q(iso.err_in_max, 90), iso_rec_gnss_med=_q(iso.rec_gnss_s, 50),
                          iso_rec_gnss_p90=_q(iso.rec_gnss_s, 90))
            if len(req):
                rr.update(n_det_required=len(req), det_rate=float(req.det_fired.astype(bool).mean()),
                          det_delay_med=_q(req.det_delay_s, 50), det_delay_p90=_q(req.det_delay_s, 90),
                          exp_family=';'.join(sorted(set(req.exp_family))))
            det_any = w[w.exp_family != 'none']
            if len(det_any):
                rr['det_rate_all'] = float(det_any.det_fired.astype(bool).mean())
        rows.append(rr)
    df = pd.DataFrame(rows)
    df.to_csv(vd / 'scenario_summary.csv', index=False)
    return df


def false_alarm_summary(variant: str) -> pd.DataFrame:
    vd = RESULTS / variant
    cr = pd.read_csv(vd / 'clean_runs.csv')
    ep = pd.read_csv(vd / 'false_alarm_episodes.csv') if (vd / 'false_alarm_episodes.csv').exists() and \
        (vd / 'false_alarm_episodes.csv').stat().st_size > 5 else pd.DataFrame(columns=['bag', 'family'])
    hours = float(cr.hours.sum())
    hours_mov = float(cr.hours_moving.sum())
    rows = []
    for fam in FA_FAMILIES:
        e = ep[ep.family == fam] if len(ep) else ep
        nat = e['natural'].fillna('').astype(str) != '' if len(e) else pd.Series(dtype=bool)
        true_gap = (pd.to_numeric(e['input_gap_s'], errors='coerce') > 0.6) if len(e) else pd.Series(dtype=bool)
        start = e['startup'].astype(bool) if len(e) and 'startup' in e else pd.Series(False, index=e.index)
        e_run = e[~start] if len(e) else e             # start-up transients excluded from the rates
        nat, true_gap = (nat[~start], true_gap[~start]) if len(e) else (nat, true_gap)
        unexplained = e_run[~(nat | true_gap)] if len(e_run) else e_run
        rows.append({'family': fam, 'startup_episodes': int(start.sum()) if len(e) else 0,
                     'episodes': len(e_run), 'per_hour': len(e_run) / hours if hours else float('nan'),
                     'per_moving_hour': len(e_run) / hours_mov if hours_mov else float('nan'),
                     'near_natural_anomaly': int(nat.sum()) if len(e_run) else 0,
                     'real_input_gap': int(true_gap.sum()) if len(e_run) else 0,
                     'unexplained': len(unexplained),
                     'unexplained_per_hour': len(unexplained) / hours if hours else float('nan'),
                     'flagged_s': float(e_run.duration_s.sum()) if len(e_run) else 0.0,
                     'bags_with_episodes': int(e_run.bag.nunique()) if len(e_run) else 0,
                     'hours': hours, 'hours_moving': hours_mov, 'n_bags': len(cr)})
    df = pd.DataFrame(rows)
    df.to_csv(vd / 'false_alarms.csv', index=False)
    return df


def compare_variants(variants: list[str], base: str = 'base') -> pd.DataFrame:
    """One long table: per variant, clean accuracy / false alarms (scenario '_clean') and per-scenario
    robustness metrics, restricted to scenarios the variant ran, with deltas vs the base variant."""
    rows = []
    metrics = ['v_rmse_med', 'dv_vs_clean_max', 'err_in_rms_med', 'err_in_max_p90', 'err_in_max_max',
               'err_after_max_p90', 'rec_gnss_med', 'rec_gnss_p90', 'rec_clean_p90', 'not_recovered', 'det_rate',
               'det_delay_med', 'wrong_family_rate', 'ds_growth_abs_p90', 'ds_growth_abs_max', 'dp_end_max',
               'n_diverged', 'model_only_frac_med']
    for v in variants:
        vd = RESULTS / v
        if not (vd / 'clean_runs.csv').exists():
            continue
        meta = json.loads((vd / 'meta.json').read_text()) if (vd / 'meta.json').exists() else {}
        cr = pd.read_csv(vd / 'clean_runs.csv')
        good = cr[cr.ref_good.astype(bool)]
        fa = pd.read_csv(vd / 'false_alarms.csv') if (vd / 'false_alarms.csv').exists() else pd.DataFrame()
        r = {'variant': v, 'sets': json.dumps(meta.get('sets', {})), 'scenario': '_clean',
             'clean_v_rmse_mean': good.v_rmse.mean(), 'clean_v_rmse_max': good.v_rmse.max(),
             'clean_v_p99_mean': good.v_p99.mean(), 'clean_along_rmse_mean': good.along_rmse.mean(),
             'clean_along_rmse_med': good.along_rmse.median(), 'clean_end_err_mean': good.end_err.mean(),
             'clean_p3_rmse_mean': good.p3_rmse.mean(), 'clean_model_only_frac': cr.model_only_frac.mean(),
             'n_good_bags': len(good)}
        for _, f in fa.iterrows():
            r[f'fa_{f.family}_per_h'] = f.per_hour
            r[f'fa_{f.family}_unexpl_per_h'] = f.unexplained_per_hour
        rows.append(r)
        if (vd / 'scenario_summary.csv').exists():
            s = pd.read_csv(vd / 'scenario_summary.csv')
            for _, x in s.iterrows():
                rows.append({'variant': v, 'sets': json.dumps(meta.get('sets', {})), 'scenario': x.scenario,
                             **{m: x.get(m) for m in metrics}})
    df = pd.DataFrame(rows)
    if len(df) and base in set(df.variant):
        b = df[df.variant == base].set_index('scenario')
        for m in metrics + ['clean_v_rmse_mean', 'clean_along_rmse_mean', 'clean_end_err_mean']:
            if m in df:
                df[f'd_{m}'] = [row[m] - b.loc[row['scenario'], m] if row['scenario'] in b.index and m in b and
                                pd.notna(row[m]) else np.nan for _, row in df.iterrows()]
    df.to_csv(RESULTS / 'variants_compare.csv', index=False)
    return df


FINAL_METRICS = ['n_windows', 'err_in_rms_med', 'err_in_max_p90', 'err_in_max_max', 'err_after_rms_med',
                 'err_after_max_p90', 'rec_gnss_med', 'rec_gnss_p90', 'rec_speed_med', 'rec_speed_p90',
                 'not_recovered', 'det_rate', 'det_delay_med',
                 'wrong_family_rate', 'ds_growth_abs_p90', 'ds_growth_abs_max', 'dv_vs_clean_max', 'n_diverged',
                 'n_crash', 'n_nonfinite_runs', 'n_absurd_runs', 'v_rmse_med', 'dp_end_max', 'n_glitch_0p3_sum',
                 'rejected_stamps_max', 'rate_hz_min', 'backward_stamps', 'proc_us_p99_max']


def final_tables(variants: list[str]):
    """Side-by-side per-scenario table, false alarms and clean accuracy for the given variants
    (a scenario a variant did not run is taken from 'base' only if the variant's parameters cannot
    affect it - never: missing cells stay empty)."""
    rows = {}
    for v in variants:
        s = pd.read_csv(RESULTS / v / 'scenario_summary.csv')
        for _, x in s.iterrows():
            r = rows.setdefault(x.scenario, {'scenario': x.scenario,
                                             'description': (SCENARIOS[x.scenario].description or '').strip()
                                             if x.scenario in SCENARIOS else ''})
            for m in FINAL_METRICS:
                r[f'{v}:{m}'] = x.get(m)
    df = pd.DataFrame(list(rows.values())).sort_values('scenario')
    df.to_csv(RESULTS / 'final_scenario_table.csv', index=False)
    fa_rows, acc_rows = [], []
    for v in variants:
        fa = pd.read_csv(RESULTS / v / 'false_alarms.csv')
        fa.insert(0, 'variant', v)
        fa_rows.append(fa)
        cr = pd.read_csv(RESULTS / v / 'clean_runs.csv')
        cr.insert(0, 'variant', v)
        acc_rows.append(cr[['variant', 'bag', 'ref_good', 'v_rmse', 'v_mae', 'v_p99', 'v_max', 'p3_rmse', 'along_rmse',
                            'along_max', 'cross_rmse', 'end_err', 'drift_pct', 'model_only_frac', 'rate_hz',
                            'n_backward_stamps']])
    pd.concat(fa_rows).to_csv(RESULTS / 'final_false_alarms.csv', index=False)
    pd.concat(acc_rows).to_csv(RESULTS / 'final_clean_accuracy.csv', index=False)
    return df


# ------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', default='base')
    ap.add_argument('--set', action='append', default=[], help='estimator parameter override key=value')
    ap.add_argument('--bags', nargs='+', default=['good'], help="anomaly-suite bags ('good' = 7 good-reference val bags)")
    ap.add_argument('--fa-bags', nargs='+', default=['val'], help='clean runs for false alarms (default: all val)')
    ap.add_argument('--scenarios', nargs='+', default=['all'], help='scenario names or prefixes (S01, X0, ...)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--keep-out', action='store_true', help='keep compact outputs of every run (npz)')
    ap.add_argument('--skip-clean', action='store_true', help='reuse existing clean runs of this variant')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--rescore', action='store_true', help='recompute metrics from saved outputs (--keep-out run)')
    ap.add_argument('--slip-accuracy', action='store_true',
                    help='published slip ratio vs injected slip (saved outputs of slip/slide scenarios)')
    ap.add_argument('--compare', nargs='+', help='write results_cpp/variants_compare.csv for these variants')
    ap.add_argument('--final', nargs='+', help='write results_cpp/final_*.csv for these variants')
    ap.add_argument('--exe', help='replay binary to use instead of bin/tbo_replay_snapshot.exe '
                                  '(e.g. bin/tbo_replay_patched.exe with the fix_* prototypes)')
    args = ap.parse_args(argv)
    if args.exe:
        global SNAP_EXE
        SNAP_EXE = Path(args.exe).resolve()
        os.environ['TBO_REPLAY_EXE'] = str(SNAP_EXE)
    if args.final:
        print(final_tables(args.final).shape)
        return
    if args.compare:
        df = compare_variants(args.compare)
        pd.set_option('display.width', 250)
        print(df[[c for c in ['variant', 'scenario', 'clean_v_rmse_mean', 'clean_along_rmse_mean', 'err_in_max_p90',
                              'rec_gnss_p90', 'det_rate', 'ds_growth_abs_p90', 'dv_vs_clean_max'] if c in df]]
              .round(3).to_string(index=False))
        return

    sets = {}
    for kv in args.set:
        k, v = kv.split('=', 1)
        sets[k] = v
    p = _paths(args.variant)
    p['dir'].mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(summarize(args.variant).to_string())
        print(false_alarm_summary(args.variant).to_string())
        return
    if args.rescore:
        s = rescore(args.variant, args.workers)
        print(s.round(3).to_string(index=False))
        return
    if args.slip_accuracy:
        scns = [s for s in select_scenarios(args.scenarios) if any(
            i.get('type') in ('slip', 'slide') for i in SCENARIOS[s].injectors)]
        df = slip_accuracy(args.variant, args.workers, scns, bag_list(args.bags))
        pd.set_option('display.width', 250)
        print(df.groupby(['scenario', 'type']).agg(n=('rmse', 'size'), rmse_med=('rmse', 'median'),
                                                   bias_med=('bias', 'median'), corr_med=('corr', 'median'),
                                                   true_peak_med=('true_peak', 'median'),
                                                   est_peak_med=('est_peak', 'median')).round(3).to_string())
        return
    bags = bag_list(args.bags)
    fa_bags = bag_list(args.fa_bags) if args.fa_bags != ['none'] else []
    scns = [s for s in select_scenarios(args.scenarios) if s != 'S00_baseline_gnss_cut']
    exe_hash = hashlib.sha256(SNAP_EXE.read_bytes()).hexdigest()
    meta = {'variant': args.variant, 'sets': sets, 'bags': bags, 'fa_bags': fa_bags, 'scenarios': scns,
            'exe': str(SNAP_EXE), 'exe_sha256': exe_hash, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
            'rec_tol': REC_TOL, 'rec_hold': REC_HOLD, 'after_s': AFTER_S, 'tail_s': TAIL_S,
            'det_pre': DET_PRE, 'det_grace': DET_GRACE}
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # ---- phase 1: clean-input runs (reference for "vs clean" metrics + false alarms) ----
        clean_bags = [b for b in dict.fromkeys(bags + fa_bags)]
        if not args.skip_clean:
            futs = {ex.submit(task_clean, args.variant, sets, b, args.keep_out): b for b in clean_bags}
            rows, eps = [], []
            for fu in as_completed(futs):
                r = fu.result()
                rows.append(r['row'])
                eps += r['episodes']
                rr = r['row']
                print(f"[clean] {rr['bag']} crash={rr.get('crash')} v_rmse={rr.get('v_rmse', float('nan')):.4f} "
                      f"along_rmse={rr.get('along_rmse', float('nan')):.3f} wall={rr.get('wall_s')}", flush=True)
            cr = pd.DataFrame(rows).sort_values('bag')
            cr.to_csv(p['dir'] / 'clean_runs.csv', index=False)
            ep = pd.DataFrame(eps, columns=['bag', 'family', 't0_rel', 'duration_s', 'v_est', 'a_model', 'natural',
                                            'input_gap_s', 'startup'])
            ep.to_csv(p['dir'] / 'false_alarm_episodes.csv', index=False)
            fa = false_alarm_summary(args.variant)
            print(fa[['family', 'episodes', 'per_hour', 'unexplained_per_hour']].to_string(index=False), flush=True)
        # ---- phase 2: scenario runs ----
        tasks = [(b, s) for s in scns for b in bags]
        futs = {ex.submit(task_scenario, args.variant, sets, b, s, args.keep_out): (b, s) for b, s in tasks}
        rows, wins = [], []
        done = 0
        for fu in as_completed(futs):
            r = fu.result()
            rows.append(r['row'])
            wins += r['windows']
            done += 1
            rr = r['row']
            if done % 10 == 0 or rr.get('crash'):
                print(f"[{done}/{len(tasks)}] {rr['scenario']} {rr['bag']} crash={rr.get('crash')} "
                      f"v_rmse={rr.get('v_rmse', float('nan')):.3f} dv_max={rr.get('dv_vs_clean_max', float('nan')):.2f} "
                      f"({time.time() - t_all:.0f}s)", flush=True)
    if rows:
        _merge_write(p['dir'] / 'runs.csv', parse_counters(pd.DataFrame(rows)), ['scenario', 'bag'])
    if wins:
        wdf = pd.DataFrame(wins)
        done_keys = pd.DataFrame(rows)[['scenario', 'bag']]
        if (p['dir'] / 'windows.csv').exists():
            old = pd.read_csv(p['dir'] / 'windows.csv', low_memory=False)
            keep = ~pd.MultiIndex.from_frame(old[['scenario', 'bag']]).isin(pd.MultiIndex.from_frame(done_keys))
            wdf = pd.concat([old[keep], wdf], ignore_index=True)
        wdf.to_csv(p['dir'] / 'windows.csv', index=False)
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    meta['elapsed_s'] = round(time.time() - t_all, 1)
    if (p['dir'] / 'meta.json').exists():   # partial re-run: keep the union of scenarios / runs
        try:
            old = json.loads((p['dir'] / 'meta.json').read_text(encoding='utf-8'))
            if old.get('sets') == meta['sets']:
                meta['scenarios'] = sorted(set(old.get('scenarios', [])) | set(meta['scenarios']))
                meta['runs'] = old.get('runs', []) + [{k: meta[k] for k in ('started', 'finished', 'elapsed_s')}]
        except (json.JSONDecodeError, OSError):
            pass
    (p['dir'] / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    shutil.rmtree(p['tmp'], ignore_errors=True)
    if (p['dir'] / 'runs.csv').exists():
        s = summarize(args.variant)
        cols = [c for c in ['scenario', 'n_crash', 'n_diverged', 'v_rmse_med', 'err_in_max_p90', 'err_after_max_p90',
                            'rec_gnss_med', 'rec_gnss_p90', 'det_rate', 'det_delay_med', 'ds_growth_abs_p90'] if c in s]
        pd.set_option('display.width', 250)
        print(s[cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
