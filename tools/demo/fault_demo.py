#!/usr/bin/env python3
"""Live fault-injection demo of the tram backup odometry (C++ estimator, offline replay).

The frozen replay binary is run twice on one bag: on the clean inputs and on inputs corrupted by
``tools/harness/faults.py`` (the GNSS reference is never touched). For every fault window the tool
writes a multi-panel PNG (faulted bogie speeds, GNSS truth, estimate, naive mean-of-bogies baseline,
speed error, along-track error accumulated since the fault onset, IMM mode probabilities, health
flags, driver notch), an animated GIF of the most interesting window and a metrics table
(speed RMSE and along-track error inside / after the window, estimator vs naive).

Usage (from the repository root):
    python tools/demo/fault_demo.py --bag 30618_2f104a1d \
        --faults "slip:wheels@170+4:peak=0.3:phase=1,dropout:wheels@690+5" --out tools/demo/out/try
    python tools/demo/fault_demo.py --bag 30618_2f104a1d --suggest      # candidate traction/braking windows

Fault syntax (tools/harness/faults.py): ``kind:topic@t0+dur[:key=val...]``, faults separated by ','.
``t0`` is seconds from the bag start or a percentage (``40%``); ``phase=1`` / ``phase=-1`` moves the
window start to the first traction / braking moment at or after t0 (and limits slip / slide to it).

Conventions: time axis = seconds from the bag start on the bag receive clock (the clock the fault
windows are defined on); estimator outputs are placed at the receive time of the input that
triggered their publication. Speed / position errors use the harness definitions
(``harness.cpp_estimator.per_sample_errors``: nearest header stamp within 50 ms, reference =
GNSS master Doppler |v| and master fixes projected on the reference path of the same bag).
The estimator publishes antenna-1 positions in ENU from the first master fix
(``output_frame=enu, base_link_along_m=0, base_link_height_m=0``) so that it is comparable
with the master-antenna reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tools'))

from harness.loader import BagData, load_bag, T_BAG, V_COL, VX, VY, VZ, LAT, LON, ALT, STATUS  # noqa: E402
from harness.reference import RefConfig, Reference, build_reference, flag_fix_outliers, select_origin  # noqa: E402
from harness.replay import EvalConfig  # noqa: E402
from harness.faults import apply_faults  # noqa: E402
from harness.cpp_estimator import export_harness_events, per_sample_errors  # noqa: E402
from harness import metrics as M  # noqa: E402

PKG = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry'
# frozen snapshot of the estimator used for the demo (Linux / jury: build it, see docs/DEMO.md)
DEFAULT_EXE = ROOT / 'build_core' / ('tbo_replay_fix5.exe' if os.name == 'nt' else 'tbo_replay')
# TRAIN-only map, branches, stop landmarks, traction cut-offs and disturbance field: val bags are not in them
DEFAULT_MAPS = ROOT / 'analysis' / 'validation_maps'
DEFAULT_TRACTION = PKG / 'config' / 'traction_lut.csv'
DEFAULT_OUT = ROOT / 'tools' / 'demo' / 'out'
MAX_PARALLEL_REPLAYS = 2
KMH = 3.6

# ----------------------------------------------------------------------------------------------
# Estimator health flags (core/include/tbo/types.hpp) and IMM modes
# ----------------------------------------------------------------------------------------------
FLAG_LABELS = {
    0: 'буксование П', 1: 'буксование З', 2: 'юз П', 3: 'юз З',
    4: 'пропуск П', 5: 'пропуск З', 6: 'пропуск контроллера',
    7: 'залипание П', 8: 'залипание З', 9: 'недостоверно П', 10: 'недостоверно З',
    11: 'только модель', 12: 'стоянка (ZUPT)', 13: 'не инициализировано', 14: 'нет карты',
    15: 'перепривязка к колёсам', 16: 'опоздавшие данные', 17: 'необъясн. ускорение',
    18: 'контроллер не согласован', 19: 'виртуальная балиса',
}
FLAG_MODEL_ONLY = 1 << 11
# bits that mean "the estimator noticed something wrong with its inputs"
INDICATOR_MASK = sum(1 << b for b in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18))
# rows always shown in the flag raster (others only when active)
ALWAYS_ROWS = (11,)
HIDDEN_ROWS = (13, 14)          # init / map flags: irrelevant mid-bag unless active
MODE_LABELS = ['норма', 'сбой передней', 'сбой задней', 'обе врут: только модель', 'манёвр']

# ----------------------------------------------------------------------------------------------
# Style: paper and ink + one accent (estimator); categorical slots validated for CVD separation
# ----------------------------------------------------------------------------------------------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
C_EST = '#2a78d6'          # estimator on faulted inputs (accent)
C_EST_CLEAN = '#b7d3f6'    # estimator on clean inputs (lighter step of the same ramp, drawn as a halo)
C_NAIVE = '#eb6834'        # naive mean of the two bogies
C_FAULT = '#d03b3b'        # fault window wash (status: critical)
MODE_COLORS = ['#e1e0d9', '#1baf7a', '#eda100', '#e34948', '#4a3aa7']
FLAG_COLOR = INK2
FLAG_COLOR_MODEL_ONLY = '#e34948'
FLAG_COLOR_CONTEXT = '#b3b1aa'


def _setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = [f for f in ('Segoe UI', 'Arial', 'Liberation Sans', 'DejaVu Sans') if f in names] or ['DejaVu Sans']
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': sans + ['DejaVu Sans'], 'font.size': 9.5,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'axes.titlesize': 10, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left', 'axes.titlepad': 4,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'legend.frameon': False, 'legend.fontsize': 8.5, 'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round', 'axes.unicode_minus': True,
    })
    return plt


# ----------------------------------------------------------------------------------------------
# Replay of the C++ estimator
# ----------------------------------------------------------------------------------------------
def file_md5(path: Path) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


@dataclass
class ReplaySetup:
    """Binary + map / table inputs + parameter overrides of one replay configuration."""
    exe: Path = DEFAULT_EXE
    maps: Path = DEFAULT_MAPS
    traction: Path = DEFAULT_TRACTION
    gnss_seconds: float = 5.0
    sets: Dict[str, str] = field(default_factory=dict)
    use_dfield: bool = True              # maps/dfield.csv (learned disturbance field), as in config/params.yaml

    def inputs(self) -> Dict[str, Path]:
        d = {'exe': Path(self.exe), 'track_map': Path(self.maps) / 'track_map.csv', 'traction': Path(self.traction)}
        for p in sorted(Path(self.maps).glob('branch_*.csv')):
            d[p.stem] = p
        for name in ('landmarks', 'cutoffs') + (('dfield',) if self.use_dfield else ()):
            p = Path(self.maps) / f'{name}.csv'
            if p.exists():
                d[name] = p
        return d

    def binary_sets(self) -> Dict[str, str]:
        # antenna-1 point in ENU from the first master fix = the master-antenna reference frame
        s = {'output_frame': 'enu', 'base_link_along_m': '0', 'base_link_height_m': '0',
             'gnss_init_window_s': f'{self.gnss_seconds:g}'}
        if 'dfield' in self.inputs():
            s['dfield_file'] = str(self.inputs()['dfield'])
        s.update({k: str(v) for k, v in self.sets.items()})
        return s

    def command(self, ev_csv: Path, out_csv: Path) -> List[str]:
        inp = self.inputs()
        cmd = [str(self.exe), '--in', str(ev_csv), '--out', str(out_csv),
               '--map', str(inp['track_map']), '--traction', str(inp['traction'])]
        branches = [str(p) for k, p in inp.items() if k.startswith('branch_')]
        if branches:
            cmd += ['--branches', ','.join(branches)]
        if 'landmarks' in inp:
            cmd += ['--landmarks', str(inp['landmarks'])]
        if 'cutoffs' in inp:
            cmd += ['--cutoffs', str(inp['cutoffs'])]
        for k, v in self.binary_sets().items():
            cmd += ['--set', f'{k}={v}']
        return cmd

    def provenance(self) -> dict:
        return {k: {'path': _rel(p), 'md5': file_md5(p)} for k, p in self.inputs().items()}

    def key(self) -> str:
        """Hash of everything that changes the binary's output (for the replay cache)."""
        h = hashlib.md5(json.dumps({'inputs': {k: v['md5'] for k, v in self.provenance().items()},
                                    'sets': self.binary_sets()}, sort_keys=True).encode())
        return h.hexdigest()[:10]


def _rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


@dataclass
class Replay:
    df: pd.DataFrame
    stderr: str
    wall_s: float
    events: dict
    cached: bool = False


def run_replay(setup: ReplaySetup, bag: BagData, ev_csv: Path, out_csv: Path, reuse: bool = False) -> Replay:
    """Export ``bag`` (possibly fault-injected) as the harness event stream and run the binary on it."""
    meta = out_csv.with_suffix('.json')
    if reuse and out_csv.exists() and meta.exists():
        info = json.loads(meta.read_text(encoding='utf-8'))
        return Replay(pd.read_csv(out_csv), info.get('stderr', ''), 0.0, info.get('events', {}), cached=True)
    cfg = EvalConfig(gnss_seconds=setup.gnss_seconds)
    info = export_harness_events(bag, cfg, ev_csv, gnss_vel=True, est_window_s=setup.gnss_seconds)
    info.pop('_fix_hdr_master', None)
    cmd = setup.command(ev_csv, out_csv)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f'replay failed (rc={res.returncode}): {res.stderr.strip()}\ncmd: {" ".join(cmd)}')
    df = pd.read_csv(out_csv)
    meta.write_text(json.dumps({'stderr': res.stderr.strip(), 'events': info, 'cmd': cmd}, default=str),
                    encoding='utf-8')
    return Replay(df, res.stderr.strip(), wall, info)


# ----------------------------------------------------------------------------------------------
# Analysis containers
# ----------------------------------------------------------------------------------------------
@dataclass
class Run:
    """Published estimator outputs of one replay."""
    t: np.ndarray        # bag clock of publication (receive time of the triggering input) [s]
    stamp: np.ndarray    # published header stamp [s]
    v: np.ndarray        # speed [m/s]
    mu: np.ndarray       # (N, 5) IMM mode probabilities
    flags: np.ndarray    # health flag bits
    log: M.OutputLog


def make_run(df: pd.DataFrame) -> Run:
    stamp = df['stamp_ns'].to_numpy(np.int64) * 1e-9
    recv = df['recv_ns'].to_numpy(np.int64) * 1e-9
    xyz = df[['x', 'y', 'z']].to_numpy(float).copy()
    if 'pos_valid' in df.columns:
        xyz[df['pos_valid'].to_numpy() == 0] = np.nan          # position not published yet
    v = df['v'].to_numpy(float)
    log = M.OutputLog(stamp=stamp, v=v, xyz=xyz, emit_tbag=recv, emit_proc=np.zeros(len(df)),
                      frame_id=['map'] * len(df))
    mu = df[[f'mu{i}' for i in range(5)]].to_numpy(float)
    return Run(t=recv, stamp=stamp, v=v, mu=mu, flags=df['flags'].to_numpy(np.int64), log=log)


@dataclass
class Demo:
    bag: BagData          # clean inputs
    fbag: BagData         # fault-injected inputs
    ref: Reference
    clean: Run
    fault: Run
    pse_c: dict           # per-sample errors of the clean run
    pse_f: dict           # per-sample errors of the faulted run (naive = mean of the FAULTED bogies)
    windows: List[dict]   # applied faults with t_start / t_end on the bag clock

    @property
    def t0(self) -> float:
        return self.bag.t_start


def hold_mean_speed(bag: BagData, tq: np.ndarray) -> np.ndarray:
    """Naive odometry [m/s] on the bag clock: mean of the latest front and rear readings (NaN ignored,
    stale values held during dropouts) - what a node without any filtering would publish."""
    vals = []
    for name in ('front', 'rear'):
        w = bag[name]
        if len(w) == 0:
            vals.append(np.full(len(tq), np.nan))
            continue
        vals.append(M.sample_hold(w[:, T_BAG], w[:, V_COL] / KMH, tq))
    st = np.stack(vals)
    n = np.sum(np.isfinite(st), axis=0)
    return np.where(n > 0, np.nansum(st, axis=0) / np.maximum(n, 1), np.nan)


def _sorted_by(t: np.ndarray, *arrs):
    o = np.argsort(t, kind='stable')
    return (t[o],) + tuple(a[o] for a in arrs)


def along_increments(demo: Demo, a: float, t_end: float, dt: float = 0.01) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Along-track error accumulated since the fault onset ``a`` (bag clock) at the reference fixes in
    [a, t_end]. Estimator (faulted / clean): published positions projected on the reference path,
    minus their value at ``a``. Naive: s_ref(a) + integral of the mean bogie speed - s_ref(t)."""
    out = {}
    for name, pse in (('est', demo.pse_f), ('clean', demo.pse_c)):
        tp, al = pse['tp'], pse['along']
        ok = np.isfinite(al)
        tp, al = tp[ok], al[ok]
        if len(tp) < 2 or tp[0] > a:
            out[name] = (np.array([a]), np.array([np.nan]))
            continue
        a0 = float(np.interp(a, tp, al))
        m = (tp > a) & (tp <= t_end)
        out[name] = (np.r_[a, tp[m]], np.r_[0.0, al[m] - a0])
    pm = demo.ref.pos_mask()
    tb, s = _sorted_by(demo.ref.tbag_pos[pm], demo.ref.s_pos[pm])
    grid = np.arange(a, t_end + dt, dt)
    vn = np.nan_to_num(hold_mean_speed(demo.fbag, grid), nan=0.0)
    dist = np.r_[0.0, np.cumsum(vn[:-1] * dt)]
    m = (tb > a) & (tb <= t_end)
    s_a = float(np.interp(a, tb, s))
    out['naive'] = (np.r_[a, tb[m]], np.r_[0.0, s_a + np.interp(tb[m], grid, dist) - s[m]])
    return out


def indicator(run: Run) -> np.ndarray:
    return ((run.flags & INDICATOR_MASK) != 0) | (run.mu[:, 0] < 0.5)


def flag_names(bits: int, mask: int = INDICATOR_MASK) -> List[str]:
    return [FLAG_LABELS[b] for b in sorted(FLAG_LABELS) if (bits & mask) >> b & 1]


def reaction(demo: Demo, a: float, t_end: float) -> Tuple[Optional[float], str]:
    """First moment in [a, t_end] when the faulted run raises an anomaly indicator (flag or
    P(nominal) < 0.5) that the clean run does not have at the same stamp."""
    f, c = demo.fault, demo.clean
    idx = np.flatnonzero((f.t >= a) & (f.t <= t_end))
    if len(idx) == 0 or len(c.stamp) == 0:
        return None, ''
    j = np.clip(np.searchsorted(c.stamp, f.stamp[idx]), 0, len(c.stamp) - 1)
    j0 = np.clip(j - 1, 0, len(c.stamp) - 1)
    j = np.where(np.abs(c.stamp[j0] - f.stamp[idx]) < np.abs(c.stamp[j] - f.stamp[idx]), j0, j)
    new = indicator(f)[idx] & ~indicator(c)[j]
    k = np.flatnonzero(new)
    if len(k) == 0:
        return None, ''
    i, jc = idx[k[0]], j[k[0]]
    names = flag_names(int(f.flags[i]) & ~int(c.flags[jc]))
    if not names:
        names = [f'режим «{MODE_LABELS[int(np.argmax(f.mu[i]))]}»']
    return float(f.t[i] - a), ', '.join(names)


def flagged_time(run: Run, lo: float, hi: float, bit_mask: int) -> float:
    idx = np.flatnonzero((run.t >= lo) & (run.t <= hi))
    if len(idx) == 0:
        return 0.0
    dt = np.clip(np.diff(np.r_[run.t[idx], hi]), 0.0, 0.25)
    return float(np.sum(dt[(run.flags[idx] & bit_mask) != 0]))


def _err_stats(pse: dict, key: str, lo: float, hi: float, left_open: bool = False) -> Tuple[float, float, int]:
    tv = pse['tv']
    m = ((tv > lo) if left_open else (tv >= lo)) & (tv <= hi)
    e = pse[key][m]
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return float('nan'), float('nan'), 0
    return float(np.sqrt(np.mean(e * e))), float(np.max(np.abs(e))), int(len(e))


def _at(series: Tuple[np.ndarray, np.ndarray], t: float) -> float:
    ts, ys = series
    if len(ts) < 2 or not np.isfinite(ys).any():
        return float('nan')
    return float(np.interp(t, ts, ys))


def _maxabs(series: Tuple[np.ndarray, np.ndarray]) -> float:
    y = series[1][np.isfinite(series[1])]
    return float(np.max(np.abs(y))) if len(y) else float('nan')


def spec_string(w: dict) -> str:
    t0 = w['t0']
    s = f"{w['kind']}:{w['topic']}@{t0 if isinstance(t0, str) else f'{t0:g}'}+{w['dur']:g}"
    extra = [f'{k}={v:g}' for k, v in w.items()
             if k not in ('kind', 'topic', 't0', 'dur', 't_start', 't_end') and isinstance(v, (int, float))]
    return s + (':' + ':'.join(extra) if extra else '')


def describe_fault(w: dict) -> str:
    """Short Russian description of a fault spec."""
    kind, topic = w['kind'], w['topic']
    who = {'front': 'передней тележки', 'rear': 'задней тележки', 'wheels': 'обеих тележек',
           'cmd': 'контроллера', 'inputs': 'всех входов (тележки + контроллер)'}.get(topic, topic)
    sensor = {'wheels': 'датчиков обеих тележек'}.get(topic, f'датчика {who}')
    ph = w.get('phase')
    if kind == 'slip':
        s = f"Буксование {who}{' под тягой' if ph and ph > 0 else ''}: показания до +{100 * w.get('peak', 0.4):.0f}%"
    elif kind == 'slide':
        s = f"Юз {who}{' при торможении' if ph and ph < 0 else ''}: показания до −{100 * w.get('peak', 0.6):.0f}%"
    elif kind == 'freeze':
        s = f'Залипание {sensor}: показание заморожено'
    elif kind == 'zero':
        s = f'Отказ {sensor}: показывает 0'
    elif kind == 'dropout':
        s = f'Пропуск всех сообщений {who}'
    elif kind == 'spike':
        s = f"Выбросы {who}: ±{w.get('amp', 20):g} км/ч в {100 * w.get('p', 0.05):g}% сообщений"
    elif kind == 'noise':
        s = f"Шум {who}, σ = {w.get('sigma', 1.0):g} км/ч"
    elif kind == 'nan':
        s = f'NaN вместо показаний {who}'
    elif kind == 'scale':
        s = f"Ошибка масштаба {sensor}: ×{w.get('factor', 1.2):g}"
    elif kind == 'delay':
        s = f"Задержка доставки {who} на {w.get('lag', 0.3):g} с"
    elif kind == 'stamp_jump':
        s = f"Скачок меток времени {who} на {w.get('offset', 1.0):+g} с"
    elif kind == 'stamp_zero':
        s = f'Нулевые метки времени {who}'
    elif kind == 'dup':
        s = f'Дублирование сообщений {who}'
    else:
        s = f'{kind} {who}'
    return f"{s}, {w['dur']:g} с"


def regime(demo: Demo, a: float, b: float) -> dict:
    """Driving context of the window from the clean inputs and the reference."""
    cmd = demo.bag['cmd']
    m = (cmd[:, T_BAG] >= a) & (cmd[:, T_BAG] <= b)
    notch = cmd[m, V_COL] if m.any() else np.array([np.nan])
    tv, v = _sorted_by(demo.ref.tbag_vel, demo.ref.v)
    mv = (tv >= a) & (tv <= b)
    va, vb = float(np.interp(a, tv, v)), float(np.interp(b, tv, v))
    frac_tr, frac_br = float(np.mean(notch > 0)), float(np.mean(notch < 0))
    name = 'тяга' if frac_tr >= 0.5 else ('торможение' if frac_br >= 0.5 else
                                         ('стоянка' if np.nanmax(v[mv], initial=0.0) < 0.3 else 'выбег'))
    return {'v_start_kmh': va * KMH, 'v_end_kmh': vb * KMH,
            'v_mean_kmh': float(np.mean(v[mv]) * KMH) if mv.any() else float('nan'),
            'accel_mps2': (vb - va) / max(b - a, 1e-6), 'notch_mean': float(np.nanmean(notch)),
            'frac_traction': frac_tr, 'frac_braking': frac_br, 'regime': name}


def window_metrics(demo: Demo, w: dict, after_s: float) -> dict:
    a, b = float(w['t_start']), float(w['t_end'])
    t_end = b + after_s
    r = {'spec': spec_string(w), 'kind': w['kind'], 'topic': w['topic'], 'description': describe_fault(w),
         't_start_s': a - demo.t0, 't_end_s': b - demo.t0, 'after_s': after_s}
    r.update(regime(demo, a, b))
    for tag, lo, hi, lopen in (('in', a, b, False), ('after', b, t_end, True)):
        for name, pse, key in (('est', demo.pse_f, 'ev'), ('naive', demo.pse_f, 'env'), ('clean', demo.pse_c, 'ev')):
            rmse, mx, n = _err_stats(pse, key, lo, hi, lopen)
            r[f'v_rmse_{tag}_{name}'] = rmse
            r[f'v_max_{tag}_{name}'] = mx
        r[f'n_speed_{tag}'] = n
    inc = along_increments(demo, a, t_end)
    for name, series in inc.items():
        r[f'along_end_{name}'] = _at(series, b)            # accumulated by the end of the window
        r[f'along_after_{name}'] = _at(series, t_end)      # ... and after_s later
        r[f'along_maxabs_{name}'] = _maxabs(series)
    rt, what = reaction(demo, a, t_end)
    r['reaction_s'] = rt
    r['reaction_flags'] = what
    r['model_only_s'] = flagged_time(demo.fault, a, t_end, FLAG_MODEL_ONLY)
    r['interest'] = (r['along_maxabs_naive'] - r['along_maxabs_est']) + \
        max(r['v_rmse_in_naive'] - r['v_rmse_in_est'], 0.0) * (b - a)
    return r


# ----------------------------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------------------------
def gap_break(t: np.ndarray, y: np.ndarray, max_gap: float) -> Tuple[np.ndarray, np.ndarray]:
    """Insert NaN where consecutive samples are further apart than ``max_gap`` (no line across gaps)."""
    if len(t) < 2:
        return t, y
    g = np.flatnonzero(np.diff(t) > max_gap)
    if len(g) == 0:
        return t, y
    return np.insert(t, g + 1, t[g] + 1e-6), np.insert(y.astype(float), g + 1, np.nan)


def flag_intervals(t: np.ndarray, flags: np.ndarray, bit: int, max_gap: float = 0.3) -> List[Tuple[float, float]]:
    on = ((flags >> bit) & 1).astype(bool)
    if not on.any():
        return []
    d = np.diff(np.r_[0, on.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = []
    for s, e in zip(starts, ends):
        t0 = t[s]
        t1 = t[e] if e < len(t) and t[e] - t[e - 1] <= max_gap else t[e - 1] + 0.05
        out.append((t0, max(t1 - t0, 0.05)))
    return out


def _slice(t: np.ndarray, lo: float, hi: float, *arrs, pad: float = 1.0):
    m = (t >= lo - pad) & (t <= hi + pad)
    return (t[m],) + tuple(a[m] for a in arrs)


@dataclass
class WindowData:
    """Everything drawn for one window, on the relative time axis (s from the bag start)."""
    x0: float
    x1: float
    a: float
    b: float
    t_after: float
    truth: Tuple[np.ndarray, np.ndarray]
    front: Tuple[np.ndarray, np.ndarray]
    rear: Tuple[np.ndarray, np.ndarray]
    naive: Tuple[np.ndarray, np.ndarray]
    est: Tuple[np.ndarray, np.ndarray]
    est_clean: Tuple[np.ndarray, np.ndarray]
    err_est: Tuple[np.ndarray, np.ndarray]
    err_naive: Tuple[np.ndarray, np.ndarray]
    err_clean: Tuple[np.ndarray, np.ndarray]
    along: Dict[str, Tuple[np.ndarray, np.ndarray]]
    mode_t: np.ndarray
    mode_mu: np.ndarray
    flags_t: np.ndarray
    flags: np.ndarray
    notch: Tuple[np.ndarray, np.ndarray]


def window_data(demo: Demo, w: dict, pre: float, post: float, after_s: float) -> WindowData:
    t0 = demo.t0
    a, b = float(w['t_start']), float(w['t_end'])
    lo, hi = a - pre, b + max(post, after_s + 4.0)
    tv, v = _sorted_by(demo.ref.tbag_vel, demo.ref.v)
    tv, v = _slice(tv, lo, hi, v)
    wheels = []
    for name in ('front', 'rear'):
        arr = demo.fbag[name]
        tt, vv = _slice(arr[:, T_BAG], lo, hi, arr[:, V_COL])
        wheels.append(gap_break(tt - t0, vv, 0.35))
    grid = np.arange(lo, hi, 0.02)
    naive = hold_mean_speed(demo.fbag, grid) * KMH
    ft, fv, fmu, ffl = _slice(demo.fault.t, lo, hi, demo.fault.v, demo.fault.mu, demo.fault.flags)
    ct, cv = _slice(demo.clean.t, lo, hi, demo.clean.v)
    errs = []
    for pse, key in ((demo.pse_f, 'ev'), (demo.pse_f, 'env'), (demo.pse_c, 'ev')):
        tt, ee = _sorted_by(pse['tv'], pse[key])
        tt, ee = _slice(tt, lo, hi, ee)
        errs.append(gap_break(tt - t0, ee * KMH, 0.5))
    inc = {k: (ts - t0, ys) for k, (ts, ys) in along_increments(demo, a, b + after_s).items()}
    cmd = demo.fbag['cmd']
    nt, nn = _slice(cmd[:, T_BAG], lo, hi, cmd[:, V_COL])
    return WindowData(x0=lo - t0, x1=hi - t0, a=a - t0, b=b - t0, t_after=b + after_s - t0,
                      truth=(tv - t0, v * KMH), front=wheels[0], rear=wheels[1],
                      naive=(grid - t0, naive), est=gap_break(ft - t0, fv * KMH, 0.5),
                      est_clean=gap_break(ct - t0, cv * KMH, 0.5),
                      err_est=errs[0], err_naive=errs[1], err_clean=errs[2], along=inc,
                      mode_t=ft - t0, mode_mu=fmu, flags_t=ft - t0, flags=ffl,
                      notch=gap_break(nt - t0, nn, 0.5))


def _fault_band(ax, wd: WindowData, label: bool = False):
    ax.axvspan(wd.a, wd.b, color=C_FAULT, alpha=0.08, lw=0, zorder=0)
    for x in (wd.a, wd.b):
        ax.axvline(x, color=C_FAULT, lw=0.7, alpha=0.55, zorder=0.5)
    if label:
        ax.text((wd.a + wd.b) / 2, 1.0, 'сбой', transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=8.5, color=INK2)


def _flag_rows(wd: WindowData) -> List[int]:
    rows = []
    for bit in sorted(FLAG_LABELS):
        active = bool(np.any((wd.flags >> bit) & 1))
        if bit in ALWAYS_ROWS or (active and bit not in HIDDEN_ROWS):
            rows.append(bit)
    return rows


def _draw_flags(ax, wd: WindowData, rows: List[int], t_max: Optional[float] = None):
    for i, bit in enumerate(rows):
        color = FLAG_COLOR_MODEL_ONLY if bit == 11 else (FLAG_COLOR_CONTEXT if bit in (12, 15, 19) else FLAG_COLOR)
        iv = flag_intervals(wd.flags_t, wd.flags, bit)
        if t_max is not None:
            iv = [(s, min(w, t_max - s)) for s, w in iv if s < t_max]
        if iv:
            ax.broken_barh(iv, (i - 0.32, 0.64), facecolors=color, lw=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([FLAG_LABELS[b] for b in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(False, axis='y')


def _draw_modes(ax, t: np.ndarray, mu: np.ndarray):
    if len(t) == 0:
        return []
    return ax.stackplot(t, mu.T, colors=MODE_COLORS, labels=MODE_LABELS, lw=0, step=None)


def _end_label(ax, series, color, text, dx=0.3):
    ts, ys = series
    ok = np.isfinite(ys)
    if not ok.any():
        return
    ax.plot(ts[ok][-1], ys[ok][-1], 'o', ms=4.5, color=color, mec=SURFACE, mew=1.2, zorder=5)
    ax.annotate(text, (ts[ok][-1], ys[ok][-1]), xytext=(6, 0), textcoords='offset points', va='center',
                fontsize=8.5, color=INK2)


def _legend_right(ax, **kw):
    kw.setdefault('handlelength', 2.4)
    return ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, **kw)


def plot_window(demo: Demo, w: dict, stats: dict, wd: WindowData, path: Path, dpi: int = 110):
    plt = _setup_matplotlib()
    from matplotlib.ticker import MaxNLocator
    rows = _flag_rows(wd)
    h_flags = 0.42 + 0.21 * len(rows)
    ratios = [3.3, 1.5, 1.5, 1.05, h_flags, 0.8]
    fig_h = 1.45 + sum(ratios) * 1.12
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(12.0, fig_h),
                             gridspec_kw={'height_ratios': ratios, 'hspace': 0.42})
    ax_v, ax_e, ax_s, ax_m, ax_f, ax_n = axes

    # 1: speeds
    ax_v.plot(*wd.front, color=MUTED, lw=0.9, label='тележка П', zorder=2)
    ax_v.plot(*wd.rear, color=MUTED, lw=0.9, ls=(0, (1.2, 1.6)), label='тележка З', zorder=2)
    ax_v.plot(*wd.truth, color=INK, lw=1.8, label='эталон GNSS', zorder=3)
    ax_v.plot(*wd.naive, color=C_NAIVE, lw=1.4, ls=(0, (4, 2)), label='наивная\n(среднее тележек)', zorder=3)
    ax_v.plot(*wd.est_clean, color=C_EST_CLEAN, lw=4.0, label='оценка без сбоя', zorder=3.5)
    ax_v.plot(*wd.est, color=C_EST, lw=1.7, label='оценка при сбое', zorder=4)
    ax_v.set_title('Скорость, км/ч (тележки — с инъекцией сбоя)')
    yv = np.r_[wd.truth[1], wd.est[1][np.isfinite(wd.est[1])]]
    yw = np.r_[wd.front[1][np.isfinite(wd.front[1])], wd.rear[1][np.isfinite(wd.rear[1])]]
    top = max(np.nanmax(yv, initial=0) * 1.08 + 3, (np.percentile(yw, 99.7) + 2) if len(yw) else 0)
    ax_v.set_ylim(min(-1.5, np.nanmin(yv, initial=0) - 1.5), top)
    _legend_right(ax_v)

    # 2: speed error
    ax_e.axhline(0, color=AXIS, lw=0.9, zorder=1)
    ax_e.plot(*wd.err_clean, color=C_EST_CLEAN, lw=3.6, label='оценка без сбоя', zorder=2)
    ax_e.plot(*wd.err_naive, color=C_NAIVE, lw=1.3, label='наивная', zorder=3)
    ax_e.plot(*wd.err_est, color=C_EST, lw=1.6, label='оценка при сбое', zorder=4)
    ax_e.set_title('Ошибка скорости относительно GNSS, км/ч')
    _legend_right(ax_e)

    # 3: along-track error accumulated since the fault onset
    ax_s.axhline(0, color=AXIS, lw=0.9, zorder=1)
    al = wd.along
    ax_s.plot(*al['clean'], color=C_EST_CLEAN, lw=3.6, label='оценка без сбоя', zorder=2)
    ax_s.plot(*al['naive'], color=C_NAIVE, lw=1.4, label='наивная\n(интеграл среднего)', zorder=3)
    ax_s.plot(*al['est'], color=C_EST, lw=1.7, label='оценка при сбое', zorder=4)
    ax_s.axvline(wd.t_after, color=AXIS, lw=0.9, zorder=1)
    ax_s.text(wd.t_after, 1.0, f'+{stats["after_s"]:g} с после окна', transform=ax_s.get_xaxis_transform(),
              ha='center', va='bottom', fontsize=8, color=MUTED)
    na, ea = stats['along_after_naive'], stats['along_after_est']
    _end_label(ax_s, al['naive'], C_NAIVE, f'{na:+.1f} м')
    _end_label(ax_s, al['est'], C_EST, f'{ea:+.1f} м')
    ax_s.set_title('Ошибка положения вдоль пути, накопленная с начала сбоя, м')
    _legend_right(ax_s)

    # 4: IMM mode probabilities
    _draw_modes(ax_m, wd.mode_t, wd.mode_mu)
    ax_m.set_ylim(0, 1)
    ax_m.set_yticks([0, 0.5, 1])
    ax_m.grid(False)
    ax_m.set_title('Вероятности режимов фильтра (IMM)')
    _legend_right(ax_m, fontsize=8, handlelength=1.2)

    # 5: health flags
    _draw_flags(ax_f, wd, rows)
    ax_f.set_title('Флаги диагностики оценщика')

    # 6: driver controller
    nt, nn = wd.notch
    ax_n.axhline(0, color=AXIS, lw=0.9)
    ax_n.fill_between(nt, 0, nn, where=nn > 0, step='post', color=INK2, alpha=0.12, lw=0)
    ax_n.fill_between(nt, 0, nn, where=nn < 0, step='post', color=INK2, alpha=0.25, lw=0)
    ax_n.step(nt, nn, where='post', color=INK2, lw=1.0)
    lim = max(4.0, np.nanmax(np.abs(nn), initial=0) + 1)
    ax_n.set_ylim(-lim, lim)
    ax_n.set_title('Позиция контроллера водителя (+ тяга, − торможение)')
    ax_n.set_xlabel('время от начала записи, с')

    for i, ax in enumerate(axes):
        _fault_band(ax, wd, label=(i == 0))
        ax.set_xlim(wd.x0, wd.x1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    top_in = 1.22                                          # inches above the first axes: title + subtitle
    fig.text(0.07, 1 - 0.22 / fig_h, describe_fault(w), ha='left', va='top', fontsize=13.5, fontweight='semibold',
             color=INK)
    sub = (f'{demo.bag.name} · окно {stats["t_start_s"]:.1f}–{stats["t_end_s"]:.1f} с · {stats["regime"]}, '
           f'{stats["v_start_kmh"]:.0f}→{stats["v_end_kmh"]:.0f} км/ч · {stats["spec"]}\n'
           f'RMSE скорости в окне: оценка {stats["v_rmse_in_est"]:.2f} м/с, наивная {stats["v_rmse_in_naive"]:.2f} м/с · '
           f'ошибка вдоль пути через {stats["after_s"]:g} с после окна: оценка {ea:+.2f} м, наивная {na:+.2f} м')
    fig.text(0.07, 1 - 0.55 / fig_h, sub, ha='left', va='top', fontsize=9.2, color=INK2, linespacing=1.5)
    fig.subplots_adjust(left=0.15, right=0.83, top=1 - top_in / fig_h, bottom=0.45 / fig_h)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Animated GIF of one window (progressive reveal, as if streamed live)
# ----------------------------------------------------------------------------------------------
def _blit_pillow_writer_cls():
    from matplotlib.animation import PillowWriter

    class BlitPillowWriter(PillowWriter):
        """PillowWriter that renders the static figure once and composes every frame from the cached
        background plus the animated artists (a full redraw per frame is ~10x slower), quantizes all
        frames to one shared palette (small inter-frame deltas -> small GIF) and holds the last frame."""

        def __init__(self, animated, fps, hold_s=0.0, colors=200, swatches=()):
            super().__init__(fps=fps)
            self._animated = sorted(animated, key=lambda a: a.get_zorder())
            self._hold_s = hold_s
            self._colors = colors
            self._swatches = list(swatches)
            self._bg = None

        def grab_frame(self, **savefig_kwargs):
            from PIL import Image
            fig, canvas = self.fig, self.fig.canvas
            if self._bg is None:
                # static background: animated artists hidden (while saving, Axes.draw would include them)
                vis = [a.get_visible() for a in self._animated]
                for a in self._animated:
                    a.set_visible(False)
                canvas.draw()
                self._bg = canvas.copy_from_bbox(fig.bbox)
                for a, v in zip(self._animated, vis):
                    a.set_visible(v)
                canvas.restore_region(self._bg)
            else:
                canvas.restore_region(self._bg)
            for a in self._animated:
                fig.draw_artist(a)
            wpx, hpx = canvas.get_width_height(physical=True)
            self._frames.append(Image.frombuffer('RGBA', (wpx, hpx), bytes(canvas.buffer_rgba()), 'raw', 'RGBA', 0, 1)
                                .convert('RGB'))

        def finish(self):
            from PIL import Image, ImageColor
            # shared palette from the last frame plus solid swatches of the style colors (small areas such as
            # legend keys would otherwise be snapped to a neighbouring hue)
            last = self._frames[-1]
            sw = 48
            probe = Image.new('RGB', (last.width, last.height + sw), SURFACE)
            probe.paste(last, (0, 0))
            for k, c in enumerate(self._swatches):
                probe.paste(ImageColor.getrgb(c), (k * sw, last.height, (k + 1) * sw, last.height + sw))
            pal = probe.quantize(colors=self._colors, method=Image.Quantize.MEDIANCUT)
            frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in self._frames]
            dur = [int(round(1000 / self.fps))] * len(frames)
            dur[-1] += int(round(self._hold_s * 1000))
            frames[0].save(self.outfile, save_all=True, append_images=frames[1:], duration=dur, loop=0)

    return BlitPillowWriter


def make_gif(demo: Demo, w: dict, stats: dict, wd: WindowData, path: Path, step: float = 0.2, fps: int = 10,
             dpi: int = 90, pre: float = 6.0, max_mb: float = 8.0) -> dict:
    plt = _setup_matplotlib()
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MaxNLocator

    x0, x1 = max(wd.x0, wd.a - pre), wd.t_after + 1.0
    frames_t = np.arange(x0 + step, x1 + 1e-9, step)
    fig = plt.figure(figsize=(10, 6.4), dpi=dpi)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.45, 0.9], hspace=0.45, left=0.07, right=0.8, top=0.79,
                          bottom=0.09)
    ax_v = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1], sharex=ax_v)
    ax_m = fig.add_subplot(gs[2], sharex=ax_v)
    animated = []

    def anim(a):
        a.set_animated(True)
        animated.append(a)
        return a

    for ax in (ax_v, ax_s, ax_m):
        ax.set_xlim(x0, x1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.axvline(wd.a, color=C_FAULT, lw=0.7, alpha=0.55, zorder=0.5)
    bands = [anim(ax.add_patch(Rectangle((wd.a, 0), 0, 1, transform=ax.get_xaxis_transform(), color=C_FAULT,
                                         alpha=0.08, lw=0, zorder=0.4))) for ax in (ax_v, ax_s)]
    yv = np.r_[wd.truth[1], wd.est[1][np.isfinite(wd.est[1])]]
    yw = np.r_[wd.front[1][np.isfinite(wd.front[1])], wd.rear[1][np.isfinite(wd.rear[1])]]
    ax_v.set_ylim(min(-1.0, np.nanmin(yv) - 2), max(np.nanmax(yv) * 1.08 + 3, (np.percentile(yw, 99.7) + 2) if len(yw) else 0))
    al = wd.along
    ya = np.r_[al['naive'][1], al['est'][1]]
    ya = ya[np.isfinite(ya)]
    pad = max(0.5, 0.12 * (np.ptp(ya) if len(ya) else 1.0))
    ax_s.set_ylim(min(-0.5, ya.min() - pad) if len(ya) else -1, max(0.5, ya.max() + pad) if len(ya) else 1)
    ax_s.axhline(0, color=AXIS, lw=0.9, zorder=1)
    ax_m.set_ylim(0, 1)
    ax_m.set_yticks([0, 1])
    ax_m.grid(False)
    ax_m.stackplot(wd.mode_t, wd.mode_mu.T, colors=MODE_COLORS, lw=0)         # static, revealed by the cover
    cover = anim(ax_m.add_patch(Rectangle((x0, 0), x1 - x0, 1, facecolor=SURFACE, lw=0, zorder=5)))

    lines = {
        'front': anim(ax_v.plot([], [], color=MUTED, lw=0.9, label='тележка П', zorder=2)[0]),
        'rear': anim(ax_v.plot([], [], color=MUTED, lw=0.9, ls=(0, (1.2, 1.6)), label='тележка З', zorder=2)[0]),
        'truth': anim(ax_v.plot([], [], color=INK, lw=1.9, label='эталон GNSS', zorder=3)[0]),
        'naive': anim(ax_v.plot([], [], color=C_NAIVE, lw=1.5, ls=(0, (4, 2)), label='наивная\n(среднее тележек)',
                                zorder=3)[0]),
        'est': anim(ax_v.plot([], [], color=C_EST, lw=2.0, label='оценка фильтра', zorder=4)[0]),
        's_naive': anim(ax_s.plot([], [], color=C_NAIVE, lw=1.6, label='наивная', zorder=3)[0]),
        's_est': anim(ax_s.plot([], [], color=C_EST, lw=2.0, label='оценка фильтра', zorder=4)[0]),
    }
    src = {'front': wd.front, 'rear': wd.rear, 'truth': wd.truth, 'naive': wd.naive, 'est': wd.est,
           's_naive': al['naive'], 's_est': al['est']}
    ax_v.set_title('Скорость, км/ч')
    ax_s.set_title('Ошибка положения вдоль пути с начала сбоя, м')
    ax_m.set_title('Режимы фильтра (IMM)')
    ax_m.set_xlabel('время от начала записи, с')
    _legend_right(ax_v)
    _legend_right(ax_s)
    ax_m.legend(handles=[Patch(color=c, label=l) for c, l in zip(MODE_COLORS, MODE_LABELS)], loc='upper left',
                bbox_to_anchor=(1.01, 1.25), borderaxespad=0, fontsize=7.5, handlelength=1.0)
    cursors = [anim(ax.axvline(x0, color=INK, lw=0.8, alpha=0.6, zorder=6)) for ax in (ax_v, ax_s, ax_m)]
    fig.text(0.07, 0.975, describe_fault(w), ha='left', va='top', fontsize=12.5, fontweight='semibold', color=INK)
    fig.text(0.07, 0.93, f'{demo.bag.name} · {stats["regime"]}, {stats["v_start_kmh"]:.0f}→{stats["v_end_kmh"]:.0f} км/ч · '
             f'повтор записи с инъекцией сбоя, ускорено ×{step * fps:g}', ha='left', va='top', fontsize=9, color=INK2)
    status = anim(fig.text(0.07, 0.89, '', ha='left', va='top', fontsize=9.5, color=INK, linespacing=1.5))

    mt, mmu, ffl = wd.mode_t, wd.mode_mu, wd.flags

    def cut(series, t):
        ts, ys = series
        k = np.searchsorted(ts, t, side='right')
        return ts[:k], ys[:k]

    def value_at(series, t):
        ts, ys = series
        k = np.searchsorted(ts, t, side='right') - 1
        while k >= 0 and not np.isfinite(ys[k]):
            k -= 1
        return ys[k] if k >= 0 else float('nan')

    def update(i):
        t = frames_t[min(i, len(frames_t) - 1)]
        for k, ln in lines.items():
            ln.set_data(*cut(src[k], t))
        for c in cursors:
            c.set_xdata([t, t])
        for r in bands:
            r.set_width(max(0.0, min(t, wd.b) - wd.a))
        cover.set_x(t)
        cover.set_width(max(x1 - t, 0.0) + 1.0)
        j = np.searchsorted(mt, t, side='right') - 1
        if j >= 0:
            mode = int(np.argmax(mmu[j]))
            fl = flag_names(int(ffl[j]), INDICATOR_MASK)
            txt_mode = f'режим: {MODE_LABELS[mode]} ({mmu[j, mode]:.2f})'
            txt_flags = 'флаги: ' + (', '.join(fl) if fl else 'нет')
        else:
            txt_mode, txt_flags = 'режим: —', 'флаги: —'
        if t < wd.a:
            phase = f'до сбоя {wd.a - t:.1f} с'
        elif t <= wd.b:
            phase = f'СБОЙ идёт {t - wd.a:.1f} с из {wd.b - wd.a:g}'
        else:
            phase = f'после сбоя +{t - wd.b:.1f} с'
        e_est, e_nv = value_at(wd.err_est, t), value_at(wd.err_naive, t)
        line2 = f'ошибка скорости: оценка {e_est:+.1f} км/ч, наивная {e_nv:+.1f} км/ч'
        if t >= wd.a:
            line2 += (f'    ·    ошибка положения: оценка {value_at(src["s_est"], t):+.2f} м, '
                      f'наивная {value_at(src["s_naive"], t):+.2f} м')
        status.set_text(f'{phase}    ·    {txt_mode}    ·    {txt_flags}\n{line2}')
        return animated

    writer = _blit_pillow_writer_cls()(animated, fps=fps, hold_s=3.0,
                                       swatches=[INK, INK2, MUTED, GRID, AXIS, C_EST, C_NAIVE, '#f8eceb'] + MODE_COLORS)
    ani = FuncAnimation(fig, update, frames=len(frames_t), blit=False, repeat=False, cache_frame_data=False)
    fig.canvas.draw_idle = lambda *a, **k: None       # frames are composed by the writer (no full redraws)
    ani.save(str(path), writer=writer, dpi=dpi)
    plt.close(fig)
    size_mb = path.stat().st_size / 1e6
    info = {'path': _rel(path), 'frames': int(len(frames_t)), 'fps': fps, 'step_s': step, 'dpi': dpi,
            'size_mb': size_mb, 'speedup': step * fps}
    if size_mb > max_mb and dpi > 50:
        return make_gif(demo, w, stats, wd, path, step=step * 1.5, fps=fps, dpi=int(dpi * 0.85), pre=pre,
                        max_mb=max_mb)
    return info


# ----------------------------------------------------------------------------------------------
# Metrics table
# ----------------------------------------------------------------------------------------------
def _f(x, fmt='{:.2f}', dash='—'):
    return dash if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def metrics_markdown(bag_name: str, rows: List[dict], after_s: float, header: bool = True) -> str:
    out = []
    if header:
        out.append(f'### {bag_name}\n')
    out.append('| # | Сбой | Окно, с | Режим, км/ч | RMSE скорости в окне, м/с<br>оценка / наивная (без сбоя) '
               f'| RMSE скорости {after_s:g} с после, м/с<br>оценка / наивная '
               '| Ошибка вдоль пути к концу окна, м<br>оценка / наивная '
               f'| … через {after_s:g} с после окна, м<br>оценка / наивная | Реакция, с |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(rows, 1):
        out.append(
            f"| {i} | {r['description']} | {r['t_start_s']:.1f}–{r['t_end_s']:.1f} "
            f"| {r['regime']}, {r['v_start_kmh']:.0f}→{r['v_end_kmh']:.0f} "
            f"| **{_f(r['v_rmse_in_est'])}** / {_f(r['v_rmse_in_naive'])} ({_f(r['v_rmse_in_clean'])}) "
            f"| **{_f(r['v_rmse_after_est'])}** / {_f(r['v_rmse_after_naive'])} "
            f"| **{_f(r['along_end_est'], '{:+.2f}')}** / {_f(r['along_end_naive'], '{:+.2f}')} "
            f"| **{_f(r['along_after_est'], '{:+.2f}')}** / {_f(r['along_after_naive'], '{:+.2f}')} "
            f"| {_f(r['reaction_s'], '{:.2f}')} |")
    return '\n'.join(out) + '\n'


CSV_COLS = ['bag', 'window', 'spec', 'kind', 'topic', 'description', 't_start_s', 't_end_s', 'regime', 'v_start_kmh',
            'v_end_kmh', 'v_mean_kmh', 'accel_mps2', 'notch_mean', 'n_speed_in',
            'v_rmse_in_est', 'v_rmse_in_naive', 'v_rmse_in_clean', 'v_max_in_est', 'v_max_in_naive', 'v_max_in_clean',
            'v_rmse_after_est', 'v_rmse_after_naive', 'v_rmse_after_clean',
            'along_end_est', 'along_end_naive', 'along_end_clean', 'along_after_est', 'along_after_naive',
            'along_after_clean', 'along_maxabs_est', 'along_maxabs_naive', 'along_maxabs_clean',
            'reaction_s', 'reaction_flags', 'model_only_s', 'after_s', 'interest', 'png', 'gif']


# ----------------------------------------------------------------------------------------------
# Window suggestions (where the tram really accelerates / brakes)
# ----------------------------------------------------------------------------------------------
def suggest(bag: BagData, min_run_s: float = 5.0) -> List[dict]:
    """Traction / braking episodes (runs of constant notch sign >= min_run_s) with their speed change."""
    cmd = bag['cmd']
    vel = bag['vel_master']
    if len(vel):
        tv, v = vel[:, T_BAG], np.hypot(vel[:, VX], vel[:, VY])
    else:
        tv, v = bag['front'][:, T_BAG], bag['front'][:, V_COL] / KMH
    sgn = np.sign(cmd[:, V_COL]).astype(int)
    edges = np.flatnonzero(np.diff(sgn) != 0) + 1
    out = []
    for s, e in zip(np.r_[0, edges], np.r_[edges, len(sgn)]):
        if sgn[s] == 0:
            continue
        ta, tb = cmd[s, T_BAG], cmd[e - 1, T_BAG]
        if tb - ta < min_run_s:
            continue
        va, vb = np.interp(ta, tv, v) * KMH, np.interp(tb, tv, v) * KMH
        out.append({'regime': 'traction' if sgn[s] > 0 else 'braking', 't0_s': ta - bag.t_start, 'dur_s': tb - ta,
                    'v0_kmh': va, 'v1_kmh': vb, 'notch_median': float(np.median(cmd[s:e, V_COL]))})
    return out


def print_suggestions(bag: BagData):
    rows = suggest(bag)
    print(f'{bag.name}: duration {bag.duration:.0f} s, {len(rows)} traction/braking episodes >= 5 s')
    for r in rows:
        t0, d = r['t0_s'], r['dur_s']
        if r['regime'] == 'traction':
            hint = f"slip:wheels@{t0 + 2:.0f}+{min(5.0, d - 2):.0f}:peak=0.3:phase=1  dropout:wheels@{t0 + 2:.0f}+5"
        else:
            hint = f"slide:wheels@{t0 + 2:.0f}+{min(5.0, d - 2):.0f}:peak=0.4:phase=-1  freeze:front@{t0 + 1:.0f}+{d:.0f}"
        strong = (r['regime'] == 'traction' and r['v1_kmh'] - r['v0_kmh'] > 15) or \
                 (r['regime'] == 'braking' and r['v0_kmh'] > 25 and r['v0_kmh'] - r['v1_kmh'] > 15)
        print(f"  {'*' if strong else ' '} {r['regime']:8s} t0={t0:7.1f}s dur={d:5.1f}s "
              f"v {r['v0_kmh']:5.1f}->{r['v1_kmh']:5.1f} km/h notch~{r['notch_median']:+.0f}   e.g. {hint}")


# ----------------------------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------------------------
def parse_fault_list(text: str) -> List[str]:
    return [s.strip() for s in text.replace(';', ',').split(',') if s.strip()]


def demo_reference(bag: BagData) -> Reference:
    """GNSS master reference (harness definitions) with a stricter position part: when the bag is mostly
    RTK, only RTK fixes (status 2) that pass the Doppler-consistency outlier check are used, so that
    short-window along-track increments are not polluted by float-solution jumps. The frame origin stays
    the first valid master fix (the estimator's own ENU origin)."""
    fix = bag['fix_master']
    oi = select_origin(fix, 0)
    rtk = float(np.mean(fix[:, STATUS] == 2)) if len(fix) else 0.0
    cfg = RefConfig(frame='enu', clean=True, min_status=2 if rtk >= 0.5 else 0,
                    origin=(float(fix[oi, LAT]), float(fix[oi, LON]), float(fix[oi, ALT])))
    ref = build_reference(bag, cfg)
    if cfg.min_status == 2:
        # RTK fixes are cm-level, but the receiver sometimes re-emits a stale position and then delivers
        # fixes ~0.4 s late for a few seconds (2-3 m along-track steps, below the harness 3 m gate):
        # a tighter Doppler-consistency gate removes them from the position truth
        vel = bag['vel_master']
        o = np.argsort(ref.tbag_pos, kind='stable')
        bad = flag_fix_outliers(ref.tbag_pos[o], ref.xyz[o], vel[:, T_BAG], vel[:, [VX, VY, VZ]], thr_h=1.5)
        extra = np.zeros(len(o), bool)
        extra[o] = bad
        ref.diag['n_outlier_tight'] = int(np.sum(extra & ~ref.outlier))
        ref.outlier = ref.outlier | extra
    return ref


def run_demo(bag_name: str, faults: Sequence[str], out_dir: Path, seed: int = 0, setup: ReplaySetup = None,
             pre: float = 10.0, post: float = 14.0, after_s: float = 10.0, gif: bool = True,
             gif_window: Optional[int] = None, gif_step: float = 0.2, gif_fps: int = 10, gif_dpi: int = 90,
             jobs: int = 2, replay_dir: Optional[Path] = None, reuse: bool = False, dpi: int = 110,
             png: bool = True, quiet: bool = False) -> dict:
    """Clean + faulted replay of one bag, per-window figures, GIF and metrics. Returns the metrics dict."""
    setup = setup or ReplaySetup()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    t_all = time.perf_counter()

    bag = load_bag(bag_name)
    fbag = apply_faults(bag, list(faults), seed=seed)
    windows = fbag.meta.get('faults', [])
    ref = demo_reference(bag)

    tmp = None
    if replay_dir is None:
        tmp = Path(tempfile.mkdtemp(prefix=f'fault_demo_{bag_name}_'))
        rdir = tmp
    else:
        rdir = Path(replay_dir)
        rdir.mkdir(parents=True, exist_ok=True)
    key = setup.key()
    fkey = hashlib.md5(json.dumps([list(faults), seed]).encode()).hexdigest()[:8]
    jobs_spec = [('clean', bag, rdir / f'{bag_name}_{key}_clean_events.csv', rdir / f'{bag_name}_{key}_clean_out.csv'),
                 ('fault', fbag, rdir / f'{bag_name}_{key}_f{fkey}_events.csv', rdir / f'{bag_name}_{key}_f{fkey}_out.csv')]
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(jobs, MAX_PARALLEL_REPLAYS))) as ex:
            futs = {name: ex.submit(run_replay, setup, b, ev, out, reuse) for name, b, ev, out in jobs_spec}
            reps = {name: f.result() for name, f in futs.items()}
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    for name, r in reps.items():
        log(f'[{bag_name}] replay {name}: {"cached" if r.cached else f"{r.wall_s:.1f} s"}  {r.stderr}')

    clean, fault = make_run(reps['clean'].df), make_run(reps['fault'].df)
    demo = Demo(bag=bag, fbag=fbag, ref=ref, clean=clean, fault=fault,
                pse_c=per_sample_errors(ref, clean.log, bag), pse_f=per_sample_errors(ref, fault.log, fbag),
                windows=windows)

    rows, wds = [], []
    for i, w in enumerate(windows, 1):
        st = window_metrics(demo, w, after_s)
        st['bag'] = bag_name
        st['window'] = i
        wd = window_data(demo, w, pre, post, after_s)
        st['png'] = ''
        if png:
            png_path = out_dir / f"w{i:02d}_{w['kind']}_{w['topic']}.png"
            plot_window(demo, w, st, wd, png_path, dpi=dpi)
            st['png'] = _rel(png_path)
        st['gif'] = ''
        rows.append(st)
        wds.append(wd)
        log(f"[{bag_name}] w{i:02d} {st['spec']:40s} v_rmse est {st['v_rmse_in_est']:.3f} naive {st['v_rmse_in_naive']:.3f}"
            f" | along+{after_s:g}s est {st['along_after_est']:+.2f} naive {st['along_after_naive']:+.2f} m"
            f" | reaction {_f(st['reaction_s'])} s")

    gif_info = None
    if gif and rows:
        k = (gif_window - 1) if gif_window else int(np.nanargmax([r['interest'] for r in rows]))
        w, st = windows[k], rows[k]
        gpath = out_dir / f"live_w{k + 1:02d}_{w['kind']}_{w['topic']}.gif"
        gif_info = make_gif(demo, w, st, wds[k], gpath, step=gif_step, fps=gif_fps, dpi=gif_dpi)
        gif_info['window'] = k + 1
        st['gif'] = gif_info['path']
        log(f"[{bag_name}] gif: window {k + 1} -> {gif_info['path']} ({gif_info['size_mb']:.2f} MB, "
            f"{gif_info['frames']} frames)")

    md = [f'# Инъекция сбоев: {bag_name}\n',
          f'Сбои: `{",".join(faults)}` (seed {seed}). Время — секунды от начала записи (часы bag). '
          f'Наивная оценка — среднее последних показаний двух тележек без фильтрации. «Без сбоя» — тот же '
          f'оценщик на исходных данных в тех же окнах. Ошибка вдоль пути накапливается с начала окна '
          f'(у наивной — интеграл её скорости от эталонного положения в начале окна). Реакция — время от '
          f'начала окна до первого флага аномалии или P(норма) < 0.5, которых нет в прогоне без сбоя.\n',
          metrics_markdown(bag_name, rows, after_s, header=False)]
    for i, r in enumerate(rows, 1):
        if r['reaction_flags']:
            md.append(f"- окно {i}: первым сработало «{r['reaction_flags']}»; «только модель» {r['model_only_s']:.1f} с")
    (out_dir / 'metrics.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    pd.DataFrame(rows).reindex(columns=CSV_COLS).to_csv(out_dir / 'metrics.csv', index=False, float_format='%.6g')
    result = {
        'bag': bag_name, 'faults': list(faults), 'seed': seed, 'after_s': after_s,
        'windows': rows, 'gif': gif_info,
        'replay': {'setup': setup.provenance(), 'sets': setup.binary_sets(), 'gnss_seconds': setup.gnss_seconds,
                   'clean': {'stderr': reps['clean'].stderr, 'events': reps['clean'].events},
                   'fault': {'stderr': reps['fault'].stderr, 'events': reps['fault'].events}},
        'reference': {'antenna': 'master', 'frame': 'enu', 'time_base': 'header', 'match_tol_s': 0.05,
                      'position_min_status': ref.cfg.min_status, 'position_outliers_removed': ref.cfg.clean,
                      'rtk_frac_bag': float(np.mean(bag['fix_master'][:, STATUS] == 2)),
                      'outlier_frac': float(np.mean(ref.outlier)), 'n_outlier_tight': ref.diag.get('n_outlier_tight')},
        'wall_s': time.perf_counter() - t_all,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(clean_json(result), indent=1, ensure_ascii=False, default=_json_default),
                                          encoding='utf-8')
    log(f'[{bag_name}] done in {result["wall_s"]:.1f} s -> {_rel(out_dir)}')
    return result


def clean_json(o):
    """NaN / inf -> None (strict JSON), numpy scalars -> Python."""
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True, help='bag name (data/npz/<bag>.npz)')
    ap.add_argument('--faults', default='', help='comma-separated fault specs (tools/harness/faults.py syntax)')
    ap.add_argument('--out', default=None, help='output directory (default tools/demo/out/<bag>)')
    ap.add_argument('--seed', type=int, default=0, help='fault RNG seed (spike / noise / nan)')
    ap.add_argument('--suggest', action='store_true', help='list traction / braking episodes and exit')
    ap.add_argument('--exe', default=str(DEFAULT_EXE), help='tbo_replay binary')
    ap.add_argument('--maps', default=str(DEFAULT_MAPS), help='directory with track_map / branch_* / landmarks / cutoffs')
    ap.add_argument('--traction', default=str(DEFAULT_TRACTION))
    ap.add_argument('--no-dfield', action='store_true', help='do not pass <maps>/dfield.csv to the binary')
    ap.add_argument('--gnss-seconds', type=float, default=5.0)
    ap.add_argument('--set', dest='sets', action='append', default=[], help='binary override key=value (repeatable)')
    ap.add_argument('--pre', type=float, default=10.0, help='context before each window in the figure [s]')
    ap.add_argument('--post', type=float, default=14.0, help='context after each window in the figure [s] (at least --after + 4)')
    ap.add_argument('--after', type=float, default=10.0, help='"after the window" horizon of the metrics [s]')
    ap.add_argument('--no-gif', action='store_true')
    ap.add_argument('--no-png', action='store_true', help='metrics only (no per-window figures)')
    ap.add_argument('--gif-window', type=int, default=None, help='1-based window for the GIF (default: most interesting)')
    ap.add_argument('--gif-step', type=float, default=0.2, help='bag seconds per GIF frame')
    ap.add_argument('--gif-fps', type=int, default=10)
    ap.add_argument('--gif-dpi', type=int, default=90)
    ap.add_argument('--dpi', type=int, default=110, help='PNG resolution')
    ap.add_argument('--jobs', type=int, default=2, help=f'parallel replays (max {MAX_PARALLEL_REPLAYS})')
    ap.add_argument('--replay-dir', default=None, help='keep event / output CSVs here (default: temp dir, deleted)')
    ap.add_argument('--reuse', action='store_true', help='reuse replay outputs found in --replay-dir')
    return ap


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    if a.suggest:
        print_suggestions(load_bag(a.bag))
        return None
    faults = parse_fault_list(a.faults)
    if not faults:
        raise SystemExit('no faults given (--faults "kind:topic@t0+dur[:k=v],...")')
    setup = ReplaySetup(exe=Path(a.exe), maps=Path(a.maps), traction=Path(a.traction), gnss_seconds=a.gnss_seconds,
                        sets=dict(kv.split('=', 1) for kv in a.sets), use_dfield=not a.no_dfield)
    out = Path(a.out) if a.out else DEFAULT_OUT / a.bag
    res = run_demo(a.bag, faults, out, seed=a.seed, setup=setup, pre=a.pre, post=a.post, after_s=a.after,
                   gif=not a.no_gif, gif_window=a.gif_window, gif_step=a.gif_step, gif_fps=a.gif_fps,
                   gif_dpi=a.gif_dpi, jobs=a.jobs, replay_dir=Path(a.replay_dir) if a.replay_dir else None,
                   reuse=a.reuse, dpi=a.dpi, png=not a.no_png)
    print(f"command: python tools/demo/fault_demo.py {' '.join(shlex.quote(x) for x in (argv or sys.argv[1:]))}")
    return res


if __name__ == '__main__':
    main()
