"""Shared helpers for the filter-consistency analysis (analysis/consistency/*).

Replays go through tools/replay/quick_eval.eval_bag() with the frozen binary
build_core/tbo_replay_fix2.exe and the same default settings the team uses
(ENU at the first master fix, antenna-1 point, landmarks + cut-offs, full map).
Outputs are cached as parquet under analysis/consistency/cache/ (git-ignored).

Reference conventions (identical to quick_eval):
  speed    = horizontal |v| of /sensing/gnss/master/vel at its header stamps
  position = /sensing/gnss/master/fix in ENU at the first finite master fix
  matching = nearest output stamp within 0.05 s (position: pos_valid outputs only)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONS = ROOT / 'analysis' / 'consistency'
CACHE = CONS / 'cache'
RESULTS = CONS / 'results'
EXE = ROOT / 'build_core' / 'tbo_replay_fix2.exe'
NPZ = ROOT / 'data' / 'npz'

sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge  # noqa: E402
import quick_eval  # noqa: E402

match_nearest = quick_eval.match_nearest
enu = quick_eval.enu

# Estimator constants used by the reconstructions (must match the frozen binary; checked
# against `tbo_replay_fix2.exe --dump-params` in run_replays.py).
KMH = 1.00037 / 3.6            # wheel_kmh_to_ms
SIGMA_WHEEL = 0.05             # m/s
SIGMA_WHEEL_REL = 0.004        # fraction of speed
SIGMA_ACCEL = 0.12             # m/s^2
Q_DIST = 0.004                 # (m/s^2)^2 / s
CUSUM_SLIP = 0.60              # m/s^2
CUSUM_SLIDE = 0.80             # m/s^2
CUSUM_H = 0.30                 # m/s
CMD_FAULT_ACCEL = 0.50         # m/s^2
CMD_FAULT_H = 0.40             # m/s
D_MAX = 0.60                   # disturbance_max (clamp of d in the monitor reference)
STANDSTILL_KMH = 0.15

# Health flag bits (core/include/tbo/types.hpp)
F_FRONT_SLIP, F_REAR_SLIP, F_FRONT_SLIDE, F_REAR_SLIDE = 1 << 0, 1 << 1, 1 << 2, 1 << 3
F_FRONT_DROP, F_REAR_DROP, F_CMD_DROP = 1 << 4, 1 << 5, 1 << 6
F_FRONT_STUCK, F_REAR_STUCK, F_FRONT_INV, F_REAR_INV = 1 << 7, 1 << 8, 1 << 9, 1 << 10
F_MODEL_ONLY, F_STANDSTILL, F_NOT_INIT, F_NO_MAP = 1 << 11, 1 << 12, 1 << 13, 1 << 14
F_RECOVERED, F_LATE, F_UNMODELED, F_CMD_INCONS, F_LANDMARK = 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19
F_ANY_WHEEL_PROBLEM = (F_FRONT_SLIP | F_REAR_SLIP | F_FRONT_SLIDE | F_REAR_SLIDE | F_FRONT_DROP | F_REAR_DROP |
                       F_FRONT_STUCK | F_REAR_STUCK | F_FRONT_INV | F_REAR_INV | F_MODEL_ONLY | F_RECOVERED)

REGIMES = ['stop', 'accel', 'cruise', 'brake']
REGIME_V_STOP = 0.2    # m/s, smoothed reference speed below -> stop
REGIME_A = 0.2         # m/s^2, |smoothed reference acceleration| above -> accel / brake


def splits() -> dict:
    return json.load(open(ROOT / 'data' / 'splits.json'))


def ensure_dirs():
    for p in (CACHE, RESULTS):
        p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- replays

def replay_file(bag: str, tag: str = 'base') -> Path:
    return CACHE / 'replay' / tag / f'{bag}.parquet'


def run_bag(bag: str, tag: str = 'base', sets: dict | None = None, force: bool = False) -> dict:
    """Replays one bag with the frozen binary (via quick_eval) and caches outputs + metrics."""
    out = replay_file(bag, tag)
    meta = out.with_suffix('.json')
    if out.exists() and meta.exists() and not force:
        return json.load(open(meta))
    out.parent.mkdir(parents=True, exist_ok=True)
    cpp_bridge.REPLAY_EXE = EXE
    quick_eval.TMP = CACHE / 'tmp' / tag   # private temp dir: no clash with other jobs
    m, o = quick_eval.eval_bag(bag, sets=sets)
    m['stderr'] = o.attrs.get('stderr', '')
    o = o.drop(columns=['proc_ns'])
    o.to_parquet(out, index=False)
    json.dump(m, open(meta, 'w'), indent=1)
    for suffix in ('_events.csv', '_out.csv'):
        try:
            os.remove(quick_eval.TMP / f'{bag}{suffix}')
        except OSError:
            pass
    return m


def load_replay(bag: str, tag: str = 'base') -> tuple[dict, pd.DataFrame]:
    out = replay_file(bag, tag)
    return json.load(open(out.with_suffix('.json'))), pd.read_parquet(out)


def load_npz(bag: str):
    return np.load(NPZ / f'{bag}.npz')


# ----------------------------------------------------------------------------- reference

def smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centered moving average of odd length n (edges: shrinking window)."""
    if n <= 1:
        return x.copy()
    k = np.ones(n)
    num = np.convolve(np.nan_to_num(x), k, mode='same')
    den = np.convolve(np.isfinite(x).astype(float), k, mode='same')
    return num / np.maximum(den, 1)


def reference_speed(d) -> pd.DataFrame:
    """Master GNSS velocity at header stamps with a regime label from the smoothed reference."""
    mv = d['sensing__gnss__master__vel']
    t = mv[:, 1]
    order = np.argsort(t, kind='stable')
    t = t[order]
    v = np.hypot(mv[order, 2], mv[order, 3])
    vs = smooth(v, 11)                                # 1 s at 10 Hz
    a = np.gradient(vs, t)
    a = smooth(a, 5)
    reg = np.full(len(t), 'cruise', dtype=object)
    reg[a > REGIME_A] = 'accel'
    reg[a < -REGIME_A] = 'brake'
    reg[vs < REGIME_V_STOP] = 'stop'
    return pd.DataFrame({'t': t, 'v_ref': v, 'v_ref_s': vs, 'a_ref': a, 'regime': reg})


def label_at(t_query: np.ndarray, ref: pd.DataFrame, col: str = 'regime', tol: float = 0.1) -> np.ndarray:
    # nearest reference epoch for every query time
    rt = ref.t.to_numpy()
    idx = np.clip(np.searchsorted(rt, t_query), 1, len(rt) - 1)
    left = rt[idx - 1]
    right = rt[idx]
    pick = np.where(np.abs(t_query - left) <= np.abs(right - t_query), idx - 1, idx)
    lab = ref[col].to_numpy()[pick].copy()
    far = np.abs(rt[pick] - t_query) > tol
    if lab.dtype == object:
        lab[far] = 'none'
    else:
        lab = lab.astype(float)
        lab[far] = np.nan
    return lab


def speed_pairs(o: pd.DataFrame, d) -> pd.DataFrame:
    """Output speed vs reference at reference epochs (quick_eval matching)."""
    ref = reference_speed(d)
    out_t = o.stamp_ns.to_numpy() * 1e-9
    pick, ok = match_nearest(ref.t.to_numpy(), out_t)
    r = ref[ok].reset_index(drop=True)
    oo = o.iloc[pick[ok]].reset_index(drop=True)
    return pd.DataFrame({
        't': r.t, 'v_ref': r.v_ref, 'v_ref_s': r.v_ref_s, 'a_ref': r.a_ref, 'regime': r.regime,
        'v': oo.v, 'v_var': oo.v_var, 'flags': oo['flags'].astype(np.int64), 'mu0': oo.mu0, 'mu3': oo.mu3,
        'k': oo.k, 'err': oo.v - r.v_ref})


def position_pairs(o: pd.DataFrame, d) -> pd.DataFrame:
    """Published position vs master fix (quick_eval matching), with the fix status."""
    mf = d['sensing__gnss__master__fix']
    mf = mf[np.isfinite(mf[:, 2])]
    p_ref = enu(mf[:, 2], mf[:, 3], mf[:, 4], mf[0, 2], mf[0, 3], mf[0, 4])
    op = o[o.pos_valid == 1]
    out_tp = op.stamp_ns.to_numpy() * 1e-9
    pick, ok = match_nearest(mf[:, 1], out_tp)
    oo = op.iloc[pick[ok]].reset_index(drop=True)
    e = oo[['x', 'y', 'z']].to_numpy() - p_ref[ok]
    yaw = oo.yaw.to_numpy()
    c, s = np.cos(yaw), np.sin(yaw)
    along = e[:, 0] * c + e[:, 1] * s
    cross = -e[:, 0] * s + e[:, 1] * c
    cxx, cxy, cyy = oo.cov_xx.to_numpy(), oo.cov_xy.to_numpy(), oo.cov_yy.to_numpy()
    var_along = cxx * c * c + 2 * cxy * c * s + cyy * s * s
    var_cross = cxx * s * s - 2 * cxy * c * s + cyy * c * c
    det = cxx * cyy - cxy * cxy
    nees2 = (cyy * e[:, 0] ** 2 - 2 * cxy * e[:, 0] * e[:, 1] + cxx * e[:, 1] ** 2) / det
    ref = reference_speed(d)
    t_fix = mf[ok, 1]
    return pd.DataFrame({
        't': t_fix, 'status': mf[ok, 5], 'ex': e[:, 0], 'ey': e[:, 1], 'ez': e[:, 2],
        'along': along, 'cross': cross, 'var_along': var_along, 'var_cross': var_cross,
        's_var': oo.s_var, 'cov_xx': cxx, 'cov_xy': cxy, 'cov_yy': cyy, 'nees2': nees2,
        'flags': oo['flags'].astype(np.int64), 'v': oo.v,
        'regime': label_at(t_fix, ref, 'regime'), 'v_ref_s': label_at(t_fix, ref, 'v_ref_s')})


def clock_anomaly_frac(bag: str) -> float:
    try:
        ck = pd.read_csv(ROOT / 'analysis' / 'timing_reference' / 'clocks_per_bag.csv').set_index('bag')
        return float(ck.loc[bag, 'gnss_vs_veh_anom_frac']) if bag in ck.index else 0.0
    except Exception:
        return float('nan')


# ----------------------------------------------------------------------------- track map

MAP_DIR = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps'
FRONT_ALONG, REAR_ALONG = 9.9, 2.35          # bogies ahead of antenna 1 (params.hpp)
BODY_REAR, BODY_FRONT = -2.1, 14.4
POSITION_LEAD = 0.045
KG_BRAKE, KG_COAST, KG_TRACTION = 8.22, 7.98, 7.36
CURV_ABS, CURV_SIGNED, CURV_SAT = 0.415, -0.054, 0.009


def enu_rotation(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([[-np.sin(lo), np.cos(lo), 0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


class TrackMap:
    """Main-cycle map (s, x, y, z, grade, curvature) re-expressed in a bag's ENU frame."""

    def __init__(self, bag_origin, path: Path = MAP_DIR / 'track_map.csv'):
        from scipy.spatial import cKDTree
        head = open(path).readlines()[:3]
        kv = dict(tok.split('=') for line in head if line.startswith('#') for tok in line[1:].split() if '=' in tok)
        lat0, lon0, h0 = float(kv['origin_lat']), float(kv['origin_lon']), float(kv['origin_h'])
        df = pd.read_csv(path, comment='#')
        p_map = df[['x', 'y', 'z']].to_numpy()
        x0 = quick_eval.ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))[0]
        xb = quick_eval.ecef(np.array([bag_origin[0]]), np.array([bag_origin[1]]), np.array([bag_origin[2]]))[0]
        ecef_pts = x0 + p_map @ enu_rotation(lat0, lon0)
        self.xyz = (ecef_pts - xb) @ enu_rotation(bag_origin[0], bag_origin[1]).T
        self.s = df.s.to_numpy()
        self.grade = df.grade.to_numpy()
        self.curv = df.curvature.to_numpy()
        self.length = float(self.s[-1] + (self.s[1] - self.s[0]))
        tang = np.gradient(self.xyz[:, :2], axis=0)
        self.heading = np.arctan2(tang[:, 1], tang[:, 0])
        self.tree = cKDTree(self.xyz[:, :2])

    def project(self, x, y, yaw, max_dist=3.0):
        """Arc length of the nearest map point with a consistent heading (NaN if none within max_dist)."""
        dist, idx = self.tree.query(np.stack([x, y], -1), k=6)
        ok_head = np.abs(np.angle(np.exp(1j * (self.heading[idx] - yaw[:, None])))) < np.radians(60)
        good = ok_head & (dist < max_dist)
        first = np.where(good.any(1), good.argmax(1), -1)
        s = np.where(first >= 0, self.s[idx[np.arange(len(x)), np.maximum(first, 0)]], np.nan)
        return s

    def _interp(self, arr, s):
        s = np.mod(s, self.length)
        return np.interp(s, np.append(self.s, self.length), np.append(arr, arr[0]))

    def curvature(self, s):
        return self._interp(self.curv, s)

    def grade_at(self, s):
        return self._interp(self.grade, s)


def fix_origin(d):
    mf = d['sensing__gnss__master__fix']
    mf = mf[np.isfinite(mf[:, 2])]
    return mf[0, 2], mf[0, 3], mf[0, 4]


def curve_factor(k):
    """Wheel curve under-reading correction c(k) applied by the estimator to each bogie."""
    return 1.0 + np.minimum(CURV_ABS * np.abs(k), CURV_SAT) + CURV_SIGNED * k


# ----------------------------------------------------------------------------- statistics

def bag_bootstrap_ci(values_by_bag: list[np.ndarray], stat, n_boot: int = 2000, seed: int = 0,
                     level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of stat(concatenated values) resampling whole bags (epochs are correlated)."""
    rng = np.random.default_rng(seed)
    nb = len(values_by_bag)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, nb, nb)
        out.append(stat(np.concatenate([values_by_bag[i] for i in pick])))
    lo, hi = np.percentile(out, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of a demeaned series, lags 0..max_lag (biased estimator)."""
    x = x - x.mean()
    n = len(x)
    den = float(np.dot(x, x))
    return np.array([float(np.dot(x[:n - k], x[k:])) / den if den > 0 else np.nan for k in range(max_lag + 1)])


def pooled_acf(segments: list[np.ndarray], max_lag: int) -> tuple[np.ndarray, int]:
    """ACF pooled over segments (each demeaned separately; lagged products never cross segments)."""
    num = np.zeros(max_lag + 1)
    den = 0.0
    n = 0
    for x in segments:
        if len(x) <= max_lag + 1:
            continue
        x = x - x.mean()
        den += float(np.dot(x, x))
        n += len(x)
        for k in range(max_lag + 1):
            num[k] += float(np.dot(x[:len(x) - k], x[k:]))
    return num / den, n


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    dm = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(dm == 1), np.flatnonzero(dm == -1)))


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x))
    json.dump(obj, open(path, 'w'), indent=1, default=conv, ensure_ascii=False)
