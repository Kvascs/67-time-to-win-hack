"""timing.py - ground-truth reference, clocks and local frames for the MosTransHack tram-odometry case.

This module reproduces, as closely as the data allows, the reference that the jury script most likely
builds from the GNSS telemetry, and provides the timing tools used in the analysis (REPORT.md).

Conventions
-----------
* Times are float64 seconds (UNIX epoch).  ``t`` / ``t_hdr`` = message ``header.stamp``;
  ``t_bag`` = rosbag2 receive time (the time at which ``ros2 bag play`` re-publishes the message).
* Wheel speeds in the bags are **km/h** (README says m/s - wrong); ``wheel()`` returns m/s = raw / 3.6.
  The residual wheel/GNSS scale (~0.999, vehicle/date dependent) is NOT corrected here.
* GNSS ``vel`` topics carry ENU (east, north, up) velocity in m/s; ``twist.angular`` is always zero.
* Local frame of the reference (judge's most likely choice): exact WGS-84 East-North-Up tangent plane
  with origin at the first valid ``/sensing/gnss/master/fix`` (by header stamp) of the run.

Main entry points
-----------------
load_bag(name)                          -> Bag
reference_trajectory(bag, ...)          -> Reference (t, x, y, z, speed, heading, s, ...)
reference_quality_mask(bag, ref)        -> epochs where the reference itself is trustworthy (own validation)
init_alignment(bag, gnss_bag_seconds)   -> what the node can derive from the first seconds of GNSS
judge_metrics(ref, t, v, x, y, z)       -> emulation of the jury metrics (speed + position)
llh_to_enu / enu_to_llh / llh_to_utm    -> geodesy (exact WGS-84; pyproj only for UTM)
estimate_lag(...)                       -> sub-ms delay between two speed signals
clock_offsets(bag)                      -> GNSS-clock vs vehicle-clock offset (detects 1-s clock steps)
match_nearest(t_ref, t_est, tol)        -> emulation of the judge's nearest-stamp pairing
along_cross_errors(ref, t, x, y)        -> along-/cross-track decomposition w.r.t. reference heading

Measured timing constants (REPORT.md, 70 bags; all w.r.t. header stamps):
  wheel speed  vs GNSS vel |v| ............ lag  0 +/- 4 ms   (VEL_REF_LAG_OF_WHEEL)
  GNSS vel |v| vs d|p|/dt of master fix ... lag 47 ms         (VEL_LAG_VS_POS)
  wheel speed  vs d|p|/dt of master fix ... lag 48 ms         (WHEEL_LAG_VS_POS)
  => speed output: no extra lead; position output: lead the wheel odometry by +48 ms.

Only numpy is required (scipy/pyproj optional), so the geodesy part can be copied into the ROS node.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------------------
# paths / constants
# ----------------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]            # C:\MosTransHack
DATA = ROOT / 'data'
NPZ_DIR = DATA / 'npz'
SPLITS_FILE = DATA / 'splits.json'

KMH = 1.0 / 3.6                                       # wheel topics are km/h

TOPIC = {
    'front': 'vehicle__front_bogie_velocity',
    'rear': 'vehicle__rear_bogie_velocity',
    'cmd': 'vehicle__driver_position_cmd',
    'master_fix': 'sensing__gnss__master__fix',
    'rover_fix': 'sensing__gnss__rover__fix',
    'master_vel': 'sensing__gnss__master__vel',
    'rover_vel': 'sensing__gnss__rover__vel',
}

# WGS-84
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)

# NavSatStatus
STATUS_NO_FIX, STATUS_FIX, STATUS_SBAS_FIX, STATUS_GBAS_FIX = -1, 0, 1, 2

# measured timing (header-stamp time base, see REPORT.md)
VEL_REF_LAG_OF_WHEEL = 0.000      # wheel(t_hdr) ~ GNSS vel |v|(t_hdr)            [s]
VEL_LAG_VS_POS = 0.047            # GNSS vel |v|(t) ~ d|p|/dt (t - 47 ms)          [s]
WHEEL_LAG_VS_POS = 0.048          # wheel(t) ~ d|p|/dt (t - 48 ms) -> position lead [s]
# offsets of receive (bag) time after header stamp, medians [s]
BAG_MINUS_HDR = dict(front=0.046, rear=0.046, cmd=0.001, master_fix=0.044, master_vel=0.080, rover_fix=0.076)

# antennas (median over bags with both receivers in RTK): rover is AHEAD of master
LEVER_ARM = {'30618': dict(ahead=12.441, left=-0.041, up=-0.005),
             '30639': dict(ahead=12.425, left=0.178, up=0.015)}
BASELINE_LEN = 12.44


# ----------------------------------------------------------------------------------------------------------
# splits / loading
# ----------------------------------------------------------------------------------------------------------
def splits() -> dict:
    with open(SPLITS_FILE, encoding='utf-8') as f:
        return json.load(f)


def list_bags(*names: str) -> list[str]:
    """Bags of the given split names (default: train + val), duplicates already excluded by splits.json."""
    s = splits()
    names = names or ('train', 'val')
    out: list[str] = []
    for n in names:
        out += [b for b in s[n] if b not in out]
    return out


@dataclass
class Bag:
    name: str
    raw: dict                       # topic key -> float64 array [t_bag, t_hdr, ...]
    t0: float = 0.0                 # first bag receive time of any message (= bag start)

    @property
    def vehicle(self) -> str:
        return self.name.split('_')[0]

    def has(self, key: str) -> bool:
        k = TOPIC.get(key, key)
        return k in self.raw and self.raw[k].shape[0] > 0

    def arr(self, key: str) -> np.ndarray:
        return self.raw[TOPIC.get(key, key)]


def load_bag(name: str) -> Bag:
    with np.load(NPZ_DIR / f'{name}.npz') as z:
        raw = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    t0 = min(a[0, 0] for a in raw.values() if a.shape[0])
    return Bag(name=name, raw=raw, t0=float(t0))


# ----------------------------------------------------------------------------------------------------------
# per-topic accessors
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Series:
    t_hdr: np.ndarray
    t_bag: np.ndarray
    v: np.ndarray                    # payload (m/s for wheels, notch for cmd)

    def t(self, base: str = 'hdr') -> np.ndarray:
        return self.t_hdr if base == 'hdr' else self.t_bag


def wheel(bag: Bag, which: str = 'front') -> Series:
    a = bag.arr(which)
    return Series(a[:, 1].copy(), a[:, 0].copy(), a[:, 2] * KMH)


def cmd(bag: Bag) -> Series:
    a = bag.arr('cmd')
    return Series(a[:, 1].copy(), a[:, 0].copy(), a[:, 2].copy())


@dataclass
class Fix:
    t_hdr: np.ndarray
    t_bag: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    status: np.ndarray

    def __len__(self):
        return len(self.t_hdr)

    def subset(self, m) -> 'Fix':
        return Fix(*(getattr(self, f)[m] for f in ('t_hdr', 't_bag', 'lat', 'lon', 'alt', 'status')))


def gnss_fix(bag: Bag, antenna: str = 'master', drop_invalid: bool = True) -> Fix:
    """NavSatFix rows in bag order.  Rows with status < 0, NaN or zero lat/lon are dropped."""
    a = bag.arr(f'{antenna}_fix') if bag.has(f'{antenna}_fix') else np.zeros((0, 9))
    if a.shape[0] == 0 or a.shape[1] < 6:
        a = np.zeros((0, 9))
    f = Fix(a[:, 1].copy(), a[:, 0].copy(), a[:, 2].copy(), a[:, 3].copy(), a[:, 4].copy(), a[:, 5].astype(int))
    if drop_invalid and len(f):
        ok = (f.status >= 0) & np.isfinite(f.lat) & np.isfinite(f.lon) & (np.abs(f.lat) > 1e-6)
        f = f.subset(ok)
    return f


@dataclass
class Vel:
    t_hdr: np.ndarray
    t_bag: np.ndarray
    ve: np.ndarray
    vn: np.ndarray
    vu: np.ndarray

    @property
    def speed(self) -> np.ndarray:           # horizontal speed (most likely reference definition)
        return np.hypot(self.ve, self.vn)

    @property
    def speed3d(self) -> np.ndarray:
        return np.sqrt(self.ve ** 2 + self.vn ** 2 + self.vu ** 2)

    @property
    def course(self) -> np.ndarray:          # course over ground, rad, ENU convention (0 = east, ccw)
        return np.arctan2(self.vn, self.ve)


def gnss_vel(bag: Bag, antenna: str = 'master') -> Vel:
    a = bag.arr(f'{antenna}_vel') if bag.has(f'{antenna}_vel') else np.zeros((0, 6))
    if a.shape[0] == 0 or a.shape[1] < 5:
        a = np.zeros((0, 6))
    return Vel(a[:, 1].copy(), a[:, 0].copy(), a[:, 2].copy(), a[:, 3].copy(), a[:, 4].copy())


# ----------------------------------------------------------------------------------------------------------
# geodesy (exact WGS-84, numpy only)
# ----------------------------------------------------------------------------------------------------------
def llh_to_ecef(lat, lon, h):
    lat = np.radians(lat); lon = np.radians(lon)
    s, c = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    x = (n + h) * c * np.cos(lon)
    y = (n + h) * c * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * s
    return x, y, z


def ecef_to_llh(x, y, z):
    """Bowring's method with two refinements (sub-mm for terrestrial points)."""
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    th = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * np.sin(th) ** 3, p - WGS84_E2 * WGS84_A * np.cos(th) ** 3)
    for _ in range(2):
        s = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def _enu_rot(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    """Exact local tangent plane (same as pymap3d.geodetic2enu / GeographicLib LocalCartesian)."""
    x, y, z = llh_to_ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(h, float))
    x0, y0, z0 = llh_to_ecef(lat0, lon0, h0)
    R = _enu_rot(lat0, lon0)
    d = np.stack([np.asarray(x) - x0, np.asarray(y) - y0, np.asarray(z) - z0])
    e, n, u = np.tensordot(R, d, axes=1)
    return e, n, u


def enu_to_llh(e, n, u, lat0, lon0, h0):
    R = _enu_rot(lat0, lon0)
    x0, y0, z0 = llh_to_ecef(lat0, lon0, h0)
    d = np.tensordot(R.T, np.stack([np.asarray(e, float), np.asarray(n, float), np.asarray(u, float)]), axes=1)
    return ecef_to_llh(d[0] + x0, d[1] + y0, d[2] + z0)


def llh_to_utm(lat, lon, epsg: int = 32637):
    """UTM (default zone 37N, EPSG:32637) via pyproj."""
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    return tr.transform(np.asarray(lon, float), np.asarray(lat, float))


def utm_convergence_scale(lat, lon, epsg: int = 32637):
    """Meridian (grid) convergence [deg] and point scale factor of the UTM projection at (lat, lon)."""
    from pyproj import Proj
    f = Proj(f'EPSG:{epsg}').get_factors(lon, lat)
    return f.meridian_convergence, f.meridional_scale


# ----------------------------------------------------------------------------------------------------------
# clocks
# ----------------------------------------------------------------------------------------------------------
def stale_mask(t_bag: np.ndarray, t_hdr: np.ndarray, thresh: float = 0.3, burst: float = 1.0) -> np.ndarray:
    """Rows of the start-up burst: received within ``burst`` s of the topic's first message but stamped
    more than ``thresh`` s earlier.

    At the start of every bag ~2.5-5 s of buffered history is delivered within the first ~0.3 s of bag time.
    Those rows are *valid* data (header stamps are right), only their receive time is compressed.
    (Mid-bag 1-s clock steps are NOT flagged here - see clock_offsets().)
    """
    if len(t_bag) == 0:
        return np.zeros(0, bool)
    return ((t_bag - t_hdr) > thresh) & (t_bag - t_bag.min() < burst)


def clock_offsets(bag: Bag, antenna: str = 'master', thresh: float = 0.3):
    """Offset of the GNSS header clock w.r.t. the vehicle header clock, evaluated at every GNSS fix.

    rel = (t_hdr - t_bag)_gnss - (t_hdr - t_bag)_cmd@same bag time.
    Normal value ~ +/- tens of ms (GNSS publication latency differences); 1-s steps and ~8 %/s chrony-like
    slews appear in a few bags.  Returns (t_hdr_gnss, rel, anomalous_mask).
    """
    f = gnss_fix(bag, antenna, drop_invalid=False)
    c = cmd(bag)
    ok_c = ~stale_mask(c.t_bag, c.t_hdr)
    off_c = np.interp(f.t_bag, c.t_bag[ok_c], (c.t_hdr - c.t_bag)[ok_c])
    rel = (f.t_hdr - f.t_bag) - off_c
    good = ~stale_mask(f.t_bag, f.t_hdr)
    med = np.median(rel[good]) if good.any() else 0.0
    anom = (np.abs(rel - med) > thresh) & good
    return f.t_hdr, rel, anom


# ----------------------------------------------------------------------------------------------------------
# reference trajectory
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Reference:
    """Most-likely jury reference: master fix -> ENU at first valid fix; speed from master/vel."""
    bag: str
    antenna: str
    time_base: str
    origin: tuple                   # (lat0, lon0, h0)
    t: np.ndarray                   # reference stamps (header by default), sorted, unique
    x: np.ndarray                   # east  [m]
    y: np.ndarray                   # north [m]
    z: np.ndarray                   # up    [m]
    status: np.ndarray
    speed: np.ndarray               # |v_EN| from <antenna>/vel interpolated at t (NaN where vel missing)
    speed_pos: np.ndarray           # central-difference speed of positions
    heading: np.ndarray             # track direction, rad ENU (0 = east, ccw), from positions (NaN at rest)
    s: np.ndarray                   # along-track distance, motion-gated cumulative horizontal path [m]
    s_naive: np.ndarray             # naive cumulative sum of |dp| (includes stand-still jitter)
    vel_t: np.ndarray = field(default=None)      # raw vel stamps/speed (reference speed samples)
    vel_speed: np.ndarray = field(default=None)

    def at(self, t):
        """Linear interpolation of x, y, z, s at arbitrary times (for diagnostics)."""
        return tuple(np.interp(t, self.t, a) for a in (self.x, self.y, self.z, self.s))


def _dedup_sorted(t: np.ndarray, *cols):
    order = np.argsort(t, kind='stable')
    t = t[order]
    keep = np.r_[True, np.diff(t) > 1e-6]
    return (t[keep],) + tuple(c[order][keep] for c in cols)


def first_valid_index(f: Fix, min_status: int = STATUS_FIX) -> int:
    ok = np.where(f.status >= min_status)[0]
    if not len(ok):
        raise ValueError('no valid fix')
    return int(ok[np.argmin(f.t_hdr[ok])])


def reference_trajectory(bag: Bag | str, antenna: str = 'master', time_base: str = 'hdr',
                         origin: tuple | None = None, min_speed_gate: float = 0.2) -> Reference:
    """Build the reference trajectory the way the jury most likely does.

    * positions: ``<antenna>/fix`` converted to exact ENU at ``origin`` (default: first valid fix by header
      stamp - note it lies ~2.5-5 s *before* the bag start because of the start-up burst);
    * speed: horizontal norm of ``<antenna>/vel`` (ENU components), interpolated to the fix stamps;
    * along-track distance ``s``: cumulative horizontal path length, accumulated only while the vehicle
      moves (GNSS |v| > ``min_speed_gate``) so stand-still jitter is not integrated.
    """
    if isinstance(bag, str):
        bag = load_bag(bag)
    f = gnss_fix(bag, antenna)
    if len(f) < 2:
        raise ValueError(f'{bag.name}: no {antenna} fixes')
    t_raw = f.t_hdr if time_base == 'hdr' else f.t_bag
    if origin is None:
        i0 = first_valid_index(f)
        origin = (float(f.lat[i0]), float(f.lon[i0]), float(f.alt[i0]))
    t, lat, lon, alt, st = _dedup_sorted(t_raw, f.lat, f.lon, f.alt, f.status)
    x, y, z = llh_to_enu(lat, lon, alt, *origin)

    v = gnss_vel(bag, antenna)
    tv = v.t_hdr if time_base == 'hdr' else v.t_bag
    tv, sp = _dedup_sorted(tv, v.speed)
    speed = np.interp(t, tv, sp, left=np.nan, right=np.nan) if len(tv) > 1 else np.full_like(t, np.nan)
    # blank interpolation across vel gaps > 0.3 s
    if len(tv) > 1:
        k = np.clip(np.searchsorted(tv, t), 1, len(tv) - 1)
        gap = (tv[k] - tv[k - 1]) > 0.3
        speed[gap] = np.nan

    # central-difference position speed / heading
    dx = np.gradient(x, t); dy = np.gradient(y, t)
    speed_pos = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx)
    moving = np.where(np.isfinite(speed), speed, speed_pos) > min_speed_gate
    heading[~moving] = np.nan

    step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    s_naive = np.cumsum(step)
    mv_step = np.r_[False, moving[1:] | moving[:-1]]
    s = np.cumsum(np.where(mv_step, step, 0.0))
    return Reference(bag.name, antenna, time_base, origin, t, x, y, z, st, speed, speed_pos, heading, s,
                     s_naive, tv, sp)


# ----------------------------------------------------------------------------------------------------------
# delay estimation
# ----------------------------------------------------------------------------------------------------------
def _gap_ok(t_src: np.ndarray, tq: np.ndarray, max_gap: float) -> np.ndarray:
    k = np.clip(np.searchsorted(t_src, tq), 1, len(t_src) - 1)
    return ((t_src[k] - t_src[k - 1]) <= max_gap) & (tq >= t_src[0]) & (tq <= t_src[-1])


def lag_cost(t_ref, v_ref, t_sig, v_sig, lag, min_speed=0.3, max_gap=0.25, trim=None, weights=None):
    """Robust LS cost of  v_ref(t) ~ k * v_sig(t + lag)  evaluated at the reference samples.

    lag > 0  <=>  the signal is *late*: what happened at reference time t shows up in the signal at t + lag.
    Returns (cost_per_sample, k, n_used, residuals, used_mask).
    """
    tq = t_ref + lag
    ok = _gap_ok(t_sig, tq, max_gap) & np.isfinite(v_ref)
    vs = np.interp(tq, t_sig, v_sig)
    ok &= (v_ref > min_speed) | (vs > min_speed)
    if weights is not None:
        ok &= weights
    if ok.sum() < 20:
        return np.inf, np.nan, 0, None, ok
    a, b = vs[ok], v_ref[ok]
    k = np.dot(a, b) / np.dot(a, a)
    r = b - k * a
    if trim is not None:
        keep = np.abs(r) < trim
        a, b = a[keep], b[keep]
        k = np.dot(a, b) / np.dot(a, a)
        r = b - k * a
        used = ok.copy(); used[np.where(ok)[0][~keep]] = False
    else:
        used = ok
    return float(np.mean(r ** 2)), float(k), int(len(r)), r, used


def estimate_lag(t_ref, v_ref, t_sig, v_sig, lo=-0.6, hi=1.2, coarse=0.005, min_speed=0.3, max_gap=0.25,
                 trim_sigma=4.0, mask=None):
    """Delay of ``v_sig`` w.r.t. ``v_ref`` (seconds, sub-ms via parabolic refinement).

    1. coarse grid search of the robust LS cost (scale factor re-fitted for every lag);
    2. trim outliers (slip/slide, dropouts) at trim_sigma * 1.4826 * MAD of the residuals at the best lag;
    3. re-scan +/-3 coarse steps on a 0.5 ms grid and fit a parabola to the minimum.
    Returns dict(lag, k, rms, n, cost_curve=(lags, costs)).
    """
    t_ref = np.asarray(t_ref, float); v_ref = np.asarray(v_ref, float)
    lags = np.arange(lo, hi + 1e-12, coarse)
    if len(t_ref) < 30 or len(t_sig) < 30:
        return dict(lag=np.nan, k=np.nan, rms=np.nan, n=0, curve=(lags, np.full_like(lags, np.nan)))
    c0 = np.array([lag_cost(t_ref, v_ref, t_sig, v_sig, L, min_speed, max_gap, weights=mask)[0] for L in lags])
    if not np.isfinite(c0).any():
        return dict(lag=np.nan, k=np.nan, rms=np.nan, n=0, curve=(lags, c0))
    L0 = lags[np.nanargmin(c0)]
    _, _, _, r, _ = lag_cost(t_ref, v_ref, t_sig, v_sig, L0, min_speed, max_gap, weights=mask)
    mad = np.median(np.abs(r - np.median(r))) * 1.4826
    trim = max(trim_sigma * mad, 0.05)
    fine = np.arange(L0 - 3 * coarse, L0 + 3 * coarse + 1e-12, 0.0005)
    # fix the trimmed sample set at L0 so that the cost curve is smooth in the lag
    _, _, _, _, used = lag_cost(t_ref, v_ref, t_sig, v_sig, L0, min_speed, max_gap, trim=trim, weights=mask)
    cf = np.array([lag_cost(t_ref, v_ref, t_sig, v_sig, L, min_speed, max_gap, weights=used)[0] for L in fine])
    i = int(np.nanargmin(cf))
    i = min(max(i, 2), len(fine) - 3)
    p = np.polyfit(fine[i - 2:i + 3] - fine[i], cf[i - 2:i + 3], 2)
    lag = fine[i] - p[1] / (2 * p[0]) if p[0] > 0 else fine[i]
    cost, k, n, r, _ = lag_cost(t_ref, v_ref, t_sig, v_sig, lag, min_speed, max_gap, weights=used)
    return dict(lag=float(lag), k=k, rms=float(np.sqrt(cost)), n=n, curve=(lags, c0), trim=trim)


# ----------------------------------------------------------------------------------------------------------
# judge emulation helpers
# ----------------------------------------------------------------------------------------------------------
def match_nearest(t_ref: np.ndarray, t_est: np.ndarray, tol: float = 0.05):
    """For every reference stamp find the nearest estimate stamp; returns (i_ref, i_est, dt) of pairs
    with |dt| <= tol (dt = t_est - t_ref).  ``t_est`` must be sorted."""
    k = np.clip(np.searchsorted(t_est, t_ref), 1, len(t_est) - 1)
    left, right = t_est[k - 1], t_est[k]
    j = np.where(np.abs(t_ref - left) <= np.abs(right - t_ref), k - 1, k)
    dt = t_est[j] - t_ref
    ok = np.abs(dt) <= tol
    return np.where(ok)[0], j[ok], dt[ok]


def filled_heading(ref: Reference) -> np.ndarray:
    """Reference heading with stand-still gaps filled by the last (or, before the first motion, next) value."""
    hd = ref.heading.copy()
    fin = np.isfinite(hd)
    if not fin.any():
        return np.zeros_like(hd)
    idx = np.where(fin, np.arange(len(hd)), 0)
    np.maximum.accumulate(idx, out=idx)
    hd = hd[idx]
    hd[:np.argmax(fin)] = ref.heading[np.argmax(fin)]
    return hd


def along_cross_errors(ref: Reference, t_est, x_est, y_est, tol: float = 0.05):
    """Along-/cross-track errors of an estimate w.r.t. the reference heading at the paired stamps."""
    t_est = np.asarray(t_est, float)
    order = np.argsort(t_est)
    t_est, x_est, y_est = t_est[order], np.asarray(x_est)[order], np.asarray(y_est)[order]
    ir, ie, dt = match_nearest(ref.t, t_est, tol)
    hd = filled_heading(ref)
    ex, ey = x_est[ie] - ref.x[ir], y_est[ie] - ref.y[ir]
    c, s = np.cos(hd[ir]), np.sin(hd[ir])
    return dict(t=ref.t[ir], along=ex * c + ey * s, cross=-ex * s + ey * c, dt=dt)


def cross_track_to_polyline(ref: Reference, ir: np.ndarray, px: np.ndarray, py: np.ndarray,
                            window_m: float = 300.0, step_m: float = 1.0):
    """Distance of points (px, py), paired with reference rows ir, to the reference polyline (a stand-in for
    the organisers' 'pathgraph' map), searching segments within +/- window_m of arc length around the paired
    reference point.  The polyline is resampled every step_m metres of along-track distance, so long stops
    do not shrink the search window."""
    s_u, iu = np.unique(ref.s, return_index=True)
    if len(s_u) < 2:
        return np.hypot(px - ref.x[ir], py - ref.y[ir])
    sg = np.arange(s_u[0], s_u[-1] + step_m, step_m)
    gx, gy = np.interp(sg, s_u, ref.x[iu]), np.interp(sg, s_u, ref.y[iu])
    n = len(sg)
    k0 = np.clip(np.searchsorted(sg, ref.s[ir]), 0, n - 1)
    W = int(np.ceil(window_m / step_m))
    best = np.full(len(ir), np.inf)
    for k in range(-W, W):
        j = np.clip(k0 + k, 0, n - 2)
        ax, ay = gx[j], gy[j]
        bx, by = gx[j + 1] - ax, gy[j + 1] - ay
        L2 = bx * bx + by * by
        u = np.clip(np.where(L2 > 1e-12, ((px - ax) * bx + (py - ay) * by) / np.maximum(L2, 1e-12), 0.0), 0.0, 1.0)
        np.minimum(best, np.hypot(px - ax - u * bx, py - ay - u * by), out=best)
    return best


def reference_quality_mask(bag: Bag, ref: Reference, max_innov: float = 0.3, clock_margin: float = 3.0):
    """Epochs (rows of ref.t) where the reference is trustworthy - for OUR validation only (the jury will
    use everything).  Excludes: status-0 fixes (if the run is mostly RTK), fixes whose step disagrees with
    the GNSS velocity by > max_innov m, and +/- clock_margin s around GNSS-vs-vehicle 1-s clock anomalies.
    Returns (mask_pos over ref.t, mask_vel over ref.vel_t)."""
    ok = np.ones(len(ref.t), bool)
    if np.mean(ref.status == STATUS_GBAS_FIX) > 0.5:
        ok &= ref.status == STATUS_GBAS_FIX
    v = gnss_vel(bag)
    tv, ve, vn = _dedup_sorted(v.t_hdr, v.ve, v.vn)
    if len(tv) > 1:
        vx, vy = np.interp(ref.t, tv, ve), np.interp(ref.t, tv, vn)
        dt = np.diff(ref.t)
        inn = np.hypot(np.diff(ref.x) - 0.5 * (vx[1:] + vx[:-1]) * dt, np.diff(ref.y) - 0.5 * (vy[1:] + vy[:-1]) * dt)
        bad = (inn > max_innov) & (dt < 0.3)
        ok &= ~np.r_[bad, False] & ~np.r_[False, bad]
    tg, rel, anom = clock_offsets(bag)
    okv = np.ones(len(ref.vel_t), bool)
    if anom.any():
        ta = np.sort(tg[anom])
        for arr, m in ((ref.t, ok), (ref.vel_t, okv)):
            k = np.clip(np.searchsorted(ta, arr), 1, max(len(ta) - 1, 1))
            d = np.minimum(np.abs(arr - ta[np.clip(k - 1, 0, len(ta) - 1)]), np.abs(arr - ta[np.clip(k, 0, len(ta) - 1)]))
            m &= d > clock_margin
    return ok, okv


def reference_grade(bag: Bag, ref: Reference | None = None) -> dict:
    """Run-level quality of the GNSS position reference (for choosing validation runs).

    frac_bad   : share of 10-Hz steps whose |dp - v dt| > 0.3 m (flip-flopping / jumping fixes)
    path_ratio : motion-gated fix path length / integral of the GNSS-vel speed (1.00 for clean runs)
    grade      : 'A' clean (frac_bad < 0.5 %, path_ratio < 1.02), 'B' usable with reference_quality_mask
                 (frac_bad < 2 %, path_ratio < 1.10), 'C' position reference unusable (speed still fine).
    On the 69 train/val runs with >= 600 fixes: 34 A, 21 B, 14 C (reference_grades.csv).
    """
    ref = ref or reference_trajectory(bag)
    v = gnss_vel(bag)
    tv, ve, vn = _dedup_sorted(v.t_hdr, v.ve, v.vn)
    vx, vy = np.interp(ref.t, tv, ve), np.interp(ref.t, tv, vn)
    dt = np.diff(ref.t)
    inn = np.hypot(np.diff(ref.x) - 0.5 * (vx[1:] + vx[:-1]) * dt, np.diff(ref.y) - 0.5 * (vy[1:] + vy[:-1]) * dt)
    ok = dt < 0.15
    frac_bad = float(np.mean(inn[ok] > 0.3)) if ok.any() else np.nan
    path_vel = float(np.trapezoid(np.hypot(vx, vy), ref.t))
    ratio = float(ref.s[-1] / path_vel) if path_vel > 10 else np.nan
    if frac_bad < 0.005 and ratio < 1.02:
        grade = 'A'
    elif frac_bad < 0.02 and ratio < 1.10:
        grade = 'B'
    else:
        grade = 'C'
    return dict(status2_frac=float(np.mean(ref.status == STATUS_GBAS_FIX)), frac_bad=frac_bad, path_ratio=ratio,
                grade=grade)


# ----------------------------------------------------------------------------------------------------------
# initial alignment (what the node can see) and jury-metric emulation
# ----------------------------------------------------------------------------------------------------------
def init_alignment(bag: Bag | str, gnss_bag_seconds: float = 2.0, antenna: str = 'master') -> dict:
    """Emulate the test condition 'GNSS only in the first few seconds' by keeping GNSS rows with
    bag time < bag start + gnss_bag_seconds (this includes the ~2.5-5 s start-up burst of history).

    Returns the judge-frame origin (first fix by header stamp - use it EXACTLY, even if it is a poor fix),
    a robust initial position in that frame, the dual-antenna heading (valid at stand-still), and flags.
    """
    if isinstance(bag, str):
        bag = load_bag(bag)
    fm = gnss_fix(bag, antenna)
    fr = gnss_fix(bag, 'rover' if antenna == 'master' else 'master')
    keep = fm.t_bag < bag.t0 + gnss_bag_seconds
    if keep.sum() == 0:
        return dict(ok=False)
    fmk = fm.subset(keep)
    i0 = first_valid_index(fmk)
    lat0, lon0, h0 = float(fmk.lat[i0]), float(fmk.lon[i0]), float(fmk.alt[i0])
    x, y, z = llh_to_enu(fmk.lat, fmk.lon, fmk.alt, lat0, lon0, h0)
    good = fmk.status == STATUS_GBAS_FIX
    use = good if good.sum() >= 5 else np.ones(len(x), bool)
    p0 = (float(np.median(x[use])), float(np.median(y[use])), float(np.median(z[use])))
    spread = float(np.sqrt(np.median((x[use] - p0[0]) ** 2 + (y[use] - p0[1]) ** 2)))
    out = dict(ok=True, origin=(lat0, lon0, h0), origin_status=int(fmk.status[i0]), p0_enu=p0,
               p0_spread_m=spread, n_fix=int(keep.sum()), n_status2=int(good.sum()),
               t_first_fix_hdr=float(fmk.t_hdr[i0]), t_last_fix_hdr=float(fmk.t_hdr.max()))
    # dual-antenna heading
    krk = fr.t_bag < bag.t0 + gnss_bag_seconds
    if krk.sum() >= 3:
        frk = fr.subset(krk)
        km = np.round(fmk.t_hdr * 10).astype(np.int64); kr = np.round(frk.t_hdr * 10).astype(np.int64)
        _, im, ir = np.intersect1d(km, kr, return_indices=True)
        xr, yr, _ = llh_to_enu(frk.lat[ir], frk.lon[ir], frk.alt[ir], lat0, lon0, h0)
        bx, by = xr - x[im], yr - y[im]
        if antenna != 'master':
            bx, by = -bx, -by
        L = np.hypot(bx, by)
        okb = np.abs(L - BASELINE_LEN) < 0.3
        if okb.sum() >= 3:
            hb = np.arctan2(by[okb], bx[okb])
            h = float(np.angle(np.mean(np.exp(1j * hb))))
            out.update(heading_enu=h, heading_std_deg=float(np.degrees(np.std((hb - h + np.pi) % (2 * np.pi) - np.pi))),
                       n_heading_pairs=int(okb.sum()))
    v = gnss_vel(bag, antenna)
    kv = v.t_bag < bag.t0 + gnss_bag_seconds
    out['gnss_speed_max'] = float(v.speed[kv].max()) if kv.any() else np.nan
    f = wheel(bag, 'front')
    out['wheel_speed_max'] = float(f.v[f.t_bag < bag.t0 + gnss_bag_seconds].max())
    out['stationary'] = bool(out['wheel_speed_max'] < 0.05 and (not np.isfinite(out['gnss_speed_max']) or out['gnss_speed_max'] < 0.2))
    return out


def judge_metrics(ref: Reference, t_out, v_out=None, x_out=None, y_out=None, z_out=None, tol: float = 0.05,
                  mask_pos=None, mask_vel=None, polyline_window: float = 300.0) -> dict:
    """Most-likely jury metrics.  Reference samples are paired with the NEAREST output stamp (|dt| <= tol).

    speed    : reference = GNSS master vel |v| at its header stamps (ref.vel_t / ref.vel_speed):
               rmse, mae, bias, max_abs; the same on accel (a > +0.3 m/s^2), brake (a < -0.3) and stand-still
               (|v_ref| < 0.1) subsets; match fraction.
    position : rmse_x/y/z, rmse_3d, mean_3d, max_3d; along-track error (projection on reference heading)
               mean/max_abs/rmse; cross-track distance to the reference polyline (map stand-in) mean/max/rmse;
               end drift = |3D error at last paired epoch| / reference path length * 100 [%].
    mask_pos / mask_vel optionally drop reference epochs (see reference_quality_mask) - NOT done by the jury.
    """
    t_out = np.asarray(t_out, float)
    o = np.argsort(t_out, kind='stable')
    t_out = t_out[o]
    res = {}
    if v_out is not None:
        v_out = np.asarray(v_out, float)[o]
        tv, sv = ref.vel_t, ref.vel_speed
        m = np.ones(len(tv), bool) if mask_vel is None else np.asarray(mask_vel, bool)
        ir, ie, dt = match_nearest(tv, t_out, tol)
        keep = m[ir]
        ir, ie = ir[keep], ie[keep]
        e = v_out[ie] - sv[ir]
        acc = np.gradient(sv, tv)
        acc = np.convolve(acc, np.ones(5) / 5, 'same')
        res['speed_match_frac'] = float(len(ir) / max(m.sum(), 1))
        res['speed_rmse'] = float(np.sqrt(np.mean(e ** 2)))
        res['speed_mae'] = float(np.mean(np.abs(e)))
        res['speed_bias'] = float(np.mean(e))
        res['speed_max_abs'] = float(np.max(np.abs(e)))
        for lab, sel in (('accel', acc[ir] > 0.3), ('brake', acc[ir] < -0.3), ('still', sv[ir] < 0.1)):
            res[f'speed_bias_{lab}'] = float(np.mean(e[sel])) if sel.any() else np.nan
            res[f'speed_rmse_{lab}'] = float(np.sqrt(np.mean(e[sel] ** 2))) if sel.any() else np.nan
    if x_out is not None:
        x_out, y_out = np.asarray(x_out, float)[o], np.asarray(y_out, float)[o]
        z_out = np.zeros_like(x_out) if z_out is None else np.asarray(z_out, float)[o]
        m = np.ones(len(ref.t), bool) if mask_pos is None else np.asarray(mask_pos, bool)
        ir, ie, dt = match_nearest(ref.t, t_out, tol)
        keep = m[ir]
        ir, ie = ir[keep], ie[keep]
        ex, ey, ez = x_out[ie] - ref.x[ir], y_out[ie] - ref.y[ir], z_out[ie] - ref.z[ir]
        e3 = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
        hd = filled_heading(ref)[ir]
        along = ex * np.cos(hd) + ey * np.sin(hd)
        cross_h = -ex * np.sin(hd) + ey * np.cos(hd)
        cross_p = cross_track_to_polyline(ref, ir, x_out[ie], y_out[ie], polyline_window)
        path = float(ref.s[-1]) if ref.s[-1] > 1 else np.nan
        res.update(pos_match_frac=float(len(ir) / max(m.sum(), 1)),
                   rmse_x=float(np.sqrt(np.mean(ex ** 2))), rmse_y=float(np.sqrt(np.mean(ey ** 2))),
                   rmse_z=float(np.sqrt(np.mean(ez ** 2))), rmse_3d=float(np.sqrt(np.mean(e3 ** 2))),
                   mean_3d=float(np.mean(e3)), max_3d=float(np.max(e3)),
                   along_mean=float(np.mean(along)), along_max_abs=float(np.max(np.abs(along))),
                   along_rmse=float(np.sqrt(np.mean(along ** 2))),
                   cross_heading_rmse=float(np.sqrt(np.mean(cross_h ** 2))),
                   cross_poly_mean=float(np.mean(cross_p)), cross_poly_max=float(np.max(cross_p)),
                   cross_poly_rmse=float(np.sqrt(np.mean(cross_p ** 2))),
                   end_error_3d=float(e3[-1]), end_error_2d=float(np.hypot(ex[-1], ey[-1])),
                   path_length=path, end_drift_pct=float(100 * e3[-1] / path) if path == path else np.nan,
                   end_drift_2d_pct=float(100 * np.hypot(ex[-1], ey[-1]) / path) if path == path else np.nan)
    return res


# ----------------------------------------------------------------------------------------------------------
# CLI: python timing.py <bag> [gnss_seconds]
# ----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else '30618_e2dcf65f'
    gs = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    b = load_bag(name)
    r = reference_trajectory(b)
    mp, mv = reference_quality_mask(b, r)
    print(f'{name}: {len(r.t)} master fixes, first fix {r.t[0] - b.t0:+.2f} s rel. bag start, origin {r.origin}')
    print(f'  ENU extent x [{r.x.min():.1f}, {r.x.max():.1f}]  y [{r.y.min():.1f}, {r.y.max():.1f}]  z [{r.z.min():.1f}, {r.z.max():.1f}] m')
    print(f'  path (motion-gated) {r.s[-1]:.1f} m, naive {r.s_naive[-1]:.1f} m; status-2 {np.mean(r.status == 2):.3f}; '
          f'trustworthy epochs {mp.mean():.3f}')
    tg, rel, an = clock_offsets(b)
    print(f'  GNSS-vs-vehicle clock anomalies: {an.sum()} epochs')
    ia = init_alignment(b, gs)
    print(f'  init ({gs:.0f} s of GNSS): n_fix {ia.get("n_fix")}, status2 {ia.get("n_status2")}, stationary {ia.get("stationary")}, '
          f'p0 {np.round(ia.get("p0_enu"), 3)}, heading {np.degrees(ia.get("heading_enu", np.nan)):.2f} deg '
          f'(std {ia.get("heading_std_deg", np.nan):.3f})')
