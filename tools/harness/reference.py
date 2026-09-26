"""Ground-truth reference built from GNSS: local metric frame, reference path, speed, regimes.

Main entry point: :func:`build_reference` -> :class:`Reference`.

Frames (``RefConfig.frame``):
  * ``'enu'`` - local East-North-Up tangent plane at the origin (exact WGS84, ECEF based).
  * ``'utm'`` - UTM (zone from origin longitude, 37N here) easting/northing minus origin,
    z = alt - alt0. NOTE: the UTM grid is rotated ~1.3 deg w.r.t. true north near Moscow and
    scaled by ~0.9997, so 'enu' and 'utm' positions differ by up to ~110 m at 5 km from the
    origin. The judge's frame is unknown -> the estimator must use the SAME frame kind.

Origin (``RefConfig.origin``): ``'first_fix'`` = first master fix (bag order) with
status >= min_status and finite lat/lon; or an explicit ``(lat, lon, alt)`` tuple.

Time base (``RefConfig.time_base``) used to timestamp reference samples:
  * ``'header'`` - GNSS ``header.stamp`` (measurement time; has occasional +-1 s glitches),
  * ``'bag'`` - bag receive time (what ``ros2 bag play --clock`` would give),
  * ``'header_fixed'`` - header stamps with integer-second glitches repaired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .loader import BagData, T_BAG, T_HDR, LAT, LON, ALT, STATUS, VX, VY, VZ

# ----------------------------------------------------------------------------------------------
# Geodesy
# ----------------------------------------------------------------------------------------------
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lat, lon, alt):
    lat = np.radians(np.asarray(lat, float))
    lon = np.radians(np.asarray(lon, float))
    alt = np.asarray(alt, float)
    s, c = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    x = (n + alt) * c * np.cos(lon)
    y = (n + alt) * c * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * s
    return x, y, z


def ecef_to_geodetic(x, y, z, iters: int = 5):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(iters):
        s = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


class LocalFrame:
    """Geodetic <-> local metric frame. ``kind`` in {'enu', 'utm'}; origin = (lat0, lon0, alt0)."""

    def __init__(self, kind: str, lat0: float, lon0: float, alt0: float):
        if kind not in ('enu', 'utm'):
            raise ValueError(f'unknown frame kind {kind!r}')
        self.kind, self.lat0, self.lon0, self.alt0 = kind, float(lat0), float(lon0), float(alt0)
        if kind == 'enu':
            la, lo = np.radians(self.lat0), np.radians(self.lon0)
            self._R = np.array([[-np.sin(lo), np.cos(lo), 0.0],
                                [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                                [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
            self._o = np.array(geodetic_to_ecef(self.lat0, self.lon0, self.alt0))
        else:
            from pyproj import Transformer
            zone = int((self.lon0 + 180.0) // 6) + 1
            epsg = (32600 if self.lat0 >= 0 else 32700) + zone
            self.epsg = epsg
            self._fwd = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
            self._inv = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
            self._e0, self._n0 = self._fwd.transform(self.lon0, self.lat0)

    @classmethod
    def from_fix(cls, kind: str, lat: float, lon: float, alt: float) -> 'LocalFrame':
        return cls(kind, lat, lon, alt)

    def forward(self, lat, lon, alt) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """lat/lon [deg], alt [m] -> x (east-ish), y (north-ish), z (up) [m]."""
        if self.kind == 'enu':
            x, y, z = geodetic_to_ecef(lat, lon, alt)
            d = np.stack([np.asarray(x) - self._o[0], np.asarray(y) - self._o[1], np.asarray(z) - self._o[2]])
            e = np.tensordot(self._R, d, axes=1)
            return e[0], e[1], e[2]
        e, n = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(e) - self._e0, np.asarray(n) - self._n0, np.asarray(alt, float) - self.alt0

    def inverse(self, x, y, z):
        """Local x, y, z -> lat, lon, alt."""
        if self.kind == 'enu':
            d = np.tensordot(self._R.T, np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)]), axes=1)
            return ecef_to_geodetic(d[0] + self._o[0], d[1] + self._o[1], d[2] + self._o[2])
        lon, lat = self._inv.transform(np.asarray(x, float) + self._e0, np.asarray(y, float) + self._n0)
        return np.asarray(lat), np.asarray(lon), np.asarray(z, float) + self.alt0

    def enu_rotation(self) -> float:
        """Angle [rad] to rotate an ENU vector (e.g. GNSS vel) into this frame's x/y axes."""
        if self.kind == 'enu':
            return 0.0
        dlat = 1e-3
        x0, y0, _ = self.forward(self.lat0, self.lon0, self.alt0)
        x1, y1, _ = self.forward(self.lat0 + dlat, self.lon0, self.alt0)
        # true north in grid coords points at angle atan2(dy, dx); in ENU it is +90 deg
        return float(np.arctan2(y1 - y0, x1 - x0) - np.pi / 2)

    def rotate_enu(self, vx, vy):
        a = self.enu_rotation()
        c, s = np.cos(a), np.sin(a)
        return c * np.asarray(vx) - s * np.asarray(vy), s * np.asarray(vx) + c * np.asarray(vy)

    def __repr__(self):
        return f'LocalFrame({self.kind!r}, {self.lat0:.8f}, {self.lon0:.8f}, {self.alt0:.3f})'


# ----------------------------------------------------------------------------------------------
# Time base
# ----------------------------------------------------------------------------------------------
def repair_header_glitches(t_bag: np.ndarray, t_hdr: np.ndarray, skip_start_s: float = 5.0) -> Tuple[np.ndarray, int]:
    """Remove integer-second jumps of ``header.stamp`` relative to bag time.

    Header stamps (GNSS and, more rarely, wheels) sometimes jump by exactly +-1 s for up to
    minutes. The latency ``t_hdr - t_bag`` is otherwise nearly constant (GNSS: -30..-90 ms;
    wheels drift slowly between -80 and -10 ms), so deviations from the bag-wide median that
    are within 0.15 s of a non-zero integer are subtracted. The first ``skip_start_s`` seconds
    (bag-start burst of buffered messages, header ~2.5 s older than bag time, header correct)
    are left untouched. Returns (fixed stamps, number of repaired samples).
    """
    off = t_hdr - t_bag
    if len(off) < 5:
        return t_hdr.copy(), 0
    body = t_bag - t_bag[0] > skip_start_s
    med = np.median(off[body]) if np.any(body) else np.median(off)
    dev = off - med
    k = np.round(dev)
    fix = body & (k != 0) & (np.abs(dev - k) < 0.15)
    out = t_hdr.copy()
    out[fix] -= k[fix]
    return out, int(fix.sum())


def time_vector(arr: np.ndarray, base: str) -> np.ndarray:
    if base == 'bag':
        return arr[:, T_BAG].copy()
    if base == 'header':
        return arr[:, T_HDR].copy()
    if base == 'header_fixed':
        return repair_header_glitches(arr[:, T_BAG], arr[:, T_HDR])[0]
    raise ValueError(f'unknown time base {base!r}')


# ----------------------------------------------------------------------------------------------
# Track path (polyline with arc length)
# ----------------------------------------------------------------------------------------------
_KERNEL = None              # numba kernel, compiled lazily (numba import is slow on loaded boxes)
NUMBA_MIN_CELLS = 8_000_000  # use numba only when the brute-force work is at least this large


def _numba_kernel():
    """Return the jitted projection kernel, or False if numba is unavailable."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    try:
        from numba import njit
    except ImportError:  # pragma: no cover
        _KERNEL = False
        return _KERNEL

    @njit(cache=True)
    def kernel(qx, qy, lo, hi, x0, y0, dx, dy, L, S):  # pragma: no cover (jit)
        n = qx.shape[0]
        s_out = np.empty(n)
        lat = np.empty(n)
        dist = np.empty(n)
        for i in range(n):
            best = 1e300
            bs = 0.0
            bc = 0.0
            for j in range(lo[i], hi[i] + 1):
                rx = qx[i] - x0[j]
                ry = qy[i] - y0[j]
                t = rx * dx[j] + ry * dy[j]
                if t < 0.0:
                    t = 0.0
                elif t > L[j]:
                    t = L[j]
                fx = rx - dx[j] * t
                fy = ry - dy[j] * t
                d2 = fx * fx + fy * fy
                if d2 < best:
                    best = d2
                    bs = S[j] + t
                    bc = dx[j] * ry - dy[j] * rx
            d = np.sqrt(best)
            s_out[i] = bs
            dist[i] = d
            lat[i] = d if bc >= 0.0 else -d
        return s_out, lat, dist

    _KERNEL = kernel
    return _KERNEL


class TrackPath:
    """2D polyline with cumulative arc length ``S`` and per-vertex altitude ``z``."""

    def __init__(self, xy: np.ndarray, z: Optional[np.ndarray] = None):
        xy = np.asarray(xy, float)
        z = np.zeros(len(xy)) if z is None else np.asarray(z, float)
        if len(xy) >= 2:
            L = np.hypot(*np.diff(xy, axis=0).T)
            keep = np.r_[True, L > 1e-6]
            xy, z = xy[keep], z[keep]
        if len(xy) < 2:  # degenerate (never moved): make a tiny segment so that queries work
            p = xy[0] if len(xy) else np.zeros(2)
            xy = np.vstack([p, p + [1e-3, 0.0]])
            z = np.r_[z[:1], z[:1]] if len(z) else np.zeros(2)
        self.xy, self.z = xy, z
        seg = np.diff(xy, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.seg_dir = seg / self.seg_len[:, None]
        self.S = np.r_[0.0, np.cumsum(self.seg_len)]
        # contiguous copies for the numba kernel
        self._x0 = np.ascontiguousarray(xy[:-1, 0])
        self._y0 = np.ascontiguousarray(xy[:-1, 1])
        self._dx = np.ascontiguousarray(self.seg_dir[:, 0])
        self._dy = np.ascontiguousarray(self.seg_dir[:, 1])

    @property
    def length(self) -> float:
        return float(self.S[-1])

    def _seg_index(self, s):
        return np.clip(np.searchsorted(self.S, s, side='right') - 1, 0, len(self.seg_len) - 1)

    def interp(self, s) -> Tuple[np.ndarray, np.ndarray]:
        """Points (N,2) and altitudes (N,) at arc length(s) ``s`` (clamped to [0, length])."""
        s = np.clip(np.atleast_1d(np.asarray(s, float)), 0.0, self.length)
        j = self._seg_index(s)
        u = s - self.S[j]
        p = self.xy[j] + self.seg_dir[j] * u[:, None]
        z = self.z[j] + (self.z[j + 1] - self.z[j]) * (u / self.seg_len[j])
        return p, z

    def interp_scalar(self, s: float):
        """Fast scalar version of :meth:`interp` -> (x, y, z) floats (for per-message use)."""
        import bisect
        if not hasattr(self, '_Sl'):
            self._Sl = self.S.tolist()
            self._xyl = self.xy.tolist()
            self._zl = self.z.tolist()
            self._dl = self.seg_dir.tolist()
            self._Ll = self.seg_len.tolist()
        L = self._Sl[-1]
        s = 0.0 if s < 0.0 else (L if s > L else s)
        j = min(max(bisect.bisect_right(self._Sl, s) - 1, 0), len(self._Ll) - 1)
        u = s - self._Sl[j]
        x0, y0 = self._xyl[j]
        dx, dy = self._dl[j]
        z0, z1 = self._zl[j], self._zl[j + 1]
        return x0 + dx * u, y0 + dy * u, z0 + (z1 - z0) * (u / self._Ll[j])

    def heading_scalar(self, s: float, half_chord: float = 5.0) -> float:
        """Heading [rad] of the path at arc length ``s`` (chord of +-half_chord m)."""
        x0, y0, _ = self.interp_scalar(s - half_chord)
        x1, y1, _ = self.interp_scalar(s + half_chord)
        import math
        return math.atan2(y1 - y0, x1 - x0)

    def tangent(self, s, half_chord: float = 5.0) -> np.ndarray:
        """Unit tangent (N,2) at ``s`` estimated from a chord of +-half_chord metres."""
        s = np.atleast_1d(np.asarray(s, float))
        a, _ = self.interp(s - half_chord)
        b, _ = self.interp(s + half_chord)
        d = b - a
        n = np.hypot(d[:, 0], d[:, 1])
        bad = n < 1e-6
        if np.any(bad):
            d[bad] = self.seg_dir[self._seg_index(s[bad])]
            n[bad] = 1.0
        return d / n[:, None]

    def project(self, q: np.ndarray, s_hint=None, window=np.inf, max_cells: int = 1_000_000):
        """Project points ``q`` (N,2) onto the polyline.

        Only segments whose arc-length range intersects ``[s_hint - window, s_hint + window]``
        are searched (``window`` scalar or (N,)); ``s_hint=None`` searches the whole path.
        Returns ``(s, lateral, dist)``: arc length of the foot point, signed lateral offset
        (+ = left of the path direction) and Euclidean distance to the foot point.
        """
        q = np.atleast_2d(np.asarray(q, float))
        N, K = len(q), len(self.seg_len)
        if s_hint is None:
            lo = np.zeros(N, np.int64)
            hi = np.full(N, K - 1, np.int64)
        else:
            s_hint = np.broadcast_to(np.asarray(s_hint, float), (N,))
            w = np.broadcast_to(np.asarray(window, float), (N,))
            lo = np.clip(np.searchsorted(self.S, s_hint - w, side='right') - 2, 0, K - 1).astype(np.int64)
            hi = np.clip(np.searchsorted(self.S, s_hint + w, side='left'), 0, K - 1).astype(np.int64)
            hi = np.maximum(hi, lo)
        work = int(np.sum(hi - lo + 1))
        if work >= NUMBA_MIN_CELLS:
            kern = _numba_kernel()
            if kern:
                return kern(np.ascontiguousarray(q[:, 0]), np.ascontiguousarray(q[:, 1]), lo, hi,
                            self._x0, self._y0, self._dx, self._dy, self.seg_len, self.S)
        return self._project_numpy(q, lo, hi, max_cells)

    def _project_numpy(self, q, lo, hi, max_cells):
        """Chunked brute force over the candidate segments (pure numpy fallback)."""
        N, K = len(q), len(self.seg_len)
        qx, qy = q[:, 0], q[:, 1]
        s_out = np.empty(N)
        lat_out = np.empty(N)
        d_out = np.empty(N)
        width = hi - lo + 1
        order = np.argsort(width, kind='stable')  # group similar widths -> less padding
        i = 0
        while i < N:
            n_chunk = max(1, max_cells // int(width[order[i]]))
            idx = order[i:i + n_chunk]
            wmax = int(width[idx].max())
            if wmax * len(idx) > 2 * max_cells:          # widths grew inside the chunk: shrink
                idx = order[i:i + max(1, max_cells // wmax)]
                wmax = int(width[idx].max())
            J = lo[idx, None] + np.arange(wmax)[None, :]
            invalid = J > hi[idx, None]
            np.minimum(J, K - 1, out=J)
            dx = self._dx[J]
            dy = self._dy[J]
            rx = qx[idx, None] - self._x0[J]
            ry = qy[idx, None] - self._y0[J]
            t = rx * dx
            t += ry * dy
            np.clip(t, 0.0, self.seg_len[J], out=t)
            fx = rx - dx * t
            fy = ry - dy * t
            d2 = fx * fx
            d2 += fy * fy
            d2[invalid] = np.inf
            k = np.argmin(d2, axis=1)
            r = np.arange(len(idx))
            d = np.sqrt(d2[r, k])
            cross = dx[r, k] * ry[r, k] - dy[r, k] * rx[r, k]
            s_out[idx] = self.S[J[r, k]] + t[r, k]
            d_out[idx] = d
            lat_out[idx] = np.where(cross >= 0, d, -d)
            i += len(idx)
        return s_out, lat_out, d_out


def build_path(t: np.ndarray, xyz: np.ndarray, moving: np.ndarray, step: float = 1.0) -> Tuple[TrackPath, np.ndarray]:
    """Decimate positions into a polyline: a vertex is appended when the vehicle is moving and
    is >= ``step`` m from the last vertex. Returns the path and, per input sample, the arc length
    of the last appended vertex (a monotone 'running' arc length usable as projection hint)."""
    n = len(xyz)
    keep = np.zeros(n, bool)
    s_run = np.zeros(n)
    if n == 0:
        return TrackPath(np.zeros((1, 2))), s_run
    last = xyz[0, :2]
    keep[0] = True
    S = 0.0
    step2 = step * step
    xy = xyz[:, :2]
    for k in range(1, n):
        if moving[k]:
            dx = xy[k, 0] - last[0]
            dy = xy[k, 1] - last[1]
            d2 = dx * dx + dy * dy
            if d2 >= step2:
                S += np.sqrt(d2)
                last = xy[k]
                keep[k] = True
        s_run[k] = S
    return TrackPath(xyz[keep, :2], xyz[keep, 2]), s_run


# ----------------------------------------------------------------------------------------------
# Reference outlier detection
# ----------------------------------------------------------------------------------------------
def rolling_median_time(t: np.ndarray, x: np.ndarray, win_s: float, grid_s: float = 1.0) -> np.ndarray:
    """Centered time-window running median of the columns of ``x`` (NaN ignored), evaluated on a
    ``grid_s`` grid and linearly interpolated back to ``t`` (fast, pure numpy)."""
    x = np.asarray(x, float)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    g = np.arange(t[0], t[-1] + grid_s, grid_s)
    lo = np.searchsorted(t, g - win_s / 2, 'left')
    hi = np.searchsorted(t, g + win_s / 2, 'right')
    med = np.full((len(g), x.shape[1]), np.nan)
    for i in range(len(g)):
        if hi[i] > lo[i]:
            w = x[lo[i]:hi[i]]
            ok = np.isfinite(w[:, 0])
            if ok.any():
                med[i] = np.median(w[ok], axis=0)
    out = np.empty((len(t), x.shape[1]))
    for c in range(x.shape[1]):
        good = np.isfinite(med[:, c])
        out[:, c] = np.interp(t, g[good], med[good, c]) if good.any() else np.nan
    return out[:, 0] if squeeze else out


def flag_fix_outliers(t_fix: np.ndarray, enu: np.ndarray, t_vel: np.ndarray, vel_enu: np.ndarray,
                      win_s: float = 30.0, thr_h: float = 3.0, thr_v: float = 6.0, iters: int = 2) -> np.ndarray:
    """Flag GNSS fixes inconsistent with the integrated GNSS Doppler velocity.

    offset_k = fix_k - integral(vel) is (nearly) constant for good fixes; fixes whose offset
    deviates from the running (time-window) median by > thr_h (horizontal) or > thr_v (vertical)
    are flagged. Catches multipath jumps and the 'two interleaved solutions' failure seen in
    e.g. 30618_defd0170. All times must be on a common monotone clock (bag time).
    """
    n = len(t_fix)
    bad = np.zeros(n, bool)
    if n < 10 or len(t_vel) < 10:
        return bad
    o = np.argsort(t_vel)
    tv, ve = t_vel[o], vel_enu[o]
    vi = np.stack([np.interp(t_fix, tv, ve[:, c]) for c in range(3)], axis=1)
    dt = np.clip(np.diff(t_fix), 0.0, 2.0)
    dr = np.vstack([np.zeros(3), np.cumsum(0.5 * (vi[1:] + vi[:-1]) * dt[:, None], axis=0)])
    off = enu - dr
    for _ in range(iters):
        med = rolling_median_time(t_fix, np.where(bad[:, None], np.nan, off), win_s)
        r_h = np.hypot(off[:, 0] - med[:, 0], off[:, 1] - med[:, 1])
        r_v = np.abs(off[:, 2] - med[:, 2])
        new_bad = (r_h > thr_h) | (r_v > thr_v) | ~np.isfinite(r_h)
        if np.array_equal(new_bad, bad):
            break
        bad = new_bad
    return bad


# ----------------------------------------------------------------------------------------------
# Reference
# ----------------------------------------------------------------------------------------------
@dataclass
class RefConfig:
    antenna: str = 'master'                 # 'master' | 'rover'
    frame: str = 'enu'                      # 'enu' | 'utm'
    origin: Union[str, Tuple[float, float, float]] = 'first_fix'
    time_base: str = 'header'               # 'header' | 'bag' | 'header_fixed'
    speed_dims: int = 2                     # 2: |(vx,vy)|, 3: |(vx,vy,vz)|
    min_status: int = 0                     # NavSatStatus: -1 no fix, 0 fix, 1 SBAS, 2 GBAS
    clean: bool = False                     # evaluate only on fixes that pass the outlier check
    path_step: float = 5.0                  # [m] reference path vertex spacing (1-2 m inflates the path
                                            #  length by up to +2 % on noisy non-RTK bags: GNSS zig-zag)
    moving_thresh: float = 0.3              # [m/s] path grows only above this speed
    stop_thresh: float = 0.2                # [m/s] 'stopped' regime threshold
    acc_thresh: float = 0.1                 # [m/s^2] accel/brake regime threshold
    transition_s: float = 3.0               # [s] length of departure/arrival windows


@dataclass
class Reference:
    cfg: RefConfig
    frame: LocalFrame
    origin_index: int                       # row in the antenna fix array used as origin
    # position samples (sorted by the chosen time base)
    t_pos: np.ndarray
    tbag_pos: np.ndarray
    xyz: np.ndarray
    status: np.ndarray
    outlier: np.ndarray                     # flagged by velocity consistency check
    s_pos: np.ndarray                       # arc length along ``path``
    tan_pos: np.ndarray                     # unit path tangent (N,2) at s_pos
    # speed samples (sorted by chosen time base)
    t_vel: np.ndarray
    tbag_vel: np.ndarray
    v: np.ndarray
    a: np.ndarray                           # smoothed longitudinal accel [m/s^2]
    vel_xy: np.ndarray                      # (M,2) horizontal velocity in the frame
    path: TrackPath
    distance: float                         # [m] arc length travelled over the evaluated span
    distance_vint: float                    # [m] integral of reference speed
    diag: dict = field(default_factory=dict)

    def pos_mask(self) -> np.ndarray:
        """Samples used for position metrics (all valid, or only non-outliers if cfg.clean)."""
        m = np.ones(len(self.t_pos), bool)
        if self.cfg.clean:
            m &= ~self.outlier
        return m


def _smooth_accel(t: np.ndarray, v: np.ndarray, win_s: float = 1.0) -> np.ndarray:
    """Centered local-regression slope of v(t) over +-win_s/2 on a 10 Hz grid (equivalent to a
    Savitzky-Golay first derivative, polyorder 1-2). Offline, reference only."""
    if len(t) < 7:
        return np.zeros_like(v)
    dt = 0.1
    g = np.arange(t[0], t[-1] + dt / 2, dt)
    vg = np.interp(g, t, v)
    m = max(2, int(round(win_s / dt / 2)))
    if 2 * m + 1 >= len(g):
        return np.zeros_like(v)
    k = np.arange(-m, m + 1, dtype=float)
    c = k / (np.sum(k * k) * dt)
    vp = np.r_[np.full(m, vg[0]), vg, np.full(m, vg[-1])]
    ag = np.convolve(vp, c[::-1], mode='valid')
    return np.interp(t, g, ag)


def select_origin(fix: np.ndarray, min_status: int = 0) -> int:
    ok = (fix[:, STATUS] >= min_status) & np.isfinite(fix[:, LAT]) & np.isfinite(fix[:, LON]) \
        & (np.abs(fix[:, LAT]) > 1e-6) & (np.abs(fix[:, LON]) > 1e-6)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        raise ValueError('no valid GNSS fix for the origin')
    return int(idx[0])


def build_reference(bag: BagData, cfg: RefConfig = None) -> Reference:
    """Build the evaluation reference for one bag (see module docstring)."""
    cfg = cfg or RefConfig()
    fix = bag[f'fix_{cfg.antenna}']
    vel = bag[f'vel_{cfg.antenna}']
    if len(fix) < 2 or len(vel) < 2:
        raise ValueError(f'{bag.name}: no GNSS reference')

    valid = (fix[:, STATUS] >= cfg.min_status) & np.isfinite(fix[:, LAT]) & np.isfinite(fix[:, LON]) \
        & (np.abs(fix[:, LAT]) > 1e-6)
    if isinstance(cfg.origin, str):
        if cfg.origin != 'first_fix':
            raise ValueError(cfg.origin)
        # origin is always the MASTER first valid fix (what the estimator sees first)
        mfix = bag['fix_master']
        oi = select_origin(mfix, cfg.min_status)
        lat0, lon0, alt0 = mfix[oi, LAT], mfix[oi, LON], mfix[oi, ALT]
    else:
        oi = -1
        lat0, lon0, alt0 = cfg.origin
    frame = LocalFrame(cfg.frame, lat0, lon0, alt0)

    fix = fix[valid]
    x, y, z = frame.forward(fix[:, LAT], fix[:, LON], fix[:, ALT])
    xyz = np.stack([x, y, z], axis=1)

    # outlier flags on the bag-time axis, in ENU (independent of output frame)
    enu_frame = frame if cfg.frame == 'enu' else LocalFrame('enu', lat0, lon0, alt0)
    e = np.stack(enu_frame.forward(fix[:, LAT], fix[:, LON], fix[:, ALT]), axis=1) if cfg.frame != 'enu' else xyz
    outlier = flag_fix_outliers(fix[:, T_BAG], e, vel[:, T_BAG], vel[:, [VX, VY, VZ]])

    # speed (bag-time order), regimes computed on the bag-time axis (robust to stamp glitches)
    vh = np.hypot(vel[:, VX], vel[:, VY])
    v = vh if cfg.speed_dims == 2 else np.sqrt(vh ** 2 + vel[:, VZ] ** 2)
    a = _smooth_accel(vel[:, T_BAG], v)
    vxy = np.stack(frame.rotate_enu(vel[:, VX], vel[:, VY]), axis=1)

    # path from clean fixes, moving only
    v_at_fix = np.interp(fix[:, T_BAG], vel[:, T_BAG], v)
    moving = (v_at_fix > cfg.moving_thresh) & ~outlier
    clean = ~outlier
    path, s_run = build_path(fix[clean, T_BAG], xyz[clean], moving[clean], cfg.path_step)
    s_pos = np.empty(len(fix))
    s_c, _, _ = path.project(xyz[clean, :2], s_hint=s_run, window=25.0)
    s_c = np.maximum.accumulate(s_c)          # the tram never reverses
    s_pos[clean] = s_c
    if np.any(outlier):
        s_pos[outlier] = np.interp(fix[outlier, T_BAG], fix[clean, T_BAG], s_c)
    tan_pos = path.tangent(s_pos)

    # time base + sorting
    t_pos = time_vector(fix, cfg.time_base)
    t_vel = time_vector(vel, cfg.time_base)
    op = np.argsort(t_pos, kind='stable')
    ov = np.argsort(t_vel, kind='stable')

    dist = float(s_c[-1] - s_c[0]) if len(s_c) else 0.0
    tb = vel[:, T_BAG]
    dvint = float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(tb)))   # linear interpolation across vel gaps

    ref = Reference(cfg=cfg, frame=frame, origin_index=oi,
                    t_pos=t_pos[op], tbag_pos=fix[op, T_BAG], xyz=xyz[op], status=fix[op, STATUS],
                    outlier=outlier[op], s_pos=s_pos[op], tan_pos=tan_pos[op],
                    t_vel=t_vel[ov], tbag_vel=vel[ov, T_BAG], v=v[ov], a=a[ov], vel_xy=vxy[ov],
                    path=path, distance=dist, distance_vint=dvint)
    ref.diag = reference_diagnostics(bag, ref)
    return ref


def reference_diagnostics(bag: BagData, ref: Reference) -> dict:
    """Quality indicators of the GNSS reference itself (not of the estimator)."""
    fix = bag[f'fix_{ref.cfg.antenna}']
    vel = bag[f'vel_{ref.cfg.antenna}']
    d = {
        'n_fix': int(len(ref.t_pos)),
        'n_vel': int(len(ref.t_vel)),
        'frac_status2': float(np.mean(ref.status == 2)) if len(ref.status) else 0.0,
        'frac_outlier': float(np.mean(ref.outlier)) if len(ref.outlier) else 0.0,
        'hdr_glitch_fix': repair_header_glitches(fix[:, T_BAG], fix[:, T_HDR])[1],
        'hdr_glitch_vel': repair_header_glitches(vel[:, T_BAG], vel[:, T_HDR])[1],
        'distance_path_m': ref.distance,
        'distance_vint_m': ref.distance_vint,
        'path_length_m': ref.path.length,
        'alt_range_m': float(np.ptp(ref.xyz[~ref.outlier, 2])) if np.any(~ref.outlier) else 0.0,
        'nonmono_t_pos': int(np.sum(np.diff(time_vector(fix, ref.cfg.time_base)) <= 0)),
    }
    return d


def frame_discrepancy(bag: BagData, kinds=('enu', 'utm')) -> dict:
    """Max/final horizontal distance between the same fixes expressed in two frame kinds
    (both with origin at the first master fix) - quantifies the cost of a frame mix-up."""
    fix = bag['fix_master']
    oi = select_origin(fix)
    f1 = LocalFrame(kinds[0], *fix[oi, [LAT, LON, ALT]])
    f2 = LocalFrame(kinds[1], *fix[oi, [LAT, LON, ALT]])
    a = np.stack(f1.forward(fix[:, LAT], fix[:, LON], fix[:, ALT]), 1)
    b = np.stack(f2.forward(fix[:, LAT], fix[:, LON], fix[:, ALT]), 1)
    d2 = np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])
    r = np.hypot(a[:, 0], a[:, 1])
    return {'max_m': float(d2.max()), 'final_m': float(d2[-1]), 'max_range_m': float(r.max()),
            'z_max_m': float(np.abs(a[:, 2] - b[:, 2]).max())}
