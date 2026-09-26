"""Derived kinematic signals of a clean run and random event placement.

The context is always built from the *clean* run (before any injection) so that the placement of
later injectors does not depend on earlier corruption. All signals live on a uniform grid in
*measurement time* = sanitised header-stamp time (header stamps of the vehicle topics are the
acquisition times; bag time is the arrival time and contains bursts at the start of each bag).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .constants import CMD, FRONT, KMH_PER_MS, REAR
from .run import Run, Stream

GRID_DT = 0.05


def sanitize_stamps(t_bag: np.ndarray, t_hdr: np.ndarray, max_dev: float = 0.25, win: int = 31) -> np.ndarray:
    """Return header times with glitches (zero stamps, +-1 s roll-over errors, NaN) replaced.

    Latency ``t_bag - t_hdr`` is compared to its running median; outliers are re-stamped with
    ``t_bag - running_median``. Smooth latency changes (start-of-bag burst, clock drift) survive.
    """
    t_hdr = np.asarray(t_hdr, float)
    t_bag = np.asarray(t_bag, float)
    if len(t_hdr) == 0:
        return t_hdr.copy()
    lat = t_bag - t_hdr
    good = np.isfinite(lat) & (t_hdr > 1.0)
    lat_f = lat.copy()
    if good.any():
        lat_f[~good] = np.median(lat[good])
    else:
        return t_bag.copy()
    w = min(win, len(lat_f) | 1)
    med = median_filter(lat_f, size=w, mode='nearest')
    bad = (~good) | (np.abs(lat_f - med) > max_dev)
    out = t_hdr.copy()
    out[bad] = t_bag[bad] - med[bad]
    return out


def _series_on_grid(t: np.ndarray, y: np.ndarray, grid: np.ndarray, max_gap: float = 0.35) -> np.ndarray:
    """Linear interpolation onto ``grid``; NaN where no sample within ``max_gap`` seconds."""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 2:
        return np.full(len(grid), np.nan)
    order = np.argsort(t, kind='stable')
    t, y = t[order], y[order]
    out = np.interp(grid, t, y)
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    dist = np.minimum(np.abs(grid - t[j - 1]), np.abs(t[j] - grid))
    out[dist > max_gap] = np.nan
    return out


def zoh(t: np.ndarray, y: np.ndarray, grid: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Zero-order hold of samples (t, y) evaluated on ``grid``."""
    if len(t) == 0:
        return np.full(len(grid), default)
    order = np.argsort(t, kind='stable')
    t, y = t[order], y[order]
    i = np.searchsorted(t, grid, side='right') - 1
    out = y[np.clip(i, 0, len(y) - 1)].astype(float)
    out[i < 0] = default
    return out


def fill_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float).copy()
    ok = np.isfinite(y)
    if ok.all() or not ok.any():
        return np.nan_to_num(y)
    idx = np.arange(len(y))
    y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y


@dataclass
class Context:
    t_start: float        # bag time of the first message (run reference for relative times)
    grid: np.ndarray      # measurement-time grid [s]
    v: np.ndarray         # vehicle speed [m/s], from clean wheels (smoothed, >= 0)
    a: np.ndarray         # acceleration [m/s^2]
    notch: np.ndarray     # controller position (zero-order hold)
    x: np.ndarray         # travelled distance [m]
    lat_wheel: float      # median t_bag - t_hdr of wheel messages [s] (maps header -> bag time)
    lat_cmd: float
    dt: float = GRID_DT

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(cls, run: Run, dt: float = GRID_DT) -> 'Context':
        f, r, c = run.streams.get(FRONT), run.streams.get(REAR), run.streams.get(CMD)
        series = []
        t_all = []
        lats = []
        for s in (f, r):
            if s is None or len(s) < 2:
                continue
            tm = sanitize_stamps(s.t_bag, s.t_hdr0)
            series.append((tm, s.clean[:, 0]))
            t_all.append(tm)
            lats.append(np.median(s.t_bag - tm))
        if not series:
            raise ValueError(f'{run.name}: no wheel data')
        t_lo = min(t.min() for t in t_all)
        t_hi = max(t.max() for t in t_all)
        grid = np.arange(t_lo, t_hi + dt, dt)
        vs = np.vstack([_series_on_grid(t, y, grid) for t, y in series])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN columns inside natural gaps
            v_kmh = np.nanmean(vs, axis=0) if len(vs) > 1 else vs[0]
        v = fill_nan(v_kmh) / KMH_PER_MS
        v = np.maximum(uniform_filter1d(v, 5, mode='nearest'), 0.0)
        a = np.gradient(uniform_filter1d(v, 9, mode='nearest'), dt)
        x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
        if c is not None and len(c):
            tc = sanitize_stamps(c.t_bag, c.t_hdr0)
            notch = zoh(tc, c.clean[:, 0], grid)
            lat_cmd = float(np.median(c.t_bag - tc))
        else:
            notch = np.zeros_like(grid)
            lat_cmd = 0.0
        return cls(t_start=run.t_start, grid=grid, v=v, a=a, notch=notch, x=x,
                   lat_wheel=float(np.median(lats)), lat_cmd=lat_cmd, dt=dt)

    # ------------------------------------------------------------------ helpers
    def at(self, name: str, t) -> np.ndarray:
        """Interpolate a context signal (``v``, ``a``, ``x``) at measurement times ``t``."""
        return np.interp(t, self.grid, getattr(self, name))

    def notch_at(self, t) -> np.ndarray:
        i = np.clip(np.round((np.asarray(t) - self.grid[0]) / self.dt).astype(int), 0, len(self.grid) - 1)
        return self.notch[i]

    def rel(self, t: float) -> float:
        return float(t - self.t_start)

    @property
    def t_lo(self) -> float:
        return float(self.grid[0])

    @property
    def t_hi(self) -> float:
        return float(self.grid[-1])

    # ------------------------------------------------------------------ placement
    def mask(self, when: str = 'any', min_speed: float | None = None, max_speed: float | None = None,
             min_notch: float | None = None, max_notch: float | None = None) -> np.ndarray:
        """Boolean mask over the grid for a driving-state condition."""
        v, n, a = self.v, self.notch, self.a
        moving = v > 0.5
        m = {
            'any': np.ones_like(v, bool),
            'motion': moving,
            'standstill': v < 0.05,
            'traction': moving & (n > 0),
            'braking': moving & (n < 0),
            'coast': moving & (n == 0),
            'accel': moving & (a > 0.2),
            'decel': moving & (a < -0.2),
            'departure': self._crossing(up=True),
            'arrival': self._crossing(up=False),
        }.get(when)
        if m is None:
            raise ValueError(f'unknown "when": {when!r}')
        m = m.copy()
        if min_speed is not None:
            m &= v >= min_speed
        if max_speed is not None:
            m &= v <= max_speed
        if min_notch is not None:
            m &= n >= min_notch
        if max_notch is not None:
            m &= n <= max_notch
        return m

    def _crossing(self, up: bool, thr: float = 0.3) -> np.ndarray:
        above = self.v > thr
        d = np.diff(above.astype(int))
        idx = np.flatnonzero(d == (1 if up else -1)) + 1
        m = np.zeros_like(above)
        m[idx] = True
        return m

    def hold_time(self, mask: np.ndarray) -> np.ndarray:
        """For every grid point, how long [s] ``mask`` stays True from there on."""
        out = np.zeros(len(mask))
        run = 0
        for i in range(len(mask) - 1, -1, -1):
            run = run + 1 if mask[i] else 0
            out[i] = run * self.dt
        return out

    def place(self, rng: np.random.Generator, mask: np.ndarray, count: int, duration, *, min_hold: float = 0.0,
              min_gap: float = 3.0, t_min: float = 10.0, t_max: float | None = None, lead: float = 0.0,
              taken: list[tuple[float, float]] | None = None, max_tries: int = 400) -> list[tuple[float, float]]:
        """Pick up to ``count`` non-overlapping intervals whose start satisfies ``mask``.

        ``duration`` is a callable ``rng -> seconds`` (or a number). ``min_hold`` requires the mask
        to stay true that long after the start. ``lead`` moves the start earlier by U(0, lead) s
        (e.g. a dropout that begins shortly before a departure). Times are measurement times.
        """
        dur_fn = duration if callable(duration) else (lambda _r, d=float(duration): d)
        m = mask.copy()
        if min_hold > 0:
            m &= self.hold_time(mask) >= min_hold
        lo = self.t_start + t_min
        hi = self.t_hi - 1.0 if t_max is None else min(self.t_hi - 1.0, self.t_start + t_max)
        m &= (self.grid >= lo) & (self.grid <= hi)
        cand = np.flatnonzero(m)
        placed: list[tuple[float, float]] = []
        blocked = list(taken or [])
        tries = 0
        while len(placed) < count and len(cand) and tries < max_tries:
            tries += 1
            i = cand[rng.integers(len(cand))]
            d = float(dur_fn(rng))
            t0 = self.grid[i] - (rng.uniform(0, lead) if lead > 0 else 0.0)
            t1 = min(t0 + d, self.t_hi)
            if any(t0 < b1 + min_gap and t1 > b0 - min_gap for b0, b1 in blocked):
                continue
            placed.append((float(t0), float(t1)))
            blocked.append((t0, t1))
        placed.sort()
        return placed


def measurement_time(stream: Stream) -> np.ndarray:
    """Sanitised clean header times of a stream (time axis used for value anomalies)."""
    return sanitize_stamps(stream.t_bag, stream.t_hdr0)
