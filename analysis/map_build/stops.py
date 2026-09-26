"""Stationary-interval detection and stop clustering along the track map.

A stop = GNSS Doppler speed < 0.1 m/s continuously for >= 5 s (brief <0.5 s interruptions tolerated).
Each stop gets: start/end time, dwell, median position of good fixes, projection (edge, s, d) on the map,
and the wheel-speed view (what the online estimator sees: both bogies ~0).
Stops are clustered along s per edge; clusters with high repeatability are platform stops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_io as D


def intervals(mask: np.ndarray, t: np.ndarray, min_dur: float, max_gap: float = 0.5):
    """Contiguous True intervals of mask (on sample times t), merging gaps shorter than max_gap seconds."""
    if len(mask) == 0:
        return []
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out = []
    a = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if t[i] - t[prev] > max_gap:
            out.append((a, prev))
            a = i
        prev = i
    out.append((a, prev))
    return [(i0, i1) for i0, i1 in out if t[i1] - t[i0] >= min_dur]


def detect_stops(r, v_thr=0.1, min_dur=5.0):
    """Stationary intervals of a Run (runs.Run). Returns list of dicts."""
    sp = np.where(np.isfinite(r.speed), r.speed, np.inf)
    st = sp < v_thr
    rows = []
    for i0, i1 in intervals(st, r.th, min_dur):
        sl = np.arange(i0, i1 + 1)
        g = sl[r.good[sl]]
        use = g if len(g) >= 3 else sl
        wf = r.wheel_f[sl] if r.wheel_f is not None else np.full(len(sl), np.nan)
        wr = r.wheel_r[sl] if r.wheel_r is not None else np.full(len(sl), np.nan)
        rows.append(dict(bag=r.bag, t0=r.th[i0], t1=r.th[i1], dwell=r.th[i1] - r.th[i0],
                         t_rel=r.th[i0] - r.th[0], t_end_rel=r.th[-1] - r.th[i1],
                         x=float(np.median(r.x[use])), y=float(np.median(r.y[use])), z=float(np.median(r.z[use])),
                         psi=float(np.angle(np.mean(np.exp(1j * r.psi[use])))),
                         n_good=len(g), n=len(sl), pos_spread=float(np.hypot(np.std(r.x[use]), np.std(r.y[use]))),
                         wheel_max=float(np.nanmax(np.maximum(wf, wr))) if len(sl) else np.nan,
                         first=(i0 == 0), last=(i1 == len(r) - 1)))
    return rows


def wheel_stops(bag: str, v_thr=0.02, min_dur=5.0):
    """Stops as seen by the wheel sensors only (km/h -> m/s), for comparison with GNSS stops."""
    w = D.wheels(bag)
    f, r = w['front'], w['rear']
    if len(f) < 2 or len(r) < 2:
        return []
    t = f[:, 0]
    vr = np.interp(t, r[:, 0], r[:, 2])
    st = (f[:, 2] / 3.6 < v_thr) & (vr / 3.6 < v_thr)
    return [(t[i0], t[i1]) for i0, i1 in intervals(st, t, min_dur, max_gap=0.5)]


def cluster_1d(s: np.ndarray, gap: float = 15.0):
    """Single-linkage clustering of sorted 1-D positions; returns labels in original order."""
    o = np.argsort(s)
    ss = s[o]
    lab_sorted = np.r_[0, np.cumsum(np.diff(ss) > gap)]
    lab = np.empty(len(s), int)
    lab[o] = lab_sorted
    return lab
