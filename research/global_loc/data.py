"""Per-bag data access for the global-localisation study: replay outputs, notch, truth on the replay grid,
and online cue extraction (stops, cut-offs) exactly as a causal on-board implementation would see them."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

import common as C

ODOMETER = os.environ.get('GL_ODOMETER', 'int_v')  # 'int_v' (integrated published speed) | 's' (estimator s)
GNSS_LEAD_S = 0.045  # GNSS fix positions lead wheel stamps (DATA_FINDINGS #5): fix(t) = pos(t + lead)


@dataclass
class Bag:
    name: str
    t: np.ndarray          # output stamps, s (header time)
    s_rel: np.ndarray      # estimator odometer since start (no GNSS -> never map-matched), m
    v: np.ndarray          # estimator speed, m/s
    d: np.ndarray          # estimator disturbance acceleration, m/s^2
    flags: np.ndarray
    mu3: np.ndarray        # P(both wheels bad) -> model-only dead reckoning
    a_model: np.ndarray
    notch_t: np.ndarray    # controller notch stamps
    notch: np.ndarray
    truth_s: np.ndarray | None = None   # true unwrapped main-cycle arc length at t (NaN where unknown)
    truth_ok: np.ndarray | None = None
    has_truth: bool = False
    meta: dict = field(default_factory=dict)


def load_bag(name: str, with_truth: bool = True) -> Bag:
    r = C.replay_nognss(name)
    t = r['t']
    # outputs are published at cmd and wheel stamps; keep a strictly increasing stamp sequence
    o = np.argsort(t, kind='stable')
    t = t[o]
    keep = np.r_[True, np.diff(t) > 0]
    idx = o[keep]
    nt, nv = C.notch_series(name)
    b = Bag(name, r['t'][idx], r['s'][idx], r['v'][idx], r['d'][idx], r['flags'][idx], r['mu3'][idx],
            r['a_model'][idx], nt, nv)
    # The snapshot's odometer s jumps backwards by up to ~16 m when the slip monitor rolls back to a
    # snapshot taken before a standstill (flags 0x40000 at departure). Integrating the published speed
    # avoids that artefact; ODOMETER='s' keeps the raw estimator distance for comparison.
    b.meta['s_core'] = b.s_rel.copy()
    if ODOMETER == 'int_v':
        b.s_rel = np.r_[0.0, np.cumsum(0.5 * (b.v[1:] + b.v[:-1]) * np.diff(b.t))] + b.s_rel[0]
    tp = C.CACHE / f'truth_{name}.npz'
    if with_truth and tp.exists():
        z = np.load(tp)
        tt, su, ok, st = z['t'], z['s_unw'], z['ok'], z['status']
        # plain (status 0) fixes lag RTK ones by up to ~0.4 s in mixed bags: use RTK only when the bag has
        # enough of it, and force the track to be monotone (the tram never reverses) by isotonic regression
        # plain/SBAS fixes can be off by tens of metres next to RTK ones: RTK only whenever the bag has RTK
        b.meta['rtk_share'] = float((st[ok] == 2).mean()) if ok.any() else 0.0
        if b.meta['rtk_share'] >= 0.05:
            ok = ok & (st == 2)
        ok = ok & (z['dist'] < 2.0)  # off the main cycle (yard fans, detour): no main-cycle truth
        tt, su = tt[ok], su[ok]
        if len(tt) > 10:
            su = isotonic(su)
            q = b.t - GNSS_LEAD_S
            j = np.clip(np.searchsorted(tt, q), 1, len(tt) - 1)
            gap = tt[j] - tt[j - 1]
            s_true = np.interp(q, tt, su)
            valid = (gap < 2.0) & (q >= tt[0]) & (q <= tt[-1])
            s_true[~valid] = np.nan
            b.truth_s = s_true
            b.truth_ok = valid
            b.has_truth = valid.mean() > 0.5
    return b


def isotonic(y: np.ndarray) -> np.ndarray:
    """Non-decreasing least-squares fit (pool adjacent violators)."""
    vals, wts, cnt = [], [], []
    for v in y:
        vals.append(float(v))
        wts.append(1.0)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w
            cnt[-2] += cnt[-1]
            vals.pop(), wts.pop(), cnt.pop()
    return np.repeat(vals, cnt)


# ----------------------------------------------------------------------------------------------
# online cue extraction (causal: each cue is emitted at the time it becomes known)
def stop_events(b: Bag, dwell_s: float = 1.5):
    """Standstill intervals from the estimator flag. A stop is emitted once it has lasted dwell_s.

    Returns list of dicts: t_emit, t0, t1 (end, may be the run end), s_rel, initial (run starts stopped)."""
    st = (b.flags & C.FLAG_STANDSTILL) != 0
    out = []
    if not st.any():
        return out
    edges = np.flatnonzero(np.diff(np.r_[0, st.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2] - 1
    for i0, i1 in zip(starts, ends):
        t0, t1 = b.t[i0], b.t[i1]
        if t1 - t0 < dwell_s:
            continue
        k = np.searchsorted(b.t, t0 + dwell_s)
        out.append(dict(t_emit=float(b.t[min(k, len(b.t) - 1)]), t0=float(t0), t1=float(t1),
                        s_rel=float(np.median(b.s_rel[i0:i1 + 1])), initial=bool(i0 == 0 or b.s_rel[i0] < 1.0),
                        final=bool(i1 >= len(b.t) - 2), i0=int(i0), i1=int(i1)))
    return out


def cutoff_events(b: Bag, notch_min: int = 4, v_min: float = 2.0):
    """Abrupt traction cut-offs: notch >= notch_min followed directly by 0, while moving faster than v_min.
    Returns list of dicts: t, s_rel (odometer at the event, + v * GNSS lead like the core), v, notch_before."""
    nt, nv = b.notch_t, b.notch
    out = []
    ch = np.flatnonzero((nv[1:] == 0) & (nv[:-1] >= notch_min)) + 1
    for i in ch:
        t = nt[i]
        if t < b.t[0] or t > b.t[-1]:
            continue
        v = float(np.interp(t, b.t, b.v))
        if v <= v_min:
            continue
        s = float(np.interp(t, b.t, b.s_rel)) + v * GNSS_LEAD_S
        out.append(dict(t=float(t), s_rel=s, v=v, notch_before=int(nv[i - 1])))
    return out
