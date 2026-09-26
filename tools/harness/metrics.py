"""Judge-like metrics: nearest-stamp matching, speed / position / robustness / real-time metrics.

All functions are vectorised numpy; they take the reference (:class:`reference.Reference`) and an
:class:`OutputLog` produced by :mod:`replay` (or any arrays with the same meaning).

Conventions
-----------
* error = estimate - reference (signed; bias > 0 means the estimator over-estimates).
* ``mean`` of a signed error is the bias, ``mean_abs`` the MAE, ``max`` is max |e|.
* matching: for every reference sample (``direction='ref2out'``, default) the output with the
  nearest ``stamp`` is taken; pairs with |dt| > tol (0.05 s) are unmatched. ``'out2ref'`` does the
  opposite (every output is scored against its nearest reference sample).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .loader import BagData, T_BAG, T_HDR, V_COL
from .reference import Reference, repair_header_glitches

KMH = 1.0 / 3.6


# ----------------------------------------------------------------------------------------------
# Output log container
# ----------------------------------------------------------------------------------------------
@dataclass
class OutputLog:
    """Everything the estimator published during one replay (arrays of equal length N)."""
    stamp: np.ndarray                   # header.stamp written by the estimator [s]
    v: np.ndarray                       # /result/velocity [m/s]
    xyz: np.ndarray                     # (N,3) /result/position pose.position [m]
    emit_tbag: np.ndarray               # replay clock (bag time) when the output was produced
    emit_proc: np.ndarray               # wall time of the producing callback [s]
    frame_id: List[Optional[str]] = field(default_factory=list)
    cov: Optional[np.ndarray] = None    # (N,3) position variances (x,y,z) or None
    v_var: Optional[np.ndarray] = None  # (N,) speed variance or None
    slip: Optional[np.ndarray] = None   # (N,) slip flag / probability or None
    yaw: Optional[np.ndarray] = None    # (N,) heading [rad] or None
    proc_all: Optional[np.ndarray] = None   # wall time of every callback (incl. silent ones) [s]
    n_callbacks: int = 0

    def __len__(self):
        return len(self.stamp)


# ----------------------------------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------------------------------
def err_stats(e: np.ndarray) -> Dict[str, float]:
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {'n': 0, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'std': np.nan, 'max': np.nan, 'p95': np.nan}
    ae = np.abs(e)
    return {'n': int(len(e)), 'rmse': float(np.sqrt(np.mean(e * e))), 'mae': float(ae.mean()),
            'bias': float(e.mean()), 'std': float(e.std()), 'max': float(ae.max()),
            'p95': float(np.percentile(ae, 95))}


def abs_stats(d: np.ndarray) -> Dict[str, float]:
    """Stats of a non-negative error magnitude (e.g. 2D/3D distance)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {'n': 0, 'mean': np.nan, 'rmse': np.nan, 'max': np.nan, 'p95': np.nan, 'median': np.nan}
    return {'n': int(len(d)), 'mean': float(d.mean()), 'rmse': float(np.sqrt(np.mean(d * d))),
            'max': float(d.max()), 'p95': float(np.percentile(d, 95)), 'median': float(np.median(d))}


def signed_stats(e: np.ndarray) -> Dict[str, float]:
    """Along/cross-track style stats: MEAN (signed), MEAN_ABS, RMSE, MAX |e|, P95 |e|."""
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {'n': 0, 'mean': np.nan, 'mean_abs': np.nan, 'rmse': np.nan, 'max': np.nan, 'p95': np.nan}
    ae = np.abs(e)
    return {'n': int(len(e)), 'mean': float(e.mean()), 'mean_abs': float(ae.mean()),
            'rmse': float(np.sqrt(np.mean(e * e))), 'max': float(ae.max()), 'p95': float(np.percentile(ae, 95))}


def match_nearest(t_query: np.ndarray, t_target_sorted: np.ndarray, tol: float):
    """Index of the nearest target stamp for every query stamp (targets must be sorted).
    Returns (idx, dt, ok) with dt = t_target[idx] - t_query and ok = |dt| <= tol."""
    tq = np.asarray(t_query, float)
    tt = np.asarray(t_target_sorted, float)
    n = len(tt)
    if n == 0 or len(tq) == 0:
        return np.zeros(len(tq), int), np.full(len(tq), np.inf), np.zeros(len(tq), bool)
    j = np.searchsorted(tt, tq)
    j0 = np.clip(j - 1, 0, n - 1)
    j1 = np.clip(j, 0, n - 1)
    d0 = np.abs(tq - tt[j0])
    d1 = np.abs(tt[j1] - tq)
    idx = np.where(d1 < d0, j1, j0)
    dt = tt[idx] - tq
    return idx, dt, np.abs(dt) <= tol


def pair(t_ref: np.ndarray, t_out: np.ndarray, out_valid: np.ndarray, tol: float, direction: str = 'ref2out'):
    """Build (ref_index, out_index) pairs. t_ref must be sorted; t_out need not be.
    Returns (iref, iout, n_candidates) where n_candidates is the denominator of the match rate."""
    vo = np.flatnonzero(out_valid)
    order = vo[np.argsort(t_out[vo], kind='stable')]
    ts = t_out[order]
    if direction == 'ref2out':
        idx, _, ok = match_nearest(t_ref, ts, tol)
        iref = np.flatnonzero(ok)
        return iref, order[idx[ok]], len(t_ref)
    if direction == 'out2ref':
        idx, _, ok = match_nearest(ts, t_ref, tol)
        return idx[ok], order[ok], len(ts)
    raise ValueError(direction)


def _runs(mask: np.ndarray):
    """(start, end_exclusive) index pairs of True runs."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def dilate_time(t: np.ndarray, mask: np.ndarray, before: float, after: float) -> np.ndarray:
    """Grow True runs of ``mask`` (on sorted times ``t``) by ``before``/``after`` seconds."""
    out = mask.copy()
    s, e = _runs(mask)
    for a, b in zip(s, e):
        lo = np.searchsorted(t, t[a] - before, 'left')
        hi = np.searchsorted(t, t[b - 1] + after, 'right')
        out[lo:hi] = True
    return out


def sample_hold(t_src: np.ndarray, v_src: np.ndarray, t_q: np.ndarray, fill=np.nan) -> np.ndarray:
    """Causal zero-order hold: value of the last source sample with t_src <= t_q."""
    j = np.searchsorted(t_src, t_q, side='right') - 1
    out = np.where(j >= 0, v_src[np.clip(j, 0, len(v_src) - 1)], fill) if len(t_src) else np.full(len(t_q), fill)
    return out


# ----------------------------------------------------------------------------------------------
# Regimes (evaluated on the reference speed samples)
# ----------------------------------------------------------------------------------------------
def speed_regimes(ref: Reference, bag: Optional[BagData] = None) -> Dict[str, np.ndarray]:
    """Boolean masks over ``ref.t_vel`` samples.

    kinematic (from reference speed / smoothed acceleration):
      stopped (v < stop_thresh), moving, accel (a > +acc), brake (a < -acc), cruise (|a| <= acc),
      depart ([-1 s, +transition_s] around each stop->move crossing),
      arrive ([-transition_s, +1 s] around each move->stop crossing), transitions = depart|arrive,
      low_speed (0 < v < 2 m/s)
    command based (driver notch, sample-and-hold on bag time; needs ``bag``):
      cmd_traction (notch > 0), cmd_brake (notch < 0), cmd_coast (notch == 0 & moving)
    """
    c = ref.cfg
    o = np.argsort(ref.tbag_vel, kind='stable')       # regimes on the monotone bag clock
    tb = ref.tbag_vel[o]
    v = ref.v[o]
    a = ref.a[o]
    stopped = v < c.stop_thresh
    moving = ~stopped
    r = {
        'all': np.ones(len(v), bool),
        'stopped': stopped,
        'moving': moving,
        'accel': moving & (a > c.acc_thresh),
        'brake': moving & (a < -c.acc_thresh),
        'cruise': moving & (np.abs(a) <= c.acc_thresh),
        'low_speed': moving & (v < 2.0),
    }
    s, e = _runs(moving)
    dep = np.zeros(len(v), bool)
    arr = np.zeros(len(v), bool)
    for a0, b0 in zip(s, e):
        if a0 > 0:   # a real departure (not the bag start)
            lo = np.searchsorted(tb, tb[a0] - 1.0)
            hi = np.searchsorted(tb, tb[a0] + c.transition_s, 'right')
            dep[lo:hi] = True
        if b0 < len(v):
            lo = np.searchsorted(tb, tb[b0 - 1] - c.transition_s)
            hi = np.searchsorted(tb, tb[b0 - 1] + 1.0, 'right')
            arr[lo:hi] = True
    r['depart'] = dep
    r['arrive'] = arr
    r['transitions'] = dep | arr
    if bag is not None and len(bag['cmd']):
        cmd = bag['cmd']
        notch = sample_hold(cmd[:, T_BAG], cmd[:, V_COL], tb, fill=0.0)
        r['cmd_traction'] = notch > 0
        r['cmd_brake'] = notch < 0
        r['cmd_coast'] = (notch == 0) & moving
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    return {k: m[inv] for k, m in r.items()}


def anomaly_regimes(ref: Reference, bag: BagData, dev_abs: float = 0.5, dev_rel: float = 0.1,
                    gap_s: float = 0.35, pad_s: float = 1.0, recovery_s: float = 5.0) -> Dict[str, np.ndarray]:
    """Masks over ``ref.t_vel`` samples where the wheel odometry is unreliable (criterion 3).

    slip_front / slip_rear: |wheel/3.6 - v_ref| > max(dev_abs, dev_rel*v_ref) (+-pad_s),
    slip_both: both bogies wrong at once (averaging cannot help), bogie_mismatch: |front-rear|
    > threshold, dropout: no message from a bogie for > gap_s, anomaly: union,
    recovery: recovery_s after the end of each anomaly window, clean: none of the above.
    The comparison uses glitch-repaired header stamps of wheels and GNSS vel (physical time).
    """
    o = np.argsort(ref.tbag_vel, kind='stable')
    tb = ref.tbag_vel[o]
    v = ref.v[o]
    # physical time of each reference sample = glitch-repaired header stamp of its vel message
    # (rows of the bag vel array are in bag-time order, same as ``o``)
    vel = bag[f'vel_{ref.cfg.antenna}']
    t_phys = repair_header_glitches(vel[:, T_BAG], vel[:, T_HDR])[0] if len(vel) == len(tb) else tb - 0.08
    out = {}
    thr = np.maximum(dev_abs, dev_rel * v)
    for name in ('front', 'rear'):
        w = bag[name]
        if len(w) < 2:
            out[f'slip_{name}'] = np.zeros(len(v), bool)
            out[f'dropout_{name}'] = np.ones(len(v), bool)
            continue
        th = repair_header_glitches(w[:, T_BAG], w[:, T_HDR])[0]
        ow = np.argsort(th, kind='stable')
        vw = np.interp(t_phys, th[ow], w[ow, V_COL]) * KMH
        bad = np.abs(vw - v) > thr
        out[f'slip_{name}'] = dilate_time(tb, bad, pad_s, pad_s)
        # dropouts on the bag clock: age of the latest message of this bogie
        j = np.searchsorted(w[:, T_BAG], tb, side='right') - 1
        age = np.where(j >= 0, tb - w[np.clip(j, 0, None), T_BAG], np.inf)
        out[f'dropout_{name}'] = dilate_time(tb, age > gap_s, pad_s, pad_s)
    out['slip_any'] = out['slip_front'] | out['slip_rear']
    out['slip_both'] = out['slip_front'] & out['slip_rear']
    out['dropout'] = out['dropout_front'] | out['dropout_rear']
    anomaly = out['slip_any'] | out['dropout']
    out['anomaly'] = anomaly
    rec = np.zeros(len(v), bool)
    s, e = _runs(anomaly)
    for a0, b0 in zip(s, e):
        if b0 < len(v):
            lo = b0
            hi = np.searchsorted(tb, tb[b0 - 1] + recovery_s, 'right')
            rec[lo:hi] = True
    out['recovery'] = rec & ~anomaly
    out['clean'] = ~anomaly & ~rec
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    return {k: m[inv] for k, m in out.items()}


def naive_wheel_speed(ref: Reference, bag: BagData) -> np.ndarray:
    """Naive raw-odometry baseline at the reference samples: mean of the latest front & rear
    speeds / 3.6, looked up on the SAME clock as the reference stamps (like a naive node that stamps
    with the wheel's own time), so it is affected by stamp conventions/glitches exactly like an
    estimator. Used to express robustness as an improvement over raw odometry."""
    from .reference import time_vector
    tq = ref.t_vel
    vals = []
    for name in ('front', 'rear'):
        w = bag[name]
        if len(w) == 0:
            vals.append(np.full(len(tq), np.nan))
            continue
        tw = time_vector(w, ref.cfg.time_base)
        o = np.argsort(tw, kind='stable')
        vals.append(sample_hold(tw[o], w[o, V_COL] * KMH, tq))
    st = np.stack(vals)
    n = np.sum(np.isfinite(st), axis=0)
    return np.where(n > 0, np.nansum(st, axis=0) / np.maximum(n, 1), np.nan)


# ----------------------------------------------------------------------------------------------
# Speed metrics
# ----------------------------------------------------------------------------------------------
def speed_metrics(ref: Reference, out: OutputLog, tol: float = 0.05, direction: str = 'ref2out',
                  regimes: Optional[Dict[str, np.ndarray]] = None) -> dict:
    valid = np.isfinite(out.v) & np.isfinite(out.stamp)
    iref, iout, n_cand = pair(ref.t_vel, out.stamp, valid, tol, direction)
    e = out.v[iout] - ref.v[iref]
    res = {'match_rate': float(len(iref) / n_cand) if n_cand else 0.0,
           'n_ref': int(len(ref.t_vel)), 'n_matched': int(len(iref)),
           'all': err_stats(e)}
    if regimes:
        for name, m in regimes.items():
            if name == 'all':
                continue
            res[name] = err_stats(e[m[iref]])
    return res


# ----------------------------------------------------------------------------------------------
# Position metrics
# ----------------------------------------------------------------------------------------------
def position_metrics(ref: Reference, out: OutputLog, tol: float = 0.05, direction: str = 'ref2out',
                     arc_window_min: float = 20.0, arc_window_max: float = 2000.0) -> dict:
    """3D/2D/z errors, along/cross-track decomposition and final drift.

    along_tan / cross_tan: error vector projected on the reference path tangent / normal
        (normal = left of travel direction) at the reference sample.
    along_arc: arc length of the estimate projected onto the reference path (searched within
        +-clip(20 + 1.6*|e2d|, 20, 2000) m of the reference arc length) minus the reference arc
        length -> 'longitudinal coordinate' error. cross_map: signed distance to that path
        (the 'pathgraph' cross-track error, with the bag's own GNSS path as the map).
    drift_pct_*: error at the last matched reference sample / travelled distance * 100.
    """
    valid = np.all(np.isfinite(out.xyz), axis=1) & np.isfinite(out.stamp)
    pm = ref.pos_mask()
    t_ref = ref.t_pos
    iref, iout, n_cand = pair(t_ref, out.stamp, valid, tol, direction)
    keep = pm[iref]
    iref, iout = iref[keep], iout[keep]
    res = {'match_rate': float(len(iref) / max(1, int(pm.sum())) if direction == 'ref2out' else len(iref) / max(1, n_cand)),
           'n_ref': int(pm.sum()), 'n_matched': int(len(iref)), 'distance_m': ref.distance}
    if len(iref) == 0:
        return res
    p_ref = ref.xyz[iref]
    p_est = out.xyz[iout]
    e = p_est - p_ref
    e2 = np.hypot(e[:, 0], e[:, 1])
    e3 = np.sqrt(e2 ** 2 + e[:, 2] ** 2)
    tan = ref.tan_pos[iref]
    along_tan = e[:, 0] * tan[:, 0] + e[:, 1] * tan[:, 1]
    cross_tan = tan[:, 0] * e[:, 1] - tan[:, 1] * e[:, 0]
    win = np.clip(arc_window_min + 1.6 * e2, arc_window_min, arc_window_max)   # arc/chord <= 1.6 on loops
    s_est, lat_est, _ = ref.path.project(p_est[:, :2], s_hint=ref.s_pos[iref], window=win)
    along_arc = s_est - ref.s_pos[iref]
    res.update({
        'err3d': abs_stats(e3), 'err2d': abs_stats(e2),
        'x': signed_stats(e[:, 0]), 'y': signed_stats(e[:, 1]), 'z': signed_stats(e[:, 2]),
        'along_tan': signed_stats(along_tan), 'cross_tan': signed_stats(cross_tan),
        'along_arc': signed_stats(along_arc), 'cross_map': signed_stats(lat_est),
    })
    # final drift: last matched sample in time (prefer non-outlier reference)
    order = np.argsort(t_ref[iref])
    cand = order[~ref.outlier[iref][order]] if np.any(~ref.outlier[iref]) else order
    k = cand[-1]
    D = ref.distance if ref.distance >= 50.0 else np.nan     # drift % meaningless on (almost) static runs
    ds = ref.s_pos[iref] - ref.s_pos[iref].min()
    rel = e2 / np.maximum(ds, 200.0) * 100.0
    res['final'] = {
        't_rel': float(t_ref[iref][k] - t_ref[0]),
        'err3d': float(e3[k]), 'err2d': float(e2[k]), 'along_arc': float(along_arc[k]),
        'along_tan': float(along_tan[k]), 'cross_map': float(lat_est[k]),
        'drift_pct_3d': float(e3[k] / D * 100.0), 'drift_pct_2d': float(e2[k] / D * 100.0),
        'drift_pct_along': float(abs(along_arc[k]) / D * 100.0),
    }
    res['max_rel_err2d_pct'] = float(rel.max())      # max over time of e2d / max(dist so far, 200 m)
    return res


def covariance_metrics(ref: Reference, out: OutputLog, tol: float = 0.05, direction: str = 'ref2out') -> dict:
    """Consistency of reported uncertainties (if the estimator provides them).
    nees2d_mean ~ 2 and cover95_2d ~ 0.95 for a consistent 2D position covariance;
    nees_v_mean ~ 1 and cover95_v ~ 0.95 for a consistent speed variance."""
    res = {}
    if out.cov is not None:
        valid = np.all(np.isfinite(out.xyz), axis=1) & np.all(np.isfinite(out.cov), axis=1) & (out.cov[:, 0] > 0)
        iref, iout, _ = pair(ref.t_pos, out.stamp, valid, tol, direction)
        if len(iref):
            e = out.xyz[iout] - ref.xyz[iref]
            c = out.cov[iout]
            nees = e[:, 0] ** 2 / c[:, 0] + e[:, 1] ** 2 / np.maximum(c[:, 1], 1e-12)
            res.update({'nees2d_mean': float(nees.mean()), 'cover95_2d': float(np.mean(nees <= 5.991)),
                        'sigma2d_median': float(np.median(np.sqrt(c[:, 0] + c[:, 1])))})
    if out.v_var is not None:
        valid = np.isfinite(out.v) & np.isfinite(out.v_var) & (out.v_var > 0)
        iref, iout, _ = pair(ref.t_vel, out.stamp, valid, tol, direction)
        if len(iref):
            e = out.v[iout] - ref.v[iref]
            nees = e ** 2 / out.v_var[iout]
            res.update({'nees_v_mean': float(nees.mean()), 'cover95_v': float(np.mean(nees <= 3.841))})
    return res


def heading_metrics(ref: Reference, out: OutputLog, tol: float = 0.05, direction: str = 'ref2out',
                    min_speed: float = 1.0) -> dict:
    """Heading (Odometry orientation) error vs the reference path tangent, moving samples only [deg]."""
    if out.yaw is None:
        return {}
    valid = np.isfinite(out.yaw) & np.isfinite(out.stamp)
    iref, iout, _ = pair(ref.t_pos, out.stamp, valid, tol, direction)
    if len(iref) == 0:
        return {}
    v_at = np.interp(ref.tbag_pos[iref], np.sort(ref.tbag_vel), ref.v[np.argsort(ref.tbag_vel)])
    m = v_at > min_speed
    if not np.any(m):
        return {}
    tan = ref.tan_pos[iref[m]]
    e = np.degrees(np.angle(np.exp(1j * (out.yaw[iout[m]] - np.arctan2(tan[:, 1], tan[:, 0])))))
    st = signed_stats(e)
    return {'yaw_err_deg': st}


# ----------------------------------------------------------------------------------------------
# Robustness (criterion 3)
# ----------------------------------------------------------------------------------------------
def robustness_metrics(ref: Reference, bag: BagData, out: OutputLog, tol: float = 0.05,
                       direction: str = 'ref2out', anomalies: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """Speed error of the estimator vs the naive wheel average inside anomaly windows."""
    anomalies = anomalies if anomalies is not None else anomaly_regimes(ref, bag)
    valid = np.isfinite(out.v) & np.isfinite(out.stamp)
    iref, iout, _ = pair(ref.t_vel, out.stamp, valid, tol, direction)
    e = out.v[iout] - ref.v[iref]
    naive = naive_wheel_speed(ref, bag)
    en = naive[iref] - ref.v[iref]
    res = {}
    for name in ('anomaly', 'slip_any', 'slip_both', 'dropout', 'recovery', 'clean'):
        m = anomalies[name][iref]
        st = err_stats(e[m])
        nv = err_stats(en[m])
        st['naive_rmse'] = nv['rmse']
        st['naive_max'] = nv['max']
        st['frac_time'] = float(np.mean(anomalies[name])) if len(anomalies[name]) else 0.0
        res[name] = st
    # error spikes: fraction of matched samples with |e| > 1 m/s (and > 2 m/s)
    res['spike_frac_1mps'] = float(np.mean(np.abs(e) > 1.0)) if len(e) else np.nan
    res['spike_frac_2mps'] = float(np.mean(np.abs(e) > 2.0)) if len(e) else np.nan
    return res


# ----------------------------------------------------------------------------------------------
# Real-time and message validity (criteria 2 'Odometry correctness' and 4)
# ----------------------------------------------------------------------------------------------
def realtime_metrics(out: OutputLog, bag: BagData, min_rate_hz: float = 10.0) -> dict:
    n = len(out)
    dur = max(bag.duration, 1e-6)
    res = {'n_out': int(n), 'rate_hz': float(n / dur)}
    if n == 0:
        return res
    te = np.sort(out.emit_tbag)
    gaps = np.diff(np.r_[bag.t_start, te, bag.t_end])
    bins = np.floor((te - bag.t_start)).astype(int)
    counts = np.bincount(bins, minlength=int(np.ceil(dur)))[:int(np.ceil(dur))]
    age = out.emit_tbag - out.stamp
    res.update({
        'max_gap_s': float(gaps.max()), 'p99_gap_s': float(np.percentile(gaps, 99)),
        'frac_1s_bins_ge_min_rate': float(np.mean(counts >= min_rate_hz)) if len(counts) else 0.0,
        # "latency" as a judge would see it if it compared its clock with header.stamp
        'stamp_age_med_ms': float(np.median(age) * 1e3), 'stamp_age_p95_ms': float(np.percentile(age, 95) * 1e3),
        'stamp_age_max_ms': float(np.max(age) * 1e3),
        'frac_stamp_age_gt_100ms': float(np.mean(age > 0.1)), 'frac_stamp_age_gt_250ms': float(np.mean(age > 0.25)),
    })
    if out.proc_all is not None and len(out.proc_all):
        p = out.proc_all * 1e3
        res.update({'proc_mean_ms': float(p.mean()), 'proc_p99_ms': float(np.percentile(p, 99)),
                    'proc_max_ms': float(p.max()), 'cpu_load_pct': float(out.proc_all.sum() / dur * 100.0)})
    return res


def validity_metrics(out: OutputLog, bag: BagData, jump_m: float = 5.0) -> dict:
    """Sanity of published messages: NaN/inf, zero or out-of-range stamps, non-monotonic stamps,
    negative speeds, empty frame_id, position jumps between consecutive outputs."""
    n = len(out)
    res = {'n_out': int(n)}
    if n == 0:
        return res
    t_lo = min(bag.t_start, min(a[0, T_HDR] for a in (bag['front'], bag['rear'], bag['cmd']) if len(a))) - 5.0
    t_hi = bag.t_end + 5.0
    fid = [f for f in out.frame_id] if out.frame_id else [None] * n
    o = np.argsort(out.emit_tbag, kind='stable')
    st = out.stamp[o]
    p = out.xyz[o]
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1) if n > 1 else np.zeros(0)
    res.update({
        'n_nan_v': int(np.sum(~np.isfinite(out.v))),
        'n_nan_pos': int(np.sum(~np.all(np.isfinite(out.xyz), axis=1))),
        'n_zero_stamp': int(np.sum(~np.isfinite(out.stamp) | (out.stamp <= 0))),
        'n_stamp_out_of_range': int(np.sum((out.stamp < t_lo) | (out.stamp > t_hi))),
        'n_stamp_backwards': int(np.sum(np.diff(st) < 0)),
        'n_stamp_repeat': int(np.sum(np.diff(st) == 0)),
        'n_v_negative': int(np.sum(out.v < 0)),
        'n_frame_id_empty': int(sum(1 for f in fid if not f)),
        'n_pos_jump': int(np.sum(dp > jump_m)), 'max_pos_step_m': float(np.nanmax(dp)) if len(dp) else 0.0,
    })
    res['ok'] = bool(res['n_nan_v'] == 0 and res['n_nan_pos'] == 0 and res['n_zero_stamp'] == 0
                     and res['n_stamp_out_of_range'] == 0 and res['n_v_negative'] == 0)
    return res


# ----------------------------------------------------------------------------------------------
# Summary / aggregation
# ----------------------------------------------------------------------------------------------
HEADLINE = [
    # key, path in result dict
    ('v_rmse', 'speed.all.rmse'), ('v_mae', 'speed.all.mae'), ('v_bias', 'speed.all.bias'),
    ('v_max', 'speed.all.max'),
    ('v_rmse_accel', 'speed.accel.rmse'), ('v_bias_accel', 'speed.accel.bias'),
    ('v_rmse_brake', 'speed.brake.rmse'), ('v_bias_brake', 'speed.brake.bias'),
    ('v_rmse_trans', 'speed.transitions.rmse'), ('v_bias_trans', 'speed.transitions.bias'),
    ('v_rmse_stop', 'speed.stopped.rmse'),
    ('v_rmse_anom', 'robust.anomaly.rmse'), ('v_naive_rmse_anom', 'robust.anomaly.naive_rmse'),
    ('pos_rmse3d', 'pos.err3d.rmse'), ('pos_rmse2d', 'pos.err2d.rmse'), ('pos_max3d', 'pos.err3d.max'),
    ('z_rmse', 'pos.z.rmse'),
    ('along_mean', 'pos.along_arc.mean'), ('along_mean_abs', 'pos.along_arc.mean_abs'),
    ('along_rmse', 'pos.along_arc.rmse'), ('along_max', 'pos.along_arc.max'),
    ('along_tan_rmse', 'pos.along_tan.rmse'),
    ('cross_map_rmse', 'pos.cross_map.rmse'), ('cross_map_max', 'pos.cross_map.max'),
    ('final_err3d', 'pos.final.err3d'), ('drift_pct_3d', 'pos.final.drift_pct_3d'),
    ('drift_pct_2d', 'pos.final.drift_pct_2d'), ('drift_pct_along', 'pos.final.drift_pct_along'),
    ('dist_m', 'pos.distance_m'),
    ('match_v', 'speed.match_rate'), ('match_pos', 'pos.match_rate'),
    ('rate_hz', 'rt.rate_hz'), ('max_gap_s', 'rt.max_gap_s'), ('proc_p99_ms', 'rt.proc_p99_ms'),
    ('stamp_age_p95_ms', 'rt.stamp_age_p95_ms'),
    ('yaw_err_mean_abs_deg', 'heading.yaw_err_deg.mean_abs'),
]


def get_path(d: dict, path: str, default=np.nan):
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def headline(result: dict) -> Dict[str, float]:
    return {k: get_path(result, p) for k, p in HEADLINE}


def aggregate(results: List[dict]) -> dict:
    """Across-bag aggregate: mean / median / max of every headline number, plus pooled
    (sample-weighted) RMSEs and distance-weighted drift."""
    rows = [headline(r) for r in results if 'error' not in r]
    agg = {'n_bags': len(rows)}
    if not rows:
        return agg
    for k in rows[0]:
        vals = np.array([r[k] for r in rows], float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            agg[k] = {'mean': float(vals.mean()), 'median': float(np.median(vals)), 'max': float(vals.max()),
                      'min': float(vals.min())}

    def pooled(path_rmse, path_n):
        num = den = 0.0
        for r in results:
            if 'error' in r:
                continue
            rm, n = get_path(r, path_rmse), get_path(r, path_n)
            if np.isfinite(rm) and n and np.isfinite(n):
                num += n * rm * rm
                den += n
        return float(np.sqrt(num / den)) if den else np.nan

    agg['pooled'] = {
        'v_rmse': pooled('speed.all.rmse', 'speed.all.n'),
        'v_rmse_accel': pooled('speed.accel.rmse', 'speed.accel.n'),
        'v_rmse_brake': pooled('speed.brake.rmse', 'speed.brake.n'),
        'v_rmse_trans': pooled('speed.transitions.rmse', 'speed.transitions.n'),
        'pos_rmse3d': pooled('pos.err3d.rmse', 'pos.err3d.n'),
        'along_rmse': pooled('pos.along_arc.rmse', 'pos.along_arc.n'),
    }
    num = sum(get_path(r, 'pos.final.err3d', 0.0) for r in results if 'error' not in r)
    den = sum(get_path(r, 'pos.distance_m', 0.0) for r in results if 'error' not in r)
    agg['pooled']['drift_pct_3d_distance_weighted'] = float(num / den * 100) if den else np.nan
    return agg
