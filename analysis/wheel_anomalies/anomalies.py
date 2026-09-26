"""Wheel-speed sensor characterization and anomaly detectors for the tram backup-odometry case.

Layers:
1. Loading                      : `load_bag`, `Stream`, `Bag`
2. Stream integrity (GNSS-free) : `stream_events`, `stamp_anomalies`, `startup_burst`, `gaps`, `frozen_runs`,
                                  `hampel_spikes`, `cmd_checks`, `causal_accept_mask`
3. GNSS reference (offline)     : `GnssRef` (cleaned + fused master/rover Doppler speed), `align_bag`
                                  (per-receiver lag model for the +-1 s GNSS header-clock excursions),
                                  `gnss_curvature`
4. Scale factors                : `bag_scale`, `ratio_estimates`, `distance_ratio`, `segment_ratios`
5. Truth labelling (GNSS)       : `label_episodes`, `episode_table`, `classify_episode`
6. GNSS-free detection          : `CausalWheelMonitor` + `MonitorParams` (streaming, O(1)/message),
                                  `replay_monitor`, feature helpers `fr_mismatch`, `causal_rate`
7. Robustness harness           : `inject` (12 synthetic fault types)

Key facts behind the defaults (all 97 unique bags, see the CSV/PNG outputs next to this file):
* wheel topics are in km/h; k = wheel_kmh / true_ms: fleet median 3.597 (straight track), per bag
  3.545..3.659, constant within a bag (inter-stop IQR -0.04..+0.07 %), speed-independent (+-0.1 %),
  front == rear (<= 0.06 %); curves |kappa|>0.015 1/m read 0.9-1.1 % low vs GNSS.
* header stamps are the measurement times (lag vs GNSS 0.00 s); front/rear stamps identical; 9.3 Hz
  effective (7 % single drops), header jitter 0.06-0.14 s; dead band 0.15 km/h; noise ~0.006 m/s.
* true motion: accel <= 1.45 m/s^2 (traction p99.99 1.23), service decel >= -2.2 (p0.01 -1.94),
  emergency -4.4 once; notch 0/-8/-9..-15 are ambiguous (tram seen accelerating at 1 m/s^2 there).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA = Path(r'C:\MosTransHack\data')
NPZ = DATA / 'npz'
KMH = 3.6                      # nominal km/h per m/s
WHEEL_DT = 0.1                 # nominal wheel period (s); real median header dt is 0.100-0.106 s
CMD_DT = 0.05                  # nominal controller period (s)

WHEEL_KEYS = {'front': 'vehicle__front_bogie_velocity', 'rear': 'vehicle__rear_bogie_velocity'}
CMD_KEY = 'vehicle__driver_position_cmd'


# ----------------------------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------------------------
@dataclass
class Stream:
    """One topic: bag receive time, header stamp, payload (1-D or 2-D)."""
    t_bag: np.ndarray
    t_hdr: np.ndarray
    v: np.ndarray

    def __len__(self):
        return len(self.t_bag)

    def sel(self, m):
        return Stream(self.t_bag[m], self.t_hdr[m], self.v[m])


@dataclass
class Bag:
    name: str
    vehicle: str
    front: Stream
    rear: Stream
    cmd: Stream
    gnss_vel: dict = field(default_factory=dict)   # 'master'/'rover' -> Stream, v = [vx, vy, vz, wz]
    gnss_fix: dict = field(default_factory=dict)   # 'master'/'rover' -> Stream, v = [lat, lon, alt, status]
    t0: float = 0.0                                  # first bag time of any vehicle topic

    @property
    def has_gnss(self):
        return any(len(s) > 50 for s in self.gnss_vel.values())


def splits():
    return json.loads((DATA / 'splits.json').read_text())


def load_bag(name: str) -> Bag:
    z = np.load(NPZ / f'{name}.npz')

    def mk(key, cols):
        a = z[key]
        if len(a) == 0:
            return Stream(np.zeros(0), np.zeros(0), np.zeros((0,) + ((len(cols),) if len(cols) > 1 else ())))
        v = a[:, cols] if len(cols) > 1 else a[:, cols[0]]
        return Stream(a[:, 0].copy(), a[:, 1].copy(), v.copy())

    b = Bag(name=name, vehicle=name.split('_')[0],
            front=mk(WHEEL_KEYS['front'], [2]), rear=mk(WHEEL_KEYS['rear'], [2]), cmd=mk(CMD_KEY, [2]))
    for rx in ('master', 'rover'):
        b.gnss_vel[rx] = mk(f'sensing__gnss__{rx}__vel', [2, 3, 4, 5])
        b.gnss_fix[rx] = mk(f'sensing__gnss__{rx}__fix', [2, 3, 4, 5])
    b.t0 = min(s.t_bag[0] for s in (b.front, b.rear, b.cmd) if len(s))
    return b


# ----------------------------------------------------------------------------------------------
# 2. Stream integrity checks (all GNSS-free)
# ----------------------------------------------------------------------------------------------
def runs(mask: np.ndarray):
    """Return list of (start_idx, end_idx_inclusive) of True runs."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0] - 1
    return list(zip(s, e))


def rolling_median(x, w):
    """Centered rolling median with odd window w (edges use shrinking windows)."""
    from scipy.ndimage import median_filter
    return median_filter(x, size=w, mode='nearest')


def startup_burst(s: Stream, max_gap=0.02, min_span=0.3):
    """Messages delivered in one burst at recorder start (bag time compressed, header spread).
    Returns number of messages and header span of the burst."""
    if len(s) < 5:
        return 0, 0.0
    dtb = np.diff(s.t_bag)
    n = 1
    while n < len(s) and dtb[n - 1] < max_gap:
        n += 1
    span = s.t_hdr[n - 1] - s.t_hdr[0]
    if span < min_span:
        return 0, 0.0
    return n, span


def stamp_anomalies(s: Stream, excursion_thr=0.25, burst_skip=True):
    """Header-stamp integrity: non-monotonic stamps, duplicates, clock excursions.

    latency = t_bag - t_hdr. A message is an 'excursion' when its latency deviates from the running
    median latency (61 samples) by more than `excursion_thr` seconds. Start-up burst messages are
    excluded from the excursion test (their latency is large by construction)."""
    out = {}
    n = len(s)
    if n < 5:
        return dict(n=n)
    lat = s.t_bag - s.t_hdr
    nb, span = startup_burst(s) if burst_skip else (0, 0.0)
    out['burst_n'], out['burst_span'] = nb, span
    base = np.median(lat[nb:]) if n - nb > 10 else np.median(lat)
    med = rolling_median(lat, 61)
    exc = np.abs(lat - med) > excursion_thr
    exc[:nb] = False
    # slow excursions (ramps): running median itself departs from global baseline
    slow = np.abs(med - base) > excursion_thr
    slow[:nb] = False
    dth = np.diff(s.t_hdr)
    out.update(lat_median=float(base), lat_p50=float(np.median(lat[nb:])), lat_p99=float(np.percentile(lat[nb:], 99)),
               lat_min=float(lat[nb:].min()), lat_max=float(lat[nb:].max()),
               n_nonmono=int((dth < 0).sum()), n_dup_stamp=int((dth == 0).sum()),
               n_excursion_msgs=int(exc.sum()), n_slow_excursion_msgs=int(slow.sum()),
               excursion_runs=[(float(s.t_bag[a] ), float(s.t_bag[b]), float(np.median(lat[a:b + 1] - base)))
                               for a, b in runs(exc | slow)])
    return out


def stamp_ok_mask(s: Stream):
    """Offline 'clean timeline' mask: keeps every message except exact duplicate header stamps.
    Vehicle header stamps were verified to be self-consistent (the +-1 s latency excursions come from
    steps of the *recorder* clock, i.e. bag time, while header stamps stay continuous), so offline
    analysis should sort by header stamp (see `clean_order`) rather than drop 'late' messages."""
    n = len(s)
    ok = np.ones(n, bool)
    if n < 2:
        return ok
    o = np.argsort(s.t_hdr, kind='stable')
    dup = np.r_[False, np.diff(s.t_hdr[o]) <= 0]
    ok[o[dup]] = False
    return ok


def causal_accept_mask(s: Stream, max_future=None):
    """What a causal node that processes messages in *arrival (bag) order* and keeps only messages with
    a header stamp newer than the last accepted one would use. Returns boolean mask (arrival order)."""
    o = np.argsort(s.t_bag, kind='stable')
    ok = np.zeros(len(s), bool)
    last = -np.inf
    for i in o:
        if s.t_hdr[i] > last:
            ok[i] = True
            last = s.t_hdr[i]
    return ok


def gaps(t: np.ndarray, thr: float):
    """Gaps between consecutive (sorted) timestamps larger than thr: array of (t_start, dt)."""
    if len(t) < 2:
        return np.zeros((0, 2))
    ts = np.sort(t)
    d = np.diff(ts)
    i = np.where(d > thr)[0]
    return np.c_[ts[i], d[i]]


def gap_histogram(t, edges=(0.15, 0.25, 0.35, 0.5, 1.0, 2.0, 5.0, 1e9)):
    d = np.diff(np.sort(t))
    return {f'>{edges[i]:g}s': int(((d > edges[i]) & (d <= edges[i + 1])).sum()) for i in range(len(edges) - 1)}


def frozen_runs(t, v, min_dur=1.0, min_abs=0.3):
    """Runs of bit-identical consecutive values with |v| >= min_abs lasting >= min_dur seconds.
    (A real moving wheel at ~10 Hz practically never repeats a float64 value exactly.)"""
    out = []
    if len(v) < 3:
        return out
    same = np.r_[False, v[1:] == v[:-1]]
    # group: a run of identical values includes the first element
    for a, b in runs(same):
        a0 = a - 1
        if abs(v[a0]) < min_abs:
            continue
        dur = t[b] - t[a0]
        if dur >= min_dur:
            out.append((float(t[a0]), float(dur), int(b - a0 + 1), float(v[a0])))
    return out


def repeat_stats(v, min_abs=0.3):
    """Fraction of consecutive exactly-repeated values among samples with |v|>=min_abs."""
    m = np.abs(v[1:]) >= min_abs
    if m.sum() == 0:
        return 0.0
    return float(np.mean((v[1:] == v[:-1])[m]))


def hampel_spikes(t, v, half_window=5, n_sigma=6.0, min_abs_dev=0.5):
    """Isolated outliers: |v - rolling median| > max(n_sigma * 1.4826 * MAD, min_abs_dev).
    v in m/s. Returns indices."""
    n = len(v)
    if n < 2 * half_window + 1:
        return np.zeros(0, int)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.r_[np.full(half_window, v[0]), v, np.full(half_window, v[-1])]
    W = sliding_window_view(pad, 2 * half_window + 1)
    med = np.median(W, axis=1)
    mad = 1.4826 * np.median(np.abs(W - med[:, None]), axis=1)
    dev = np.abs(v - med)
    return np.where(dev > np.maximum(n_sigma * mad, min_abs_dev))[0]


def diff_rate(t, v):
    """Finite-difference derivative with guard against dt<=0."""
    dt = np.diff(t)
    dv = np.diff(v)
    r = np.full(len(dt), np.nan)
    ok = dt > 1e-3
    r[ok] = dv[ok] / dt[ok]
    return r


def stream_events(b: Bag, gap_thr=0.35, frozen_min=1.0):
    """Discrete GNSS-free integrity events for the three input topics of a bag.
    Returns list of dicts: topic, kind, t_rel (s from bag start, bag time), dur, detail."""
    ev = []
    t0 = b.t0
    for topic, s in (('front', b.front), ('rear', b.rear), ('cmd', b.cmd)):
        if len(s) < 5:
            ev.append(dict(topic=topic, kind='missing_topic', t_rel=0.0, dur=np.nan, detail=f'n={len(s)}'))
            continue
        nb, span = startup_burst(s)
        if nb:
            ev.append(dict(topic=topic, kind='startup_burst', t_rel=float(s.t_bag[0] - t0), dur=span,
                           detail=f'{nb} msgs delivered at once, header span {span:.2f}s'))
        ok = stamp_ok_mask(s)
        # gaps on the header-sorted timeline (true sampling gaps)
        o = np.where(ok)[0]
        o = o[np.argsort(s.t_hdr[o], kind='stable')]
        th = s.t_hdr[o]; vv = s.v[o]
        off = np.median(s.t_bag - s.t_hdr)
        d = np.diff(th)
        for i in np.where(d > (gap_thr if topic != 'cmd' else 0.2))[0]:
            ev.append(dict(topic=topic, kind='gap', t_rel=float(th[i] + off - t0), dur=float(d[i]),
                           detail=f'v_before={vv[i]:.2f} v_after={vv[i + 1]:.2f}'))
        # gaps as seen in arrival order by a causal node that drops stale (older-stamp) messages
        ca = causal_accept_mask(s)
        tb_c = np.sort(s.t_bag[ca])
        dc = np.diff(tb_c)
        for i in np.where(dc > (gap_thr if topic != 'cmd' else 0.2) + 0.25)[0]:
            ev.append(dict(topic=topic, kind='arrival_gap', t_rel=float(tb_c[i] - t0), dur=float(dc[i]),
                           detail='no new-stamp message in arrival order'))
        # end-of-topic silence (topic stops while others continue)
        t_end_all = max(x.t_bag[-1] for x in (b.front, b.rear, b.cmd) if len(x))
        if t_end_all - s.t_bag[-1] > 1.0:
            ev.append(dict(topic=topic, kind='gap', t_rel=float(s.t_bag[-1] - t0), dur=float(t_end_all - s.t_bag[-1]),
                           detail='topic silent until end of bag'))
        st = stamp_anomalies(s)
        for a, e, off in st.get('excursion_runs', []):
            ev.append(dict(topic=topic, kind='stamp_excursion', t_rel=float(a - t0), dur=float(e - a),
                           detail=f'latency offset {off:+.2f}s'))
        dth = np.diff(s.t_hdr)
        for i in np.where(dth < 0)[0]:
            ev.append(dict(topic=topic, kind='stamp_nonmonotonic', t_rel=float(s.t_bag[i + 1] - t0), dur=0.0,
                           detail=f'dt_hdr={dth[i]:+.3f}'))
        if topic in ('front', 'rear'):
            v = s.v
            for i in np.where(~np.isfinite(v))[0]:
                ev.append(dict(topic=topic, kind='nonfinite', t_rel=float(s.t_bag[i] - t0), dur=0.0, detail=str(v[i])))
            for i in np.where(v < 0)[0]:
                ev.append(dict(topic=topic, kind='negative', t_rel=float(s.t_bag[i] - t0), dur=0.0,
                               detail=f'v={v[i]:.3f} km/h'))
            for tt, dur, n, val in frozen_runs(s.t_bag, v, frozen_min, min_abs=0.3 * KMH):
                ev.append(dict(topic=topic, kind='frozen', t_rel=float(tt - t0),
                               dur=dur, detail=f'{n} identical samples v={val:.3f} km/h'))
        else:
            v = s.v
            bad = (v < -15) | (v > 15) | (np.abs(v - np.round(v)) > 1e-9)
            for i in np.where(bad)[0]:
                ev.append(dict(topic=topic, kind='cmd_invalid', t_rel=float(s.t_bag[i] - t0), dur=0.0, detail=f'{v[i]}'))
            dv = np.diff(v)
            for i in np.where(np.abs(dv) > 3)[0]:
                ev.append(dict(topic=topic, kind='cmd_jump', t_rel=float(s.t_bag[i + 1] - t0), dur=0.0,
                               detail=f'{v[i]:+.0f} -> {v[i + 1]:+.0f}'))
    return ev


def cmd_checks(s: Stream):
    """Controller topic integrity: range, integer-ness, step sizes, repeated messages."""
    v = s.v
    out = dict(n=len(v))
    if len(v) == 0:
        return out
    d = np.diff(v)
    out.update(min=float(v.min()), max=float(v.max()),
               n_out_of_range=int(((v < -15) | (v > 15)).sum()),
               n_non_integer=int((np.abs(v - np.round(v)) > 1e-9).sum()),
               n_steps=int((d != 0).sum()),
               step_hist={int(k): int(c) for k, c in zip(*np.unique(d[d != 0], return_counts=True))},
               n_big_steps=int((np.abs(d) > 2).sum()),
               frac_zero=float(np.mean(v == 0)), frac_trac=float(np.mean(v > 0)), frac_brake=float(np.mean(v < 0)))
    return out


# ----------------------------------------------------------------------------------------------
# 3. GNSS reference
# ----------------------------------------------------------------------------------------------
def gnss_speed_stream(b: Bag, rx='master', dims=2, vmax=25.0, hampel_w=5, hampel_thr=0.5):
    """Cleaned GNSS speed stream of one receiver, sorted by header stamp.
    Removed: exact-zero velocity vectors (receiver 'no solution' marker; seen while moving in 5 bags),
    |v| > vmax, and Hampel spikes (|v - rolling median(2*hampel_w+1)| > max(hampel_thr, 5*MAD))."""
    s = b.gnss_vel[rx]
    if len(s) == 0:
        return None
    vx, vy, vz = s.v[:, 0], s.v[:, 1], s.v[:, 2]
    sp = np.hypot(vx, vy) if dims == 2 else np.sqrt(vx * vx + vy * vy + vz * vz)
    ok = stamp_ok_mask(s) & np.isfinite(sp)
    ok &= ~((vx == 0) & (vy == 0) & (vz == 0))
    ok &= np.hypot(vx, vy) <= vmax
    idx = np.where(ok)[0]
    idx = idx[np.argsort(s.t_hdr[idx], kind='stable')]
    x = np.hypot(vx, vy)[idx]
    if len(x) > 2 * hampel_w + 1:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.r_[np.full(hampel_w, x[0]), x, np.full(hampel_w, x[-1])]
        W = sliding_window_view(pad, 2 * hampel_w + 1)
        med = np.median(W, axis=1)
        mad = 1.4826 * np.median(np.abs(W - med[:, None]), axis=1)
        spike = np.abs(x - med) > np.maximum(hampel_thr, 5 * mad)
        idx = idx[~spike]
    return Stream(s.t_bag[idx], s.t_hdr[idx], sp[idx])


def interp_nogap(tq, t, v, max_gap=0.25):
    """Linear interpolation that returns NaN where the bracketing samples are > max_gap apart
    or tq is outside the data range."""
    out = np.full(len(tq), np.nan)
    if len(t) < 2:
        return out
    i = np.searchsorted(t, tq)
    ok = (i > 0) & (i < len(t))
    i1 = np.clip(i, 1, len(t) - 1)
    t0, t1 = t[i1 - 1], t[i1]
    ok &= (t1 - t0) <= max_gap
    w = np.where(t1 > t0, (tq - t0) / np.where(t1 > t0, t1 - t0, 1), 0)
    out[ok] = (v[i1 - 1] + w * (v[i1] - v[i1 - 1]))[ok]
    return out


class GnssRef:
    """Cleaned master/rover GNSS speed streams (sorted by stamp) with fused evaluation."""

    def __init__(self, b: Bag, time_base='hdr'):
        self.streams = {}
        self.latency = {}
        for dims in (2, 3):
            for rx in ('master', 'rover'):
                s = gnss_speed_stream(b, rx, dims)
                if s is None or len(s) < 20:
                    continue
                t = s.t_hdr if time_base == 'hdr' else s.t_bag
                o = np.argsort(t, kind='stable')
                self.streams[(rx, dims)] = (t[o], s.v[o])
                if dims == 2:
                    raw = b.gnss_vel[rx]
                    nb, _ = startup_burst(raw)
                    t_first_ok = raw.t_hdr[nb] if nb < len(raw) else -np.inf
                    oo = o[s.t_hdr[o] >= t_first_ok]
                    if len(oo) < 21:
                        oo = o
                    self.latency[rx] = (s.t_hdr[oo], rolling_median(s.t_bag[oo] - s.t_hdr[oo], 21))

    def receivers(self):
        return [rx for rx in ('master', 'rover') if (rx, 2) in self.streams]

    def single(self, rx, tq, dims=2, max_gap=0.25):
        if (rx, dims) not in self.streams:
            return np.full(len(tq), np.nan)
        t, v = self.streams[(rx, dims)]
        return interp_nogap(tq, t, v, max_gap)

    def at(self, tq, dims=2, max_gap=0.25, disagree_thr=0.25, lags=None):
        """Fused speed at tq (+ per-receiver lag arrays `lags` dict). Quality: 0 none, 1 single receiver,
        2 both agree, 3 both disagree (value set to NaN: truth unknown)."""
        vals = []
        for rx in ('master', 'rover'):
            if (rx, dims) in self.streams:
                t, v = self.streams[(rx, dims)]
                L = 0.0 if lags is None else lags.get(rx, 0.0)
                vals.append(interp_nogap(tq + L, t, v, max_gap))
            else:
                vals.append(np.full(len(tq), np.nan))
        m, r = vals
        both = np.isfinite(m) & np.isfinite(r)
        sp = np.where(np.isfinite(m), m, r)
        q = np.where(np.isfinite(sp), 1, 0)
        agree = both & (np.abs(m - r) <= disagree_thr)
        q[agree] = 2
        q[both & ~agree] = 3
        sp[agree] = 0.5 * (m[agree] + r[agree])
        sp[both & ~agree] = np.nan
        return sp, q


def gnss_curvature(b: Bag, tq, rx='master', vmin=1.0, win_s=1.5):
    """Signed path curvature kappa = yaw_rate / speed (1/m; >0 = left turn in ENU) from GNSS velocity
    heading, evaluated at header times tq (apply the same lag as the speed reference)."""
    from scipy.signal import savgol_filter
    s = b.gnss_vel[rx]
    if len(s) < 50:
        return np.full(len(tq), np.nan)
    o = np.argsort(s.t_hdr, kind='stable')
    t = s.t_hdr[o]; vx = s.v[o, 0]; vy = s.v[o, 1]
    sp = np.hypot(vx, vy)
    psi = np.unwrap(np.arctan2(vy, vx))
    # uniform resample 0.1 s
    tu = np.arange(t[0], t[-1], 0.1)
    psu = np.interp(tu, t, psi); spu = np.interp(tu, t, sp)
    n = int(round(win_s / 0.1)) | 1
    yr = savgol_filter(psu, n, 2, deriv=1, delta=0.1)
    kap = np.where(spu > vmin, yr / np.maximum(spu, vmin), np.nan)
    # blank across GNSS gaps
    gap = np.interp(tu, t[1:], np.diff(t)) > 0.3
    kap[gap] = np.nan
    return interp_nogap(tq, tu, kap, 0.25)


def fused_gnss_speed(b: Bag, tq, time_base='hdr', dims=2, lag=0.0, max_gap=0.25, disagree_thr=0.2):
    """Convenience wrapper (rebuilds GnssRef each call; use GnssRef directly in loops)."""
    return GnssRef(b, time_base).at(tq + lag, dims, max_gap, disagree_thr)


def wheel_grid(b: Bag, dt=0.1, skip_start=0.0):
    """Common grid in wheel header time covering both wheel streams."""
    t_start = max(b.front.t_hdr[0] if len(b.front) else np.inf, b.rear.t_hdr[0] if len(b.rear) else np.inf)
    t_start = min(t_start, np.nanmin([b.front.t_hdr[0] if len(b.front) else np.nan,
                                      b.rear.t_hdr[0] if len(b.rear) else np.nan]))
    t_end = np.nanmax([b.front.t_hdr[-1] if len(b.front) else np.nan, b.rear.t_hdr[-1] if len(b.rear) else np.nan])
    return np.arange(t_start + skip_start, t_end, dt)


def clean_wheel(s: Stream):
    ok = stamp_ok_mask(s) & np.isfinite(s.v)
    return s.sel(ok)


def wheel_on_grid(s: Stream, tg, max_gap=0.35, time_base='hdr'):
    """Wheel speed (m/s, nominal 3.6) interpolated on grid without bridging gaps > max_gap."""
    c = clean_wheel(s)
    t = c.t_hdr if time_base == 'hdr' else c.t_bag
    o = np.argsort(t, kind='stable')
    return interp_nogap(tg, t[o], c.v[o] / KMH, max_gap)


def local_lag(tg, w, g, win=16.0, step=2.0, lags=np.arange(-1.5, 1.5001, 0.02), min_info=0.4, dt=0.1,
              max_cost=0.12, min_gain=0.6):
    """Windowed lag between wheel speed w(t) and GNSS speed g(t) on a uniform grid tg (spacing dt).
    Lag L means: g(t + L) best matches w(t). Scale is fitted per window (so it is lag only).
    A window estimate is accepted only if (i) wheel speed varies (std >= min_info m/s),
    (ii) the robust cost (median |residual|, m/s) at the optimum is <= max_cost,
    (iii) the optimum is interior (not on the search boundary) and
    (iv) the cost at +-0.5 s from the optimum is >= optimum/min_gain (identifiable minimum).
    Returns window centers, lag estimates (NaN if rejected), optimum cost."""
    n = len(tg)
    wl = int(round(win / dt)); st = int(round(step / dt))
    # g may be an array on tg (then lags are rounded to the grid) or a callable g(tq) (exact lags)
    if callable(g):
        G = np.vstack([g(tg + L) for L in lags])
    else:
        shifts = np.round(lags / dt).astype(int)
        G = np.full((len(lags), n), np.nan)
        for j, sh in enumerate(shifts):
            if sh >= 0:
                G[j, :n - sh] = g[sh:]
            else:
                G[j, -sh:] = g[:n + sh]
    centers, est, cost = [], [], []
    for a in range(0, max(n - wl, 1), st):
        seg = slice(a, a + wl)
        ww = w[seg]
        wv = ww[np.isfinite(ww)]
        centers.append(tg[min(a + wl // 2, n - 1)])
        if len(wv) < wl * 0.6 or np.std(wv) < min_info:
            est.append(np.nan); cost.append(np.nan)
            continue
        cs = np.full(len(lags), np.inf)
        GG = G[:, seg]
        for j in range(len(lags)):
            gg = GG[j]
            m = np.isfinite(gg) & np.isfinite(ww)
            if m.sum() < wl * 0.6:
                continue
            kk = np.sum(ww[m] * gg[m]) / max(np.sum(gg[m] ** 2), 1e-6)
            cs[j] = np.median(np.abs(ww[m] - gg[m] * kk))
        j = int(np.argmin(cs))
        c = cs[j]
        side = int(round(0.5 / (lags[1] - lags[0])))
        neigh = [cs[jj] for jj in (j - side, j + side) if 0 <= jj < len(cs)]
        ok = (np.isfinite(c) and c <= max_cost and 0 < j < len(lags) - 1 and np.isfinite(cs[j - 1])
              and np.isfinite(cs[j + 1]) and len(neigh) > 0 and min(neigh) * min_gain >= c)
        est.append(lags[j] if ok else np.nan); cost.append(c)
    return np.array(centers), np.array(est), np.array(cost)


def smooth_lag(tc, lag, tg, k=5, default=0.0):
    """Median-filter valid window lags and interpolate (nearest-hold) on tg."""
    ok = np.isfinite(lag)
    if ok.sum() == 0:
        return np.full(len(tg), default)
    tl, ll = tc[ok], lag[ok]
    if len(ll) >= k:
        ll = rolling_median(ll, k)
    return np.interp(tg, tl, ll)


@dataclass
class Aligned:
    """Wheel and reference on a common grid (wheel header time)."""
    bag: str
    t: np.ndarray             # grid (header time of wheels)
    front: np.ndarray         # m/s (nominal /3.6), NaN in gaps
    rear: np.ndarray
    notch: np.ndarray         # controller (zero-order hold, header time), NaN if cmd gap > 0.5 s
    ref: np.ndarray           # fused GNSS speed aligned to wheel time (m/s), NaN if unavailable
    ref_q: np.ndarray         # quality code of ref
    lag: np.ndarray           # applied lag (s)
    ref3d: np.ndarray = None  # 3-D speed (incl. vz)


def hold_on_grid(s: Stream, tg, max_age=0.5, time_base='hdr'):
    """Zero-order hold of a (possibly irregular) stream on grid; NaN if last sample older than max_age."""
    c = s.sel(stamp_ok_mask(s))
    t = c.t_hdr if time_base == 'hdr' else c.t_bag
    o = np.argsort(t, kind='stable'); t = t[o]; v = c.v[o]
    i = np.searchsorted(t, tg, side='right') - 1
    out = np.full(len(tg), np.nan)
    ok = i >= 0
    out[ok] = v[i[ok]]
    age = np.full(len(tg), np.inf); age[ok] = tg[ok] - t[i[ok]]
    out[age > max_age] = np.nan
    return out


def align_bag(b: Bag, dt=0.1, win=16.0, step=4.0) -> Aligned:
    """Wheel/controller/GNSS on a common grid in wheel header time. The GNSS reference of each receiver is
    shifted by lag_rx(t) = lat_wheel(t) - lat_rx(t) + c_rx, where lat = bag_time - header_stamp (rolling
    median) captures the +-1 s GNSS header-clock excursions, and c_rx is calibrated from data-driven
    windowed lag estimates (`local_lag`)."""
    tg = wheel_grid(b, dt)
    f = wheel_on_grid(b.front, tg)
    r = wheel_on_grid(b.rear, tg)
    notch = hold_on_grid(b.cmd, tg)
    if not b.has_gnss:
        nan = np.full(len(tg), np.nan)
        al = Aligned(b.name, tg, f, r, notch, nan, np.zeros(len(tg), int), nan, nan)
        al.lag_info, al.lags, al.lag_windows = {}, {}, (np.zeros(0), np.zeros(0))
        return al
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            w = np.nanmean(np.c_[f, r], axis=1)
    G = GnssRef(b, 'hdr')
    # wheel latency (bag - header) on grid, robust; start-up burst messages excluded
    c = clean_wheel(b.front if len(b.front) >= len(b.rear) else b.rear)
    nb, _ = startup_burst(c)
    o = np.argsort(c.t_hdr[nb:], kind='stable') + nb
    lw = np.interp(tg, c.t_hdr[o], rolling_median(c.t_bag[o] - c.t_hdr[o], 21))
    lags, lag_dd, lag_info = {}, {}, {}
    for rx in G.receivers():
        # data-driven windows for this receiver
        tc, lg, _ = local_lag(tg, w, lambda tq, rx=rx: G.single(rx, tq), win=win, step=step, dt=dt)
        # latency-based prediction: lag = lat_wheel - lat_gnss + c_rx
        tl, ll = G.latency[rx]
        pred = lw - np.interp(tg, tl, ll)
        pred_c = np.interp(tc, tg, pred)
        ok = np.isfinite(lg)
        c_rx = float(np.median(lg[ok] - pred_c[ok])) if ok.sum() >= 5 else 0.04
        lag_rx = np.clip(pred + c_rx, -1.3, 1.3)   # observed excursions are within +-1.1 s
        # 5 s rolling median: removes 1-3 s latency glitches (recorder-clock steps, delivery bursts) while
        # following the ~10 s header-clock ramps and >20 s plateaus
        lag_rx = rolling_median(lag_rx, int(round(5.0 / dt)) | 1)
        resid = lg - (pred_c + c_rx)
        lags[rx] = lag_rx
        lag_dd[rx] = (tc, lg)
        lag_info[rx] = dict(c=c_rx, n_windows=int(ok.sum()),
                            resid_mad=float(np.median(np.abs(resid[ok]))) if ok.any() else np.nan,
                            frac_bad=float(np.mean(np.abs(resid[ok]) > 0.15)) if ok.any() else np.nan)
    ref, q = G.at(tg, lags=lags)
    ref3, _ = G.at(tg, dims=3, lags=lags)
    lag_main = lags.get('master', lags.get('rover', np.zeros(len(tg))))
    al = Aligned(b.name, tg, f, r, notch, ref, q, lag_main, ref3)
    al.lag_windows = lag_dd.get('master', lag_dd.get('rover', (np.zeros(0), np.zeros(0))))
    al.lag_info = lag_info
    al.lags = lags
    rx0 = 'master' if 'master' in lags else 'rover'
    al.kappa = gnss_curvature(b, tg + lags[rx0], rx=rx0)
    return al


# ----------------------------------------------------------------------------------------------
# 4. Scale factors
# ----------------------------------------------------------------------------------------------
def savgol_deriv(v, dt, win_s=1.0, order=2):
    from scipy.signal import savgol_filter
    n = int(round(win_s / dt)) | 1
    n = max(n, order + 2 | 1)
    x = np.array(v, float)
    m = np.isfinite(x)
    if m.sum() < n:
        return np.full(len(x), np.nan)
    xi = np.interp(np.arange(len(x)), np.where(m)[0], x[m])
    d = savgol_filter(xi, n, order, deriv=1, delta=dt)
    d[~m] = np.nan
    return d


def steady_mask(al: Aligned, vmin=2.0, amax=0.25):
    """Samples suitable for scale estimation: moving, low acceleration (GNSS), reference agreed."""
    a = savgol_deriv(al.ref, al.t[1] - al.t[0], 2.0)
    return (al.ref > vmin) & (np.abs(a) < amax) & (al.ref_q >= 1) & np.isfinite(al.ref)


def ratio_estimates(al: Aligned, mask, sensor='front'):
    """Speed ratio k = v_wheel(km/h) / v_ref(m/s) (i.e. 3.6*(1+eps)).
    Returns LS ratio, median ratio, n."""
    w = getattr(al, sensor) * KMH
    m = mask & np.isfinite(w) & np.isfinite(al.ref)
    if m.sum() < 20:
        return np.nan, np.nan, int(m.sum())
    ls = np.sum(w[m] * al.ref[m]) / np.sum(al.ref[m] ** 2)
    med = np.median(w[m] / al.ref[m])
    return float(ls), float(med), int(m.sum())


def distance_ratio(al: Aligned, sensor='front', exclude=None, ref=None):
    """Integrated wheel distance / integrated GNSS distance over samples where both exist.
    Returns ratio in km/h per m/s units (so comparable with speed ratio), wheel dist (m, nominal),
    ref dist (m)."""
    w = getattr(al, sensor)
    g = al.ref if ref is None else ref
    m = np.isfinite(w) & np.isfinite(g)
    if exclude is not None:
        m &= ~exclude
    dt = al.t[1] - al.t[0]
    dw = np.sum(w[m]) * dt
    dg = np.sum(g[m]) * dt
    return (KMH * dw / dg if dg > 0 else np.nan), dw, dg


def episode_mask(al: Aligned, eps, pad_s=2.0):
    """Boolean mask covering labelled episodes (any sensor) with padding."""
    m = np.zeros(len(al.t), bool)
    dt = al.t[1] - al.t[0]
    pad = int(round(pad_s / dt))
    for ep in eps:
        m[max(ep['i0'] - pad, 0):ep['i1'] + pad + 1] = True
    return m


def lag_transition_mask(al: Aligned, rate_thr=0.05, pad_s=5.0):
    """Samples where the wheel<->GNSS lag is changing fast (GNSS header-clock excursion transitions):
    the reference speed is less reliable there during accelerations."""
    dt = al.t[1] - al.t[0]
    lag = rolling_median(np.nan_to_num(al.lag), 11)
    r = np.abs(np.gradient(lag, dt)) > rate_thr
    pad = int(round(pad_s / dt))
    m = np.zeros(len(al.t), bool)
    for a, b in runs(r):
        m[max(a - pad, 0):b + pad + 1] = True
    return m


def bag_scale(al: Aligned, p: 'EpisodeParams' = None, n_iter=2, kappa=None, kappa_straight=0.003):
    """Per-bag scale factors (km/h per m/s) for both sensors with episode exclusion.
    k_speed_*    : LS speed ratio on steady (|a|<0.25), moving (>2 m/s), *straight* (|kappa|<0.003) samples
                   = intrinsic sensor scale (curves read ~0.9 % low, see REPORT)
    k_speedall_* : same but including curves
    k_dist_*     : distance ratio over all moving samples (curves included; standstill excluded because
                   GNSS noise integrates to spurious distance there), episodes/lag transitions excluded
    k_dist_all_* : distance ratio over everything moving (nothing excluded) = what position drift sees
    Returns dict + the episode list from the final iteration (key 'eps')."""
    p = p or EpisodeParams()
    excl = lag_transition_mask(al)
    straight = np.ones(len(al.t), bool) if kappa is None else (np.abs(kappa) < kappa_straight)
    steady_all = steady_mask(al) & ~excl
    steady = steady_all & straight
    k = {}
    for s in ('front', 'rear'):
        k[s] = ratio_estimates(al, steady, s)[0]
    eps = []
    for _ in range(n_iter):
        if not all(np.isfinite(list(k.values()))):
            break
        eps = label_episodes(al, k, p)
        em = episode_mask(al, eps) | excl
        for s in ('front', 'rear'):
            k[s] = ratio_estimates(al, steady & ~em, s)[0]
    out = dict(eps=eps)
    em = episode_mask(al, eps) | excl if eps else excl
    moving = (np.nan_to_num(al.ref) > 0.1) | (np.nan_to_num(al.front) > 0.1) | (np.nan_to_num(al.rear) > 0.1)
    for s in ('front', 'rear'):
        ls, med, n = ratio_estimates(al, steady & ~em, s)
        lsa, _, _ = ratio_estimates(al, steady_all & ~em, s)
        dist, dw, dg = distance_ratio(al, s, exclude=em | ~moving)
        dist_all, dw_all, dg_all = distance_ratio(al, s, exclude=~moving)
        dist_str, _, _ = distance_ratio(al, s, exclude=em | ~moving | ~straight)
        al3 = Aligned(al.bag, al.t, al.front, al.rear, al.notch, al.ref3d, al.ref_q, al.lag)
        ls3, _, _ = ratio_estimates(al3, steady & ~em, s)
        dist3, _, _ = distance_ratio(al3, s, exclude=em | ~moving)
        out.update({f'k_speed_{s}': ls, f'k_speed_med_{s}': med, f'n_steady_{s}': n, f'k_speedall_{s}': lsa,
                    f'k_dist_{s}': dist, f'k_dist_all_{s}': dist_all, f'k_dist_straight_{s}': dist_str,
                    f'dist_wheel_{s}_m': dw_all, f'k_speed3d_{s}': ls3, f'k_dist3d_{s}': dist3})
        out['dist_ref_m'] = dg_all
    out['frac_dist_curve'] = float(np.nansum(np.where(moving & ~straight, al.ref, 0)) /
                                   max(np.nansum(np.where(moving, al.ref, 0)), 1e-9)) if kappa is not None else np.nan
    out['k'] = {s: out[f'k_speed_{s}'] for s in ('front', 'rear')}
    return out


def segment_ratios(al: Aligned, k_mask=None, vstop=0.05, min_dist=100.0):
    """Distance ratio per inter-stop movement segment (km/h per m/s). Returns rows with t_mid,
    distance, ratio front/rear, mean speed."""
    rows = []
    dt = al.t[1] - al.t[0]
    for a, b in movement_segments(al.ref, al.t, vstop, 10.0):
        seg = slice(a, b + 1)
        m = np.isfinite(al.ref[seg])
        if k_mask is not None:
            m &= ~k_mask[seg]
        dg = np.nansum(np.where(m, al.ref[seg], 0)) * dt
        if dg < min_dist:
            continue
        r = dict(t_mid=float(0.5 * (al.t[a] + al.t[b])), dist=float(dg), v_mean=float(np.nanmean(al.ref[seg])),
                 dur=float(al.t[b] - al.t[a]))
        for s in ('front', 'rear'):
            w = getattr(al, s)[seg]
            mm = m & np.isfinite(w)
            dgm = np.sum(al.ref[seg][mm]) * dt
            r[f'k_{s}'] = float(KMH * np.sum(w[mm]) * dt / dgm) if dgm > 0 else np.nan
            r[f'cov_{s}'] = float(np.mean(mm))
        rows.append(r)
    return rows


def movement_segments(v, t, vstop=0.05, min_len_s=10.0):
    """Segments between standstills (v <= vstop) of at least min_len_s duration: list of (i0,i1)."""
    mv = np.isfinite(v) & (v > vstop)
    out = []
    for a, b in runs(mv):
        if t[b] - t[a] >= min_len_s:
            out.append((a, b))
    return out


# ----------------------------------------------------------------------------------------------
# 5. Truth labelling vs GNSS (offline only)
# ----------------------------------------------------------------------------------------------
@dataclass
class EpisodeParams:
    thr_abs: float = 0.30      # m/s: episode core threshold on |wheel/k - ref|
    thr_rel: float = 0.05      # relative to ref speed (core threshold = max(abs, rel*ref))
    thr_ext: float = 0.12      # m/s: hysteresis to extend the core
    min_dur: float = 0.25      # s
    merge_gap: float = 0.6     # s: merge episodes separated by less
    lock_v: float = 0.1        # m/s: wheel reading below -> 'lock/zero'


def label_episodes(al: Aligned, k: dict, p: EpisodeParams = EpisodeParams(), invalid=None):
    """Return list of dict episodes where a wheel sensor deviates from the GNSS reference.
    k: {'front': ratio, 'rear': ratio} (km/h per m/s) used to de-bias the wheel before comparing.
    invalid: optional mask of samples where the reference must not be trusted (default: fast lag
    transitions of the GNSS header clock)."""
    eps = []
    dt = al.t[1] - al.t[0]
    ref = al.ref
    if invalid is None:
        invalid = lag_transition_mask(al)
    for sensor in ('front', 'rear'):
        w = getattr(al, sensor) * KMH / k[sensor]
        e = w - ref
        valid = np.isfinite(e) & (al.ref_q >= 1) & ~invalid
        thr = np.maximum(p.thr_abs, p.thr_rel * np.nan_to_num(ref))
        core = valid & (np.abs(e) > thr)
        ext = valid & (np.abs(e) > p.thr_ext)
        # extend cores within ext runs
        lab = np.zeros(len(e), bool)
        for a, bb in runs(ext):
            if core[a:bb + 1].any():
                lab[a:bb + 1] = True
        # merge close runs
        rr = runs(lab)
        merged = []
        for a, bb in rr:
            if merged and (a - merged[-1][1]) * dt <= p.merge_gap:
                merged[-1] = (merged[-1][0], bb)
            else:
                merged.append((a, bb))
        for a, bb in merged:
            dur = (bb - a + 1) * dt
            if dur < p.min_dur:
                continue
            seg = slice(a, bb + 1)
            ee = e[seg]
            i_pk = a + int(np.nanargmax(np.abs(ee)))
            eps.append(dict(sensor=sensor, i0=a, i1=bb, t0=al.t[a], t1=al.t[bb], dur=dur,
                            e_peak=float(e[i_pk]), e_mean=float(np.nanmean(ee)),
                            i_peak=i_pk))
    return eps


def mode_int(x):
    x = x[np.isfinite(x)].astype(int)
    if len(x) == 0:
        return np.nan
    vals, cnt = np.unique(x, return_counts=True)
    return int(vals[np.argmax(cnt)])


def episode_table(al: Aligned, k: dict, p: EpisodeParams = EpisodeParams(), t0=None, pad=1.0):
    """Label episodes and compute descriptive + GNSS-free features for each.
    Returns list of dict rows (one per sensor episode)."""
    dt = al.t[1] - al.t[0]
    t0 = al.t[0] if t0 is None else t0
    eps = label_episodes(al, k, p)
    ws = {s: getattr(al, s) * KMH / k[s] for s in ('front', 'rear')}
    e = {s: ws[s] - al.ref for s in ws}
    fr = ws['front'] - ws['rear']
    acc = {s: causal_rate(ws[s], 2, dt) for s in ws}          # 0.2 s causal difference
    jerk = {s: causal_rate(acc[s], 2, dt) for s in ws}
    a_ref = savgol_deriv(al.ref, dt, 1.0)
    lag_rate = np.abs(np.gradient(al.lag, dt))
    rows = []
    npad = int(round(pad / dt))
    for ep in eps:
        s = ep['sensor']; o = 'rear' if s == 'front' else 'front'
        a, b = ep['i0'], ep['i1']
        seg = slice(a, b + 1)
        segp = slice(max(a - npad, 0), min(b + 1 + npad, len(al.t)))
        ref = al.ref[seg]; w = ws[s][seg]
        notch = al.notch[seg]
        e_s = e[s][seg]
        e_o = e[o][seg]
        ip = ep['i_peak']
        v_at = float(al.ref[ip]) if np.isfinite(al.ref[ip]) else np.nan
        lock = bool(np.nanmin(w) < p.lock_v and np.nanmax(ref) > 1.0) if np.isfinite(w).any() else False
        # sensor stuck at zero (never left 0 during the episode while the other bogie / GNSS moved):
        # the 30639 start-of-motion sensor fault, usually followed by a long silence of that topic
        w_o = ws[o][segp]
        stuck0 = bool(np.isfinite(w).any() and np.nanmax(w) < 0.05 and np.nanmax(w_o) > 0.25)
        if ep['e_peak'] > 0:
            kind = 'slip'
        elif stuck0:
            kind = 'stuck_zero'
        elif lock:
            kind = 'lock'
        else:
            kind = 'slide'
        nmin = np.nanmin(notch) if np.isfinite(notch).any() else np.nan
        nmax = np.nanmax(notch) if np.isfinite(notch).any() else np.nan
        nm = mode_int(notch)
        regime = 'traction' if nm > 0 else ('brake' if nm < 0 else 'coast') if np.isfinite(nm) else 'unknown'
        other_max = float(np.nanmax(np.abs(e_o))) if np.isfinite(e_o).any() else np.nan
        with np.errstate(all='ignore'):
            rows.append(dict(
                bag=al.bag, sensor=s, kind=kind, regime=regime,
                t_start=float(al.t[a] - t0), t_end=float(al.t[b] - t0), dur=float(ep['dur']),
                t_start_hdr=float(al.t[a]),
                e_peak=float(ep['e_peak']), e_mean=float(ep['e_mean']),
                e_rel_peak=float(ep['e_peak'] / v_at) if v_at and v_at > 0.2 else np.nan,
                v_ref_peak=v_at, v_ref_mean=float(np.nanmean(ref)), v_wheel_min=float(np.nanmin(w)),
                v_wheel_max=float(np.nanmax(w)),
                notch_mode=nm, notch_min=nmin, notch_max=nmax,
                other_sensor_max_abs_e=other_max,
                both=bool(np.isfinite(other_max) and other_max > max(p.thr_abs, 0.5 * abs(ep['e_peak']))),
                ref_q_min=int(np.min(al.ref_q[seg])), frac_ref_agree=float(np.mean(al.ref_q[seg] == 2)),
                lag=float(np.nanmedian(al.lag[seg])), lag_rate_max=float(np.nanmax(lag_rate[segp])),
                a_ref_max=float(np.nanmax(np.abs(a_ref[segp]))),
                # GNSS-free signatures (on padded window so onset/recovery are included)
                fr_max_abs=float(np.nanmax(np.abs(fr[segp]))),
                a_wheel_max=float(np.nanmax(acc[s][segp])), a_wheel_min=float(np.nanmin(acc[s][segp])),
                jerk_wheel_max_abs=float(np.nanmax(np.abs(jerk[s][segp]))),
                gap_in_sensor=bool(np.isnan(ws[s][seg]).any()),
                dist_err_m=float(np.nansum(e_s) * dt),
                kappa_abs_max=float(np.nanmax(np.abs(al.kappa[seg]))) if getattr(al, 'kappa', None) is not None
                and np.isfinite(al.kappa[seg]).any() else np.nan,
            ))
    return rows


def classify_episode(r: dict):
    """Validity of a GNSS-labelled episode row (from `episode_table`).
    Returns (validity, confidence, note):
      validity 'wheel'        genuine wheel anomaly (slip/slide/lock/stuck_zero)
               'ref_artifact' reference problem (GNSS glitch, GNSS clock transition) - not a wheel fault
    Rules: identical deviation on both bogies (|F-R| < 0.1 m/s) with unphysical GNSS acceleration
    (> 3 m/s^2), a single-receiver reference, or a fast GNSS-clock transition => reference artifact."""
    same_both = bool(r.get('both')) and r.get('fr_max_abs', 1.0) < 0.1
    if same_both and (r.get('a_ref_max', 0) > 3.0 or r.get('lag_rate_max', 0) > 0.05 or r.get('frac_ref_agree', 1) < 0.5):
        return 'ref_artifact', 'high', 'both bogies identical, reference implausible'
    if same_both:
        # both bogies identical but reference plausible: all-axle slip/slide or residual reference error
        return 'wheel', 'low', 'both bogies identical (all-axle event or reference error)'
    if r.get('kind') == 'stuck_zero':
        return 'wheel', 'high', 'sensor stuck at 0 at motion start (then topic silent)'
    if r.get('fr_max_abs', 0) > 0.2:
        return 'wheel', 'high', 'front/rear disagree'
    return 'wheel', 'medium', ''


def causal_rate(x, n, dt):
    """Causal finite-difference rate (x[i]-x[i-n])/(n*dt)."""
    out = np.full(len(x), np.nan)
    out[n:] = (x[n:] - x[:-n]) / (n * dt)
    return out


# ----------------------------------------------------------------------------------------------
# 6. GNSS-free features / detectors
# ----------------------------------------------------------------------------------------------
def fr_mismatch(al: Aligned, kf=KMH, kr=KMH):
    """Front-rear difference (m/s) after per-sensor scale."""
    return al.front * KMH / kf - al.rear * KMH / kr


def local_std(x, n=10):
    """Causal rolling std of first differences (noise proxy) over n samples."""
    d = np.r_[np.nan, np.diff(x)]
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        seg = d[i - n + 1:i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= n // 2:
            out[i] = np.std(seg)
    return out


@dataclass
class MonitorParams:
    """Thresholds for the causal wheel monitor (derivations in REPORT.md, sections 4-6)."""
    k_front: float = 3.597        # km/h per m/s (fleet median, straight track; per-bag range 3.545 .. 3.659)
    k_rear: float = 3.597
    v_max: float = 25.0           # m/s absolute plausibility (observed max true speed 14.4 m/s)
    v_neg_tol: float = 0.3        # m/s tolerated negative reading (real roll-back down to -0.11 m/s observed)
    gap_s: float = 0.5            # header gap => dropout (resets rate history)
    stale_s: float = 0.5          # other sensor considered lost if silent this long
    a_max: float = 1.5            # m/s^2 max vehicle acceleration (true p99.99 1.23 in traction, max 1.45)
    a_min: float = -2.2           # m/s^2 max service deceleration (true p0.01 -1.94)
    a_emergency: float = -5.0     # m/s^2 emergency braking accepted when BOTH bogies agree (-4.4 seen, 616ec56b)
    a_margin: float = 0.6         # m/s^2 allowance: clean wheel 0.2 s differences reach +1.45 / -2.07
    fr_abs: float = 0.20          # m/s front-rear tolerance (bogie kinematics in R~20 m loops: up to ~0.25)
    fr_rel: float = 0.03          # relative front-rear tolerance
    gate0: float = 0.30           # m/s innovation gate around the predicted speed interval
    gate_rate: float = 0.5        # m/s per s since the last trusted update (model uncertainty growth)
    frozen_n: int = 6             # identical consecutive values while moving (real data: max run 5)
    recover_n: int = 3            # consecutive consistent samples to re-trust a sensor
    survivor_s: float = 1.5       # sole live sensor rejected this long but self-consistent => re-trust
    zero_other: float = 0.30      # m/s: other sensor above this while this one reads 0 => stuck_zero
    stamp_dev: float = 0.30       # s: header vs arrival latency deviation => stamp anomaly (if only this topic)
    a_tau: float = 2.0            # s: decay of extrapolated acceleration during outages
    # acceleration envelopes (m/s^2) from 'notch held >= 2 s' statistics (physics_accel_by_notch_steady.csv).
    # Notch 0 and -8..-15 are AMBIGUOUS: the tram was observed accelerating at ~1 m/s^2 and emergency-braking
    # at -4.4 m/s^2 with the handle there, so they get the full physical envelope.
    env_trac: tuple = (-0.5, 1.5)       # notch +1..+15
    env_brake: tuple = (-1.3, 0.3)      # notch -1..-7
    env_ambiguous: tuple = (-3.0, 1.5)  # notch 0, -8..-15


class CausalWheelMonitor:
    """Streaming, GNSS-free wheel-speed plausibility monitor (message-driven, O(1) per message).

    Feed messages in arrival order: `wheel(sensor, t_hdr, v_kmh, t_arr=None)` and `cmd(t_hdr, notch)`.
    Internal model: fused trusted speed v_hat at t_hat and its slope a_hat (from the last ~0.5 s of trusted
    updates). Prediction interval at t: [v_hat + lo*dt, v_hat + hi*dt] +- (gate0 + gate_rate*dt), where
    (lo, hi) is the acceleration envelope of the controller regime. Classification of each wheel sample:
      'ok'         consistent
      'invalid'    NaN/inf/out-of-range payload or stamp
      'stale'      header stamp not newer than the last accepted one of that sensor (reordered/duplicate)
      'stamp'      header jumped vs arrival time on this topic only (re-stamped from arrival, still used)
      'frozen'     bit-identical value repeated >= frozen_n times while moving
      'stuck_zero' reads 0 while the other bogie reads > zero_other and is plausible
      'rate'       own acceleration over >= 0.2 s beyond physical limits (+margin); emergency braking seen by
                   both bogies is accepted
      'gate'       outside the predicted interval
      'fr'         front-rear mismatch and this sensor judged the wrong one: first by distance to the
                   extrapolated trajectory v_hat + a_hat*dt, tie-break by regime (traction: faster bogie
                   slips; service braking: slower bogie slides)
    `trusted_speed(t)`: fused trusted wheel speed (m/s) or None; `estimate(t)`: trusted speed or, if none,
    the extrapolation v_hat + a_hat*tau*(1-exp(-dt/tau)) (what an estimator without a traction model does).
    This is a detector + measurement selector; the EKF should use the flags to inflate the measurement noise
    / skip the update and rely on the traction model while flags are active.
    """

    def __init__(self, p: MonitorParams = MonitorParams()):
        self.p = p
        self.v_hat = 0.0
        self.a_hat = 0.0
        self.t_hat = None
        self.traj = []                                   # recent (t, v_hat) of trusted fused updates
        self.notch = 0
        self.last = {'front': None, 'rear': None}        # (t, v_ms) last accepted sample
        self.hist = {'front': [], 'rear': []}            # recent (t, v) for rate check
        self.same = {'front': 0, 'rear': 0}
        self.good_run = {'front': 0, 'rear': 0}
        self.bad_since = {'front': None, 'rear': None}
        self.trusted = {'front': True, 'rear': True}
        self.lat = {'front': [], 'rear': []}             # recent arrival-header latencies
        self.lat_dev = {'front': 0.0, 'rear': 0.0}
        self._prev_arr = {}                              # last arrival time per sensor (burst detection)
        self.counts = {}

    # ---- helpers --------------------------------------------------------------------------------
    def _regime(self):
        """+1 traction, -1 service braking (-1..-7), 0 ambiguous (0, -8..-15)."""
        if self.notch > 0:
            return 1
        if -7 <= self.notch < 0:
            return -1
        return 0

    def _env(self):
        r = self._regime()
        return self.p.env_trac if r > 0 else (self.p.env_brake if r < 0 else self.p.env_ambiguous)

    def extrapolate(self, t):
        if self.t_hat is None:
            return 0.0
        dt = max(t - self.t_hat, 0.0)
        tau = self.p.a_tau
        return max(0.0, self.v_hat + self.a_hat * tau * (1.0 - np.exp(-dt / tau)))

    def predict(self, t):
        """Predicted speed interval at time t."""
        if self.t_hat is None:
            return -np.inf, np.inf
        dt = max(t - self.t_hat, 0.0)
        lo, hi = self._env()
        g = self.p.gate0 + self.p.gate_rate * dt + 0.03 * self.v_hat
        return max(0.0, self.v_hat + lo * dt) - g, self.v_hat + hi * dt + g

    def cmd(self, t_hdr, notch):
        if notch is None or not np.isfinite(notch) or abs(notch) > 15:
            return 'invalid'
        self.notch = int(round(notch))
        return 'ok'

    def _count(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1

    def _rate(self, sensor, t, v):
        h = self.hist[sensor]
        h.append((t, v))
        while len(h) > 1 and t - h[0][0] > 0.6:
            h.pop(0)
        rate = None
        for (tt, vv) in h[:-1]:
            if t - tt >= 0.2:
                rate = (v - vv) / (t - tt)
        return rate

    def _stamp_check(self, sensor, t_hdr, t_arr):
        """Returns corrected header (or original) and whether this topic alone jumped."""
        if t_arr is None or not np.isfinite(t_arr):
            return t_hdr, False
        lat = t_arr - t_hdr
        L = self.lat[sensor]
        other = 'rear' if sensor == 'front' else 'front'
        # skip delivery bursts (start-up, catch-up after an outage): arrival spacing < 20 ms
        prev_arr = self._prev_arr.get(sensor)
        self._prev_arr[sensor] = t_arr
        if prev_arr is not None and t_arr - prev_arr < 0.02:
            return t_hdr, False
        ref = float(np.median(L)) if len(L) >= 21 else None
        jumped = False
        if ref is not None:
            dev = lat - ref
            self.lat_dev[sensor] = dev
            if (abs(dev) > self.p.stamp_dev and abs(self.lat_dev[other]) < 0.1 and len(self.lat[other]) >= 21):
                jumped = True               # only this topic's stamps moved -> stamp fault, re-stamp
                t_hdr = t_arr - ref
        # rolling median over ALL non-burst messages: short stamp faults (< ~2.5 s) do not move it, a
        # persistent re-basing of the clock is adopted after ~26 messages
        L.append(lat)
        if len(L) > 51:
            L.pop(0)
        return t_hdr, jumped

    def _fuse(self, a, b):
        reg = self._regime()
        if reg > 0:
            return min(a, b)
        if reg < 0:
            return max(a, b)
        return 0.5 * (a + b)

    def _update_model(self, t, v):
        if self.t_hat is not None and t < self.t_hat:
            return
        self.v_hat, self.t_hat = v, t
        self.traj.append((t, v))
        while len(self.traj) > 1 and t - self.traj[0][0] > 0.6:
            self.traj.pop(0)
        if len(self.traj) >= 3 and t - self.traj[0][0] >= 0.3:
            n = len(self.traj)
            mt = sum(x[0] - t for x in self.traj) / n
            mv = sum(x[1] for x in self.traj) / n
            sxx = sum((x[0] - t - mt) ** 2 for x in self.traj)
            if sxx > 1e-9:
                a = sum((x[0] - t - mt) * (x[1] - mv) for x in self.traj) / sxx
                self.a_hat = float(min(max(a, self.p.a_emergency), self.p.a_max))

    # ---- main entry ---------------------------------------------------------------------------------
    def wheel(self, sensor, t_hdr, v_kmh, t_arr=None):
        """Process one wheel message; returns classification string."""
        p = self.p
        k = p.k_front if sensor == 'front' else p.k_rear
        if v_kmh is None or t_hdr is None or not np.isfinite(v_kmh) or not np.isfinite(t_hdr):
            self._count('invalid'); return 'invalid'
        v = v_kmh / k
        if v > p.v_max or v < -p.v_neg_tol:
            self._count('invalid'); return 'invalid'
        v = max(v, 0.0)
        t_hdr, stamp_jump = self._stamp_check(sensor, t_hdr, t_arr)
        last = self.last[sensor]
        if last is not None and t_hdr <= last[0]:
            self._count('stale'); return 'stale'
        if last is not None and v == last[1] and v > 0.3:
            self.same[sensor] += 1
        else:
            self.same[sensor] = 0
        gap = last is None or (t_hdr - last[0]) > p.gap_s
        if gap:
            self.hist[sensor] = []
        rate = self._rate(sensor, t_hdr, v)
        self.last[sensor] = (t_hdr, v)
        other = 'rear' if sensor == 'front' else 'front'
        lo_o = self.last[other]
        other_live = lo_o is not None and (t_hdr - lo_o[0]) < p.stale_s
        other_sync = lo_o is not None and abs(lo_o[0] - t_hdr) < 0.15
        vlo, vhi = self.predict(t_hdr)
        inside = vlo <= v <= vhi
        agree_o = other_sync and abs(v - lo_o[1]) <= p.fr_abs + p.fr_rel * max(v, lo_o[1])
        cls = 'ok'
        if self.same[sensor] >= p.frozen_n - 1:
            cls = 'frozen'
        elif v < 0.05 and other_sync and lo_o[1] > p.zero_other and vlo <= lo_o[1] <= vhi:
            cls = 'stuck_zero'
        elif rate is not None and (rate > p.a_max + p.a_margin or rate < p.a_min - p.a_margin):
            # (not for the final drop to ~0: all observed both-bogie locks happened below 2 m/s)
            emergency = rate < 0 and agree_o and rate >= p.a_emergency and v > 1.0
            cls = 'ok' if emergency else 'rate'
        elif not inside and not (agree_o and self.trusted[other] and lo_o[1] >= vlo - 0.5 and lo_o[1] <= vhi + 0.5):
            cls = 'gate'
        if cls == 'ok' and other_sync and self.trusted[other] and not agree_o:
            # front-rear arbitration: trajectory consistency first, regime rule as tie-breaker
            v_ext = self.extrapolate(t_hdr)
            r_s, r_o = abs(v - v_ext), abs(lo_o[1] - v_ext)
            if min(r_s, r_o) < 0.5 * max(r_s, r_o) and max(r_s, r_o) > 0.15:
                wrong_self = r_s > r_o
            else:
                reg = self._regime()
                d = v - lo_o[1]
                wrong_self = (d > 0) if reg > 0 else ((d < 0) if reg < 0 else r_s > r_o)
            if wrong_self:
                cls = 'fr'
        if cls == 'ok' and stamp_jump:
            cls = 'stamp'                   # value used with the re-stamped time
        # sole-survivor recovery: other sensor dead, this one rejected for long but physically smooth
        if cls == 'gate' and not other_live:
            if self.bad_since[sensor] is None:
                self.bad_since[sensor] = t_hdr
            elif (t_hdr - self.bad_since[sensor] > p.survivor_s and rate is not None
                  and p.a_min - p.a_margin <= rate <= p.a_max + p.a_margin):
                cls = 'ok'
                self.trusted[sensor] = True
                self.good_run[sensor] = p.recover_n
        usable = cls in ('ok', 'stamp')
        if usable:
            self.bad_since[sensor] = None
            self.good_run[sensor] += 1
            if not self.trusted[sensor] and self.good_run[sensor] >= p.recover_n:
                self.trusted[sensor] = True
        else:
            if self.bad_since[sensor] is None:
                self.bad_since[sensor] = t_hdr
            self.good_run[sensor] = 0
            self.trusted[sensor] = False
        if usable and self.trusted[sensor]:
            if other_sync and self.trusted[other] and agree_o:
                self._update_model(t_hdr, self._fuse(v, lo_o[1]))
            else:
                self._update_model(t_hdr, v)
        self._count(cls)
        return cls

    def anomaly_kind(self, sensor, cls, v_kmh):
        """Map a rejected sample to slip/slide/lock semantics using the model and notch."""
        if cls in ('ok', 'stale', 'invalid', 'frozen', 'stuck_zero', 'stamp'):
            return cls
        k = self.p.k_front if sensor == 'front' else self.p.k_rear
        v = v_kmh / k
        if v < 0.1 and self.v_hat > 1.0:
            return 'lock'
        if v > self.v_hat:
            return 'slip' if self.notch >= 0 else 'overspeed'
        return 'slide' if self.notch <= 0 else 'underspeed'

    def trusted_speed(self, t_now):
        """Fused trusted wheel speed (m/s) from trusted sensors with fresh (<0.3 s) samples, else None."""
        vals = [self.last[s][1] for s in ('front', 'rear')
                if self.last[s] is not None and self.trusted[s] and t_now - self.last[s][0] < 0.3]
        if not vals:
            return None
        if len(vals) == 2:
            return self._fuse(vals[0], vals[1])
        return vals[0]

    def estimate(self, t_now):
        v = self.trusted_speed(t_now)
        return self.extrapolate(t_now) if v is None else v


def replay_monitor(b: Bag, p: MonitorParams = MonitorParams(), use_arrival=True):
    """Feed a bag's input messages to CausalWheelMonitor in arrival (bag-time) order.
    Returns (rows, monitor); rows = (sensor, t_bag, t_hdr, v_ms, cls, kind, v_hat, notch, estimate)."""
    ev = []
    for s_name, s in (('front', b.front), ('rear', b.rear)):
        for i in range(len(s)):
            ev.append((s.t_bag[i], 0, s_name, s.t_hdr[i], s.v[i]))
    for i in range(len(b.cmd)):
        ev.append((b.cmd.t_bag[i], 1, 'cmd', b.cmd.t_hdr[i], b.cmd.v[i]))
    ev.sort(key=lambda x: (x[0], x[1]))
    mon = CausalWheelMonitor(p)
    rows = []
    for tb, _, nm, th, val in ev:
        if nm == 'cmd':
            mon.cmd(th, val)
            continue
        cls = mon.wheel(nm, th, val, tb if use_arrival else None)
        kind = mon.anomaly_kind(nm, cls, val) if np.isfinite(val) else cls
        tv = mon.estimate(th) if np.isfinite(th) else np.nan
        rows.append((nm, tb, th, val / (p.k_front if nm == 'front' else p.k_rear), cls, kind, mon.v_hat, mon.notch, tv))
    return rows, mon


# ----------------------------------------------------------------------------------------------
# 7. Synthetic anomaly injection (robustness test harness for the estimator node)
# ----------------------------------------------------------------------------------------------
def inject(b: Bag, kind: str, t_rel: float, dur: float, sensor='front', mag=None, seed=0) -> Bag:
    """Return a copy of bag with one synthetic anomaly injected into an input topic.
    kinds: 'dropout' (remove messages), 'spike' (single outliers), 'noise' (gaussian, mag m/s),
    'freeze' (hold last value), 'zero' (sensor reads 0), 'slip' (additive smooth hump, mag m/s),
    'slide' (negative hump), 'scale' (multiply by 1+mag), 'nan' (NaN payloads), 'stamp_jump'
    (header stamps +mag s), 'dup' (duplicate messages), 'cmd_dropout', 'cmd_spike'.
    t_rel: seconds from bag start (bag time)."""
    import copy
    rng = np.random.default_rng(seed)
    nb = copy.deepcopy(b)
    tgt = nb.cmd if kind.startswith('cmd') else getattr(nb, sensor)
    t0 = nb.t0 + t_rel
    m = (tgt.t_bag >= t0) & (tgt.t_bag < t0 + dur)
    if kind in ('dropout', 'cmd_dropout'):
        keep = ~m
        new = tgt.sel(keep)
    elif kind == 'spike' or kind == 'cmd_spike':
        idx = np.where(m)[0][::max(int(round(1.0 / 0.1)), 1)]
        v = tgt.v.copy()
        v[idx] = (mag if mag is not None else (60.0 if kind == 'spike' else 15.0))
        new = Stream(tgt.t_bag, tgt.t_hdr, v)
    elif kind == 'noise':
        v = tgt.v.copy(); v[m] = v[m] + rng.normal(0, (mag or 0.5) * KMH, m.sum()); new = Stream(tgt.t_bag, tgt.t_hdr, v)
    elif kind == 'freeze':
        v = tgt.v.copy(); i = np.where(m)[0]
        if len(i):
            v[i] = v[i[0]]
        new = Stream(tgt.t_bag, tgt.t_hdr, v)
    elif kind == 'zero':
        v = tgt.v.copy(); v[m] = 0.0; new = Stream(tgt.t_bag, tgt.t_hdr, v)
    elif kind in ('slip', 'slide'):
        v = tgt.v.copy(); i = np.where(m)[0]
        if len(i):
            ph = (tgt.t_bag[i] - t0) / dur
            hump = np.sin(np.pi * ph) ** 2 * (mag or 1.5) * KMH
            v[i] = np.maximum(v[i] + (hump if kind == 'slip' else -hump), 0)
        new = Stream(tgt.t_bag, tgt.t_hdr, v)
    elif kind == 'scale':
        v = tgt.v.copy(); v[m] *= (1 + (mag or 0.1)); new = Stream(tgt.t_bag, tgt.t_hdr, v)
    elif kind == 'nan':
        v = tgt.v.copy(); v[m] = np.nan; new = Stream(tgt.t_bag, tgt.t_hdr, v)
    elif kind == 'stamp_jump':
        th = tgt.t_hdr.copy(); th[m] += (mag or 1.0); new = Stream(tgt.t_bag, th, tgt.v)
    elif kind == 'dup':
        i = np.where(m)[0]
        tb = np.r_[tgt.t_bag, tgt.t_bag[i] + 1e-4]; th = np.r_[tgt.t_hdr, tgt.t_hdr[i]]; v = np.r_[tgt.v, tgt.v[i]]
        o = np.argsort(tb, kind='stable'); new = Stream(tb[o], th[o], v[o])
    else:
        raise ValueError(kind)
    if kind.startswith('cmd'):
        nb.cmd = new
    else:
        setattr(nb, sensor, new)
    return nb
