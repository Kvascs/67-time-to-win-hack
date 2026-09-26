"""Anomaly injectors.

Every injector is configured from a dict (one entry of a scenario's ``injectors`` list), draws its
random parameters from the generator it is given, modifies the :class:`~anomaly_sim.run.Run` in
place and returns a list of event records (dicts) describing what was injected.

Injectors run in ``stage`` order (value-level physics first, then sensor faults, then delivery and
timing faults, GNSS removal last) so that e.g. a dropout removes already-slipping samples and the
physics always sees clean timing.

Topic specs (``topics``):
    ``wheels`` / ``[front, cmd]``        -> fixed set, affected jointly
    ``{front: 0.5, rear: 0.5}``           -> one alias drawn per event
Placement keys shared by event-based injectors:
    ``count`` (int or [lo, hi]), ``duration`` (s or [lo, hi]), ``when`` (any | motion | standstill |
    traction | braking | coast | accel | decel | departure | arrival), ``min_speed`` / ``max_speed``
    [m/s], ``min_notch`` / ``max_notch``, ``min_hold`` [s], ``min_gap`` [s], ``t_min`` / ``t_max``
    [s from run start], ``lead`` [s], ``at`` (explicit list of start times, s from run start).
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from .constants import (CMD, FRONT, GNSS, INT8_MAX, INT8_MIN, KMH_PER_MS, NOMINAL_PERIOD, REAR, WHEEL_DEADBAND_KMH,
                        WHEELS, Label, resolve_topics)
from .context import Context, sanitize_stamps
from .physics import (AdhesionParams, ControllerParams, notch_demand, patch_profile, patch_profile_space,
                      simulate_slip_delta)
from .run import Run, Stream
from .sampling import sample_bool, sample_choice, sample_float, sample_int

REGISTRY: dict[str, type['Injector']] = {}
DT_SIM = 0.004

PLACEMENT_KEYS = {'count', 'duration', 'when', 'min_speed', 'max_speed', 'min_notch', 'max_notch', 'min_hold',
                  'min_gap', 't_min', 't_max', 'lead', 'at'}
COMMON_KEYS = {'type', 'enabled', 'name'}


def register(cls):
    REGISTRY[cls.type_name] = cls
    return cls


def build_injector(spec: dict) -> 'Injector':
    typ = spec.get('type')
    if typ not in REGISTRY:
        raise ValueError(f'unknown injector type {typ!r}; known: {sorted(REGISTRY)}')
    return REGISTRY[typ](spec)


# ------------------------------------------------------------------------------------------ helpers

def _f32(x: np.ndarray) -> np.ndarray:
    """Round to float32 precision (the recorded wheel speeds are float32 values stored as float64)."""
    return np.asarray(x, np.float32).astype(np.float64)


def _deadband(vals: np.ndarray) -> np.ndarray:
    vals = vals.copy()
    vals[(vals >= 0) & (vals < WHEEL_DEADBAND_KMH)] = 0.0
    return vals


def _meas_time(s: Stream) -> np.ndarray:
    return sanitize_stamps(s.t_bag, s.t_hdr0)


def _lat(ctx: Context, key: str) -> float:
    return ctx.lat_cmd if key == CMD else ctx.lat_wheel


def _rows_between(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return np.flatnonzero((t >= t0) & (t <= t1))


def _resolve_topic_choice(spec, rng) -> tuple[str, ...]:
    """Fixed topic set, or one alias drawn from a weighted dict."""
    if isinstance(spec, dict):
        return resolve_topics(sample_choice(spec, rng))
    return resolve_topics(spec)


class Injector:
    type_name: ClassVar[str] = ''
    stage: ClassVar[int] = 0
    defaults: ClassVar[dict] = {}
    placement_defaults: ClassVar[dict] = {}

    def __init__(self, spec: dict):
        spec = dict(spec)
        allowed = set(self.defaults) | set(self.placement_defaults) | PLACEMENT_KEYS | COMMON_KEYS
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f'{self.type_name}: unknown parameter(s) {sorted(unknown)}; allowed {sorted(allowed)}')
        base = {'count': 1, 'duration': 1.0, 'when': 'any', 'min_speed': None, 'max_speed': None,
                'min_notch': None, 'max_notch': None, 'min_hold': 0.0, 'min_gap': 3.0, 't_min': 10.0,
                't_max': None, 'lead': 0.0, 'at': None}
        self.p = {**base, **self.placement_defaults, **self.defaults, **spec}
        self.name = spec.get('name', self.type_name)
        self.enabled = spec.get('enabled', True)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.p})'

    # ----------------------------------------------------------------------- placement / events
    def _windows(self, ctx: Context, rng: np.random.Generator, mask: np.ndarray | None = None,
                 taken=None) -> list[tuple[float, float]]:
        p = self.p
        dur = (lambda r: sample_float(p['duration'], r))
        if p['at'] is not None:
            return [(ctx.t_start + float(a), ctx.t_start + float(a) + dur(rng)) for a in p['at']]
        count = sample_int(p['count'], rng)
        if count <= 0:
            return []
        if mask is None:
            mask = ctx.mask(p['when'], p['min_speed'], p['max_speed'], p['min_notch'], p['max_notch'])
        placed = ctx.place(rng, mask, count, dur, min_hold=float(p['min_hold']), min_gap=float(p['min_gap']),
                           t_min=float(p['t_min']), t_max=p['t_max'], lead=float(p['lead']), taken=taken)
        self.placement = {'requested': int(count), 'placed': len(placed)}
        return placed

    def _event(self, ctx: Context, topics, t0: float, t1: float, params: dict | None = None,
               stats: dict | None = None) -> dict:
        return {'type': self.type_name, 'name': self.name, 'topics': list(topics), 't0': float(t0), 't1': float(t1),
                't0_rel': round(float(t0 - ctx.t_start), 4), 't1_rel': round(float(t1 - ctx.t_start), 4),
                'params': params or {}, 'stats': stats or {}}

    def apply(self, run: Run, ctx: Context, rng: np.random.Generator) -> list[dict]:  # pragma: no cover
        raise NotImplementedError


# ============================================================================== value: physics

@register
class ScaleDrift(Injector):
    """Wheel-diameter error: ``val *= 1 + eps(t)`` (wear, re-profiling, wrong calibration)."""
    type_name = 'scale_drift'
    stage = 10
    defaults = {'topics': 'wheels', 'start': 0.0, 'end': [-0.02, 0.02], 'profile': 'linear',
                'label_threshold': 0.002}

    def apply(self, run, ctx, rng):
        events = []
        for key in resolve_topics(self.p['topics']):
            if key not in WHEELS:
                raise ValueError('scale_drift applies to wheel topics only')
            s = run[key]
            e0 = sample_float(self.p['start'], rng)
            e1 = sample_float(self.p['end'], rng)
            prof = sample_choice(self.p['profile'], rng)
            tm = _meas_time(s)
            T0, T1 = ctx.t_lo, ctx.t_hi
            if prof == 'linear':
                eps = e0 + (e1 - e0) * np.clip((tm - T0) / max(T1 - T0, 1e-9), 0, 1)
            elif prof == 'constant':
                eps = np.full(len(tm), e1)
            elif prof == 'step':
                ts = T0 + (T1 - T0) * rng.uniform(0.2, 0.8)
                eps = np.where(tm < ts, e0, e1)
            else:
                raise ValueError(f'scale_drift profile {prof!r}')
            fin = np.isfinite(s.val[:, 0])
            old = s.val[:, 0].copy()
            s.val[fin, 0] = _f32(s.val[fin, 0] * (1.0 + eps[fin]))
            changed = (s.val[:, 0] != old) & fin
            s.label[changed | (np.abs(eps) >= float(self.p['label_threshold']))] |= np.uint32(Label.SCALE_DRIFT)
            events.append(self._event(ctx, [key], T0, T1, {'start': e0, 'end': e1, 'profile': prof}))
        return events


class _AdhesionInjector(Injector):
    """Common machinery for traction slip (+1) and braking slide (-1) events."""
    stage = 20
    direction: ClassVar[int] = +1

    def _sim_bogie(self, ctx: Context, ts: np.ndarray, t0: float, t1: float, rng, *, ctrl: ControllerParams,
                   rho: float, G: float, adh: AdhesionParams, spatial_offset: float | None,
                   x_patch: tuple[float, float] | None, model: str, peak: float, lock_hold: float):
        v = ctx.at('v', ts)
        if model == 'physical':
            a = ctx.at('a', ts)
            notch = ctx.notch_at(ts)
            f0 = abs(float(notch_demand(ctx.notch_at([t0]))[0]))
            mu_low = max(rho * f0, 0.005)
            if spatial_offset is None:
                mu = patch_profile(ts, t0, t1, adh.mu_dry, mu_low, edge=0.3)
            else:
                x = ctx.at('x', ts) - spatial_offset
                mu = patch_profile_space(x, x_patch[0], x_patch[1], adh.mu_dry, mu_low, edge_m=1.0)
            sim = simulate_slip_delta(ts, v, a, notch, mu, ctrl, adh, G)
            return sim.w, sim.locked, {'mu_low': mu_low, 'f_demand_t0': f0}
        # kinematic reference shapes (useful for simple, jury-style "multiply by k" tests)
        edge = 0.1 if model == 'step' else 0.3 * (t1 - t0)
        box = np.clip(np.minimum((ts - t0) / max(edge, 1e-3), (t1 - ts) / max(edge, 1e-3)), 0, 1)
        box[(ts < t0) | (ts > t1)] = 0.0
        w = self.direction * peak * v * box
        if lock_hold > 0:
            w = -v * box
        locked = (w <= -v + 0.05) & (v > 0.3)
        return w, locked, {}

    def apply(self, run, ctx, rng):
        p = self.p
        mask = ctx.mask('traction' if self.direction > 0 else 'braking', p['min_speed'], p['max_speed'],
                        p['min_notch'], p['max_notch'])
        windows = self._windows(ctx, rng, mask)
        events = []
        for (t0, t1) in windows:
            bogie = sample_choice(p['bogie'], rng)
            keys = {'front': (FRONT,), 'rear': (REAR,), 'both': (FRONT, REAR)}[bogie]
            model = sample_choice(p['model'], rng)
            both_mode = sample_choice(p['both_mode'], rng) if len(keys) == 2 else 'single'
            spacing = sample_float(p['bogie_spacing'], rng)
            magnitude = sample_float(p['peak_rel'] if self.direction > 0 else p['depth_rel'], rng)
            lock = self.direction < 0 and sample_bool(p.get('lock_prob', 0.0), rng)
            lock_hold = sample_float(p['lock_duration'], rng) if lock else 0.0
            if (lock or model != 'physical') and both_mode == 'spatial':
                both_mode = 'simultaneous'  # an unprotected (lock) patch met again later would run away
            v0 = float(ctx.at('v', [t0])[0])
            if lock and model == 'physical':  # patch long enough to decelerate to 0 and stay locked
                t1 = t0 + 1.5 * v0 / 5.0 + lock_hold + 0.3
            elif lock:                        # kinematic lock: wheel reads 0 for exactly lock_hold
                t1 = t0 + lock_hold
            elif model == 'physical':  # the patch ends with the traction (braking) phase it was placed in
                i0 = int(np.clip(np.searchsorted(ctx.grid, t0), 0, len(ctx.grid) - 1))
                same_phase = (ctx.notch[i0:] * self.direction) > 0
                j = np.flatnonzero(~same_phase)
                if len(j):
                    t1 = min(t1, float(ctx.grid[i0 + j[0]]) + 0.3)
            tail = float(p['recovery'])
            extra = 0.0
            x_patch = None
            if both_mode == 'spatial':
                x0 = float(ctx.at('x', [t0])[0])
                x1 = float(ctx.at('x', [t1])[0])
                x_patch = (x0, max(x1, x0 + 0.5))
                # extend until the rear bogie has passed the patch (at most 20 s)
                xs = ctx.x
                j = np.searchsorted(xs, x_patch[1] + spacing)
                t_rear_out = ctx.grid[min(j, len(xs) - 1)]
                extra = float(np.clip(t_rear_out - t1, 0.0, 20.0))
            ts = np.arange(max(ctx.t_lo, t0 - 1.0), min(ctx.t_hi, t1 + tail + extra), DT_SIM)
            if len(ts) < 10:
                continue
            ev_stats: dict = {}
            ev_params: dict = {'bogie': bogie, 'model': model, 'both_mode': both_mode,
                               ('peak_rel' if self.direction > 0 else 'depth_rel'): magnitude,
                               'lock': bool(lock), 'lock_duration': lock_hold, 'v_start': v0}
            for bi, key in enumerate(keys):
                ctrl_kind = 'none' if lock else sample_choice(p['controller'], rng)
                adh = AdhesionParams(mu_dry=sample_float(p['mu_dry'], rng), s_c=sample_float(p['s_c'], rng),
                                     A=sample_float(p['A'], rng), B=sample_float(p['B'], rng))
                rho = sample_float(p['rho_lock'] if lock else p['rho'], rng)
                if bi == 1:
                    rho *= rng.uniform(0.9, 1.15)
                G = sample_float(p['G'], rng)
                w_on = max(0.2, magnitude * max(v0, 1.0) * 0.75)
                u_min = sample_float(p['u_min'], rng)
                if ctrl_kind == 'cutoff':
                    # the cut must be deep enough for re-adhesion: u_min * f_dem < adhesion left at w_on
                    left = rho * (adh.A + (1 - adh.A) * np.exp(-adh.B * w_on))
                    u_min = min(u_min, 0.8 * left)
                elif ctrl_kind == 'creep':
                    u_min = 0.02  # a slip-regulating converter can cut torque as deep as needed
                ctrl = ControllerParams(kind=ctrl_kind, w_on=w_on, w_off=0.3 * w_on, s_target=magnitude,
                                        u_min=u_min, r_cut=sample_float(p['r_cut'], rng),
                                        r_up=sample_float(p['r_up'], rng), lock_hold=lock_hold)
                offset = (spacing if key == REAR else 0.0) if both_mode == 'spatial' else None
                w, locked, info = self._sim_bogie(ctx, ts, t0, t1, rng, ctrl=ctrl, rho=rho, G=G, adh=adh,
                                                  spatial_offset=offset, x_patch=x_patch, model=model,
                                                  peak=magnitude, lock_hold=lock_hold)
                st = self._apply_rows(run[key], ctx, ts, w, locked)
                # kinematic shapes have no controller (the draws above are kept for reproducibility)
                st.update({'controller': ctrl_kind if model == 'physical' else model, 'rho': rho, 'G': G, **info})
                ev_stats[key] = st
            if not any(st.get('n_rows', 0) for st in ev_stats.values()):
                continue
            t_first = min(st['t_first'] for st in ev_stats.values() if st.get('n_rows'))
            t_last = max(st['t_last'] for st in ev_stats.values() if st.get('n_rows'))
            events.append(self._event(ctx, keys, t_first, t_last, ev_params,
                                      {'patch_t0': t0, 'patch_t1': t1, **ev_stats}))
        return events

    def _apply_rows(self, s: Stream, ctx: Context, ts: np.ndarray, w: np.ndarray, locked: np.ndarray) -> dict:
        tm = _meas_time(s)
        idx = _rows_between(tm, ts[0], ts[-1])
        if len(idx) == 0:
            return {'n_rows': 0}
        dw = np.interp(tm[idx], ts, w)
        lk = np.interp(tm[idx], ts, locked.astype(float)) > 0.5
        touched = (np.abs(dw) > 1e-3) | lk        # invariant: label == 0  <=>  value unchanged
        idx, dw, lk = idx[touched], dw[touched], lk[touched]
        if len(idx) == 0:
            return {'n_rows': 0}
        v_row = ctx.at('v', tm[idx])
        old = s.val[idx, 0]
        new = np.where(np.isfinite(old), old + KMH_PER_MS * dw, old)
        new = np.where(np.isfinite(new), np.maximum(new, 0.0), new)
        new[lk] = 0.0
        new = _deadband(_f32(new))
        s.val[idx, 0] = new
        active = np.abs(dw) > np.maximum(0.05, 0.02 * v_row)   # "significant" part, used for stats
        changed = new != old
        s.label[idx[changed & (dw > 0)]] |= np.uint32(Label.SLIP)
        s.label[idx[changed & (dw < 0)]] |= np.uint32(Label.SLIDE)
        s.label[idx[lk]] |= np.uint32(Label.LOCK)
        rel = dw / np.maximum(v_row, 1.0)
        act_idx = idx[active]
        return {'n_rows': int(active.sum()),
                't_first': float(tm[act_idx[0]]) if len(act_idx) else float(ts[0]),
                't_last': float(tm[act_idx[-1]]) if len(act_idx) else float(ts[0]),
                'peak_dw': float(dw[np.argmax(np.abs(dw))]),
                'peak_rel': float(rel[np.argmax(np.abs(rel))]),
                'active_s': float(active.sum() * NOMINAL_PERIOD[s.key]),
                'locked_s': float(lk.sum() * NOMINAL_PERIOD[s.key])}


@register
class Slip(_AdhesionInjector):
    """Traction wheel slip (wheel over-reads) on a low-adhesion patch with anti-slip control."""
    type_name = 'slip'
    direction = +1
    placement_defaults = {'count': [2, 5], 'duration': [1.0, 10.0], 'min_speed': 1.0, 'max_speed': 16.0,
                          'min_notch': 5, 'min_hold': 1.0, 'min_gap': 8.0}
    defaults = {'bogie': {'front': 0.4, 'rear': 0.4, 'both': 0.2}, 'peak_rel': [0.05, 0.30],
                'controller': {'cutoff': 0.5, 'creep': 0.5}, 'model': 'physical',
                'both_mode': {'simultaneous': 0.7, 'spatial': 0.3}, 'bogie_spacing': [12.0, 20.0],
                'G': [40.0, 100.0], 'rho': [0.55, 0.85], 'rho_lock': [0.2, 0.45], 'mu_dry': 0.30,
                's_c': [0.004, 0.010], 'A': [0.3, 0.5], 'B': [0.2, 0.5], 'u_min': [0.3, 0.6],
                'r_cut': [2.0, 5.0], 'r_up': [0.3, 1.0], 'recovery': 5.0, 'depth_rel': 0.3,
                'lock_prob': 0.0, 'lock_duration': 0.0}


@register
class Slide(_AdhesionInjector):
    """Braking slide / skid (wheel under-reads), with WSP cycling or wheel lock (``lock_prob``)."""
    type_name = 'slide'
    direction = -1
    placement_defaults = {'count': [2, 5], 'duration': [0.5, 3.0], 'min_speed': 2.0, 'max_speed': 16.0,
                          'max_notch': -3, 'min_hold': 0.8, 'min_gap': 8.0}
    defaults = {'bogie': {'front': 0.35, 'rear': 0.35, 'both': 0.3}, 'depth_rel': [0.10, 0.50],
                'controller': {'cutoff': 0.6, 'creep': 0.4}, 'model': 'physical',
                'both_mode': {'simultaneous': 0.8, 'spatial': 0.2}, 'bogie_spacing': [12.0, 20.0],
                'G': [60.0, 120.0], 'rho': [0.4, 0.8], 'rho_lock': [0.15, 0.4], 'mu_dry': 0.30,
                's_c': [0.004, 0.010], 'A': [0.3, 0.5], 'B': [0.2, 0.5], 'u_min': [0.1, 0.4],
                'r_cut': [4.0, 8.0], 'r_up': [0.5, 1.5], 'recovery': 4.0, 'lock_prob': 0.25,
                'lock_duration': [0.5, 3.0], 'peak_rel': 0.3}


# ============================================================================== value: sensor faults

@register
class Noise(Injector):
    """Increased measurement noise: white + proportional + coloured (AR(1)) components."""
    type_name = 'noise'
    stage = 30
    placement_defaults = {'count': 0, 'duration': [30.0, 120.0], 'min_gap': 5.0}
    defaults = {'topics': 'wheels', 'sigma_kmh': [0.5, 2.0], 'rel_sigma': 0.0, 'ar_sigma_kmh': 0.0,
                'ar_tau': 1.0, 'include_standstill': False, 'independent': True}

    def apply(self, run, ctx, rng):
        p = self.p
        count = sample_int(p['count'], rng) if p['at'] is None else len(p['at'])
        windows = [(ctx.t_lo, ctx.t_hi)] if count == 0 else self._windows(ctx, rng)
        events = []
        for (t0, t1) in windows:
            keys = _resolve_topic_choice(p['topics'], rng)
            sig = sample_float(p['sigma_kmh'], rng)
            rel = sample_float(p['rel_sigma'], rng)
            ar_sig = sample_float(p['ar_sigma_kmh'], rng)
            tau = sample_float(p['ar_tau'], rng)
            for key in keys:
                s = run[key]
                tm = _meas_time(s)
                idx = _rows_between(tm, t0, t1)
                if len(idx) == 0:
                    continue
                clean = s.clean[idx, 0]
                n = len(idx)
                noise = rng.normal(0, sig, n) + rel * np.abs(clean) * rng.normal(0, 1, n)
                if ar_sig > 0:
                    dtt = np.diff(tm[idx], prepend=tm[idx][0])
                    phi = np.exp(-np.maximum(dtt, 0) / max(tau, 1e-3))
                    e = rng.normal(0, 1, n)
                    ar = np.zeros(n)
                    for i in range(1, n):
                        ar[i] = phi[i] * ar[i - 1] + np.sqrt(1 - phi[i] ** 2) * ar_sig * e[i]
                    noise += ar
                old = s.val[idx, 0]
                if key == CMD:
                    new = np.clip(np.rint(old + noise), -15, 15)
                else:
                    new = old + noise
                    if not sample_bool(p['include_standstill'], rng):
                        still = clean <= WHEEL_DEADBAND_KMH
                        new[still] = old[still]
                    new = np.where(np.isfinite(new), np.maximum(new, 0.0), new)
                    new = _deadband(_f32(new))
                s.val[idx, 0] = new
                s.label[idx] |= np.uint32(Label.NOISE)
            events.append(self._event(ctx, keys, t0, t1, {'sigma_kmh': sig, 'rel_sigma': rel, 'ar_sigma_kmh': ar_sig,
                                                          'ar_tau': tau}))
        return events


@register
class Frozen(Injector):
    """Stuck sensor: keeps publishing the last value (or 0) with fresh stamps (optionally stuck stamps)."""
    type_name = 'frozen'
    stage = 40
    placement_defaults = {'count': [1, 3], 'duration': [2.0, 20.0], 'when': 'motion', 'min_gap': 5.0}
    defaults = {'topics': {'front': 0.45, 'rear': 0.45, 'cmd': 0.1}, 'mode': {'hold': 0.7, 'zero': 0.3},
                'stamp_frozen': 0.0}

    def apply(self, run, ctx, rng):
        events = []
        for (t0, t1) in self._windows(ctx, rng):
            keys = _resolve_topic_choice(self.p['topics'], rng)
            mode = sample_choice(self.p['mode'], rng)
            stamps = sample_bool(self.p['stamp_frozen'], rng)
            st = {}
            for key in keys:
                s = run[key]
                tm = _meas_time(s)
                idx = _rows_between(tm, t0, t1)
                if len(idx) == 0:
                    continue
                hold = s.val[idx[0] - 1, 0] if idx[0] > 0 else s.val[idx[0], 0]
                value = 0.0 if mode == 'zero' else hold
                s.val[idx, 0] = value
                s.label[idx] |= np.uint32(Label.FROZEN)
                if mode == 'zero' and key != CMD:
                    s.label[idx[s.clean[idx, 0] > 1.0]] |= np.uint32(Label.ZERO)
                if stamps:
                    s.t_hdr[idx] = s.t_hdr[idx[0]]
                st[key] = {'n_rows': int(len(idx)), 'value': float(value)}
            events.append(self._event(ctx, keys, t0, t1, {'mode': mode, 'stamp_frozen': stamps}, st))
        return events


@register
class NotchFault(Injector):
    """Driver-controller faults: stuck notch, spurious jumps, invalid codes, encoder offset."""
    type_name = 'notch_fault'
    stage = 40
    placement_defaults = {'count': [2, 4], 'duration': [0.5, 5.0], 'when': 'motion', 'min_gap': 5.0}
    defaults = {'kinds': {'stuck': 0.4, 'jump': 0.25, 'invalid': 0.2, 'offset': 0.15},
                'invalid_values': [127, -128, 100, -100, 16, -16, 99, -99]}

    def apply(self, run, ctx, rng):
        s = run[CMD]
        tm = _meas_time(s)
        events = []
        for (t0, t1) in self._windows(ctx, rng):
            kind = sample_choice(self.p['kinds'], rng)
            idx = _rows_between(tm, t0, t1)
            if len(idx) == 0:
                continue
            if kind == 'stuck':
                s.val[idx, 0] = s.val[idx[0] - 1, 0] if idx[0] > 0 else s.val[idx[0], 0]
            elif kind == 'jump':
                k = idx[: rng.integers(1, 4)]
                s.val[k, 0] = rng.integers(-15, 16, len(k))
                idx = k
            elif kind == 'invalid':
                k = idx[: rng.integers(1, 4)]
                s.val[k, 0] = rng.choice(self.p['invalid_values'], len(k))
                idx = k
            elif kind == 'offset':
                off = int(rng.choice([-2, -1, 1, 2]))
                s.val[idx, 0] = np.clip(s.val[idx, 0] + off, -15, 15)
            else:
                raise ValueError(f'notch_fault kind {kind!r}')
            s.val[idx, 0] = np.clip(np.rint(s.val[idx, 0]), INT8_MIN, INT8_MAX)
            changed = idx[s.val[idx, 0] != s.clean[idx, 0]]
            s.label[idx] |= np.uint32(Label.NOTCH_FAULT)
            events.append(self._event(ctx, [CMD], float(tm[idx[0]]), float(tm[idx[-1]]), {'kind': kind},
                                      {'n_rows': int(len(idx)), 'n_changed': int(len(changed))}))
        return events


@register
class Outliers(Injector):
    """Isolated corrupted samples: spikes, spurious zeros, sign flips, NaN, +-inf, absurd codes."""
    type_name = 'outliers'
    stage = 50
    placement_defaults = {'count': None, 'when': 'any'}
    defaults = {'topics': 'wheels', 'rate': 0.005, 'burst': [1, 3], 'independent': True,
                'kinds': {'spike': 0.4, 'zero': 0.2, 'negative': 0.1, 'nan': 0.1, 'inf': 0.05, 'absurd': 0.15},
                'spike_kmh': [5.0, 60.0], 'absurd_values': [6553.5, 1.0e6, -1.0e6, 999.0, 1.0e300, -3.4e38],
                'cmd_invalid': [127, -128, 100, -100, 16, -16]}

    def apply(self, run, ctx, rng):
        p = self.p
        keys = resolve_topics(p['topics'])
        mask = ctx.mask(p['when'], p['min_speed'], p['max_speed'], p['min_notch'], p['max_notch'])
        t_lo = ctx.t_start + float(p['t_min'])
        t_hi = ctx.t_hi if p['t_max'] is None else ctx.t_start + float(p['t_max'])
        shared_starts = None
        events = []
        for key in keys:
            s = run[key]
            tm = _meas_time(s)
            ok = mask[np.clip(np.round((tm - ctx.grid[0]) / ctx.dt).astype(int), 0, len(mask) - 1)]
            ok &= (tm >= t_lo) & (tm <= t_hi)
            cand = np.flatnonzero(ok)
            if len(cand) == 0:
                continue
            if shared_starts is not None and not p['independent']:
                starts_t = shared_starts
                starts = np.unique(np.clip(np.searchsorted(tm, starts_t), 0, len(s) - 1))
            else:
                n = sample_int(p['count'], rng) if p['count'] is not None else \
                    int(rng.poisson(float(p['rate']) * len(cand)))
                starts = np.sort(rng.choice(cand, size=min(n, len(cand)), replace=False)) if n > 0 else \
                    np.array([], int)
                shared_starts = tm[starts]
            kinds_count: dict[str, int] = {}
            for i0 in starts:
                b = sample_int(p['burst'], rng)
                kind = sample_choice(p['kinds'], rng)
                rows = np.arange(i0, min(i0 + b, len(s)))
                self._corrupt(s, rows, kind, rng)
                kinds_count[kind] = kinds_count.get(kind, 0) + len(rows)
                events.append(self._event(ctx, [key], float(tm[rows[0]]), float(tm[rows[-1]]),
                                          {'kind': kind, 'burst': int(len(rows))},
                                          {'values': [float(x) for x in s.val[rows, 0]]}))
        return events

    def _corrupt(self, s: Stream, rows: np.ndarray, kind: str, rng):
        p = self.p
        old = s.val[rows, 0]
        if s.key == CMD:
            if kind == 'spike':
                new = rng.integers(-15, 16, len(rows)).astype(float)
            elif kind == 'zero':
                new = np.zeros(len(rows))
            elif kind == 'negative':
                new = -old
            else:  # nan / inf / absurd are not representable in int8 -> invalid codes
                new = rng.choice(p['cmd_invalid'], len(rows)).astype(float)
            s.val[rows, 0] = np.clip(np.rint(new), INT8_MIN, INT8_MAX)
            s.label[rows] |= np.uint32(Label.NOTCH_FAULT)
            return
        if kind == 'spike':
            mag = np.array([sample_float(p['spike_kmh'], rng) for _ in rows])
            sign = rng.choice([-1.0, 1.0], len(rows))
            new = _f32(old + sign * mag)
            s.label[rows] |= np.uint32(Label.SPIKE)
        elif kind == 'zero':
            new = np.zeros(len(rows))
            s.label[rows] |= np.uint32(Label.ZERO)
        elif kind == 'negative':
            new = np.where(np.abs(old) > 0.5, -np.abs(old), -rng.uniform(1.0, 10.0, len(rows)))
            s.label[rows] |= np.uint32(Label.NEGATIVE)
        elif kind == 'nan':
            new = np.full(len(rows), np.nan)
            s.label[rows] |= np.uint32(Label.NAN)
        elif kind == 'inf':
            new = rng.choice([np.inf, -np.inf], len(rows))
            s.label[rows] |= np.uint32(Label.INF)
        elif kind == 'absurd':
            new = rng.choice(np.asarray(p['absurd_values'], float), len(rows))
            s.label[rows] |= np.uint32(Label.ABSURD)
        else:
            raise ValueError(f'outlier kind {kind!r}')
        s.label[rows[np.isfinite(new) & (new < 0)]] |= np.uint32(Label.NEGATIVE)
        s.val[rows, 0] = new


# ============================================================================== delivery faults

@register
class Dropout(Injector):
    """Missing messages. ``mode: drop`` loses them; ``mode: stall`` delivers them late in a burst
    that drains at ``drain_factor`` x nominal period (the pattern seen in real bags)."""
    type_name = 'dropout'
    stage = 60
    placement_defaults = {'count': [3, 6], 'duration': [0.2, 10.0], 'when': 'motion', 'min_gap': 3.0}
    defaults = {'topics': {'front': 0.3, 'rear': 0.3, 'wheels': 0.25, 'cmd': 0.1, 'vehicle': 0.05},
                'mode': 'drop', 'drain_factor': 0.9}

    def apply(self, run, ctx, rng):
        events = []
        for (t0, t1) in self._windows(ctx, rng):
            keys = _resolve_topic_choice(self.p['topics'], rng)
            mode = sample_choice(self.p['mode'], rng)
            st = {}
            for key in keys:
                s = run[key]
                b0, b1 = t0 + _lat(ctx, key), t1 + _lat(ctx, key)
                inside = (s.t_bag >= b0) & (s.t_bag <= b1)
                n_in = int(inside.sum())
                if n_in == 0:
                    continue
                if mode == 'drop':
                    after = np.flatnonzero(s.t_bag > b1)
                    if len(after):
                        s.label[after[0]] |= np.uint32(Label.RESUME)
                    run.streams[key] = s.keep(~inside)
                    st[key] = {'n_dropped': n_in}
                elif mode == 'stall':
                    st[key] = self._stall(s, b0, b1, float(self.p['drain_factor']) * NOMINAL_PERIOD[key])
                    run.streams[key] = s.sort()
                else:
                    raise ValueError(f'dropout mode {mode!r}')
            events.append(self._event(ctx, keys, t0, t1, {'mode': mode}, st))
        return events

    @staticmethod
    def _stall(s: Stream, b0: float, b1: float, drain_dt: float) -> dict:
        order = np.argsort(s.t_bag, kind='stable')
        tb = s.t_bag[order]
        i = int(np.searchsorted(tb, b0))
        new = tb.copy()
        prev = b1
        k = i
        max_delay = 0.0
        while k < len(tb):
            t_new = max(tb[k], prev if k == i else prev + drain_dt)
            if k > i and t_new <= tb[k] + 1e-9:
                break
            new[k] = t_new
            max_delay = max(max_delay, t_new - tb[k])
            prev = t_new
            k += 1
        delayed = order[i:k]
        s.t_bag[order] = new
        s.label[delayed] |= np.uint32(Label.STALL)
        return {'n_delayed': int(len(delayed)), 'max_delay': float(max_delay)}


@register
class Duplicates(Injector):
    """Duplicated messages (identical stamp and payload), arriving ``delay_ms`` after the original."""
    type_name = 'duplicates'
    stage = 70
    placement_defaults = {'when': 'any'}
    defaults = {'topics': 'vehicle', 'rate': 0.02, 'delay_ms': [0.0, 5.0]}

    def apply(self, run, ctx, rng):
        events = []
        for key in resolve_topics(self.p['topics']):
            s = run[key]
            n = int(rng.binomial(len(s), float(self.p['rate'])))
            if n == 0:
                continue
            rows = np.sort(rng.choice(len(s), size=n, replace=False))
            dup = s.take(rows)
            dup.t_bag = dup.t_bag + np.array([sample_float(self.p['delay_ms'], rng) for _ in rows]) * 1e-3
            dup.label = dup.label | np.uint32(Label.DUPLICATE)
            run.streams[key] = Stream.concat([s, dup]).sort()
            events.append(self._event(ctx, [key], ctx.t_lo, ctx.t_hi, {'rate': self.p['rate']}, {'n_dup': n}))
        return events


@register
class Reorder(Injector):
    """Out-of-order delivery: some messages arrive ``delay`` seconds late (after newer ones)."""
    type_name = 'reorder'
    stage = 70
    defaults = {'topics': 'vehicle', 'rate': 0.01, 'delay': [0.15, 0.6]}

    def apply(self, run, ctx, rng):
        events = []
        for key in resolve_topics(self.p['topics']):
            s = run[key]
            n = int(rng.binomial(len(s), float(self.p['rate'])))
            if n == 0:
                continue
            rows = np.sort(rng.choice(len(s), size=n, replace=False))
            delays = np.array([sample_float(self.p['delay'], rng) for _ in rows])
            s.t_bag[rows] += delays
            s.label[rows] |= np.uint32(Label.OUT_OF_ORDER)
            run.streams[key] = s.sort()
            events.append(self._event(ctx, [key], ctx.t_lo, ctx.t_hi, {'rate': self.p['rate']},
                                      {'n_delayed': n, 'max_delay': float(delays.max())}))
        return events


# ============================================================================== timing faults

@register
class StampJitter(Injector):
    """Header-stamp jitter (Gaussian + heavy tail) and optional arrival (bag-time) jitter."""
    type_name = 'stamp_jitter'
    stage = 80
    placement_defaults = {'count': 0, 'duration': [30.0, 120.0]}
    defaults = {'topics': 'vehicle', 'sigma_ms': 15.0, 'tail_prob': 0.01, 'tail_ms': [50.0, 300.0],
                'arrival_sigma_ms': 0.0}

    def apply(self, run, ctx, rng):
        p = self.p
        count = sample_int(p['count'], rng) if p['at'] is None else len(p['at'])
        windows = [(ctx.t_lo - 10, ctx.t_hi + 10)] if count == 0 else self._windows(ctx, rng)
        events = []
        for (t0, t1) in windows:
            for key in resolve_topics(p['topics']):
                s = run[key]
                idx = _rows_between(s.t_hdr0, t0, t1)
                if len(idx) == 0:
                    continue
                sig = sample_float(p['sigma_ms'], rng) * 1e-3
                jit = rng.normal(0, sig, len(idx))
                tail = rng.random(len(idx)) < float(p['tail_prob'])
                jit[tail] += rng.choice([-1, 1], tail.sum()) * np.array(
                    [sample_float(p['tail_ms'], rng) for _ in range(tail.sum())]) * 1e-3
                s.t_hdr[idx] += jit
                s.label[idx] |= np.uint32(Label.STAMP_JITTER)
                asig = sample_float(p['arrival_sigma_ms'], rng) * 1e-3
                if asig > 0:
                    s.t_bag[idx] += np.abs(rng.normal(0, asig, len(idx)))
                    s.label[idx] |= np.uint32(Label.ARRIVAL_JITTER)
                    run.streams[key] = s.sort()
                events.append(self._event(ctx, [key], max(t0, ctx.t_lo), min(t1, ctx.t_hi),
                                          {'sigma_ms': sig * 1e3, 'arrival_sigma_ms': asig * 1e3},
                                          {'n_rows': int(len(idx)), 'n_tail': int(tail.sum())}))
        return events


@register
class ZeroStamps(Injector):
    """header.stamp = 0: random isolated messages (``rate``) and/or whole windows (``count``)."""
    type_name = 'zero_stamps'
    stage = 85  # after jitter / glitches / offsets: a missing stamp stays exactly 0
    placement_defaults = {'count': 0, 'duration': [1.0, 5.0], 'when': 'motion'}
    defaults = {'topics': 'vehicle', 'rate': 0.005}

    def apply(self, run, ctx, rng):
        events = []
        keys = resolve_topics(self.p['topics'])
        for key in keys:
            s = run[key]
            n = int(rng.binomial(len(s), float(self.p['rate'])))
            if n:
                rows = rng.choice(len(s), size=n, replace=False)
                s.t_hdr[rows] = 0.0
                s.label[rows] |= np.uint32(Label.ZERO_STAMP)
                events.append(self._event(ctx, [key], ctx.t_lo, ctx.t_hi, {'rate': self.p['rate']}, {'n_rows': n}))
        for (t0, t1) in self._windows(ctx, rng):
            st = {}
            for key in keys:
                s = run[key]
                idx = _rows_between(s.t_hdr0, t0, t1)
                s.t_hdr[idx] = 0.0
                s.label[idx] |= np.uint32(Label.ZERO_STAMP)
                st[key] = {'n_rows': int(len(idx))}
            events.append(self._event(ctx, keys, t0, t1, {'window': True}, st))
        return events


@register
class StampGlitch(Injector):
    """+-1 s header-stamp errors on isolated messages of all vehicle topics at once
    (sec/nanosec roll-over bug observed in bags 30618_2255aade and 30639_9c362687)."""
    type_name = 'stamp_glitch'
    stage = 80
    placement_defaults = {'count': [2, 5], 'duration': [0.3, 3.0], 'when': 'any'}
    defaults = {'topics': 'vehicle', 'offset_s': {'choice': [1.0, 1.0, -1.0]},
                'pattern': {'single': 0.5, 'alternating': 0.5}}

    def apply(self, run, ctx, rng):
        events = []
        for (t0, t1) in self._windows(ctx, rng):
            off = sample_float(self.p['offset_s'], rng)
            pattern = sample_choice(self.p['pattern'], rng)
            st = {}
            for key in resolve_topics(self.p['topics']):
                s = run[key]
                idx = _rows_between(s.t_hdr0, t0, t1)
                if len(idx) == 0:
                    continue
                rows = idx[:1] if pattern == 'single' else idx[::2]
                s.t_hdr[rows] += off
                s.label[rows] |= np.uint32(Label.STAMP_GLITCH)
                st[key] = {'n_rows': int(len(rows))}
            events.append(self._event(ctx, resolve_topics(self.p['topics']), t0, t1,
                                      {'offset_s': off, 'pattern': pattern}, st))
        return events


@register
class ClockOffset(Injector):
    """Header clock offset + drift over a window, ending with a step back (clock re-sync)."""
    type_name = 'clock_offset'
    stage = 80
    placement_defaults = {'count': 1, 'duration': [10.0, 60.0], 'when': 'any'}
    defaults = {'topics': 'vehicle', 'offset_s': [-1.0, 1.0], 'drift': [-0.05, 0.05]}

    def apply(self, run, ctx, rng):
        events = []
        for (t0, t1) in self._windows(ctx, rng):
            off = sample_float(self.p['offset_s'], rng)
            drift = sample_float(self.p['drift'], rng)
            st = {}
            for key in resolve_topics(self.p['topics']):
                s = run[key]
                idx = _rows_between(s.t_hdr0, t0, t1)
                if len(idx) == 0:
                    continue
                s.t_hdr[idx] += off + drift * (s.t_hdr0[idx] - t0)
                s.label[idx] |= np.uint32(Label.CLOCK_OFFSET)
                st[key] = {'n_rows': int(len(idx))}
            events.append(self._event(ctx, resolve_topics(self.p['topics']), t0, t1,
                                      {'offset_s': off, 'drift': drift}, st))
        return events


# ============================================================================== GNSS

@register
class GnssCut(Injector):
    """Remove all GNSS messages received later than ``keep_s`` seconds after the run start."""
    type_name = 'gnss_cut'
    stage = 90
    defaults = {'keep_s': 5.0, 'topics': 'gnss'}

    def apply(self, run, ctx, rng):
        keep = sample_float(self.p['keep_s'], rng)
        t_cut = run.t_start + keep
        st = {}
        for key in resolve_topics(self.p['topics']):
            if key not in GNSS or key not in run.streams:
                continue
            s = run.streams[key]
            n0 = len(s)
            run.streams[key] = s.keep(s.t_bag <= t_cut)
            st[key] = {'kept': len(run.streams[key]), 'removed': n0 - len(run.streams[key])}
        return [self._event(ctx, resolve_topics(self.p['topics']), t_cut, run.t_end, {'keep_s': keep}, st)]
