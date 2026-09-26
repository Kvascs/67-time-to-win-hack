"""Wheel-rail adhesion, wheelset dynamics and anti-slip / wheel-slide-protection controllers.

Model (per bogie, forces normalised by the bogie normal load N = m_b g, i.e. in "adhesion units"):

    w      = v_wheel - v                         slip velocity [m/s]
    s      = w / max(v, v_eps)                   creep (longitudinal slip ratio)
    f_a(w) = mu_s * ((1-A) exp(-B|w|) + A) * (2/pi) atan(s / s_c)      simplified Polach curve
    dv_wheel/dt = G (f_m - f_a)   =>   dw/dt = G (f_m - f_a) - a_vehicle
    f_m    = u * f_demand(notch)                  u in [u_min, 1] from the controller

* ``f_a`` rises linearly with creep (Kalker-like region, slope ~ mu_s*2/(pi s_c)), saturates at
  ~mu_s and then *decreases* with slip velocity (Polach's A, B parameters: wet / contaminated rail),
  which produces the characteristic run-away once demand exceeds available adhesion.
* ``G = N r^2 / J`` [m/s^2] is the ratio of adhesion force to rotating inertia (motor + gear +
  wheelsets referred to the rim).  G ~ 60-120 gives wheel run-away accelerations of 1-4 m/s^2 as
  seen in the natural slip episodes of the dataset.
* ``f_demand`` comes from the controller notch through tables measured on the training bags
  (median vehicle acceleration per notch), i.e. slip happens where the driver commands high torque.
* Controllers: ``cutoff`` = classic re-adhesion / WSP (cut torque or dump brake pressure when the
  slip exceeds ``w_on``, re-apply with a ramp once below ``w_off``) -> saw-tooth / hump trains;
  ``creep`` = modern slip-regulating control (PI on creep, holds ``s_target``) -> smooth plateau;
  ``none`` = no protection -> run-away in traction, wheel *lock* in braking.

The ground-truth vehicle motion is NOT altered (the recorded GNSS stays the reference); only the
wheel-speed measurement is corrupted.  This is the usual assumption of sensor fault injection.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

G_ACC = 9.81
#: Median vehicle acceleration [m/s^2] per notch (v > 1 m/s, notch taken 0.8 s earlier), 20 train bags.
TRACTION_ACC = np.array([0.0, 0.04, 0.09, 0.17, 0.25, 0.40, 0.60, 0.73, 0.84, 0.88, 0.88, 0.88, 0.88, 0.88,
                         0.88, 0.88])
BRAKE_ACC = np.array([0.0, 0.14, 0.52, 0.61, 0.65, 0.68, 0.73, 0.81, 0.90, 1.11, 1.30, 1.30, 1.33, 1.36, 1.45,
                      1.60])
RUNNING_RESISTANCE = 0.03  # [m/s^2] equivalent


def notch_demand(notch: np.ndarray) -> np.ndarray:
    """Adhesion demand (normalised tractive/braking force) implied by the notch."""
    n = np.clip(np.rint(np.nan_to_num(np.asarray(notch, float))), -15, 15).astype(int)
    f = np.zeros(n.shape, float)
    tr = n > 0
    br = n < 0
    f[tr] = (TRACTION_ACC[n[tr]] + RUNNING_RESISTANCE) / G_ACC
    f[br] = -np.maximum(BRAKE_ACC[-n[br]] - RUNNING_RESISTANCE, 0.0) / G_ACC
    return f


@dataclass
class AdhesionParams:
    mu_dry: float = 0.30   # adhesion level outside the low-adhesion patch
    s_c: float = 0.006     # creep scale of the atan curve (creep at ~0.5 mu_s is ~0.6 %)
    A: float = 0.40        # Polach A: mu(large slip velocity) / mu_s
    B: float = 0.35        # Polach B [s/m]
    v_eps: float = 0.5     # [m/s] creep regularisation at very low speed


@dataclass
class ControllerParams:
    kind: str = 'cutoff'   # 'cutoff' | 'creep' | 'none'
    w_on: float = 1.0      # [m/s] slip velocity that triggers torque cut / brake release
    w_off: float = 0.3     # [m/s] re-adhesion threshold
    s_target: float = 0.1  # creep controller set-point |s|
    u_min: float = 0.3     # lowest torque / brake fraction
    r_cut: float = 3.0     # [1/s] cut rate
    r_up: float = 0.5      # [1/s] re-application rate
    kp: float = 15.0       # creep PI gains (on creep error); tuned for ~10-30 % overshoot of s_target
    ki: float = 40.0
    tau: float = 0.05      # [s] torque / brake actuator lag
    lock_hold: float = 0.0  # [s] for kind='none' in braking: hold a locked wheel this long, then release


@dataclass
class WheelSim:
    """Result of one wheelset simulation."""
    t: np.ndarray
    w: np.ndarray                  # slip velocity [m/s] (v_wheel - v)
    u: np.ndarray                  # actuator fraction
    locked: np.ndarray             # bool: wheel at ~0 while vehicle moves
    info: dict = field(default_factory=dict)


def adhesion(w: float, v: float, mu_s: float, p: AdhesionParams) -> float:
    s = w / max(v, p.v_eps)
    return mu_s * ((1.0 - p.A) * math.exp(-p.B * abs(w)) + p.A) * (2.0 / math.pi) * math.atan(s / p.s_c)


_KIND_CODE = {'cutoff': 0, 'creep': 1, 'none': 2}


def _adh(w, v, mu, s_c, A, B, v_eps):
    s = w / max(v, v_eps)
    return mu * ((1.0 - A) * math.exp(-B * abs(w)) + A) * (2.0 / math.pi) * math.atan(s / s_c)


def _wheel_loop(dt, v, a, f_dem, mu_s, kind, w_on, w_off, s_target, u_min, r_cut, r_up, kp, ki, tau, lock_hold,
                s_c, A, B, v_eps, G, w_max, w_out, u_out, locked):
    """Core integration loop (plain Python; JIT-compiled with numba when available)."""
    n = v.shape[0]
    w = 0.0
    u = 1.0
    u_act = 1.0
    cut = False
    integ = 1.0
    lock_time = 0.0
    released = False
    eps = 1e-4
    for k in range(n):
        vk = v[k]
        fd = f_dem[k]
        mu = mu_s[k]
        sgn = 1.0 if fd > 0 else (-1.0 if fd < 0 else 0.0)
        ws = w * sgn  # excess slip in the direction of the applied force
        # ---------------------------------------------------------------- controller
        if sgn == 0.0:
            cut = False
            u = min(1.0, u + r_up * dt)
            integ = 1.0
        elif kind == 0:      # cutoff (re-adhesion control / classic WSP)
            if (not cut) and ws > w_on:
                cut = True
            elif cut and ws < w_off:
                cut = False
            if cut:
                u = max(u_min, u - r_cut * dt)
            else:
                u = min(1.0, u + r_up * dt)
        elif kind == 1:      # creep (slip-regulating PI)
            s = ws / max(vk, v_eps)
            e = s_target - s
            integ = min(1.0, max(u_min, integ + ki * e * dt))
            u = min(1.0, max(u_min, integ + kp * e))
        else:                # none: traction run-away / braking lock
            if sgn < 0 and lock_hold > 0:
                if vk + w < 0.05 and vk > 0.3:
                    lock_time += dt
                if lock_time >= lock_hold:
                    released = True
                if released:
                    # brake released (driver / back-up WSP); re-applied once the wheel has recovered
                    if abs(w) < w_off and lock_time > 0:
                        lock_time = 0.0
                        released = False
                        u = min(1.0, u + r_up * dt)
                    else:
                        u = max(0.0, u - 3.0 * dt)
                else:
                    u = min(1.0, u + r_up * dt)
            else:
                u = 1.0
        u_act += (u - u_act) * min(1.0, dt / max(tau, 1e-3))
        fm = u_act * fd
        # ---------------------------------------------------------------- wheelset dynamics
        fa = _adh(w, vk, mu, s_c, A, B, v_eps)
        dfa = (_adh(w + eps, vk, mu, s_c, A, B, v_eps) - _adh(w - eps, vk, mu, s_c, A, B, v_eps)) / (2 * eps)
        rhs = G * (fm - fa) - a[k]
        w_new = w + dt * rhs / (1.0 + dt * G * max(dfa, 0.0))
        if vk + w_new < 0.0:          # a braked wheel cannot turn backwards: locked
            w_new = -vk
        w = min(w_new, w_max)
        w_out[k] = w
        u_out[k] = u_act
        locked[k] = (vk > 0.3) and (vk + w < 0.05)


HAVE_NUMBA = False
if os.environ.get('ANOMALY_SIM_NUMBA', '0') == '1':  # opt-in: ~100x faster loop, but ~6-10 s import/JIT cost
    try:
        import numba as _nb
        _adh = _nb.njit(cache=True)(_adh)
        _wheel_loop = _nb.njit(cache=True)(_wheel_loop)
        HAVE_NUMBA = True
    except Exception:  # pragma: no cover - numba missing or broken
        HAVE_NUMBA = False


def simulate_wheel(t: np.ndarray, v: np.ndarray, a: np.ndarray, f_dem: np.ndarray, mu_s: np.ndarray,
                   ctrl: ControllerParams, adh: AdhesionParams, G: float = 90.0, w_max: float = 15.0) -> WheelSim:
    """Integrate the slip velocity of one bogie on a uniform grid ``t``.

    ``v``, ``a``: true vehicle speed / acceleration; ``f_dem``: demanded force (adhesion units,
    >0 traction, <0 braking); ``mu_s``: available adhesion level along time.
    Linearly-implicit Euler keeps the stiff creep region stable at dt of a few ms.
    """
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 0.002
    w_out = np.zeros(n)
    u_out = np.ones(n)
    locked = np.zeros(n, np.bool_)
    f64 = lambda x: np.ascontiguousarray(x, dtype=np.float64)  # noqa: E731
    _wheel_loop(dt, f64(v), f64(a), f64(f_dem), f64(mu_s), _KIND_CODE[ctrl.kind], float(ctrl.w_on),
                float(ctrl.w_off), float(ctrl.s_target), float(ctrl.u_min), float(ctrl.r_cut), float(ctrl.r_up),
                float(ctrl.kp), float(ctrl.ki), float(ctrl.tau), float(ctrl.lock_hold), float(adh.s_c),
                float(adh.A), float(adh.B), float(adh.v_eps), float(G), float(w_max), w_out, u_out, locked)
    return WheelSim(t=t, w=w_out, u=u_out, locked=locked)


def smooth_step(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def patch_profile(t: np.ndarray, t0: float, t1: float, mu_dry: float, mu_low: float, edge: float = 0.3) -> np.ndarray:
    """Available adhesion along time for a temporal low-adhesion patch [t0, t1] with smooth edges."""
    down = smooth_step((t - t0) / edge)
    up = smooth_step((t - t1) / edge)
    inside = down * (1 - up)
    return mu_dry + (mu_low - mu_dry) * inside


def patch_profile_space(x: np.ndarray, x0: float, x1: float, mu_dry: float, mu_low: float,
                        edge_m: float = 1.0) -> np.ndarray:
    """Available adhesion along the track for a spatial contamination patch [x0, x1] (metres)."""
    return patch_profile(x, x0, x1, mu_dry, mu_low, edge=edge_m)


def simulate_slip_delta(t: np.ndarray, v: np.ndarray, a: np.ndarray, notch: np.ndarray, mu_patch: np.ndarray,
                        ctrl: ControllerParams, adh: AdhesionParams, G: float) -> WheelSim:
    """Slip velocity caused by the patch only: simulation with patch minus simulation without.

    The clean recording already contains the natural micro-creep (~0.5-1.5 %), so only the
    difference is injected.
    """
    f_dem = notch_demand(notch)
    with_patch = simulate_wheel(t, v, a, f_dem, mu_patch, ctrl, adh, G)
    base = simulate_wheel(t, v, a, f_dem, np.full_like(mu_patch, adh.mu_dry), ctrl, adh, G)
    delta = with_patch.w - base.w
    return WheelSim(t=t, w=delta, u=with_patch.u, locked=with_patch.locked,
                    info={'f_dem_max': float(np.max(np.abs(f_dem))) if len(f_dem) else 0.0})
