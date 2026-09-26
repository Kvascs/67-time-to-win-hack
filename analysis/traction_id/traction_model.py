"""Tram traction / braking model for the backup-odometry estimator (model-based speed prediction).

    notch u(t) (int -15..15), speed v, grade g(s)  ->  longitudinal acceleration a  [m/s^2]

Identified on 52 train bags, validated on 17 val bags (traction_id analysis). Final model = "OE-LUT":
a 2-D table fitted by differentiable open-loop simulation (output-error), i.e. optimised for exactly
the use-case of bridging wheel slip / slide / dropouts, plus optional online traction/brake gains.

    c(t)   = F(u(t), v(t))                        static map: table[31][NV] over v_knots (linear in v),
                                                  c = 0 when v <= v_still and u <= 0 (holding brake)
    y_r    = lag_tau(c * 1[regime(u) == r])       one first-order lag per regime r in {trac, brake, coast}
                                                  (their sum is the lag of c; no measurable dead time)
    a(t)   = g_t*y_trac + g_b*y_brake + y_coast - kg[regime(u)] * g(t) + b
    g(t)   = grade(s) * dir                       track grade (+ uphill), s from the western terminus,
                                                  dir = +1 eastbound (s increasing) / -1 westbound
    g_t, g_b = online gains (GainAdapter), 1.0 by default;  b = optional bias (0 recommended)

Speed propagation for bridging:  v <- v + a dt;  if v <= v_still and a < 0: v = 0 (no rolling back).

All parameters are plain arrays / scalars in traction_model_params.json -> direct C++ port:
    lut.u_min, lut.v_knots[NV], lut.table[31][NV], kg[3] (brake, coast, traction), tau, v_still,
    grade.{s0, ds, values[]}, adapter.{T, lam, g_min, g_max}, window_s, noise model, detector.

C++ port of TractionModel.step(dt, u, v, grade):
    r      = u < 0 ? BRAKE : (u == 0 ? COAST : TRAC)
    c      = (v <= v_still && u <= 0) ? 0 : lerp(v_knots, table[u - u_min], v)
    alpha  = 1 - exp(-dt / tau)
    for k in {TRAC, BRAKE, COAST}: y[k] += alpha * ((k == r ? c : 0) - y[k])
    a      = g_t*y[TRAC] + g_b*y[BRAKE] + y[COAST] - kg[r] * grade + bias
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "traction_model_params.json"
TRAC, BRAKE, COAST = 0, 1, 2


def regime(u: int) -> int:
    return BRAKE if u < 0 else (COAST if u == 0 else TRAC)


class GradeProfile:
    """grade(s) on a uniform grid; s = along-track distance from the western terminus [m]."""

    def __init__(self, s0: float, ds: float, values):
        self.s0, self.ds = float(s0), float(ds)
        self.g = np.asarray(values, dtype=float)

    def at(self, s, direction: float = 1.0) -> float:
        if s is None or not math.isfinite(s):
            return 0.0
        x = (s - self.s0) / self.ds
        if x <= 0:
            g = self.g[0]
        elif x >= len(self.g) - 1:
            g = self.g[-1]
        else:
            i = int(x)
            f = x - i
            g = self.g[i] * (1 - f) + self.g[i + 1] * f
        return float(g) * direction


class Track:
    """Centerline (1 m polyline, UTM 37N) -> along-track s and travel direction from two positions."""

    def __init__(self, csv_path=HERE / "centerline.csv"):
        cl = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self.s, self.x, self.y = cl[:, 0], cl[:, 1], cl[:, 2]

    def project(self, x: float, y: float) -> float:
        i = int(np.argmin((self.x - x) ** 2 + (self.y - y) ** 2))
        j0, j1 = max(i - 1, 0), min(i + 1, len(self.s) - 1)
        tx, ty = self.x[j1] - self.x[j0], self.y[j1] - self.y[j0]
        n = math.hypot(tx, ty) or 1.0
        return float(self.s[i] + ((x - self.x[i]) * tx + (y - self.y[i]) * ty) / n)

    def direction(self, x0, y0, x1, y1) -> float:
        """+1 if moving towards increasing s (eastbound), -1 otherwise (needs >= ~5 m of motion)."""
        return 1.0 if self.project(x1, y1) >= self.project(x0, y0) else -1.0


class TractionModel:
    """Causal, stateful model; call step() at any rate (e.g. on every wheel / notch message)."""

    def __init__(self, params: dict | str | Path = PARAMS):
        if not isinstance(params, dict):
            params = json.loads(Path(params).read_text())
        self.p = params
        L = params["lut"]
        self.u_min = int(L["u_min"])
        self.v_knots = np.asarray(L["v_knots"], dtype=float)
        self.table = np.asarray(L["table"], dtype=float)
        kg = params["kg"]  # [brake, coast, traction] as exported
        self.kg = {BRAKE: float(kg[0]), COAST: float(kg[1]), TRAC: float(kg[2])}
        self.tau = float(params["tau"])
        self.v_still = float(params.get("v_still", 0.05))
        g = params.get("grade")
        self.grade = GradeProfile(g["s0"], g["ds"], g["values"]) if g else None
        self.g_t = 1.0
        self.g_b = 1.0
        self.reset()

    def reset(self, y=(0.0, 0.0, 0.0)):
        self.y = [float(y[0]), float(y[1]), float(y[2])]  # TRAC, BRAKE, COAST channel states

    def static_accel(self, u: int, v: float) -> float:
        """Steady-state flat-track acceleration for notch u at speed v (incl. running resistance)."""
        iu = int(min(max(u, self.u_min), self.u_min + self.table.shape[0] - 1)) - self.u_min
        return float(np.interp(v, self.v_knots, self.table[iu]))

    def command(self, u: int, v: float) -> float:
        return 0.0 if (v <= self.v_still and u <= 0) else self.static_accel(u, v)

    def step(self, dt: float, u: int, v: float, grade: float = 0.0, bias: float = 0.0) -> float:
        """Advance the lag states by dt with current notch u and speed v; return acceleration."""
        u = int(u)
        r = regime(u)
        c = self.command(u, v)
        alpha = 1.0 - math.exp(-dt / self.tau)
        for k in (TRAC, BRAKE, COAST):
            self.y[k] += alpha * ((c if k == r else 0.0) - self.y[k])
        return self.accel(u, grade, bias)

    def accel(self, u: int, grade: float = 0.0, bias: float = 0.0) -> float:
        """Acceleration from the current states (no update)."""
        return (self.g_t * self.y[TRAC] + self.g_b * self.y[BRAKE] + self.y[COAST]
                - self.kg[regime(int(u))] * grade + bias)

    def predict_speed(self, v0: float, notches, dt: float, s0: float | None = None, direction: float = 1.0,
                      bias: float = 0.0) -> np.ndarray:
        """Open-loop speed prediction from v0 over a known notch sequence (state is restored)."""
        saved = list(self.y)
        v, s = float(v0), s0
        out = np.empty(len(notches))
        for k, u in enumerate(notches):
            g = self.grade.at(s, direction) if (self.grade is not None and s is not None) else 0.0
            a = self.step(dt, int(u), v, g, bias)
            vn = v + a * dt
            if vn <= self.v_still and a < 0:
                vn = 0.0
            if s is not None:
                s += direction * 0.5 * (v + vn) * dt
            v = vn
            out[k] = v
        self.y = saved
        return out


class WindowedAccel:
    """Causal 1-s window: LS slope of measured speed and the model channels averaged with the SAME
    (parabolic) weights, so a_meas and the model are compared on an equal footing. Feed at a fixed
    rate (e.g. 20 Hz) with speeds from slip-free, bogie-consistent wheel data."""

    def __init__(self, dt: float = 0.05, window_s: float = 1.0):
        W = int(round(window_s / dt))
        tau = (np.arange(W) - (W - 1) / 2) * dt
        self.k_slope = tau / np.sum(tau ** 2)
        k_acc = -np.cumsum(self.k_slope)[:-1] * dt
        self.k_acc = k_acc / k_acc.sum()
        self.buf_v = deque(maxlen=W)
        self.buf_m = deque(maxlen=W - 1)

    def push(self, v_meas: float, model_terms):
        """model_terms: tuple (y_trac, y_brake, offset) with offset = y_coast - kg*grade."""
        self.buf_v.append(v_meas)
        self.buf_m.append(tuple(model_terms))
        if len(self.buf_v) < self.buf_v.maxlen:
            return None
        a_meas = float(np.dot(self.k_slope, np.asarray(self.buf_v)))
        M = np.asarray(self.buf_m)
        yt, yb, off = (float(np.dot(self.k_acc, M[:, j])) for j in range(3))
        return a_meas, yt, yb, off


class GainAdapter:
    """Online traction / brake gains (passenger load, vehicle-to-vehicle spread): exponentially
    forgotten least squares  a_meas - off ~ g_t*y_t + g_b*y_b  with a prior (1, 1) of weight lam.
    Update only on clean samples (both bogies consistent, no slip/slide, model-validity detector ok)."""

    def __init__(self, T: float = 300.0, lam: float = 0.02, g_min: float = 0.7, g_max: float = 1.3):
        self.T, self.lam, self.g_min, self.g_max = T, lam, g_min, g_max
        self.S = np.zeros(5)  # Stt, Sbb, Stb, Stz, Sbz

    def update(self, dt: float, a_meas: float, y_t: float, y_b: float, off: float, valid: bool = True):
        f = math.exp(-dt / self.T)
        self.S *= f
        if valid:
            z = a_meas - off
            self.S += (1 - f) * np.array([y_t * y_t, y_b * y_b, y_t * y_b, y_t * z, y_b * z])

    @property
    def gains(self):
        a11, a22, a12 = self.S[0] + self.lam, self.S[1] + self.lam, self.S[2]
        b1, b2 = self.S[3] + self.lam, self.S[4] + self.lam
        det = a11 * a22 - a12 * a12
        gt = (a22 * b1 - a12 * b2) / det
        gb = (a11 * b2 - a12 * b1) / det
        return float(np.clip(gt, self.g_min, self.g_max)), float(np.clip(gb, self.g_min, self.g_max))


def load(params_path=PARAMS) -> TractionModel:
    return TractionModel(params_path)
