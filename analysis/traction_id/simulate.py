"""Open-loop speed prediction ("bridging") evaluation, common to all model families.

Protocol (per validation bag):
  * start points i0 every STRIDE seconds where the sample is clean, not in a notch-not-in-control
    episode, and GNSS speed is valid;
  * the model's internal states (lag / delay line) are obtained by running the model causally on the
    recorded data up to i0 (in the online estimator the model runs continuously in parallel);
  * from i0 the speed is propagated open-loop from the TRUE speed v(i0): v <- max(0, v + a dt),
    position s <- s + dir v dt (grade is looked up at the *predicted* position, as it would be online);
  * error e_H = v_pred(i0+H) - v_ref(i0+H) for H in HORIZONS, and the distance error
    d_H = integral(v_pred - v_ref) over the window;
  * windows touching an auto episode or with invalid reference at i0+H are excluded;
  * windows where the vehicle is stationary throughout (v_ref < 0.2 and notch <= 0) are tagged
    'standstill' and excluded from the headline numbers (trivial).
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from route_map import grade_profile
from tm_core import DT, get_arrays

HORIZONS = (1.0, 3.0, 5.0, 10.0, 30.0)
V_STILL = 0.05  # [m/s] standstill threshold used consistently in fit / causal pass / rollout
STRIDE = 1.0


def _grade_grid():
    prof = grade_profile()
    s = prof["s"].to_numpy()
    g = prof["grade"].to_numpy()
    return float(s[0]), float(s[1] - s[0]), g.astype(np.float64)


@nb.njit(cache=True)
def _grade_lookup(s, s0, ds, g):
    x = (s - s0) / ds
    if x <= 0:
        return g[0]
    if x >= len(g) - 1:
        return g[len(g) - 1]
    i = int(x)
    f = x - i
    return g[i] * (1 - f) + g[i + 1] * f


@nb.njit(cache=True)
def _lut_lookup(table, v0, dv, iu, v):
    x = (v - v0) / dv
    nv = table.shape[1]
    if x <= 0:
        return table[iu, 0]
    if x >= nv - 1:
        return table[iu, nv - 1]
    i = int(x)
    f = x - i
    return table[iu, i] * (1 - f) + table[iu, i + 1] * f


@nb.njit(cache=True)
def run_states(table, tv0, tdv, umin, kd, a_up, a_dn, u, v, jmax_dt=1e9):
    """Causal pass over the recorded data with the TRUE speed -> lag state y at every sample.
    jmax_dt: optional rate limit on the lag state per step (jerk limit * dt)."""
    n = len(u)
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        j = i - kd
        uu = u[j] if j >= 0 else u[0]
        if v[i] <= V_STILL and uu <= 0:
            c = 0.0  # brake at standstill produces no acceleration (holding brake)
        else:
            c = _lut_lookup(table, tv0, tdv, uu - umin, v[i])
        al = a_up if c > acc else a_dn
        dy = al * (c - acc)
        if dy > jmax_dt:
            dy = jmax_dt
        elif dy < -jmax_dt:
            dy = -jmax_dt
        acc = acc + dy
        y[i] = acc
    return y


@nb.njit(cache=True)
def sim_hammerstein(table, tv0, tdv, umin, kg, kd, a_up, a_dn, u, dirn, i0s, v0s, s0s, y0s, b0s,
                    nsteps, dt, gs0, gds, gg, use_grade, rec_steps, jmax_dt=1e9, gts=None, gbs=None):
    """Vectorised open-loop rollouts. Returns speed and distance at the requested step counts."""
    N = len(i0s)
    R = len(rec_steps)
    vout = np.full((N, R), np.nan)
    dout = np.full((N, R), np.nan)
    n = len(u)
    for k in range(N):
        i0 = i0s[k]
        v = v0s[k]
        s = s0s[k]
        y = y0s[k]
        b = b0s[k]
        gt = 1.0 if gts is None else gts[k]
        gb = 1.0 if gbs is None else gbs[k]
        d = 0.0
        dr = dirn[i0]
        r = 0
        for step in range(1, nsteps + 1):
            i = i0 + step
            if i >= n:
                break
            j = i - kd
            uu = u[j] if j >= 0 else u[0]
            if v <= V_STILL and uu <= 0:
                c = 0.0
            else:
                c = _lut_lookup(table, tv0, tdv, uu - umin, v)
                if uu > 0:
                    c *= gt
                elif uu < 0:
                    c *= gb
            al = a_up if c > y else a_dn
            dy = al * (c - y)
            if dy > jmax_dt:
                dy = jmax_dt
            elif dy < -jmax_dt:
                dy = -jmax_dt
            y = y + dy
            ui = u[i]
            cls = 0 if ui < 0 else (1 if ui == 0 else 2)
            gr = _grade_lookup(s, gs0, gds, gg) * dr if use_grade else 0.0
            a = y - kg[cls] * gr + b
            vn = v + a * dt
            if vn <= V_STILL and a < 0.0:
                vn = 0.0  # snap to standstill; holding brake / static friction: no rolling back
            d += 0.5 * (v + vn) * dt
            s += dr * 0.5 * (v + vn) * dt
            v = vn
            while r < R and rec_steps[r] == step:
                vout[k, r] = v
                dout[k, r] = d
                r += 1
    return vout, dout


def start_indices(A, stride=STRIDE, dt=DT):
    idx = np.arange(int(2.0 / dt), A.n, int(round(stride / dt)))
    ok = A.fit[idx] & np.isfinite(A.v[idx]) & np.isfinite(A.s[idx])
    return idx[ok]


def reference_windows(A, i0s, horizons=HORIZONS, dt=DT):
    """Reference speed/distance at each horizon + validity + tags."""
    n = A.n
    vref = np.where(np.isfinite(A.v), A.v, np.nan)
    vfill = pd.Series(vref).interpolate(limit_area="inside").to_numpy()
    dist = np.r_[0, np.cumsum(0.5 * (vfill[1:] + vfill[:-1]) * dt)]
    autoc = np.r_[0, np.cumsum(A.auto.astype(np.int64))]
    uposc = np.r_[0, np.cumsum((A.u > 0).astype(np.int64))]
    out = {}
    for H in horizons:
        h = int(round(H / dt))
        i1 = i0s + h
        valid = i1 < n
        i1c = np.minimum(i1, n - 1)
        v1 = np.where(valid, vref[i1c], np.nan)
        d1 = np.where(valid, dist[i1c] - dist[i0s], np.nan)
        auto_in = (autoc[np.minimum(i1c + 1, n)] - autoc[i0s]) > 0
        vmax = np.array([np.nanmax(vfill[a:b + 1]) if b > a else np.nan for a, b in zip(i0s, i1c)])
        trac_in = (uposc[np.minimum(i1c + 1, n)] - uposc[i0s]) > 0
        still = (vmax < 0.2) & ~trac_in
        ok = valid & np.isfinite(v1) & ~auto_in
        out[H] = dict(h=h, v1=v1, d1=d1, ok=ok, still=still, auto_in=auto_in & valid)
    return out


def summarize(errs: pd.DataFrame, by=("model", "H")) -> pd.DataFrame:
    g = errs[~errs.still].groupby(list(by))
    return g.agg(n=("ev", "size"), rmse=("ev", lambda x: float(np.sqrt(np.mean(x ** 2)))),
                 mae=("ev", lambda x: float(np.mean(np.abs(x)))), bias=("ev", "mean"),
                 p95=("ev", lambda x: float(np.percentile(np.abs(x), 95))),
                 d_rmse=("ed", lambda x: float(np.sqrt(np.mean(x ** 2)))),
                 d_mae=("ed", lambda x: float(np.mean(np.abs(x)))),
                 d_bias=("ed", "mean")).reset_index()


class HammersteinSim:
    """Adapter: any model exposing a static map F(u, v) (tabulated), kg[3], delay, tau_up/tau_dn."""

    def __init__(self, table, tv0, tdv, kg, delay, tau_up, tau_dn, name, use_grade=True, umin=-15, jerk=None):
        self.jerk = jerk  # optional rate limit [m/s^3] on the lagged command
        self.table = np.ascontiguousarray(table, dtype=np.float64)
        self.tv0, self.tdv = float(tv0), float(tdv)
        self.kg = np.asarray(kg, dtype=np.float64)
        self.delay, self.tau_up, self.tau_dn = delay, tau_up, tau_dn
        self.name = name
        self.use_grade = use_grade
        self.umin = umin

    def params(self, dt=DT):
        return int(round(self.delay / dt)), 1 - np.exp(-dt / self.tau_up), 1 - np.exp(-dt / self.tau_dn)

    def static(self, u, v):
        """Vectorised static map F(u, v) (steady-state flat-track acceleration)."""
        iu = np.clip(np.asarray(u, dtype=np.int64) - self.umin, 0, self.table.shape[0] - 1)
        x = np.clip((np.asarray(v, dtype=np.float64) - self.tv0) / self.tdv, 0, self.table.shape[1] - 1 - 1e-9)
        i = x.astype(np.int64)
        f = x - i
        return self.table[iu, i] * (1 - f) + self.table[iu, i + 1] * f

    def accel_series(self, A, dt=DT):
        kd, a_up, a_dn = self.params(dt)
        vv = np.where(np.isfinite(A.v), A.v, A.vw)
        jd = 1e9 if self.jerk is None else self.jerk * dt
        y = run_states(self.table, self.tv0, self.tdv, self.umin, kd, a_up, a_dn, A.u.astype(np.int64), vv, jd)
        cls = np.where(A.u < 0, 0, np.where(A.u == 0, 1, 2))
        gr = np.nan_to_num(A.gr) if self.use_grade else 0.0
        return y - self.kg[cls] * gr, y

    def rollout(self, A, i0s, horizons=HORIZONS, bias=None, dt=DT, gains=None):
        """gains: optional (g_trac[N], g_brake[N], y0[N]) per start point (online gain adaptation)."""
        kd, a_up, a_dn = self.params(dt)
        _, y = self.accel_series(A, dt)
        gs0, gds, gg = _grade_grid()
        rec = np.array([int(round(H / dt)) for H in horizons], dtype=np.int64)
        b0 = np.zeros(len(i0s)) if bias is None else bias
        y0 = y[i0s].astype(np.float64)
        gts = gbs = None
        if gains is not None:
            gts, gbs, y0 = (np.asarray(x, dtype=np.float64) for x in gains)
        vout, dout = sim_hammerstein(self.table, self.tv0, self.tdv, self.umin, self.kg, kd, a_up, a_dn,
                                     A.u.astype(np.int64), A.dirn, i0s.astype(np.int64),
                                     A.v[i0s].astype(np.float64), A.s[i0s].astype(np.float64),
                                     y0, b0.astype(np.float64), int(rec.max()), dt,
                                     gs0, gds, gg, self.use_grade, rec, 1e9 if self.jerk is None else self.jerk * dt,
                                     gts, gbs)
        return vout, dout


def evaluate(sim, bags, horizons=HORIZONS, stride=STRIDE, bias_fn=None, name=None, gain_fn=None):
    rows = []
    for b in bags:
        A = get_arrays(b)
        i0s = start_indices(A, stride)
        if len(i0s) == 0:
            continue
        bias = bias_fn(A, i0s, sim) if bias_fn is not None else None
        if gain_fn is not None:
            vout, dout = sim.rollout(A, i0s, horizons, bias=bias, gains=gain_fn(A, i0s, sim))
        else:
            vout, dout = sim.rollout(A, i0s, horizons, bias=bias)
        ref = reference_windows(A, i0s, horizons)
        for r, H in enumerate(horizons):
            R = ref[H]
            ok = R["ok"] & np.isfinite(vout[:, r])
            rows.append(pd.DataFrame(dict(model=name or sim.name, bag=b, group=A.group, H=H, i0=i0s[ok],
                                          v0=A.v[i0s[ok]], u0=A.u[i0s[ok]], ev=vout[ok, r] - R["v1"][ok],
                                          ed=dout[ok, r] - R["d1"][ok], still=R["still"][ok])))
    return pd.concat(rows, ignore_index=True)


class BaselineSim:
    """Naive bridging baselines.
    'hold_v' : speed held at v(i0) (what a wheel-only estimator does during a dropout)
    'hold_a' : causal acceleration estimate at i0 (backward 1-s difference of speed) held, clamped at 0
    """

    def __init__(self, kind="hold_v"):
        self.kind = kind
        self.name = f"baseline_{kind}"

    def rollout(self, A, i0s, horizons=HORIZONS, bias=None, dt=DT):
        v0 = A.v[i0s]
        if self.kind == "hold_v":
            a0 = np.zeros(len(i0s))
        else:
            k = int(round(1.0 / dt))
            vb = A.v[np.maximum(i0s - k, 0)]
            a0 = np.nan_to_num((v0 - vb) / (k * dt))
        vout = np.zeros((len(i0s), len(horizons)))
        dout = np.zeros_like(vout)
        for r, H in enumerate(horizons):
            # v(t) = max(0, v0 + a0 t): distance with clamp
            t_stop = np.where(a0 < 0, -v0 / np.where(a0 < 0, a0, -1), np.inf)
            tt = np.minimum(H, t_stop)
            vout[:, r] = np.maximum(0.0, v0 + a0 * H)
            dout[:, r] = v0 * tt + 0.5 * a0 * tt ** 2
        return vout, dout
