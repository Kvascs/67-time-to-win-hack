"""Model family (c): gradient boosting / small NN on causal notch-history features + speed (+ grade).

Features at sample i (all causal, computable online from the notch stream + current state):
  exogenous (notch history, precomputable):
    u(i - L) for L in LAGS (samples at 20 Hz: 0..3 s), EMA of u with several time constants,
    EMA of max(u,0) and min(u,0), time since last notch change (clipped), previous notch,
    min / max of u over the last 1 s and 3 s
  state (from the rollout): v, grade felt gr (from map at predicted position), v^2, 1/max(v,1)
Two targets are supported:
  'direct'   : a
  'residual' : a - a_base (a_base from a Hammerstein model -> hybrid physics + ML)
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from simulate import HORIZONS, V_STILL, _grade_grid, _grade_lookup, reference_windows, start_indices
from tm_core import DT, apply_matched, get_arrays
from tid_pool import time_since_change

LAGS = (0, 2, 4, 6, 8, 10, 14, 20, 30, 40, 60)
EMA_TAUS = (0.25, 0.5, 1.0, 2.0, 4.0)


def ema(x, tau, dt=DT):
    from tm_core import lag1
    return lag1(x.astype(np.float64), 1 - np.exp(-dt / tau), float(x[0]))


def exog_features(u: np.ndarray, t: np.ndarray, dt=DT) -> tuple[np.ndarray, list[str]]:
    u = u.astype(np.float64)
    n = len(u)
    cols, names = [], []
    for L in LAGS:
        cols.append(np.r_[np.full(L, u[0]), u[:n - L]] if L else u)
        names.append(f"u_l{L}")
    for tau in EMA_TAUS:
        cols.append(ema(u, tau))
        names.append(f"u_ema{tau}")
    up = np.maximum(u, 0)
    dn = np.minimum(u, 0)
    for tau in (0.5, 2.0):
        cols.append(ema(up, tau)); names.append(f"up_ema{tau}")
        cols.append(ema(dn, tau)); names.append(f"dn_ema{tau}")
    tsc, prev, _ = time_since_change(t, u.astype(np.int64))
    cols.append(np.minimum(tsc, 10.0)); names.append("tsc")
    cols.append(prev.astype(np.float64)); names.append("prev_u")
    s = pd.Series(u)
    for w in (20, 60):
        cols.append(s.rolling(w, min_periods=1).max().to_numpy()); names.append(f"umax{w}")
        cols.append(s.rolling(w, min_periods=1).min().to_numpy()); names.append(f"umin{w}")
    return np.stack(cols, 1), names


STATE_NAMES = ["v", "gr", "v2", "inv_v"]


def state_features(v, gr):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v, gr, v * v, 1.0 / np.maximum(v, 1.0)], -1)


def bag_design(bag: str, base_sim=None):
    A = get_arrays(bag)
    X_ex, names = exog_features(A.u, A.t)
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    X = np.c_[X_ex, state_features(vv, np.nan_to_num(A.gr))]
    base = None
    if base_sim is not None:
        base, _ = base_sim.accel_series(A)
    return A, X, names + STATE_NAMES, base


def training_matrix(bags, base_sim=None, decim=2, target="direct"):
    Xs, ys, ws, gs = [], [], [], []
    for b in bags:
        A, X, names, base = bag_design(b, base_sim)
        vv = np.where(np.isfinite(A.v), A.v, A.vw)
        m = A.fit & ((vv > 0.3) | (A.u > 0))
        idx = np.flatnonzero(m)[::decim]
        y = A.a.copy()
        if target == "residual":
            y = y - apply_matched(base)
        Xs.append(X[idx])
        ys.append(y[idx])
        gs.append(np.full(len(idx), b))
    return np.concatenate(Xs), np.concatenate(ys), names, np.concatenate(gs)


# ------------------------------------------------------------------------------------------------
# generic rollout for a = f(exog(i), v, gr) models (+ optional Hammerstein base)
# ------------------------------------------------------------------------------------------------

class MLSim:
    def __init__(self, predict_fn, name, base_sim=None, target="direct", dt_sim=0.1):
        self.predict_fn = predict_fn  # X (N, F) -> a (N,)
        self.name = name
        self.base_sim = base_sim
        self.target = target
        self.dt_sim = dt_sim

    def accel_series(self, A):
        _, X, _, base = bag_design(A.bag, self.base_sim)
        a = self.predict_fn(X)
        if self.target == "residual":
            a = a + base
        return a, None

    def rollout(self, A, i0s, horizons=HORIZONS, bias=None, dt=DT):
        _, X, _, _ = bag_design(A.bag, None)
        X_ex = X[:, :-len(STATE_NAMES)]
        step = int(round(self.dt_sim / dt))
        rec = np.array([int(round(H / dt)) for H in horizons])
        nmax = rec.max()
        N = len(i0s)
        v = A.v[i0s].astype(np.float64).copy()
        s = A.s[i0s].astype(np.float64).copy()
        dr = A.dirn[i0s]
        d = np.zeros(N)
        b0 = np.zeros(N) if bias is None else bias
        gs0, gds, gg = _grade_grid()
        vout = np.full((N, len(rec)), np.nan)
        dout = np.full((N, len(rec)), np.nan)
        # base model states
        if self.target == "residual":
            kd, a_up, a_dn = self.base_sim.params(dt)
            _, y_true = self.base_sim.accel_series(A)
            y = y_true[i0s].copy()
        n = A.n
        for k in range(step, nmax + 1, step):
            i = np.minimum(i0s + k, n - 1)
            gr = np.array([_grade_lookup(si, gs0, gds, gg) for si in s]) * dr
            Xk = np.c_[X_ex[i], state_features(v, gr)]
            a = self.predict_fn(Xk)
            if self.target == "residual":
                # propagate base Hammerstein state over 'step' fine steps with frozen v
                for q in range(step):
                    j = np.maximum(i0s + k - step + q + 1 - kd, 0)
                    uu = A.u[np.minimum(j, n - 1)]
                    c = self.base_sim.static(uu, v)
                    c = np.where((v <= V_STILL) & (uu <= 0), 0.0, c)
                    al = np.where(c > y, a_up, a_dn)
                    y = y + al * (c - y)
                cls = np.where(A.u[i] < 0, 0, np.where(A.u[i] == 0, 1, 2))
                a = a + y - self.base_sim.kg[cls] * gr
            a = a + b0
            vn = v + a * self.dt_sim
            vn = np.where((vn <= V_STILL) & (a < 0), 0.0, vn)
            vn = np.maximum(vn, 0.0)
            d += 0.5 * (v + vn) * self.dt_sim
            s += dr * 0.5 * (v + vn) * self.dt_sim
            v = vn
            for r, rs in enumerate(rec):
                if rs == k:
                    vout[:, r] = v
                    dout[:, r] = d
        return vout, dout
