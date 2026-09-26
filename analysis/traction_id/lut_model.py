"""Model family (b): 2-D lookup table a_ss(notch, v) + dead time + first-order lag + grade term.

    c(t)  = LUT(u(t - d), v(t))                  (bilinear in v, exact in integer notch)
    y     = lag_tau(c)                           (first-order lag, optionally asymmetric)
    a(t)  = y(t) - k_g[class(u(t))] * gr(t)      (gravity acts instantly; class = brake/coast/traction)

For fixed (d, tau) the model is linear in the table entries and k_g, so it is fitted by regularised
linear least squares on matched-filtered regressors (normal equations accumulated per bag).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import lfilter

from tm_core import DT, apply_matched, get_arrays, lag1, lag1_asym, matched_kernel

V_STILL = 0.05

V_KNOTS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                    14.0, 15.0, 16.5, 18.0, 20.0])
U_MIN, U_MAX = -15, 15
NU = U_MAX - U_MIN + 1
NV = len(V_KNOTS)
NCELL = NU * NV


def v_weights(v: np.ndarray):
    v = np.clip(np.nan_to_num(v), V_KNOTS[0], V_KNOTS[-1])
    j = np.clip(np.searchsorted(V_KNOTS, v, side="right") - 1, 0, NV - 2)
    w1 = (v - V_KNOTS[j]) / (V_KNOTS[j + 1] - V_KNOTS[j])
    return j, w1


def shift_notch(u: np.ndarray, kd: int) -> np.ndarray:
    """u(t - kd*dt). kd > 0: causal dead time; kd < 0: look-ahead (analysis of timing offsets only)."""
    n = len(u)
    if kd > 0:
        return np.r_[np.full(kd, u[0]), u[:n - kd]]
    if kd < 0:
        k = -kd
        return np.r_[u[k:], np.full(k, u[-1])]
    return u


def u_class(u: np.ndarray) -> np.ndarray:
    return np.where(u < 0, 0, np.where(u == 0, 1, 2))


def lut_eval(table: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    iu = np.clip(u, U_MIN, U_MAX) - U_MIN
    j, w1 = v_weights(v)
    return table[iu, j] * (1 - w1) + table[iu, j + 1] * w1


@dataclass
class LutModel:
    table: np.ndarray = field(default_factory=lambda: np.zeros((NU, NV)))
    kg: np.ndarray = field(default_factory=lambda: np.array([9.0, 9.0, 9.0]))  # brake, coast, traction
    delay: float = 0.3
    tau_up: float = 0.5
    tau_dn: float = 0.5
    name: str = "lut"

    # ---- continuous-time step (used by simulators and the C++ port) -------------------------
    def kd(self, dt=DT):
        return int(round(self.delay / dt))

    def accel_series(self, u: np.ndarray, v: np.ndarray, gr: np.ndarray, dt=DT, y0=0.0):
        """Causal model acceleration along a recorded trajectory (v given)."""
        ud = shift_notch(u, self.kd(dt))
        c = np.where((v <= V_STILL) & (ud <= 0), 0.0, lut_eval(self.table, ud, v))
        a_up = 1 - np.exp(-dt / self.tau_up)
        a_dn = 1 - np.exp(-dt / self.tau_dn)
        y = lag1_asym(c, a_up, a_dn, y0)
        return y - self.kg[u_class(u)] * np.nan_to_num(gr), y

    def to_json(self) -> dict:
        return dict(type="lut", u_min=U_MIN, u_max=U_MAX, v_knots=V_KNOTS.tolist(),
                    table=np.round(self.table, 5).tolist(), kg=self.kg.tolist(), delay=self.delay,
                    tau_up=self.tau_up, tau_dn=self.tau_dn)

    @staticmethod
    def from_json(d: dict) -> "LutModel":
        return LutModel(np.array(d["table"]), np.array(d["kg"]), d["delay"], d["tau_up"], d["tau_dn"])


# ------------------------------------------------------------------------------------------------
# Linear least squares for fixed (delay, tau)
# ------------------------------------------------------------------------------------------------

def _bag_normal_eq(bag: str, kd: int, alpha: float, decim: int = 2, kg_mode: str = "class"):
    A = get_arrays(bag)
    n = A.n
    ud = shift_notch(A.u, kd)
    iu = np.clip(ud, U_MIN, U_MAX) - U_MIN
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    j, w1 = v_weights(vv)
    c0 = iu * NV + j
    still = (vv <= V_STILL) & (ud <= 0)  # holding brake at standstill: zero command
    cells = np.unique(np.r_[c0, c0 + 1])
    # time-last contiguous layout (cells x n) -> fast filtering along the last axis
    X = np.zeros((len(cells), n), dtype=np.float32)
    ci0 = np.searchsorted(cells, c0)
    ci1 = np.searchsorted(cells, c0 + 1)
    ar = np.arange(n)
    np.add.at(X, (ci0, ar), np.where(still, 0, 1 - w1).astype(np.float32))
    np.add.at(X, (ci1, ar), np.where(still, 0, w1).astype(np.float32))
    X = lfilter(np.array([alpha], np.float32), np.array([1, -(1 - alpha)], np.float32), X, axis=-1)
    K = matched_kernel().astype(np.float32)
    X = convolve1d(X, K[::-1], axis=-1, mode="nearest")
    # grade regressors (not lagged), matched filtered
    gr = np.nan_to_num(A.gr)
    cls = np.where(A.u < 0, 0, np.where(A.u == 0, 1, 2))
    if kg_mode == "class":
        G = np.stack([-(cls == k).astype(float) * gr for k in range(3)], 0)
    else:
        G = -gr[None, :]
    G = convolve1d(G, matched_kernel()[::-1], axis=-1, mode="nearest")
    rows = np.flatnonzero(A.fit & ((vv > 0.3) | (A.u > 0)))[::decim]
    Xr = np.r_[X[:, rows].astype(np.float64), G[:, rows]]
    y = A.a[rows]
    return cells, Xr @ Xr.T, Xr @ y, float(y @ y), len(rows)


def _reg_matrix(lam_v=3.0, lam_u=3.0, ridge=1e-3):
    """Smoothness penalties on the (NU x NV) table."""
    R = []
    idx = lambda iu, jv: iu * NV + jv
    # along v (2nd difference with knot spacing) per notch row
    for iu in range(NU):
        for jv in range(1, NV - 1):
            h1 = V_KNOTS[jv] - V_KNOTS[jv - 1]
            h2 = V_KNOTS[jv + 1] - V_KNOTS[jv]
            r = np.zeros(NCELL)
            r[idx(iu, jv - 1)] = 1 / h1
            r[idx(iu, jv)] = -(1 / h1 + 1 / h2)
            r[idx(iu, jv + 1)] = 1 / h2
            R.append(lam_v * r)
    # along notch (2nd difference) within brake side and traction side separately
    for side in (range(U_MIN, 0), range(1, U_MAX + 1)):
        us = list(side)
        for k in range(1, len(us) - 1):
            for jv in range(NV):
                r = np.zeros(NCELL)
                r[idx(us[k - 1] - U_MIN, jv)] = 1
                r[idx(us[k] - U_MIN, jv)] = -2
                r[idx(us[k + 1] - U_MIN, jv)] = 1
                R.append(lam_u * r)
    R = np.array(R)
    return R, ridge


def fit_lut(bags: list[str], delay: float, tau: float, lam_v=3.0, lam_u=3.0, ridge=1e-3, decim=2,
            kg_mode: str = "class", fixed_kg: float | None = None):
    kd = int(round(delay / DT))
    alpha = 1 - np.exp(-DT / tau)
    nk = 3 if kg_mode == "class" else 1
    npar = NCELL + nk
    AtA = np.zeros((npar, npar))
    Atb = np.zeros(npar)
    yy = 0.0
    nrow = 0
    for b in bags:
        cells, ata, atb, y2, nr = _bag_normal_eq(b, kd, alpha, decim, kg_mode)
        ix = np.r_[cells, NCELL + np.arange(nk)]
        AtA[np.ix_(ix, ix)] += ata
        Atb[ix] += atb
        yy += y2
        nrow += nr
    R, ridge = _reg_matrix(lam_v, lam_u, ridge)
    scale = nrow / 1e4  # regularisation relative to data volume
    Rf = np.zeros((R.shape[0], npar))
    Rf[:, :NCELL] = R
    M = AtA + scale * (Rf.T @ Rf) + scale * ridge * np.diag(np.r_[np.ones(NCELL), np.zeros(nk)])
    rhs = Atb.copy()
    if fixed_kg is not None:
        # move kg to rhs
        kgv = np.full(nk, fixed_kg)
        rhs = rhs[:NCELL] - AtA[:NCELL, NCELL:] @ kgv
        sol = np.linalg.solve(M[:NCELL, :NCELL], rhs)
        sol = np.r_[sol, kgv]
    else:
        sol = np.linalg.solve(M, rhs)
    table = sol[:NCELL].reshape(NU, NV)
    kg = sol[NCELL:]
    if kg_mode != "class":
        kg = np.repeat(kg, 3)
    sse = yy - 2 * sol @ Atb + sol @ AtA @ sol
    rmse = np.sqrt(max(sse, 0) / nrow)
    # support: diagonal of AtA per cell
    support = np.diag(AtA)[:NCELL].reshape(NU, NV)
    m = LutModel(table, kg, delay, tau, tau)
    return m, rmse, support


def score_accel(model, bags: list[str], per_bag=False):
    """One-step acceleration RMSE (matched filter) over fit samples."""
    se, n, out = 0.0, 0, {}
    for b in bags:
        A = get_arrays(b)
        vv = np.where(np.isfinite(A.v), A.v, A.vw)
        am, _ = model.accel_series(A.u, vv, A.gr)
        am_f = apply_matched(am)
        e = (am_f - A.a)[A.fit & ((vv > 0.3) | (A.u > 0))]
        se += float(e @ e)
        n += len(e)
        out[b] = (np.sqrt(np.mean(e ** 2)), np.mean(e))
    if per_bag:
        return np.sqrt(se / n), out
    return np.sqrt(se / n)
