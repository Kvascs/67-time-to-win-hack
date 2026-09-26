"""Model family (a): physics-parametric static map F(notch, v) + the same dead time / lag / grade term.

Per unit mass [m/s^2], flat track, steady state (running resistance included):
  resistance     R(v)   = c0 + c1 v + c2 v^2                                  (Davis A + Bv + Cv^2)
  coast   u = 0: F = -R(v)
  brake   u < 0: F = -(b0 + b1 |u|) * h(v) - R(v),  h(v) = h0 + (1 - h0) * min(v, v_h) / v_h
                 (electrodynamic brake with reduced effect at very low speed / blending)
  traction u > 0: F = min(x(u) * A_max, P_max / v) - R(v)        ('demand' form, variant 'dem')
              or F = x(u) * min(A_max, P_max / v) - R(v)          ('scaled' form, variant 'sc')
                 x(u): monotone piecewise-linear demand fraction, knots u = 1,3,5,7,10,15 (x(15)=1)
Dynamics (delay, tau_up, tau_dn) and grade coefficients kg[3] are shared with the LUT model.
The parameters are fitted by support-weighted least squares projection onto the data-fitted LUT
(the LUT is the unconstrained least-squares estimate of the same static map), then scored on val.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import least_squares

from lut_model import U_MIN, V_KNOTS, LutModel
from simulate import HammersteinSim
from tid_data import OUT

XK = np.array([1, 3, 5, 7, 10, 15], dtype=float)
PNAMES = ["c0", "c1", "c2", "b0", "b1", "h0", "v_h", "A_max", "P_max", "x1", "x3", "x5", "x7", "x10"]


def traction_demand(u, p):
    xs = np.r_[np.abs(p[9:14]), 1.0]
    xs = np.maximum.accumulate(xs)  # monotone
    return np.interp(u, XK, xs)


def static_map(u, v, p, form="dem"):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    c0, c1, c2, b0, b1, h0, v_h, A_max, P_max = p[:9]
    R = c0 + c1 * v + c2 * v * v
    k = np.abs(u)
    h = h0 + (1 - h0) * np.minimum(v, abs(v_h)) / abs(v_h)
    brake = -(b0 + b1 * k) * h - R
    x = traction_demand(np.clip(u, 1, 15), p)
    vv = np.maximum(v, 0.1)
    if form == "dem":
        trac = np.minimum(x * A_max, P_max / vv) - R
    else:
        trac = x * np.minimum(A_max, P_max / vv) - R
    return np.where(u < 0, brake, np.where(u == 0, -R, trac))


def fit_to_lut(lut: LutModel, support: np.ndarray, form="dem"):
    U, Vg = np.meshgrid(np.arange(U_MIN, 16), V_KNOTS, indexing="ij")
    w = np.sqrt(np.clip(support, 0, None))
    w = w / w.max()
    m = w > 1e-3
    p0 = np.array([0.03, 0.0, 0.0002, 0.25, 0.08, 0.7, 2.0, 1.0, 8.0, 0.1, 0.2, 0.3, 0.7, 0.8])

    def res(p):
        return ((static_map(U, Vg, p, form) - lut.table) * w)[m]

    lb = [-0.2, -0.05, -0.01, 0.0, 0.0, 0.0, 0.2, 0.3, 1.0, 0, 0, 0, 0, 0]
    ub = [0.3, 0.05, 0.01, 1.0, 0.3, 1.5, 12.0, 2.0, 30.0, 1.2, 1.2, 1.2, 1.2, 1.2]
    sol = least_squares(res, p0, bounds=(lb, ub), loss="soft_l1", f_scale=0.05)
    rms = np.sqrt(np.average((static_map(U, Vg, sol.x, form) - lut.table) ** 2, weights=w ** 2))
    return sol.x, rms


def physics_sim(p, lut: LutModel, form="dem", name=None) -> HammersteinSim:
    vf = np.round(np.arange(0.0, 20.0 + 1e-9, 0.1), 3)
    U, Vg = np.meshgrid(np.arange(U_MIN, 16), vf, indexing="ij")
    table = static_map(U, Vg, p, form)
    return HammersteinSim(table, 0.0, 0.1, lut.kg, lut.delay, lut.tau_up, lut.tau_dn, name or f"physics_{form}",
                          True, U_MIN)


def export(p, lut: LutModel, form, path):
    d = dict(type="physics", form=form, params=dict(zip(PNAMES, np.round(p, 6).tolist())), x_knots=XK.tolist(),
             kg=lut.kg.tolist(), delay=lut.delay, tau_up=lut.tau_up, tau_dn=lut.tau_dn,
             doc=__doc__)
    path.write_text(json.dumps(d, indent=1))
