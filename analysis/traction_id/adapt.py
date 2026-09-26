"""Per-bag gain analysis (mass / load) and online adaptation experiments.

Decomposition of a Hammerstein model output into regime channels (linear in the lagged command):
    y = y_trac + y_brake + y_coast      (each = lag(c * 1[regime]))
Per-bag regression on clean samples:
    ag ~ g_t * K*y_trac + g_b * K*y_brake + g_c * K*y_coast - kg * K*gr + b
g_t, g_b ~ 1 means the fleet model fits the bag; a torque-controlled drive with varying passenger
load would show g_t, g_b varying together (~ m_nominal / m_bag).

Online adaptation (causal) for bridging:
    residual r(t) = a_meas(t) - a_model(t) on clean samples (a_meas from wheels online; here the
    GNSS reference delayed by 0.5 s to stay causal w.r.t. the zero-phase smoothing),
    bias b(t0) = exponentially-forgotten mean of r up to t0 (time constant T_b),
    optional RLS on [g_t, g_b, b] with forgetting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from simulate import V_STILL
from tm_core import DT, apply_matched, get_arrays, lag1

CAUSAL_SHIFT = int(round(0.5 / DT))  # samples: ag is zero-phase with +-0.5 s support


def regime_channels(sim, A):
    """Return lagged command split into traction/brake/coast channels (sum = y) and grade term."""
    kd, a_up, a_dn = sim.params()
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    u = A.u
    ud = np.r_[np.full(kd, u[0]), u[:len(u) - kd]] if kd > 0 else u
    c = sim.static(ud, vv)
    c = np.where((vv <= V_STILL) & (ud <= 0), 0.0, c)
    # symmetric lag for the linear split (asymmetry is small, see report)
    alpha = 1 - np.exp(-DT / (0.5 * (sim.tau_up + sim.tau_dn)))
    ch = {}
    for name, m in (("trac", ud > 0), ("brake", ud < 0), ("coast", ud == 0)):
        ch[name] = lag1(np.where(m, c, 0.0), alpha, 0.0)
    cls = np.where(u < 0, 0, np.where(u == 0, 1, 2))
    ch["grade"] = -sim.kg[cls] * np.nan_to_num(A.gr)
    return ch


def per_bag_gains(sim, bags):
    rows = []
    for b in bags:
        A = get_arrays(b)
        ch = regime_channels(sim, A)
        vv = np.where(np.isfinite(A.v), A.v, A.vw)
        m = A.fit & ((vv > 0.3) | (A.u > 0))
        X = np.stack([apply_matched(ch[k]) for k in ("trac", "brake", "coast", "grade")] + [np.ones(A.n)], 1)[m]
        y = A.a[m]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r0 = y - X[:, :4].sum(1)  # fleet model residual (all gains 1, no bias)
        r1 = y - X @ coef
        # also gain-only on traction (x) and brake, with grade fixed at 1
        rows.append(dict(bag=b, group=A.group, g_trac=coef[0], g_brake=coef[1], g_coast=coef[2], g_grade=coef[3],
                         bias=coef[4], rmse_fleet=np.sqrt(np.mean(r0 ** 2)), rmse_bag=np.sqrt(np.mean(r1 ** 2)),
                         mean_res_fleet=r0.mean(), n=len(y), t0=A.t[0]))
    return pd.DataFrame(rows)


def causal_bias(A, i0s, sim, T_b=20.0, a_model=None):
    """Exponentially-forgotten mean residual up to (i0 - CAUSAL_SHIFT), clean samples only."""
    if a_model is None:
        a_model, _ = sim.accel_series(A)
    am_f = apply_matched(a_model)
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    ok = A.fit & ((vv > 0.3) | (A.u > 0))
    r = np.where(ok, A.a - am_f, 0.0)
    w = ok.astype(float)
    alpha = 1 - np.exp(-DT / T_b)
    num = lag1(r, alpha, 0.0)
    den = lag1(w, alpha, 0.0)
    est = np.where(den > 1e-3, num / np.maximum(den, 1e-9), 0.0)
    # shrink towards 0 when little recent clean data
    est = est * np.clip(den / 0.5, 0, 1)
    j = np.maximum(i0s - CAUSAL_SHIFT, 0)
    return est[j]


def make_bias_fn(T_b):
    def fn(A, i0s, sim):
        return causal_bias(A, i0s, sim, T_b)
    return fn


def causal_gains(A, i0s, sim, T=300.0, lam=0.02, clip=(0.7, 1.3)):
    """Exponentially-forgotten LS estimate of traction / brake gains up to (i0 - CAUSAL_SHIFT).

    z = a_meas - K*(y_coast - kg*gr) ~ g_t K*y_trac + g_b K*y_brake, prior (g_t, g_b) -> (1, 1) with weight lam.
    Returns (g_t[N], g_b[N], y0[N]) where y0 is the gain-consistent lag state at i0.
    """
    ch = regime_channels(sim, A)
    Yt = apply_matched(ch["trac"])
    Yb = apply_matched(ch["brake"])
    O = apply_matched(ch["coast"] + ch["grade"])
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    ok = A.fit & ((vv > 0.3) | (A.u > 0))
    z = np.where(ok, A.a - O, 0.0)
    Yt0 = np.where(ok, Yt, 0.0)
    Yb0 = np.where(ok, Yb, 0.0)
    al = 1 - np.exp(-DT / T)
    Stt = lag1(Yt0 * Yt0, al, 0.0)
    Sbb = lag1(Yb0 * Yb0, al, 0.0)
    Stb = lag1(Yt0 * Yb0, al, 0.0)
    Stz = lag1(Yt0 * z, al, 0.0)
    Sbz = lag1(Yb0 * z, al, 0.0)
    j = np.maximum(i0s - CAUSAL_SHIFT, 0)
    a11 = Stt[j] + lam
    a22 = Sbb[j] + lam
    a12 = Stb[j]
    b1 = Stz[j] + lam
    b2 = Sbz[j] + lam
    det = a11 * a22 - a12 * a12
    gt = np.clip((a22 * b1 - a12 * b2) / det, *clip)
    gb = np.clip((a11 * b2 - a12 * b1) / det, *clip)
    y0 = gt * ch["trac"][i0s] + gb * ch["brake"][i0s] + ch["coast"][i0s]
    return gt, gb, y0


def make_gain_fn(T, lam=0.02):
    def fn(A, i0s, sim):
        return causal_gains(A, i0s, sim, T, lam)
    return fn
