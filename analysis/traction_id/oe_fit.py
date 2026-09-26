"""Output-error (simulation-error) fitting of the physics-parametric model (family a).

Objective: open-loop speed errors v_pred(i0+H) - v_ref(i0+H), H in {2, 5, 10} s, from start points every
3 s on TRAIN bags (clean, no notch-not-in-control inside the window, non-trivial windows), robust loss.
This directly optimises what matters for bridging, instead of the one-step (equation-error) fit.
Parameters: 14 static-map parameters (physics_model) + kg[3] + tau (dead time fixed at 0).

usage: python oe_fit.py [form]
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from lut_model import LutModel
from physics_model import PNAMES, XK, fit_to_lut, physics_sim, static_map
from simulate import (_grade_grid, evaluate, reference_windows, run_states, sim_hammerstein, start_indices,
                      summarize)
from tid_data import OUT, list_bags
from tm_core import DT, apply_matched, get_arrays

HOR = (2.0, 5.0, 10.0)
V_FINE = np.round(np.arange(0.0, 20.0 + 1e-9, 0.1), 3)
UU, VV = np.meshgrid(np.arange(-15, 16), V_FINE, indexing="ij")


class OEData:
    def __init__(self, bags, stride=3.0):
        self.items = []
        for b in bags:
            A = get_arrays(b)
            i0s = start_indices(A, stride)
            ref = reference_windows(A, i0s, HOR)
            ok = np.zeros(len(i0s), bool)
            for H in HOR:
                ok |= ref[H]["ok"] & ~ref[H]["still"]
            i0s = i0s[ok]
            ref = reference_windows(A, i0s, HOR)
            vref = np.stack([ref[H]["v1"] for H in HOR], 1)
            okm = np.stack([ref[H]["ok"] & ~ref[H]["still"] for H in HOR], 1)
            vv = np.where(np.isfinite(A.v), A.v, A.vw)
            self.items.append(dict(u=A.u.astype(np.int64), dirn=A.dirn.astype(np.float64), v=vv.astype(np.float64),
                                   i0s=i0s.astype(np.int64), v0=A.v[i0s].astype(np.float64),
                                   s0=A.s[i0s].astype(np.float64), vref=vref, ok=okm))
        self.gs0, self.gds, self.gg = _grade_grid()
        self.rec = np.array([int(round(H / DT)) for H in HOR], dtype=np.int64)
        self.n = sum(int(it["ok"].sum()) for it in self.items)

    def residuals(self, table, kg, tau):
        a = 1 - np.exp(-DT / tau)
        out = []
        for it in self.items:
            y = run_states(table, 0.0, 0.1, -15, 0, a, a, it["u"], it["v"])
            vout, _ = sim_hammerstein(table, 0.0, 0.1, -15, kg, 0, a, a, it["u"], it["dirn"], it["i0s"], it["v0"],
                                      it["s0"], y[it["i0s"]], np.zeros(len(it["i0s"])), int(self.rec.max()), DT,
                                      self.gs0, self.gds, self.gg, True, self.rec)
            r = vout - it["vref"]
            out.append(r[it["ok"]])
        return np.concatenate(out)


def unpack(x):
    p = x[:14]
    kg = np.array(x[14:17])
    tau = x[17]
    return p, kg, tau


def main():
    form = sys.argv[1] if len(sys.argv) > 1 else "sc"
    tr, va = list_bags("train"), list_bags("val")
    lut = LutModel.from_json(json.loads((OUT / "lut_model.json").read_text()))
    support = np.load(OUT / "cache" / "lut_support.npy")
    p0, _ = fit_to_lut(lut, support, form)
    x0 = np.r_[p0, lut.kg, lut.tau_up]
    t0 = time.time()
    D = OEData(tr, stride=3.0)
    print("OE data: residuals", D.n, "prep %.0f s" % (time.time() - t0), flush=True)

    def fun(x):
        p, kg, tau = unpack(x)
        table = static_map(UU, VV, p, form)
        return D.residuals(table, kg, tau)

    r0 = fun(x0)
    print("initial rmse per residual %.4f" % np.sqrt(np.mean(r0 ** 2)), flush=True)
    lb = [-0.2, -0.05, -0.01, 0.0, 0.0, 0.0, 0.2, 0.3, 1.0, 0, 0, 0, 0, 0, 3.0, 3.0, 3.0, 0.05]
    ub = [0.3, 0.05, 0.01, 1.0, 0.3, 1.5, 12.0, 2.0, 30.0, 1.2, 1.2, 1.2, 1.2, 1.2, 12.0, 12.0, 12.0, 1.5]
    x0 = np.clip(x0, np.array(lb) + 1e-6, np.array(ub) - 1e-6)
    t0 = time.time()
    sol = least_squares(fun, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.5, diff_step=1e-3, max_nfev=60,
                        verbose=1)
    print("OE fit %.0f s, cost %.3f -> %.3f" % (time.time() - t0, 0.5 * np.sum(r0 ** 2), sol.cost), flush=True)
    p, kg, tau = unpack(sol.x)
    names = PNAMES + ["kg_brake", "kg_coast", "kg_trac", "tau"]
    print(dict(zip(names, np.round(sol.x, 4))), flush=True)
    m = LutModel(lut.table, kg, 0.0, tau, tau)
    sim = physics_sim(p, m, form, name=f"physics_{form}_oe")
    d = dict(type="physics_oe", form=form, params=dict(zip(PNAMES, np.round(p, 6).tolist())), x_knots=XK.tolist(),
             kg=kg.tolist(), delay=0.0, tau_up=float(tau), tau_dn=float(tau), objective="open-loop speed error H=2,5,10 s")
    (OUT / f"physics_model_{form}_oe.json").write_text(json.dumps(d, indent=1))
    # one-step on val
    se = n = 0
    for b in va:
        A = get_arrays(b)
        a, _ = sim.accel_series(A)
        vv = np.where(np.isfinite(A.v), A.v, A.vw)
        mm = A.fit & ((vv > 0.3) | (A.u > 0))
        e = (apply_matched(a) - A.a)[mm]
        se += float(e @ e)
        n += len(e)
    one = (se / n) ** 0.5
    E = evaluate(sim, va, stride=2.0)
    E.to_pickle(OUT / "cache" / f"eval_physics_{form}_oe.pkl")
    S = summarize(E)
    S["onestep"] = one
    S.to_csv(OUT / f"eval_physics_{form}_oe.csv", index=False)
    print(S.to_string(), flush=True)


if __name__ == "__main__":
    main()
