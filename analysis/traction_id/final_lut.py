"""Fit the final LUT model on all train bags, refine asymmetric lag, export JSON, evaluate on val.

usage: python final_lut.py [delay tau]      (defaults: best of lut_grid.csv)
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from lut_model import NU, NV, U_MIN, V_KNOTS, LutModel, fit_lut, score_accel
from simulate import HammersteinSim, evaluate, summarize
from tid_data import OUT, list_bags

V_FINE = np.round(np.arange(0.0, 20.0 + 1e-9, 0.1), 3)


def to_uniform(table: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(V_FINE, V_KNOTS, row) for row in table])


def lut_sim_from_model(m: LutModel, name="lut", use_grade=True) -> HammersteinSim:
    return HammersteinSim(to_uniform(m.table), 0.0, 0.1, m.kg, m.delay, m.tau_up, m.tau_dn, name, use_grade, U_MIN)


def load_lut_sim(path=None, name="lut") -> HammersteinSim:
    path = path or OUT / "lut_model.json"
    m = LutModel.from_json(json.loads(open(path).read()))
    return lut_sim_from_model(m, name)


def main():
    tr, va = list_bags("train"), list_bags("val")
    if len(sys.argv) >= 3:
        d, tau = float(sys.argv[1]), float(sys.argv[2])
    else:
        g = pd.read_csv(OUT / "lut_grid.csv")
        r = g.loc[g.rmse_val.idxmin()]
        d, tau = float(r.delay), float(r.tau)
    t0 = time.time()
    m, rmse_tr, support = fit_lut(tr, d, tau, decim=2)
    print(f"fit d={d} tau={tau}: train rmse {rmse_tr:.4f} kg {m.kg.round(3)} ({time.time() - t0:.0f} s)", flush=True)
    np.save(OUT / "cache" / "lut_support.npy", support)
    base_val = score_accel(m, va)
    print("val one-step rmse (symmetric lag)", round(base_val, 4), flush=True)
    # asymmetric lag refinement (table kept; small grid on ratio)
    best = (base_val, m.tau_up, m.tau_dn)
    for up, dn in [(tau * 0.7, tau * 1.3), (tau * 1.3, tau * 0.7), (tau * 0.5, tau * 1.5), (tau * 1.5, tau * 0.5)]:
        m2 = LutModel(m.table, m.kg, m.delay, up, dn)
        sc = score_accel(m2, va[::2])
        sc0 = score_accel(LutModel(m.table, m.kg, m.delay, best[1], best[2]), va[::2])
        print(f"  tau_up {up:.2f} tau_dn {dn:.2f}: val(sub) {sc:.4f} vs {sc0:.4f}", flush=True)
        if sc < sc0 - 1e-4:
            best = (sc, up, dn)
    m.tau_up, m.tau_dn = best[1], best[2]
    js = m.to_json()
    js["support_seconds"] = np.round(support / 20.0 * 2, 1).tolist()  # decim=2 at 20 Hz
    js["fit"] = dict(train_bags=len(tr), rmse_train=rmse_tr)
    (OUT / "lut_model.json").write_text(json.dumps(js))
    rmse_val, per_bag = score_accel(m, va, per_bag=True)
    print("final val one-step rmse", round(rmse_val, 4), "tau_up/dn", m.tau_up, m.tau_dn, flush=True)
    pd.DataFrame([(b, *v) for b, v in per_bag.items()], columns=["bag", "rmse", "bias"]).to_csv(
        OUT / "lut_val_per_bag.csv", index=False)
    # open-loop evaluation
    sims = [lut_sim_from_model(m, "lut"), lut_sim_from_model(m, "lut_nograde", use_grade=False)]
    E = pd.concat([evaluate(s, va, stride=2.0) for s in sims], ignore_index=True)
    E.to_pickle(OUT / "cache" / "eval_lut.pkl")
    S = summarize(E)
    S["onestep_rmse"] = rmse_val
    print(S.to_string(), flush=True)
    S.to_csv(OUT / "eval_lut.csv", index=False)


if __name__ == "__main__":
    main()
