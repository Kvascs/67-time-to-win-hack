"""Coordinate search of dead time d and lag tau for the LUT (Hammerstein) model. Logs to lut_grid.csv.

Uses a quarter of the train bags for speed (the machine is shared); scoring = one-step matched
acceleration RMSE on all validation bags.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from lut_model import fit_lut, score_accel
from tid_data import OUT, list_bags

tr = list_bags("train")
va = list_bags("val")
sub = tr[::4]
out = OUT / "lut_grid.csv"
rows = pd.read_csv(out).to_dict("records") if out.exists() else []


def run(d, tau):
    for r in rows:
        if abs(r["delay"] - d) < 1e-9 and abs(r["tau"] - tau) < 1e-9:
            return r["rmse_val"]
    t0 = time.time()
    m, rmse_tr, _ = fit_lut(sub, d, tau, decim=3)
    rmse_va = score_accel(m, va)
    rows.append(dict(delay=d, tau=tau, rmse_train=rmse_tr, rmse_val=rmse_va, kg_b=m.kg[0], kg_c=m.kg[1],
                     kg_t=m.kg[2], sec=time.time() - t0))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(rows[-1], flush=True)
    return rmse_va


best_d, best_tau = 0.3, 0.5
for it in range(2):
    ds = [0.0, 0.15, 0.3, 0.45, 0.6] if it == 0 else [max(0, best_d - 0.1), best_d, best_d + 0.1]
    sc = {d: run(d, best_tau) for d in ds}
    best_d = min(sc, key=sc.get)
    ts = [0.25, 0.4, 0.55, 0.75] if it == 0 else [max(0.1, best_tau - 0.1), best_tau, best_tau + 0.1]
    sc = {t: run(best_d, t) for t in ts}
    best_tau = min(sc, key=sc.get)
    print("iteration", it, "best", best_d, best_tau, flush=True)
print("FINAL best delay", best_d, "tau", best_tau, flush=True)
