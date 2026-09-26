"""Fit the physics-parametric family (a) by projection onto the data-fitted LUT, evaluate on val."""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from lut_model import LutModel
from physics_model import PNAMES, export, fit_to_lut, physics_sim
from simulate import evaluate, summarize
from tid_data import OUT, list_bags
from tm_core import apply_matched, get_arrays

va = list_bags("val")
lut = LutModel.from_json(json.loads((OUT / "lut_model.json").read_text()))
support = np.load(OUT / "cache" / "lut_support.npy")
res = []
E = []
for form in ("dem", "sc"):
    p, rms = fit_to_lut(lut, support, form)
    print(form, "projection rms vs LUT", round(rms, 4), dict(zip(PNAMES, np.round(p, 4))), flush=True)
    sim = physics_sim(p, lut, form)
    se = n = 0
    for b in va:
        A = get_arrays(b)
        a, _ = sim.accel_series(A)
        vv = np.where(np.isfinite(A.v), A.v, A.vw)
        m = A.fit & ((vv > 0.3) | (A.u > 0))
        e = (apply_matched(a) - A.a)[m]
        se += float(e @ e)
        n += len(e)
    one = (se / n) ** 0.5
    print(form, "val one-step rmse", round(one, 4), flush=True)
    export(p, lut, form, OUT / f"physics_model_{form}.json")
    ev = evaluate(sim, va, stride=2.0)
    E.append(ev)
    res.append(dict(form=form, proj_rms=rms, onestep=one))
E = pd.concat(E, ignore_index=True)
E.to_pickle(OUT / "cache" / "eval_physics.pkl")
S = summarize(E)
S = S.merge(pd.DataFrame(res).assign(model=lambda d: "physics_" + d.form)[["model", "onestep"]], on="model")
S.to_csv(OUT / "eval_physics.csv", index=False)
print(S.to_string(), flush=True)
