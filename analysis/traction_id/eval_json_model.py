"""Evaluate a LUT-type JSON model (lut_model.json format) on val: one-step + open-loop rollouts.
usage: python eval_json_model.py <json> <name> [bias_T]"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from adapt import make_bias_fn
from final_lut import lut_sim_from_model
from lut_model import LutModel
from simulate import evaluate, summarize
from tid_data import OUT, list_bags
from tm_core import apply_matched, get_arrays

path, name = sys.argv[1], sys.argv[2]
m = LutModel.from_json(json.loads((OUT / path).read_text()))
sim = lut_sim_from_model(m, name)
va = list_bags("val")
se = n = 0
for b in va:
    A = get_arrays(b)
    a, _ = sim.accel_series(A)
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    mm = A.fit & ((vv > 0.3) | (A.u > 0))
    e = (apply_matched(a) - A.a)[mm]
    se += float(e @ e)
    n += len(e)
E = [evaluate(sim, va, stride=2.0)]
for T in sys.argv[3:]:
    E.append(evaluate(sim, va, stride=2.0, bias_fn=make_bias_fn(float(T)), name=f"{name}+bias{T}s"))
import pandas as pd

E = pd.concat(E, ignore_index=True)
E.to_pickle(OUT / "cache" / f"eval_{name}.pkl")
S = summarize(E)
S["onestep"] = (se / n) ** 0.5
S.to_csv(OUT / f"eval_{name}.csv", index=False)
print(S.to_string())
