"""Online (causal) traction/brake gain adaptation on top of a LUT-type model -> rollouts on val.
usage: python run_gain_adapt.py <model json> <name>"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import pandas as pd

from adapt import make_gain_fn
from final_lut import lut_sim_from_model
from lut_model import LutModel
from simulate import evaluate, summarize
from tid_data import OUT, list_bags

path, name = sys.argv[1], sys.argv[2]
sim = lut_sim_from_model(LutModel.from_json(json.loads((OUT / path).read_text())), name)
va = list_bags("val")
E = [evaluate(sim, va, stride=2.0, name=name)]
for T in (120.0, 300.0, 900.0):
    E.append(evaluate(sim, va, stride=2.0, gain_fn=make_gain_fn(T), name=f"{name}+gain{int(T)}s"))
E = pd.concat(E, ignore_index=True)
E.to_pickle(OUT / "cache" / f"eval_gain_{name}.pkl")
S = summarize(E)
S.to_csv(OUT / f"eval_gain_{name}.csv", index=False)
print(S.pivot_table(index="model", columns="H", values="rmse").round(4).to_string())
print(S.pivot_table(index="model", columns="H", values="bias").round(4).to_string())
S3 = summarize(E, by=("model", "H", "group"))
S3.to_csv(OUT / f"eval_gain_{name}_by_group.csv", index=False)
print(S3[S3.H.isin([5.0, 10.0, 30.0])].pivot_table(index=["group", "model"], columns="H", values="rmse").round(4).to_string())
