"""Exploration: does the per-bogie value quantum q track the wheel scale k of the bag?"""
import json

import numpy as np
import pandas as pd

ROOT = r"C:\MosTransHack"
sp = json.load(open(ROOT + r"\data\splits.json"))
truth = pd.read_csv(ROOT + r"\analysis\cross_check\scale_lag_per_bag.csv").set_index("bag")


def quantum(v):
    dv = np.abs(np.diff(v))
    dv = dv[(dv > 0.0035) & (dv < 0.0062)]
    if len(dv) < 20:
        return np.nan, np.nan, 0
    med = np.median(dv)
    dv = dv[np.abs(dv - med) < 0.00005]
    # halves of the bag
    return np.median(dv), np.std(dv), len(dv)


rows = []
for b in sp["train"] + sp["val"]:
    z = np.load(ROOT + rf"\data\npz\{b}.npz")
    r = {"bag": b, "split": "train" if b in sp["train"] else "val"}
    for key, nm in (("vehicle__front_bogie_velocity", "f"), ("vehicle__rear_bogie_velocity", "r")):
        v = z[key][:, 2]
        n = len(v)
        q, s, c = quantum(v)
        q1, _, _ = quantum(v[: n // 2])
        q2, _, _ = quantum(v[n // 2:])
        r.update({f"q_{nm}": q, f"qsd_{nm}": s, f"n_{nm}": c, f"q1_{nm}": q1, f"q2_{nm}": q2})
    if b in truth.index:
        r["k_front"] = truth.loc[b, "k_front"] / 3.6 / 1.00037 - 1
        r["k_rear"] = truth.loc[b, "k_rear"] / 3.6 / 1.00037 - 1
        r["vehicle"] = truth.loc[b, "vehicle"]
        r["date"] = truth.loc[b, "date"]
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df[["bag", "vehicle", "date", "q_f", "q1_f", "q2_f", "q_r", "q1_r", "q2_r", "k_front", "k_rear"]]
      .sort_values(["vehicle", "date"]).round(7).to_string(index=False))
for nm, kc in (("f", "k_front"), ("r", "k_rear")):
    d = df.dropna(subset=[f"q_{nm}", kc])
    c = np.corrcoef(d[f"q_{nm}"], d[kc])[0, 1]
    print(nm, "corr(q, k) =", round(c, 3), "n =", len(d))
