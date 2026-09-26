"""Paired, bag-level comparison on VAL of the shipped OE table (oe_lut_s1.pt, Adam) and the L-BFGS refinements.

Same protocol as oe_torch.evaluate_torch (stride 2 s, open loop from the true speed, grade at the predicted
position); bootstrap over the 17 val bags (windows of one bag are correlated) of the RMSE ratio per horizon.

    python b_compare.py b_oe_lbfgs_warm_best.pt [b_oe_lbfgs_cold_best.pt ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
TID = Path(r"C:\MosTransHack\analysis\traction_id")
sys.path.insert(0, str(TID))
cands = sys.argv[1:]
sys.argv = [sys.argv[0], "lut"]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import oe_torch as OT  # noqa: E402
from lut_model import LutModel  # noqa: E402
from tid_data import OUT, list_bags  # noqa: E402


def run(sd_path, name):
    lut = LutModel.from_json(json.loads((OUT / "lut_model.json").read_text()))
    m = OT.Model(lut, 26, False)
    m.load_state_dict(torch.load(sd_path))
    E = OT.evaluate_torch(m, list_bags("val"), name)
    return E[~E.still]


def main():
    torch.set_num_threads(1)
    base = run(TID / "oe_lut_s1.pt", "adam_s1")
    rng = np.random.default_rng(0)
    rows = []
    for c in cands:
        E = run(HERE / c, Path(c).stem)
        for H in sorted(base.H.unique()):
            for col, lab in (("ev", "speed"), ("ed", "dist")):
                a = base[base.H == H].groupby("bag")[col].agg(lambda x: np.sum(x ** 2)).rename("a")
                b = E[E.H == H].groupby("bag")[col].agg(lambda x: np.sum(x ** 2)).rename("b")
                n = base[base.H == H].groupby("bag")[col].size().rename("n")
                d = pd.concat([a, b, n], axis=1).dropna()
                r0 = np.sqrt(d.a.sum() / d.n.sum())
                r1 = np.sqrt(d.b.sum() / d.n.sum())
                bs = []
                for _ in range(4000):
                    i = rng.integers(0, len(d), len(d))
                    bs.append(np.sqrt(d.b.to_numpy()[i].sum() / d.a.to_numpy()[i].sum()) - 1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append({"cand": c, "H": H, "what": lab, "rmse_adam": r0, "rmse_lbfgs": r1,
                             "rel_pct": 100 * (r1 / r0 - 1), "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi,
                             "bags_better": int((d.b < d.a).sum()), "bags": len(d)})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "b_compare_val.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
