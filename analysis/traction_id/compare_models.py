"""Collect all validation evaluations -> model_comparison.csv/.md + fig_model_comparison.png.
usage: python compare_models.py [final_model_name]"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tid_data import OUT

FINAL = sys.argv[1] if len(sys.argv) > 1 else "lut_oe_s1+gain300s"
files = ["eval_adapt.csv", "eval_grade_check.csv", "eval_physics.csv", "eval_physics_sc_oe.csv", "eval_gbm_direct.csv",
         "eval_gain_lut_oe.csv", "eval_oe_lut_nn_torch.csv"] + [f for f in os.listdir(OUT) if f.startswith("eval_gain_lut_oe_s") and "_by_" not in f]
rows = []
for f in files:
    p = OUT / f
    if not p.exists():
        continue
    S = pd.read_csv(p)
    if "onestep_rmse" in S and "onestep" not in S:
        S["onestep"] = S["onestep_rmse"]
    rows.append(S)
S = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["model", "H"], keep="last")
# one-step values known from logs when not stored in the csv
ONE = {"lut": 0.1040, "lut_with_grade": 0.1040, "lut_fit_without_grade": 0.1422, "gbm_direct": 0.0725,
       "lut_oe": 0.1147, "lut_oe_s1": 0.1107, "lut_oe_s1+gain300s": 0.1107}
S["onestep"] = S.apply(lambda r: r.get("onestep") if pd.notna(r.get("onestep", np.nan)) else ONE.get(r.model, np.nan),
                       axis=1)
keep = ["baseline_hold_v", "baseline_hold_a", "lut_fit_without_grade", "lut", "lut_group", "lut+bias20s",
        "physics_dem", "physics_sc", "physics_sc_oe", "gbm_direct", "oe_lut_nn", "lut_oe", "lut_oe+gain300s"]
keep += ["lut_oe_s1", "lut_oe_s1+gain300s"]
S = S[S.model.isin(keep)]
T = S.pivot_table(index="model", columns="H", values="rmse").reindex([k for k in keep if k in set(S.model)])
Tb = S.pivot_table(index="model", columns="H", values="bias").reindex(T.index)
Td = S.pivot_table(index="model", columns="H", values="d_rmse").reindex(T.index)
one = S.groupby("model").onestep.first().reindex(T.index)
tab = pd.concat({"speed RMSE [m/s]": T, "speed bias [m/s]": Tb, "distance RMSE [m]": Td}, axis=1)
tab[("one-step accel RMSE [m/s2]", "")] = one
tab.round(3).to_csv(OUT / "model_comparison.csv")
lines = ["| model | " + " | ".join(f"v RMSE {h:g}s" for h in T.columns) + " | " +
         " | ".join(f"dist RMSE {h:g}s" for h in Td.columns) + " | bias 30s | one-step a RMSE |"]
for mname in T.index:
    lines.append(f"| {mname} | " + " | ".join(f"{x:.3f}" for x in T.loc[mname]) + " | " +
                 " | ".join(f"{x:.2f}" for x in Td.loc[mname]) + f" | {Tb.loc[mname].iloc[-1]:+.3f} | " +
                 (f"{one.loc[mname]:.3f}" if pd.notna(one.loc[mname]) else "-") + " |")
print(chr(10).join(lines))  # table printed to stdout; model_comparison.csv holds the numbers

fig, axs = plt.subplots(1, 2, figsize=(18, 7))
cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
for ci, mname in enumerate(T.index):
    lw = 3 if mname == FINAL else 1.5
    ls = "--" if mname.startswith("baseline") else "-"
    col = "k" if mname == FINAL else cols[(2 * ci) % 20 if ci < 10 else (2 * ci + 1) % 20]
    axs[0].plot(T.columns, T.loc[mname], ls, marker="o", lw=lw, label=mname, color=col)
    axs[1].plot(Td.columns, Td.loc[mname], ls, marker="o", lw=lw, label=mname, color=col)
for ax, yl in zip(axs, ("speed RMSE [m/s]", "distance RMSE [m]")):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 3, 5, 10, 30])
    ax.set_xticklabels(["1", "3", "5", "10", "30"])
    ax.set_xlabel("open-loop horizon H [s] (start from true speed)")
    ax.set_ylabel(yl)
    ax.grid(True, which="both", alpha=0.4)
axs[0].legend(fontsize=8)
axs[0].set_title("validation (17 bags): open-loop speed error vs horizon")
axs[1].set_title("validation: distance error vs horizon")
plt.tight_layout()
plt.savefig(OUT / "fig_model_comparison.png", dpi=80)

# breakdown of the final model by start regime / vehicle group
pk = {"lut_oe+gain300s": "eval_gain_lut_oe.pkl", "lut_oe_s1+gain300s": "eval_gain_lut_oe_s1.pkl"}
fin_pkl = OUT / "cache" / pk.get(FINAL, f"eval_gain_{FINAL.split('+')[0]}.pkl")
if fin_pkl.exists():
    from simulate import summarize
    E = pd.read_pickle(fin_pkl)
    E = E[E.model == FINAL]
    E["regime0"] = np.where(E.u0 > 0, "traction", np.where(E.u0 < 0, "brake", "coast"))
    R1 = summarize(E, by=("regime0", "H")).pivot_table(index="regime0", columns="H", values="rmse")
    R2 = summarize(E, by=("group", "H")).pivot_table(index="group", columns="H", values="rmse")
    print("final model by start regime\n", R1.round(3).to_string(), "\nby vehicle group\n", R2.round(3).to_string())
    R1.round(4).to_csv(OUT / "final_by_regime.csv")
    R2.round(4).to_csv(OUT / "final_by_group.csv")
