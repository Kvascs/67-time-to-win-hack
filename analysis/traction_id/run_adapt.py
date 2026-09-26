"""Per-bag gains (mass/load), per-vehicle variants, online bias adaptation -> rollout evaluation.

Outputs: per_bag_gains.csv, fig_per_bag_gains.png, eval_adapt.csv, group_gains.json
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adapt import make_bias_fn, per_bag_gains, regime_channels
from final_lut import load_lut_sim
from simulate import BaselineSim, HammersteinSim, evaluate, summarize
from tid_data import OUT, list_bags
from tm_core import apply_matched, get_arrays

tr, va = list_bags("train"), list_bags("val")
sim = load_lut_sim()

# ---- 1. per-bag gains ---------------------------------------------------------------------------
G = pd.concat([per_bag_gains(sim, tr).assign(split="train"), per_bag_gains(sim, va).assign(split="val")])
G.to_csv(OUT / "per_bag_gains.csv", index=False)
print(G.groupby("group")[["g_trac", "g_brake", "g_coast", "g_grade", "bias", "rmse_fleet", "rmse_bag"]].agg(
    ["median", "std"]).round(3).T.to_string(), flush=True)
print("corr(g_trac, g_brake) per bag:", round(np.corrcoef(G.g_trac, G.g_brake)[0, 1], 3), flush=True)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
cols = {"30618": "tab:blue", "30639a": "tab:red", "30639b": "tab:green"}
for g, q in G.groupby("group"):
    axs[0].scatter(q.g_trac, q.g_brake, c=cols[g], label=g, s=25)
    axs[1].scatter(pd.to_datetime(q.t0, unit="s"), q.g_trac, c=cols[g], marker="^", s=25)
    axs[1].scatter(pd.to_datetime(q.t0, unit="s"), q.g_brake, c=cols[g], marker="v", s=25)
axs[0].axhline(1, color="k", lw=0.5)
axs[0].axvline(1, color="k", lw=0.5)
axs[0].set_xlabel("traction gain per bag")
axs[0].set_ylabel("brake gain per bag")
axs[0].legend()
axs[0].grid()
axs[0].set_title("per-bag gains of the fleet LUT (1 = fleet average)")
axs[1].set_ylabel("gain (^ traction, v brake)")
axs[1].grid()
axs[1].set_title("gains vs recording date")
plt.tight_layout()
plt.savefig(OUT / "fig_per_bag_gains.png", dpi=80)
plt.close()

# ---- 2. per-group gain scaling (fit on train bags of the group) -----------------------------------
grp = {}
for g, q in G[G.split == "train"].groupby("group"):
    grp[g] = dict(g_trac=float(np.average(q.g_trac, weights=q.n)), g_brake=float(np.average(q.g_brake, weights=q.n)),
                  g_coast=float(np.average(q.g_coast, weights=q.n)), n_bags=int(len(q)))
(OUT / "group_gains.json").write_text(json.dumps(grp, indent=1))
print("group gains", grp, flush=True)


def scaled_sim(base: HammersteinSim, gt, gb, gc, name):
    T = base.table.copy()
    U = np.arange(base.umin, base.umin + T.shape[0])
    T[U > 0] *= gt
    T[U < 0] *= gb
    T[U == 0] *= gc
    return HammersteinSim(T, base.tv0, base.tdv, base.kg, base.delay, base.tau_up, base.tau_dn, name, True, base.umin)


class GroupSim:
    """Dispatch to a per-vehicle-group scaled LUT."""

    def __init__(self, sims, name):
        self.sims = sims
        self.name = name

    def rollout(self, A, i0s, horizons, bias=None, dt=0.05):
        return self.sims[A.group].rollout(A, i0s, horizons, bias=bias)

    def accel_series(self, A, dt=0.05):
        return self.sims[A.group].accel_series(A)


gsim = GroupSim({g: scaled_sim(sim, v["g_trac"], v["g_brake"], v["g_coast"], "lut_group") for g, v in grp.items()},
                "lut_group")

# ---- 3. rollouts: baselines, fleet LUT, group LUT, online bias ------------------------------------
E = [evaluate(BaselineSim("hold_v"), va, stride=2.0), evaluate(BaselineSim("hold_a"), va, stride=2.0),
     evaluate(sim, va, stride=2.0, name="lut"), evaluate(gsim, va, stride=2.0, name="lut_group")]
for T_b in (5.0, 20.0, 60.0):
    E.append(evaluate(sim, va, stride=2.0, bias_fn=make_bias_fn(T_b), name=f"lut+bias{int(T_b)}s"))
    E.append(evaluate(gsim, va, stride=2.0, bias_fn=make_bias_fn(T_b), name=f"lut_group+bias{int(T_b)}s"))
E = pd.concat(E, ignore_index=True)
E.to_pickle(OUT / "cache" / "eval_adapt.pkl")
S = summarize(E)
S.to_csv(OUT / "eval_adapt.csv", index=False)
print(S.to_string(), flush=True)
# by regime at start
E["regime0"] = np.where(E.u0 > 0, "traction", np.where(E.u0 < 0, "brake", "coast"))
S2 = summarize(E, by=("model", "H", "regime0"))
S2.to_csv(OUT / "eval_adapt_by_regime.csv", index=False)
S3 = summarize(E, by=("model", "H", "group"))
S3.to_csv(OUT / "eval_adapt_by_group.csv", index=False)
print(S3[S3.H.isin([5.0, 10.0])].to_string(), flush=True)
