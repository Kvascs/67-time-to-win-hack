"""Where does the LUT model err? One-step matched residuals on val, conditioned on regime / transition /
time since notch change / speed / vehicle group. Output: residual_breakdown.txt, fig_residual_breakdown.png"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_lut import load_lut_sim
from tid_data import OUT, list_bags
from tid_pool import time_since_change
from tm_core import apply_matched, get_arrays

sim = load_lut_sim()
rows = []
for b in list_bags("val"):
    A = get_arrays(b)
    am, y = sim.accel_series(A)
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    m = A.fit & ((vv > 0.3) | (A.u > 0))
    tsc, prev, _ = time_since_change(A.t, A.u)
    r = A.a - apply_matched(am)
    rows.append(pd.DataFrame(dict(bag=b, group=A.group, r=r[m], u=A.u[m], prev=prev[m], tsc=tsc[m], v=vv[m],
                                  gr=A.gr[m], a=A.a[m])))
R = pd.concat(rows, ignore_index=True)
reg = lambda u: np.where(u > 0, "T", np.where(u < 0, "B", "0"))
R["reg"] = reg(R.u)
R["trans"] = reg(R.prev) + "->" + R.reg
R["tscb"] = pd.cut(R.tsc, [0, 0.25, 0.5, 1, 2, 4, 1e9], right=False)
R["vb"] = pd.cut(R.v, [0, 1, 2, 4, 6, 8, 10, 12, 16])
out = []
agg = dict(n=("r", "size"), rmse=("r", lambda x: np.sqrt(np.mean(x ** 2))), bias=("r", "mean"))
out.append("overall rmse %.4f bias %.4f n %d" % (np.sqrt(np.mean(R.r ** 2)), R.r.mean(), len(R)))
for key in ("reg", "group", "vb", "tscb"):
    out.append(f"\n--- by {key}\n" + R.groupby(key, observed=True).agg(**agg).round(4).to_string())
out.append("\n--- by transition x time-since-change\n" +
           R.groupby(["trans", "tscb"], observed=True).agg(**agg).round(4).unstack("tscb").to_string())
out.append("\n--- by notch\n" + R.groupby("u").agg(**agg).round(4).T.to_string())
# share of squared error
R["se"] = R.r ** 2
sh = R.groupby("reg").se.sum() / R.se.sum()
out.append("\nshare of squared error by regime: " + sh.round(3).to_string())
txt = "\n".join(out)
(OUT / "residual_breakdown.txt").write_text(txt)
print(txt)

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
g = R.groupby(["trans", "tscb"], observed=True).r.mean().unstack("trans")
g.index = [str(i) for i in g.index]
g.plot(ax=axs[0], marker="o")
axs[0].set_title("mean residual vs time since notch change, by transition")
axs[0].axhline(0, color="k", lw=0.5)
axs[0].grid()
g = R.groupby(["reg", "vb"], observed=True).r.mean().unstack("reg")
g.index = [str(i) for i in g.index]
g.plot(ax=axs[1], marker="o")
axs[1].set_title("mean residual vs speed by regime")
axs[1].axhline(0, color="k", lw=0.5)
axs[1].grid()
g = R.groupby("u").r.agg(["mean", "std"])
axs[2].errorbar(g.index, g["mean"], yerr=g["std"], fmt="o")
axs[2].axhline(0, color="k", lw=0.5)
axs[2].set_title("residual mean +- std by notch")
axs[2].grid()
plt.tight_layout()
plt.savefig(OUT / "fig_residual_breakdown.png", dpi=80)
