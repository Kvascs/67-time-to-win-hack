"""Identification figures and tables from settled (steady-state) samples of the train pool.

Outputs (in OUT):
  fig_heatmap_accel.png       median a (raw) and a + kg*gr (grade compensated) vs notch x speed, + counts
  fig_coast_davis.png         coasting deceleration vs speed, grade-compensated, Davis fit
  fig_brake_traction_curves.png  steady-state a vs v per notch (brake / traction), LUT overlay
  table_accel_raw.csv, table_accel_gradecomp.csv, table_counts_s.csv, davis_fit.json
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tid_data import OUT
from tid_pool import build_pool

KG = float(os.environ.get("KG", "8.3"))

P = build_pool("train")
m = P.clean & ~P.auto & P.settled & (P.vg > 0.3) & np.isfinite(P.ag) & np.isfinite(P.gr)
Q = P[m].copy()
edges = np.r_[0.3, np.arange(1, 15), 16.5]
Q["vb"] = pd.cut(Q.vg, edges)
Q["ac"] = Q.ag + KG * Q.gr
cnt = Q.pivot_table(index="notch", columns="vb", values="ag", aggfunc="size", observed=False)
raw = Q.pivot_table(index="notch", columns="vb", values="ag", aggfunc="median", observed=False)
comp = Q.pivot_table(index="notch", columns="vb", values="ac", aggfunc="median", observed=False)
MIN_N = 40  # >= 2 s of data per cell
raw.where(cnt >= MIN_N).round(3).to_csv(OUT / "table_accel_raw.csv")
comp.where(cnt >= MIN_N).round(3).to_csv(OUT / "table_accel_gradecomp.csv")
(cnt / 20).round(1).to_csv(OUT / "table_counts_s.csv")

vc = 0.5 * (edges[1:] + edges[:-1])
fig, axs = plt.subplots(1, 3, figsize=(22, 8))
for ax, T, ttl in ((axs[0], raw, "median a [m/s2] (raw)"), (axs[1], comp, f"median a + {KG}*grade [m/s2] (grade-compensated)")):
    Z = T.where(cnt >= MIN_N).to_numpy(float)
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1.1, vmax=1.1,
                   extent=[0, len(vc), T.index.min() - 0.5, T.index.max() + 0.5])
    ax.set_xticks(np.arange(len(vc)) + 0.5)
    ax.set_xticklabels([f"{x:.1f}" for x in vc], rotation=90)
    ax.set_xlabel("speed bin centre [m/s]")
    ax.set_ylabel("notch")
    ax.set_title(ttl + "\nsettled >= 2.5 s, clean, auto-episodes removed")
    for i, nn in enumerate(T.index):
        for j in range(len(vc)):
            if cnt.iloc[i, j] >= MIN_N and np.isfinite(Z[i, j]):
                ax.text(j + 0.5, nn, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax)
Cn = np.log10(cnt.to_numpy(float) / 20 + 1e-3)
im = axs[2].imshow(Cn, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=3,
                   extent=[0, len(vc), cnt.index.min() - 0.5, cnt.index.max() + 0.5])
axs[2].set_title("log10 settled seconds per cell")
axs[2].set_xticks(np.arange(len(vc)) + 0.5)
axs[2].set_xticklabels([f"{x:.1f}" for x in vc], rotation=90)
plt.colorbar(im, ax=axs[2])
plt.tight_layout()
plt.savefig(OUT / "fig_heatmap_accel.png", dpi=80)
plt.close()

# ---- coasting / Davis ------------------------------------------------------------------------
mc = P.clean & ~P.auto & (P.notch == 0) & (P.tsc >= 4) & (P.vg > 1.0) & np.isfinite(P.ag) & np.isfinite(P.gr)
C = P[mc]
y = -(C.ag + KG * C.gr).to_numpy()  # resistance deceleration (positive)
X = np.c_[np.ones(len(C)), C.vg.to_numpy(), C.vg.to_numpy() ** 2]
# robust (Huber-like IRLS)
w = np.ones(len(C))
for _ in range(10):
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    r = y - X @ coef
    s = 1.4826 * np.median(np.abs(r))
    w = np.sqrt(np.minimum(1, 1.5 * s / np.maximum(np.abs(r), 1e-9)))
# also free kg jointly
X2 = np.c_[np.ones(len(C)), C.vg.to_numpy(), C.vg.to_numpy() ** 2, C.gr.to_numpy()]
coef2, *_ = np.linalg.lstsq(X2 * w[:, None], -C.ag.to_numpy() * w, rcond=None)
davis = dict(A=coef[0], B=coef[1], C=coef[2], kg_used=KG, resid_std_robust=float(s), n=int(len(C)),
             joint_fit=dict(A=coef2[0], B=coef2[1], C=coef2[2], kg=coef2[3]))
(OUT / "davis_fit.json").write_text(json.dumps(davis, indent=1))
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].scatter(C.vg[::5], y[::5], s=1, alpha=0.25, c=C.gr[::5] * 100, cmap="coolwarm", vmin=-3, vmax=3)
vv = np.linspace(0, 16, 50)
axs[0].plot(vv, coef[0] + coef[1] * vv + coef[2] * vv ** 2, "k", lw=2,
            label=f"Davis: {coef[0]:.4f} + {coef[1]:.5f} v + {coef[2]:.6f} v^2")
bins = pd.cut(C.vg, np.arange(1, 16, 1))
med = pd.Series(y, index=C.index).groupby(bins, observed=True).median()
axs[0].plot([b.mid for b in med.index], med.values, "ro-", label="binned median")
axs[0].set_ylim(-0.2, 0.3)
axs[0].set_xlabel("v [m/s]")
axs[0].set_ylabel(f"-(a + {KG} grade) [m/s2]")
axs[0].set_title("coasting (notch 0 held >= 4 s): running resistance per unit mass")
axs[0].grid()
axs[0].legend()
axs[1].scatter(C.gr[::5] * 100, C.ag[::5], s=1, alpha=0.25, c=C.vg[::5])
gg = np.linspace(-4, 4, 10)
axs[1].plot(gg, -KG * gg / 100 - (coef[0] + coef[1] * 8 + coef[2] * 64), "k", label=f"-{KG} gr - R(8 m/s)")
axs[1].set_xlabel("grade felt [%]")
axs[1].set_ylabel("a [m/s2]")
axs[1].grid()
axs[1].legend()
axs[1].set_title("coasting acceleration vs grade")
plt.tight_layout()
plt.savefig(OUT / "fig_coast_davis.png", dpi=80)
plt.close()

# ---- brake / traction steady-state curves ---------------------------------------------------
lut = None
if (OUT / "lut_model.json").exists():
    lut = json.loads((OUT / "lut_model.json").read_text())
fig, axs = plt.subplots(1, 2, figsize=(18, 7))
cmap = plt.get_cmap("viridis")
for ax, notches, ttl in ((axs[0], range(-1, -9, -1), "brake notches"), (axs[1], [1, 2, 4, 7, 8, 9, 10, 11, 15], "traction notches")):
    for i, nn in enumerate(notches):
        col = cmap(i / max(1, len(notches) - 1))
        q = Q[Q.notch == nn]
        if len(q) > 100:
            b = pd.cut(q.vg, np.arange(0, 16.5, 1.0))
            g = q.groupby(b, observed=True).ac.agg(["median", "size"])
            g = g[g["size"] >= MIN_N]
            ax.plot([x.mid for x in g.index], g["median"], "o", color=col, ms=5)
        if lut is not None:
            vk = np.array(lut["v_knots"])
            ax.plot(vk, np.array(lut["table"])[nn - lut["u_min"]], "-", color=col, lw=1.5, label=f"notch {nn}")
    ax.set_xlabel("v [m/s]")
    ax.set_ylabel(f"a + {KG}*grade [m/s2]")
    ax.grid()
    ax.legend(fontsize=8)
    ax.set_title(ttl + ": dots = settled medians, lines = fitted LUT (dynamic fit)")
plt.tight_layout()
plt.savefig(OUT / "fig_brake_traction_curves.png", dpi=80)
plt.close()
print(json.dumps(davis, indent=1))

# ---- fitted LUT heatmap (support-masked) ------------------------------------------------------
if lut is not None:
    tab = np.array(lut["table"])
    vk = np.array(lut["v_knots"])
    sup = np.array(lut.get("support_seconds", np.ones_like(tab)))
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for ax, Z, ttl in ((axs[0], tab, "fitted LUT a_ss(notch, v) [m/s2] (flat track, incl. resistance)"),
                       (axs[1], np.where(sup >= 2.0, tab, np.nan), "same, cells with >= 2 s of support only")):
        im = ax.pcolormesh(np.r_[vk, vk[-1] + 1.5] - 0.25, np.arange(lut["u_min"], lut["u_min"] + tab.shape[0] + 1) - 0.5,
                           Z, cmap="RdBu_r", vmin=-1.6, vmax=1.6, shading="flat")
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                if np.isfinite(Z[i, j]):
                    ax.text(vk[j] + 0.5, lut["u_min"] + i, f"{Z[i, j]:.2f}", fontsize=5.5, ha="center", va="center")
        ax.set_xlabel("v knot [m/s]")
        ax.set_ylabel("notch")
        ax.set_title(ttl)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT / "fig_lut_table.png", dpi=80)
    plt.close()
