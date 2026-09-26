"""Exploration: event-triggered averages around isolated spikes of one bogie (does the other bogie
show the same spike 7.55/v seconds later / earlier?)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from bc_common import BASE, HERE, KMH, load_bag, splits

bags = splits()["train"][: int(sys.argv[1]) if len(sys.argv) > 1 else 10]
DT = 0.1
REL = np.linspace(-3, 3, 121)  # time relative to event in units of expected lag
out = {"F->R": [], "R->F": [], "F->F": [], "R->R": []}
nev = {"F": 0, "R": 0}
for b in bags:
    d = load_bag(b)
    t, F, R, vg = d["t"], d["F"] / KMH, d["R"] / KMH, d["vg"]
    ok = np.isfinite(vg)
    # residual vs a 7-sample running median (isolated spikes stand out); normalised by local MAD
    eF = F - np.r_[F[0], 0.5 * (F[:-2] + F[2:]), F[-1]]
    eR = R - np.r_[R[0], 0.5 * (R[:-2] + R[2:]), R[-1]]
    moving = ok & (vg > 4) & (vg < 16)
    sF = 1.4826 * np.median(np.abs(eF[moving]))
    sR = 1.4826 * np.median(np.abs(eR[moving]))
    dtn = np.r_[np.inf, np.diff(t)]
    dtp = np.r_[np.diff(t), np.inf]
    for src, e, s, oth, eo in (("F", eF, sF, "R", eR), ("R", eR, sR, "F", eF)):
        cand = np.flatnonzero(moving & (np.abs(e) > 5 * s) & (dtn < 0.2) & (dtp < 0.2))
        for i in cand:
            lag = BASE / vg[i]
            tq = t[i] + REL * lag
            if tq[0] < t[0] or tq[-1] > t[-1]:
                continue
            j = np.searchsorted(t, tq)
            if np.any(np.diff(t[np.clip(j.min() - 1, 0, None):j.max() + 1]) > 0.3):
                continue
            sign = np.sign(e[i])
            out[f"{src}->{oth}"].append(sign * np.interp(tq, t, eo) / (s if oth == src else (sR if oth == "R" else sF)))
            out[f"{src}->{src}"].append(sign * np.interp(tq, t, e) / s)
            nev[src] += 1
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, key in zip(ax, ("F->R", "R->F")):
    arr = np.array(out[key])
    self_key = key[0] + "->" + key[0]
    a.plot(REL, np.mean(out[self_key], 0), "k-", lw=1, label=f"{key[0]} itself (events {len(arr)})")
    a.plot(REL, arr.mean(0), "r-", lw=2, label=f"other bogie ({key[3]})")
    a.axvline(1 if key == "F->R" else -1, color="orange", ls="--", label="expected lag")
    a.set_xlabel("time from spike / (7.55 m / v)")
    a.set_ylabel("residual / MAD (sign-aligned)")
    a.set_ylim(-1, 3)
    a.legend()
    a.grid()
fig.tight_layout()
fig.savefig(HERE / "explore_spikes.png", dpi=90)
print(nev)
for key in ("F->R", "R->F"):
    arr = np.array(out[key])
    i1 = np.argmin(np.abs(REL - (1 if key == "F->R" else -1)))
    i0 = np.argmin(np.abs(REL))
    print(key, "n", len(arr), "other@0", arr[:, i0].mean().round(3), "other@expected", arr[:, i1].mean().round(3),
          "+-", (arr[:, i1].std() / np.sqrt(len(arr))).round(3))
