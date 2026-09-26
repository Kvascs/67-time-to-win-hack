"""Causal innovation statistics for gating (slip/slide) and 'notch-not-in-control' detection.

Innovation r(t) = a_w(t) - a_m(t) where
  a_w : causal least-squares slope of the mean wheel speed over [t-1s, t]
  a_m : model acceleration averaged over the same window with the identical (parabolic) weights
Both are realisable online. Outputs: innovation_stats.csv, detector_eval.json, fig_innovation.png
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_lut import load_lut_sim
from tid_data import OUT, list_bags
from tm_core import DT, get_arrays, lag1

W = int(round(1.0 / DT))  # 20 samples
tau = (np.arange(W) - (W - 1) / 2) * DT
k_slope = tau / np.sum(tau ** 2)  # slope weights on v (oldest..newest)
# equivalent weights on acceleration: slope = sum_j c_j a_j with c = cumulative of k reversed
k_acc = -np.cumsum(k_slope)[:-1] * DT  # weights on the W-1 increments
k_acc = k_acc / k_acc.sum()


def causal_filter(x, k):
    """y[t] = sum_i k[i] x[t-len(k)+1+i]  (k[0] weights the oldest sample); NaN for t < len(k)-1."""
    return np.r_[np.full(len(k) - 1, np.nan), np.convolve(x, k[::-1], mode="valid")]


sim = load_lut_sim(OUT / sys.argv[1]) if len(sys.argv) > 1 else load_lut_sim()
rows = []
for split in ("train", "val"):
    for b in list_bags(split):
        A = get_arrays(b)
        vw = A.vw
        a_w = causal_filter(vw, k_slope)
        am, _ = sim.accel_series(A)  # causal model (true speed); online: estimated speed
        a_m = causal_filter(am, k_acc)
        r = a_w - a_m
        vv = np.where(np.isfinite(A.v), A.v, A.vw)
        moving = (vv > 0.5) | (A.u > 0)
        rows.append(pd.DataFrame(dict(bag=b, split=split, r=r, clean=A.clean, auto=A.auto, moving=moving,
                                      u=A.u, v=vv, t=A.t - A.t[0])))
R = pd.concat(rows, ignore_index=True)
R = R[np.isfinite(R.r) & R.moving]
qs = [0.5, 0.9, 0.95, 0.99, 0.999]
stat = []
for name, m in (("clean_manual", R.clean & ~R.auto), ("auto_episodes", R.auto), ("not_clean(slip/slide/GNSS)", ~R.clean & ~R.auto)):
    x = R.r[m].abs()
    d = dict(subset=name, n_s=int(m.sum() * DT), rms=float(np.sqrt(np.mean(R.r[m] ** 2))), bias=float(R.r[m].mean()))
    d.update({f"|r|_q{q}": float(np.quantile(x, q)) for q in qs})
    stat.append(d)
S = pd.DataFrame(stat)
S.to_csv(OUT / "innovation_stats.csv", index=False)
print(S.round(3).to_string(), flush=True)

# detector: EMA of |r| (time constant T) above threshold for >= hold seconds
res = {}
for T, thr in ((1.0, 0.25), (1.0, 0.3), (2.0, 0.25), (2.0, 0.3), (3.0, 0.2)):
    al = 1 - np.exp(-DT / T)
    fa_time = 0.0
    tot_clean = 0.0
    det_delays = []
    missed = 0
    n_ep = 0
    for b, q in R.groupby("bag", sort=False):
        e = lag1(np.abs(q.r.to_numpy()), al, 0.0)
        alarm = e > thr
        cm = (q.clean & ~q.auto).to_numpy()
        fa_time += alarm[cm].sum() * DT
        tot_clean += cm.sum() * DT
        au = q.auto.to_numpy()
        # episodes
        st = np.flatnonzero(np.diff(np.r_[0, au.astype(int)]) == 1)
        en = np.flatnonzero(np.diff(np.r_[au.astype(int), 0]) == -1)
        for s0, e0 in zip(st, en):
            if (e0 - s0) * DT < 3:
                continue
            n_ep += 1
            hit = np.flatnonzero(alarm[s0:e0 + 1])
            if len(hit):
                det_delays.append(hit[0] * DT)
            else:
                missed += 1
    res[f"T{T}_thr{thr}"] = dict(false_alarm_frac_clean=fa_time / tot_clean, episodes=n_ep, missed=missed,
                                 det_delay_median=float(np.median(det_delays)) if det_delays else None,
                                 det_delay_p90=float(np.percentile(det_delays, 90)) if det_delays else None)
(OUT / "detector_eval.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1), flush=True)

fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(-2, 2, 161)
for name, m, c in (("clean manual", R.clean & ~R.auto, "tab:blue"), ("notch-not-in-control", R.auto, "tab:red"),
                   ("not clean (slip/slide/GNSS)", ~R.clean & ~R.auto, "tab:orange")):
    ax.hist(R.r[m], bins=bins, density=True, histtype="step", lw=2, color=c, label=name)
ax.set_yscale("log")
ax.set_xlabel("causal innovation a_wheel - a_model [m/s2] (1-s windows)")
ax.set_ylabel("density")
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_innovation.png", dpi=80)
