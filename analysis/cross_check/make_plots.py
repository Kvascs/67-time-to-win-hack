"""Figures for the cross-check: per-bag wheel scale vs time of day, and curve excess vs curvature."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("scale_lag_per_bag.csv")))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
groups = {}
for r in rows:
    groups.setdefault((r["vehicle"], r["date"]), []).append((int(r["hhmm"][:2]) + int(r["hhmm"][3:]) / 60, float(r["k_mean"])))
for (veh, date), pts in sorted(groups.items()):
    pts = sorted(pts)
    h, k = zip(*pts)
    ax[0].plot(h, k, "o-" if veh == "30639" else "s--", label=f"{veh} {date} (n={len(pts)})", ms=5)
ax[0].axhline(3.6, color="k", lw=0.8, ls=":")
ax[0].axhline(3.5956, color="gray", lw=0.8)
ax[0].set_xlabel("local time of run start [h]")
ax[0].set_ylabel("k = wheel km/h / GNSS m/s  (clean moving samples)")
ax[0].set_title("Wheel scale per run: constant within a run, varies by day (up to +/-1.6%)")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

A = np.load("curve_windows.npy")
A = A[np.all(np.isfinite(A[:, :5]), axis=1)]
bins = np.array([0, 0.001, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.045])
km, med, q1, q3 = [], [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (A[:, 0] >= lo) & (A[:, 0] < hi)
    km.append(np.median(A[m, 0])); med.append(np.median(A[m, 2]) * 100)
    q1.append(np.percentile(A[m, 2], 25) * 100); q3.append(np.percentile(A[m, 2], 75) * 100)
km = np.array(km)
ax[1].errorbar(km, med, yerr=[np.array(med) - q1, np.array(q3) - med], fmt="o", capsize=3, label="data: median, IQR (30 m windows)")
kk = np.linspace(0, 0.045, 200)
ax[1].plot(kk, 41 * kk, label="map_build: 0.41|k| (linear)")
ax[1].plot(kk, np.minimum(50 * kk, 0.95), label="wheel_anomalies: min(0.5|k|, 0.95%)")
ax[1].set_xlabel("mean |curvature| of window [1/m]")
ax[1].set_ylabel("map (antenna path) distance / wheel distance - 1  [%]")
ax[1].set_title("Wheels under-read in curves; effect saturates at ~0.9% for R < 60 m")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_scale_and_curve.png", dpi=110)
print("saved")
