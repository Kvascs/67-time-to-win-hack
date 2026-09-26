"""Exploration: what do the front/rear high-pass residuals and their cross-correlation look like?"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bc_common import BASE, HERE, KMH, load_bag, splits

bags = splits()["train"][: int(sys.argv[1]) if len(sys.argv) > 1 else 4]
DT = 0.1
W = 10.0          # window length, s
LAGS = np.arange(-25, 26)  # samples at 10 Hz


def gauss_smooth(x, sigma_samples):
    n = int(4 * sigma_samples)
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / sigma_samples) ** 2)
    k /= k.sum()
    xp = np.pad(x, n, mode="reflect")
    return np.convolve(xp, k, mode="valid")


acc = {s: [] for s in (5, 10, 20)}
acc_d = []
examples = []
for b in bags:
    d = load_bag(b)
    t, F, R, vg = d["t"], d["F"], d["R"], d["vg"]
    g = np.arange(t[0], t[-1], DT)
    j = np.searchsorted(t, g)
    j0 = np.clip(j - 1, 0, len(t) - 1)
    j1 = np.clip(j, 0, len(t) - 1)
    gap = (t[j1] - t[j0]) > 0.3
    Fg = np.interp(g, t, F) / KMH
    Rg = np.interp(g, t, R) / KMH
    Vg = np.interp(g, t, np.nan_to_num(vg))
    nw = int(W / DT)
    for i0 in range(0, len(g) - nw, nw // 2):
        sl = slice(i0, i0 + nw)
        if gap[sl].any():
            continue
        v = Vg[sl]
        if not (5 <= v.min() and v.max() <= 15 and v.max() - v.min() < 1.0):
            continue
        f, r = Fg[sl], Rg[sl]
        if np.max(np.abs(f - r)) > 0.05 * v.mean():
            continue
        lag_exp = BASE / v.mean()
        for s in acc:
            ef = f - gauss_smooth(f, s / 10 / DT * 1.0 if False else s / 10.0 / DT)
            er = r - gauss_smooth(r, s / 10.0 / DT)
            ef = (ef - ef.mean()) / (ef.std() + 1e-12)
            er = (er - er.mean()) / (er.std() + 1e-12)
            cc = np.array([np.mean(ef[max(0, -L):nw - max(0, L)] * er[max(0, L):nw - max(0, -L)]) for L in LAGS])
            acc[s].append((lag_exp, cc))
        lr = np.log(f) - np.log(r)
        lr = lr - gauss_smooth(lr, 20)
        lr = (lr - lr.mean()) / (lr.std() + 1e-12)
        ac = np.array([np.mean(lr[max(0, -L):nw - max(0, L)] * lr[max(0, L):nw - max(0, -L)]) for L in LAGS])
        acc_d.append((lag_exp, ac))
        if len(examples) < 6 and np.random.rand() < 0.05:
            examples.append((b, g[sl] - g[sl][0], f, r, v))

fig, ax = plt.subplots(3, 2, figsize=(13, 11))
for a, (s, lst) in zip(ax.flat[:3], acc.items()):
    cc = np.array([c for _, c in lst])
    le = np.array([l for l, _ in lst])
    a.plot(LAGS * DT, cc.mean(0), "k-", lw=2, label=f"mean of {len(cc)} windows")
    a.plot(LAGS * DT, np.median(cc, 0), "b--", label="median")
    a.axvspan(np.quantile(le, 0.1), np.quantile(le, 0.9), color="orange", alpha=0.3, label="expected lag p10-p90")
    a.set_title(f"CCF front(t) vs rear(t+lag), high-pass sigma={s/10:.1f}s")
    a.legend()
    a.grid()
cc = np.array([c for _, c in acc_d])
ax.flat[3].plot(LAGS * DT, cc.mean(0), "k-")
ax.flat[3].set_title(f"ACF of log(F/R) high-passed (n={len(cc)})")
ax.flat[3].grid()
for k, (b, tt, f, r, v) in enumerate(examples[:2]):
    a = ax.flat[4 + k]
    a.plot(tt, f, label="front")
    a.plot(tt, r, label="rear")
    a.plot(tt, v, label="gnss")
    a.set_title(b)
    a.legend()
fig.tight_layout()
fig.savefig(HERE / "explore_xcorr.png", dpi=90)
print("windows", len(acc_d))
