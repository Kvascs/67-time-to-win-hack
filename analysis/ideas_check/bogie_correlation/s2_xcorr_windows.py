"""Step 2. Windowed front/rear cross-correlation (the teammate's recipe) on train bags.

Uniform 10 Hz grid (linear interpolation of the shared header stamps; grid points bridging a gap > 0.3 s are
invalid). Non-overlapping windows of W = 5 / 10 / 15 s with
  GNSS speed 5..15 m/s, (vmax - vmin) / vmean < 0.10, no gap, |F - R| < 5 % of speed,
  wheel/GNSS ratio within 5 % of the bag's k (no slip/slide).
Residual variants (both bogies treated identically):
  hp05  : x - Gaussian(x, sigma 0.5 s)            teammate's high-pass
  hp10  : x - Gaussian(x, sigma 1.0 s)
  d2    : x_i - (x_{i-1} + x_{i+1}) / 2              whitening (second difference)
(GNSS-subtracted variants were tried and dropped: the common GNSS noise dominates their lag-0 CCF.)
CCF(L) = Pearson correlation of e_F[i] and e_R[i + L] over the overlap (positive L: rear lags front).
Per window:
  lag_free  : argmax over 0.3..2.0 s, parabolic sub-sample refinement (NaN if the max is on the boundary)
  lag_prior : argmax within +-20 % of the GNSS-free expectation 7.55 / v_wheel (wheel speed at nominal scale)
Outputs: s2_windows.csv (one row per window x variant), s2_stack.npz, s2_summary.txt, s2_stack.png
Option --plot: redraw s2_stack.png from s2_stack.npz without recomputing.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bc_common import BASE, HERE, KMH, load_bag, splits, true_k

DT = 0.1
WINS = (5.0, 10.0, 15.0)
VARIANTS = ("hp05", "hp10", "d2")
MAXLAG = 45                      # samples
U = np.arange(-2.5, 2.5001, 0.02)  # normalised lag grid (tau / expected lag)


def gauss(x, sig):
    n = int(np.ceil(4 * sig))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / sig) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, n, mode="reflect"), k, mode="valid")


def d2(x):
    e = np.empty_like(x)
    e[1:-1] = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    e[0] = e[-1] = 0.0
    return e


def residual(var, f, r):
    if var == "hp05":
        return f - gauss(f, 5), r - gauss(r, 5)
    if var == "hp10":
        return f - gauss(f, 10), r - gauss(r, 10)
    if var == "d2":
        return d2(f), d2(r)
    raise ValueError(var)


def ccf(ef, er, maxlag):
    """Pearson correlation of ef[i] and er[i+L] over the overlap, L = -maxlag..maxlag (vectorised)."""
    n = len(ef)
    full = np.correlate(er, ef, mode="full")          # index L + n - 1
    cf, cr = np.r_[0.0, np.cumsum(ef)], np.r_[0.0, np.cumsum(er)]
    qf, qr = np.r_[0.0, np.cumsum(ef * ef)], np.r_[0.0, np.cumsum(er * er)]
    out = np.full(2 * maxlag + 1, np.nan)
    for k, L in enumerate(range(-maxlag, maxlag + 1)):
        m = n - abs(L)
        if m < 20:
            continue
        if L >= 0:
            sa, qa = cf[n - L], qf[n - L]
            sb, qb = cr[n] - cr[L], qr[n] - qr[L]
        else:
            sa, qa = cf[n] - cf[-L], qf[n] - qf[-L]
            sb, qb = cr[n + L], qr[n + L]
        sab = full[L + n - 1]
        va, vb = qa - sa * sa / m, qb - sb * sb / m
        if va > 0 and vb > 0:
            out[k] = (sab - sa * sb / m) / np.sqrt(va * vb)
    return out


def peak(c, lags, lo, hi, strict=True):
    """argmax of c over lags in [lo, hi] (s) with parabolic sub-sample refinement.
    strict: NaN if the max sits on the range boundary (no interior peak).
    not strict: the range only constrains the discrete max (nearest grid lags included); neighbours outside
    the range are still used for the refinement."""
    if strict:
        m = (lags >= lo - 1e-9) & (lags <= hi + 1e-9) & np.isfinite(c)
    else:
        m = (lags >= lo - DT / 2) & (lags <= hi + DT / 2) & np.isfinite(c)
    idx = np.flatnonzero(m)
    if len(idx) < (3 if strict else 1):
        return np.nan, np.nan
    i = idx[np.argmax(c[idx])]
    if strict and (i == idx[0] or i == idx[-1]):
        return np.nan, c[i]
    if i == 0 or i == len(c) - 1 or not np.all(np.isfinite(c[i - 1:i + 2])):
        return lags[i], c[i]
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    return lags[i] + off * DT, y1


def per_bag(bag):
    d = load_bag(bag)
    t = d["t"]
    kt = true_k(d)
    if not np.isfinite(kt["k_front"]):
        return [], {}
    kb = (kt["k_front"], kt["k_rear"])
    g = np.arange(t[0], t[-1], DT)
    j = np.searchsorted(t, g)
    j0 = np.clip(j - 1, 0, len(t) - 1)
    j1 = np.clip(j, 0, len(t) - 1)
    bad = (t[j1] - t[j0]) > 0.3
    Fg = np.interp(g, t, d["F"]) / KMH
    Rg = np.interp(g, t, d["R"]) / KMH
    vg = d["vg"]
    okg = np.isfinite(vg)
    Vg = np.interp(g, t[okg], vg[okg])
    bad |= ~np.isfinite(np.interp(g, t, np.where(okg, 0.0, np.nan)))
    rows, stacks = [], {}
    lags = np.arange(-MAXLAG, MAXLAG + 1) * DT
    for W in WINS:
        n = int(round(W / DT))
        for i0 in range(0, len(g) - n, n):
            sl = slice(i0, i0 + n)
            if bad[sl].any():
                continue
            v = Vg[sl]
            vm = v.mean()
            if vm < 5 or vm > 15 or v.min() < 4.5 or v.max() > 15.5 or (v.max() - v.min()) / vm > 0.10:
                continue
            f, r = Fg[sl], Rg[sl]
            if np.max(np.abs(f - r)) > 0.05 * vm:
                continue
            if np.max(np.abs(f / (1 + kb[0]) / v - 1)) > 0.05 or np.max(np.abs(r / (1 + kb[1]) / v - 1)) > 0.05:
                continue
            lag_exp = BASE / vm
            vw = 0.5 * (f.mean() + r.mean())      # wheel speed, nominal scale (m/s)
            lag_w = BASE / vw                     # GNSS-free expectation
            ml = min(MAXLAG, n - 20)
            for var in VARIANTS:
                ef, er = residual(var, f, r)
                c = ccf(ef, er, MAXLAG) if ml == MAXLAG else np.r_[np.full(MAXLAG - ml, np.nan), ccf(ef, er, ml),
                                                                    np.full(MAXLAG - ml, np.nan)]
                lf, pf = peak(c, lags, 0.3, 2.0)
                lp, pp = peak(c, lags, lag_w * 0.8, lag_w * 1.2, strict=False)
                off = (np.abs(lags) > 0.25) & np.isfinite(c)
                z = pf / np.std(c[off]) if np.isfinite(pf) else np.nan
                rows.append({"bag": bag, "W": W, "var": var, "t0": g[i0], "v_gnss": vm, "v_wheel": vw,
                             "k_front": kb[0], "k_rear": kb[1], "k_mean": kt["k_mean"],
                             "lag_exp": lag_exp, "lag_w": lag_w, "lag_free": lf, "pk_free": pf, "z_free": z,
                             "lag_prior": lp, "pk_prior": pp, "c0": c[MAXLAG],
                             "c_at_exp": np.interp(lag_exp, lags, np.nan_to_num(c))})
                cu = np.interp(U * lag_exp, lags, c, left=np.nan, right=np.nan)
                key = (W, var)
                if key not in stacks:
                    stacks[key] = {"u": [], "lag": [], "v": []}
                stacks[key]["u"].append(cu)
                stacks[key]["lag"].append(c)
                stacks[key]["v"].append(vm)
    return rows, stacks


def main():
    if "--plot" in sys.argv:
        z = np.load(HERE / "s2_stack.npz")
        make_plot({k: z[k] for k in z.files})
        return
    bags = splits()["train"]
    if len(sys.argv) > 1:
        bags = bags[: int(sys.argv[1])]
    rows, stacks = [], {}
    for b in bags:
        r, s = per_bag(b)
        rows += r
        for k, v in s.items():
            if k not in stacks:
                stacks[k] = {"u": [], "lag": [], "v": []}
            for kk in v:
                stacks[k][kk] += v[kk]
        print(b, len(r), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "s2_windows.csv", index=False)
    lags = np.arange(-MAXLAG, MAXLAG + 1) * DT
    save = {"U": U, "lags": lags}
    for (W, var), s in stacks.items():
        save[f"u_{int(W)}_{var}"] = np.array(s["u"])
        save[f"lag_{int(W)}_{var}"] = np.array(s["lag"])
        save[f"v_{int(W)}_{var}"] = np.array(s["v"])
    np.savez_compressed(HERE / "s2_stack.npz", **save)

    lines = [f"train bags {df.bag.nunique()}; windows per length: "
             + ", ".join(f"{int(W)} s: {int((df.W == W).sum() / len(VARIANTS))}" for W in WINS)]
    lines.append("Per-window lag estimates vs expected 7.55/v_gnss.  rel.err = lag_hat/lag_exp - 1")
    lines.append("  interior = share of windows with an interior CCF max in 0.3..2.0 s;"
                 " hit5 = share with |rel.err| < 5 % (chance level in brackets);"
                 " robust sd = 1.4826*MAD of rel.err; prior = search restricted to +-20 % of 7.55/v_wheel")
    for W in WINS:
        for var in VARIANTS:
            s = df[(df.W == W) & (df["var"] == var)]
            e = (s.lag_free / s.lag_exp - 1).dropna()
            chance = np.mean(0.1 * s.lag_exp / (2.0 - 0.3))
            ep = (s.lag_prior / s.lag_exp - 1).dropna()
            ek = (s.v_wheel / (BASE / s.lag_prior) - 1 - s.k_mean).dropna()
            lines.append(
                f"  W={int(W):2d}s {var:5s} n={len(s):5d} interior {len(e) / len(s):5.2f} | free: median {e.median():+.3f}"
                f" robust sd {1.4826 * np.median(np.abs(e - e.median())):.3f} hit5 {np.mean(np.abs(e) < 0.05):.3f}"
                f" ({chance:.3f}) corr(lag_hat,lag_exp) {np.corrcoef(s.lag_free[e.index], s.lag_exp[e.index])[0, 1]:+.2f}"
                f" | prior: interior {len(ep) / len(s):.2f} median {ep.median():+.4f}"
                f" robust sd {1.4826 * np.median(np.abs(ep - ep.median())):.4f}"
                f" k err median {ek.median():+.4f} robust sd {1.4826 * np.median(np.abs(ek - ek.median())):.4f}"
                f" | mean CCF at lag0 {s.c0.mean():+.3f}, at expected lag {s.c_at_exp.mean():+.4f}"
                f" (+-{s.c_at_exp.std() / np.sqrt(len(s)):.4f})")
    # stacked normalised-lag CCF: value at u=1 vs background
    lines.append("Stacked CCF vs normalised lag u = tau / (7.55/v_gnss): mean at u=1 and local contrast "
                 "(u=1 minus mean of u in [0.6,0.8]U[1.2,1.4]), with standard error")
    for W in WINS:
        for var in VARIANTS:
            A = save[f"u_{int(W)}_{var}"]
            m = np.nanmean(A, 0)
            se = np.nanstd(A, 0) / np.sqrt(np.sum(np.isfinite(A), 0))
            i1 = np.argmin(np.abs(U - 1))
            bg = (np.abs(U - 0.7) <= 0.1) | (np.abs(U - 1.3) <= 0.1)
            con = A[:, i1] - np.nanmean(A[:, bg], 1)
            con = con[np.isfinite(con)]
            lines.append(f"  W={int(W):2d}s {var:5s} CCF(u=1) {m[i1]:+.4f}+-{se[i1]:.4f}  contrast {con.mean():+.4f}"
                         f"+-{con.std() / np.sqrt(len(con)):.4f} (t={con.mean() / (con.std() / np.sqrt(len(con))):.1f})"
                         f"  CCF(u=0) {m[np.argmin(np.abs(U))]:+.3f}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "s2_summary.txt").write_text(txt, encoding="utf-8")

    make_plot(save)


def make_plot(save):
    lags = save["lags"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for a, var in zip(ax.flat, VARIANTS):
        for W, col in zip(WINS, ("tab:blue", "k", "tab:red")):
            A = save[f"u_{int(W)}_{var}"]
            m = np.nanmean(A, 0)
            se = np.nanstd(A, 0) / np.sqrt(np.sum(np.isfinite(A), 0))
            a.plot(U, m, color=col, lw=1.2, label=f"W={int(W)} s (n={len(A)})")
            a.fill_between(U, m - 2 * se, m + 2 * se, color=col, alpha=0.2)
        a.axvline(1, color="orange", ls="--")
        a.axvline(-1, color="orange", ls=":")
        a.set_title(f"{var}: mean CCF front(t)*rear(t+tau)")
        a.set_xlabel("tau / (7.55 m / v_gnss)")
        a.grid(alpha=0.4)
        a.legend(fontsize=8)
        if var == "d2":
            lo = np.nanmin(np.nanmean(save[f"u_10_{var}"], 0)[np.abs(U) > 0.3])
            hi = np.nanmax(np.nanmean(save[f"u_10_{var}"], 0)[np.abs(U) > 0.3])
            a.set_ylim(lo - 0.02, hi + 0.03)
    a = ax.flat[3]
    A = save["lag_10_d2"]
    V = save["v_10_d2"]
    for lo, hi, col in ((5, 7, "tab:blue"), (7, 9, "tab:green"), (9, 11, "k"), (11, 13, "tab:orange"),
                        (13, 15, "tab:red")):
        m = (V >= lo) & (V < hi)
        if m.sum() < 10:
            continue
        a.plot(lags, np.nanmean(A[m], 0), color=col, label=f"{lo}-{hi} m/s (n={m.sum()})")
        a.axvline(BASE / (0.5 * (lo + hi)), color=col, ls="--", lw=0.8)
    a.set_xlim(0.2, 2.2)
    a.set_ylim(-0.15, 0.35)
    a.set_title("d2, W=10 s: mean CCF vs absolute lag by speed (dashed: 7.55/v)")
    a.set_xlabel("lag, s")
    a.grid(alpha=0.4)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "s2_stack.png", dpi=90)


if __name__ == "__main__":
    main()
