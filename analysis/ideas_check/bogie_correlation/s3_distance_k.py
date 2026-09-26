"""Step 3. Accuracy of the bogie-lag speed/scale observation, estimated the way it could run on the tram.

For every train and val bag:
  * whitened residuals of each bogie on its native stamps: e_i = x_i - (linear interpolation of x_{i-1}, x_{i+1} at t_i)
    (x in m/s at nominal scale; invalid if t_{i+1} - t_{i-1} > 0.3 s)
  * wheel distance s_w = integral of the mean wheel speed (nominal scale, i.e. (1 + k) * true distance)
  * distance-domain cross-correlation  C(lam) = sum_i e_F(s_i) * e_R(s_i + lam)   (rear interpolated in s)
    over the front samples with 3 <= v <= 20 m/s and |F - R| < 5 % (no slip)
    - wide range lam = -2..20 m (structure: common-mode peak at 0, track-feature peak at the bogie base)
    - narrow range lam = 6.5..8.6 m, step 0.01 m, kept per sample so that it can be summed over chunks
  * peak lam_w: max of the summed C within +-3 % of 7.55 m (k prior), least-squares parabola over +-0.2 m
  * GNSS truth: lam_w should equal (1 + k_mean) * L_eff  ->  L_bag = lam_w / (1 + k_true)
Scale estimate: k_hat = lam_w / L_eff - 1 with L_eff = median(L_bag) over TRAIN bags (applied to val as is).
Accumulation: chunks of T seconds of valid driving within a bag -> k_hat per chunk vs k_true of the bag.
Outputs: s3_per_bag.csv, s3_chunks.csv, s3_summary.txt, s3_ccf.png, s3_accumulation.png, s3_ccf_wide.npz
Option --snap: stamps replaced by the fitted regular 10 Hz ticks (bc_common.snap_times); outputs s3snap_*
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bc_common import BASE, HERE, KMH, load_bag, snap_times, splits, true_k

NARROW = np.round(np.arange(6.5, 8.6001, 0.01), 4)
WIDE = np.round(np.arange(-2.0, 20.0001, 0.02), 4)
TCHUNK = (30, 60, 120, 300, 600, 1200)
PRIOR = 0.03
VMIN, VMAX = 3.0, 20.0
SNAP = "--snap" in sys.argv          # variant: stamps replaced by the fitted regular 10 Hz ticks (no ~13 ms jitter)
TAG = "s3snap" if SNAP else "s3"


def whiten(t, x):
    e = np.full(len(x), np.nan)
    span = t[2:] - t[:-2]
    w = (t[1:-1] - t[:-2]) / span
    e[1:-1] = x[1:-1] - ((1 - w) * x[:-2] + w * x[2:])
    e[1:-1][span > 0.3] = np.nan
    return e


def peak_fit(lam, c, lo, hi, half=0.2):
    m = (lam >= lo) & (lam <= hi) & np.isfinite(c)
    if m.sum() < 5:
        return np.nan, False
    idx = np.flatnonzero(m)
    i = idx[np.argmax(c[idx])]
    edge = i in (idx[0], idx[-1])
    sel = np.abs(lam - lam[i]) <= half
    sel &= np.isfinite(c)
    p = np.polyfit(lam[sel] - lam[i], c[sel], 2)
    if p[0] >= 0:
        return lam[i], True
    x = -p[1] / (2 * p[0])
    if abs(x) > half:
        return lam[i], True
    return lam[i] + x, edge


def prepare(bag):
    d = load_bag(bag)
    kt = true_k(d)
    t = d["t"]
    if SNAP:
        ts = snap_times(t)
        keep = ts > np.maximum.accumulate(np.r_[-np.inf, ts[:-1]])   # a few samples per bag land on a taken tick
        for key in ("F", "R", "vg", "ag", "kap"):
            d[key] = d[key][keep]
        t = ts[keep]
    f, r = d["F"] / KMH, d["R"] / KMH
    v = 0.5 * (f + r)
    eF, eR = whiten(t, f), whiten(t, r)
    dt = np.r_[np.diff(t), 0.0]
    s = np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))]
    kap = d["kap"]
    # straightness at the sample: |curvature| < 1/500 over +-2 s (GNSS, diagnostic only)
    ak = np.where(np.isfinite(kap), np.abs(kap), 1.0)
    j0 = np.searchsorted(t, t - 2.0)
    j1 = np.searchsorted(t, t + 2.0)
    mx = np.array([ak[a:b].max() if b > a else 1.0 for a, b in zip(j0, j1)])
    straight = mx < 1 / 500.0
    moving_r = (v > 0.5) & np.isfinite(eR)
    # rear series in distance: moving, valid, strictly increasing s
    sr, er = s[moving_r], eR[moving_r]
    keep = np.r_[True, np.diff(sr) > 1e-6]
    sr, er = sr[keep], er[keep]
    tr = t[moving_r][keep]
    front = (v >= VMIN) & (v <= VMAX) & np.isfinite(eF) & (np.abs(f - r) < 0.05 * v) & (dt < 0.3)
    return dict(d=d, kt=kt, t=t, s=s, v=v, eF=eF, sr=sr, er=er, tr=tr, front=front, dt=dt,
                straight=straight, vg=d["vg"])


def contributions(P, lam_grid, sel):
    """per-front-sample products e_F(s_i) * e_R(s_i + lam) for lam in lam_grid (NaN-safe -> 0)."""
    sF = P["s"][sel]
    eF = P["eF"][sel]
    tF = P["t"][sel]
    sr, er, tr = P["sr"], P["er"], P["tr"]
    out = np.zeros((len(sF), len(lam_grid)))
    for k, lam in enumerate(lam_grid):
        q = sF + lam
        j = np.searchsorted(sr, q)
        ok = (j > 0) & (j < len(sr))
        jj = np.clip(j, 1, len(sr) - 1)
        # bracketing rear samples must be close in time (no gap); rear passage within 30 s of the front one
        ok &= (tr[jj] - tr[jj - 1]) < 0.3
        ok &= (tr[jj] - tF) < 30.0
        val = np.interp(q, sr, er)
        out[:, k] = np.where(ok, eF * val, 0.0)
    return out


def main():
    sp = splits()
    bags = [(b, "train") for b in sp["train"]] + [(b, "val") for b in sp["val"]]
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    if nums:
        bags = bags[: int(nums[0])]
    rows, chunks, wide = [], [], {}
    Cn_store = {}
    for b, split in bags:
        P = prepare(b)
        kt = P["kt"]
        if not np.isfinite(kt["k_mean"]):
            print(b, "no k_true")
            continue
        sel = P["front"]
        Cn = contributions(P, NARROW, sel)
        Cw = contributions(P, WIDE, sel).sum(0)
        norm = np.sqrt(np.nansum(P["eF"][sel] ** 2) * np.nansum(P["er"] ** 2) * sel.sum() / max(len(P["er"]), 1))
        wide[b] = Cw / norm
        C = Cn.sum(0)
        lam_w, edge = peak_fit(NARROW, C, BASE * (1 - PRIOR), BASE * (1 + PRIOR))
        # straight-only and speed-bin variants
        sub = {}
        for name, m in (("straight", P["straight"][sel]), ("v3_6", (P["v"][sel] < 6)),
                        ("v6_9", (P["v"][sel] >= 6) & (P["v"][sel] < 9)),
                        ("v9_12", (P["v"][sel] >= 9) & (P["v"][sel] < 12)), ("v12_20", P["v"][sel] >= 12)):
            sub[name] = peak_fit(NARROW, Cn[m].sum(0), BASE * (1 - PRIOR), BASE * (1 + PRIOR))[0] if m.sum() > 50 \
                else np.nan
            sub["n_" + name] = int(m.sum())
        T_valid = float(np.sum(P["dt"][sel]))
        rows.append({"bag": b, "split": split, **kt, "n_front": int(sel.sum()), "T_valid_s": T_valid,
                     "lam_w": lam_w, "edge": edge, "L_bag": lam_w / (1 + kt["k_mean"]),
                     "peak_rel": C.max() / np.median(np.abs(C)) if np.any(C) else np.nan,
                     **{f"lam_{k}": v for k, v in sub.items() if not k.startswith("n_")},
                     **{k: v for k, v in sub.items() if k.startswith("n_")}})
        Cn_store[b] = (Cn, P["dt"][sel], P["v"][sel])
        print(b, split, f"k_true {kt['k_mean']:+.4f} lam_w {lam_w:.3f} L_bag {lam_w / (1 + kt['k_mean']):.3f} "
              f"T {T_valid:.0f}s edge {edge}", flush=True)
    df = pd.DataFrame(rows)
    tr = df[df.split == "train"]
    L_eff = float(np.median(tr.L_bag))
    df["k_hat"] = df.lam_w / L_eff - 1
    df["k_err"] = df.k_hat - df.k_mean
    for name in ("straight", "v3_6", "v6_9", "v9_12", "v12_20"):
        Ls = np.median((tr[f"lam_{name}"] / (1 + tr.k_mean)).dropna())
        df[f"kerr_{name}"] = df[f"lam_{name}"] / Ls - 1 - df.k_mean
        df[f"Leff_{name}"] = Ls
    df.to_csv(HERE / f"{TAG}_per_bag.csv", index=False)

    # accumulation vs driving time (chunks of valid driving time inside each bag)
    for _, row in df.iterrows():
        Cn, dt, v = Cn_store[row.bag]
        ct = np.cumsum(dt)
        for T in TCHUNK:
            nchunk = int(ct[-1] // T)
            for c in range(nchunk):
                m = (ct > c * T) & (ct <= (c + 1) * T)
                lam, edge = peak_fit(NARROW, Cn[m].sum(0), BASE * (1 - PRIOR), BASE * (1 + PRIOR))
                chunks.append({"bag": row.bag, "split": row.split, "T": T, "chunk": c, "v_mean": v[m].mean(),
                               "lam_w": lam, "edge": edge, "k_true": row.k_mean,
                               "k_err": lam / L_eff - 1 - row.k_mean})
    ch = pd.DataFrame(chunks)
    ch.to_csv(HERE / f"{TAG}_chunks.csv", index=False)
    np.savez_compressed(HERE / f"{TAG}_ccf_wide.npz", lam=WIDE, bags=np.array(list(wide)),
                        C=np.array([wide[b] for b in wide]))

    rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))  # noqa: E731
    L = [f"L_eff (median over {len(tr)} train bags of lam_w / (1 + k_true)) = {L_eff:.4f} m; "
         f"spread of L_bag: robust sd {rs(tr.L_bag):.4f} m, p10..p90 {tr.L_bag.quantile(.1):.3f}..{tr.L_bag.quantile(.9):.3f}"]
    for split in ("train", "val"):
        s = df[df.split == split]
        if len(s) == 0:
            continue
        L.append(f"{split}: bags {len(s)}, valid driving per bag median {s.T_valid_s.median():.0f} s; "
                 f"per-bag k_hat - k_true: median {s.k_err.median():+.4f}, robust sd {rs(s.k_err):.4f}, "
                 f"rms {np.sqrt(np.mean(s.k_err ** 2)):.4f}, max |.| {s.k_err.abs().max():.4f}; "
                 f"|err|<0.1%: {np.mean(s.k_err.abs() < 0.001):.2f}, <0.3%: {np.mean(s.k_err.abs() < 0.003):.2f}; "
                 f"corr(k_hat, k_true) {np.corrcoef(s.k_hat, s.k_mean)[0, 1]:+.2f}; true k spread sd {s.k_mean.std():.4f}; "
                 f"peak on prior edge: {int(s.edge.sum())}")
        for name in ("straight", "v3_6", "v6_9", "v9_12", "v12_20"):
            e = s[f"kerr_{name}"].dropna()
            L.append(f"    subset {name:8s}: n_front median {s['n_' + name].median():.0f}, per-bag k err median "
                     f"{e.median():+.4f} robust sd {rs(e):.4f} (L_eff {s[f'Leff_{name}'].iloc[0]:.3f} m)")
    L.append("Accumulation: k error of chunks with T seconds of valid driving (both splits, L_eff from train)")
    acc = []
    for T in TCHUNK:
        for split in ("train", "val"):
            e = ch[(ch["T"] == T) & (ch.split == split)].k_err.dropna()
            if len(e) < 5:
                continue
            acc.append((T, split, rs(e), np.mean(np.abs(e) < 0.001), np.mean(np.abs(e) < 0.003), len(e)))
            L.append(f"  T={T:5d}s {split:5s} n={len(e):4d} median {e.median():+.4f} robust sd {rs(e):.4f} "
                     f"sd {e.std():.4f} |err|<0.1%: {np.mean(np.abs(e) < 0.001):.2f} <0.3%: {np.mean(np.abs(e) < 0.003):.2f}"
                     f" <1%: {np.mean(np.abs(e) < 0.01):.2f}")
    A = pd.DataFrame(acc, columns=["T", "split", "rsd", "p01", "p03", "n"])
    a = A[A.split == "train"]
    # fit rsd = c / sqrt(T) on train chunks
    cfit = float(np.exp(np.mean(np.log(a.rsd) + 0.5 * np.log(a["T"]))))
    L.append(f"Fit on train chunks: robust sd(T) ~ {cfit:.4f} / sqrt(T/s)  ->  sd = 0.1 % needs T = "
             f"{(cfit / 0.001) ** 2:.0f} s = {(cfit / 0.001) ** 2 / 3600:.1f} h of valid driving "
             f"(sd 0.05 %: {(cfit / 0.0005) ** 2 / 3600:.1f} h); 95 % within +-0.1 % needs "
             f"{(1.96 * cfit / 0.001) ** 2 / 3600:.1f} h")
    txt = "\n".join(L)
    print(txt)
    (HERE / f"{TAG}_summary.txt").write_text(txt, encoding="utf-8")

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    Cm = np.mean([wide[b] for b in wide if b in set(tr.bag)], 0)
    ax[0].plot(WIDE, Cm, "k-")
    ax[0].axvline(BASE, color="orange", ls="--", label="7.55 m")
    ax[0].set_xlabel("lag in wheel distance, m (rear after front)")
    ax[0].set_ylabel("normalised CCF of whitened residuals")
    ax[0].set_title(f"mean over {len(tr)} train bags")
    ax[0].legend()
    ax[0].grid(alpha=0.4)
    ax[1].scatter(df[df.split == "train"].k_mean * 100, df[df.split == "train"].k_hat * 100, s=14, label="train")
    ax[1].scatter(df[df.split == "val"].k_mean * 100, df[df.split == "val"].k_hat * 100, s=14, label="val")
    lim = [min(df.k_mean.min(), df.k_hat.min()) * 100 - 0.3, max(df.k_mean.max(), df.k_hat.max()) * 100 + 0.3]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set_xlabel("k true (GNSS, straight steady), %")
    ax[1].set_ylabel("k from bogie lag (whole bag), %")
    ax[1].set_title("per-bag scale from the lag vs truth")
    ax[1].legend()
    ax[1].grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(HERE / f"{TAG}_ccf.png", dpi=90)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, mk in (("train", "o-"), ("val", "s--")):
        s = A[A.split == split]
        ax.loglog(s["T"], s.rsd * 100, mk, label=f"{split} chunks (robust sd)")
    tt = np.logspace(np.log10(30), np.log10(3e5), 50)
    ax.loglog(tt, cfit / np.sqrt(tt) * 100, "k:", label="c/sqrt(T) fit (train)")
    ax.axhline(0.1, color="r", ls="--", label="0.1 %")
    ax.set_xlabel("valid driving time accumulated, s")
    ax.set_ylabel("k error, %")
    ax.grid(alpha=0.4, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / f"{TAG}_accumulation.png", dpi=90)


if __name__ == "__main__":
    main()
