"""Step 1. Characterise the bogie speed signals (train bags).

Per bag:
  sampling  : header dt quantiles of the joint front/rear stamps, gap fractions, share of stamps common to both
  resolution: repeated consecutive values while moving; smallest value step q (km/h) per bogie
  noise     : in 5 s windows of steady motion (GNSS speed 5..15 m/s, linear-fit slope < 0.1 m/s^2,
              GNSS speed std about the fit < 0.1 m/s):
              std of F-R about its window mean, of each signal about a linear trend, white-noise level from the
              second difference, and lag-0 correlations of high-pass residuals (F vs R, F vs GNSS)
Outputs: s1_per_bag.csv, s1_summary.txt
"""
import sys

import numpy as np
import pandas as pd

from bc_common import HERE, KMH, load_bag, splits

WIN = 5.0


def smallest_step(v):
    dv = np.abs(np.diff(v))
    dv = dv[(dv > 0.0035) & (dv < 0.0062)]
    if len(dv) < 20:
        return np.nan
    med = np.median(dv)
    return float(np.median(dv[np.abs(dv - med) < 5e-5]))


def white_sigma(x):
    """per-sample white-noise sigma from the second difference x_i - (x_{i-1}+x_{i+1})/2 (var = 1.5 sigma^2)."""
    e = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    return float(np.std(e) / np.sqrt(1.5))


def hp(x, n=5):
    k = np.ones(n) / n
    return x[n // 2: len(x) - n // 2] - np.convolve(x, k, mode="valid")


def per_bag(bag):
    d = load_bag(bag)
    t, F, R, vg, ag, kap = d["t"], d["F"], d["R"], d["vg"], d["ag"], d["kap"]
    dt = np.diff(t)
    moving = np.isfinite(vg[1:]) & (vg[1:] > 1.0)
    q = np.quantile(dt, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    qm = np.quantile(dt[moving], [0.01, 0.5, 0.99]) if moving.sum() > 100 else [np.nan] * 3
    rep_f = np.mean(np.diff(F)[moving] == 0) if moving.sum() else np.nan
    rep_r = np.mean(np.diff(R)[moving] == 0) if moving.sum() else np.nan
    res = {"bag": bag, "n_front": d["n_front"], "n_rear": d["n_rear"], "n_joint": d["n_joint"],
           "frac_front_joint": d["n_joint"] / max(d["n_front"], 1),
           "dur_s": t[-1] - t[0], "moving_s": float(np.sum(dt[moving & (dt < 0.3)])),
           "cruise_5_15_s": float(np.sum(dt[(vg[1:] >= 5) & (vg[1:] <= 15) & (dt < 0.3)])),
           **{f"dt_p{p}": v for p, v in zip((1, 5, 25, 50, 75, 95, 99), q)},
           "dt_mov_p1": qm[0], "dt_mov_p50": qm[1], "dt_mov_p99": qm[2],
           "gap_gt_0.15": np.mean(dt > 0.15), "gap_gt_0.3": np.mean(dt > 0.3), "gap_gt_1": np.mean(dt > 1.0),
           "rate_hz": d["n_joint"] / (t[-1] - t[0]),
           "repeat_front": rep_f, "repeat_rear": rep_r,
           "q_front_kmh": smallest_step(d["vf_all"]), "q_rear_kmh": smallest_step(d["vr_all"])}
    # steady windows
    rows = []
    i = 0
    n = len(t)
    while i < n:
        j = np.searchsorted(t, t[i] + WIN)
        if j >= n:
            break
        sl = slice(i, j)
        tt = t[sl]
        v, a = vg[sl], ag[sl]
        ok = (len(tt) >= 40 and np.all(np.diff(tt) < 0.3) and np.all(np.isfinite(v))
              and v.min() >= 5 and v.max() <= 15)
        if ok:
            pv = np.polyfit(tt - tt.mean(), v, 1)
            ok = abs(pv[0]) < 0.1 and np.std(v - np.polyval(pv, tt - tt.mean())) < 0.1
        if ok:
            f, r = F[sl], R[sl]
            gk = v * KMH * np.median(f / (v * KMH))  # GNSS speed on the wheel's km/h scale
            x = tt - tt.mean()
            det = lambda y: y - np.polyval(np.polyfit(x, y, 1), x)  # noqa: E731
            hf, hr, hg = hp(f), hp(r), hp(gk)
            straight = bool(np.all(np.abs(kap[sl]) < 1e-3))
            rows.append({"v": v.mean(), "straight": straight,
                         "sd_FR": np.std(f - r), "sd_FR_pct": 100 * np.std(f - r) / f.mean(),
                         "sd_F": np.std(det(f)), "sd_R": np.std(det(r)), "sd_G": np.std(det(gk)),
                         "w_F": white_sigma(f), "w_R": white_sigma(r), "w_G": white_sigma(gk),
                         "w_FR": white_sigma(f - r),
                         "c_FR": np.corrcoef(hf, hr)[0, 1], "c_FG": np.corrcoef(hf, hg)[0, 1],
                         "c_RG": np.corrcoef(hr, hg)[0, 1]})
            i = j
        else:
            i += 1
    w = pd.DataFrame(rows)
    res["n_steady_win"] = len(w)
    if len(w):
        for c in ["sd_FR", "sd_FR_pct", "sd_F", "sd_R", "sd_G", "w_F", "w_R", "w_G", "w_FR", "c_FR", "c_FG", "c_RG"]:
            res[c] = float(np.median(w[c]))
        s = w[w.straight]
        res["n_steady_straight"] = len(s)
        res["sd_FR_straight"] = float(np.median(s.sd_FR)) if len(s) else np.nan
    return res, w.assign(bag=bag)


def main():
    bags = splits()["train"]
    if len(sys.argv) > 1:
        bags = bags[: int(sys.argv[1])]
    out, wins = [], []
    for b in bags:
        r, w = per_bag(b)
        out.append(r)
        wins.append(w)
        print(b, f"rate {r['rate_hz']:.2f} Hz dt50 {r['dt_p50']:.3f} q {r['q_front_kmh']:.6f}/{r['q_rear_kmh']:.6f} "
              f"steady {r['n_steady_win']} sdFR {r.get('sd_FR', np.nan):.3f} km/h", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(HERE / "s1_per_bag.csv", index=False)
    W = pd.concat(wins, ignore_index=True)
    W.to_csv(HERE / "s1_steady_windows.csv", index=False)
    lines = []
    P = lambda s: f"{s.median():.4g} (p10 {s.quantile(.1):.4g}, p90 {s.quantile(.9):.4g})"  # noqa: E731
    lines.append(f"bags {len(df)}; total {df.dur_s.sum()/3600:.2f} h, moving {df.moving_s.sum()/3600:.2f} h, "
                 f"at 5-15 m/s {df.cruise_5_15_s.sum()/3600:.2f} h")
    lines.append("SAMPLING (joint front/rear header stamps), median over bags [min..max]:")
    for c in ["rate_hz", "dt_p1", "dt_p5", "dt_p25", "dt_p50", "dt_p75", "dt_p95", "dt_p99", "dt_mov_p1", "dt_mov_p50",
              "dt_mov_p99", "gap_gt_0.15", "gap_gt_0.3", "gap_gt_1", "frac_front_joint"]:
        lines.append(f"  {c:18s} {df[c].median():.4f} [{df[c].min():.4f}..{df[c].max():.4f}]")
    lines.append("RESOLUTION:")
    for c in ["q_front_kmh", "q_rear_kmh", "repeat_front", "repeat_rear"]:
        lines.append(f"  {c:18s} {df[c].median():.6f} [{df[c].min():.6f}..{df[c].max():.6f}]")
    lines.append(f"STEADY WINDOWS ({WIN:.0f} s, 5-15 m/s, |slope|<0.1 m/s2, sd about trend<0.1 m/s): n={len(W)} "
                 f"(straight {int(W.straight.sum())}); per-window values, median (p10, p90), km/h:")
    for c, lab in [("sd_FR", "std(F-R) about window mean"), ("sd_FR_pct", "  same, % of speed"),
                   ("sd_F", "std(F) about linear trend"), ("sd_R", "std(R) about linear trend"),
                   ("sd_G", "std(GNSS*scale) about trend"),
                   ("w_F", "white noise F (2nd diff)"), ("w_R", "white noise R (2nd diff)"),
                   ("w_G", "white noise GNSS (2nd diff)"), ("w_FR", "white noise F-R (2nd diff)"),
                   ("c_FR", "corr HP(F),HP(R) lag 0"), ("c_FG", "corr HP(F),HP(GNSS) lag 0"),
                   ("c_RG", "corr HP(R),HP(GNSS) lag 0")]:
        lines.append(f"  {lab:30s} {P(W[c])}")
    S = W[W.straight]
    lines.append(f"  std(F-R) straight only          {P(S.sd_FR)}  (n={len(S)})")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "s1_summary.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
