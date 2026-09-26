"""Cross-check 1+2: wheel units / scale factor per bag and wheel-vs-GNSS time lags.

For every train+val bag (master antenna):
  k_*      = median(wheel_kmh / |v_gnss_horizontal|) on clean moving samples (v > 3 m/s, bogies agree)
  kd_*     = distance ratio  sum(wheel_kmh*dt) / sum(v_gnss*dt) over the same clean samples
  lag_hh   = time shift tau (s) minimising RMS(wheel(t + tau)/k - v_vel(t)) with both on header stamps
             (positive tau: wheel stamps are LATE relative to GNSS vel stamps)
  lag_bb   = same with bag receive times
  lag_pos  = wheel(header) vs |dp/dt| of master fixes (header), central differences
  lag_vel_pos = GNSS vel vs |dp/dt| (header)
Only dynamic samples (|a| > 0.25 m/s^2) are used for lags, so the result is driven by transients.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\MosTransHack\analysis\cross_check")
from xc_common import OUT, bag_meta, enu, glitch_free, interp_gap, load, mono  # noqa: E402

TAUS = np.arange(-0.25, 0.2501, 0.005)


def best_lag(t_ref, v_ref, t_w, v_w, k, mask):
    """tau minimising RMS(v_w(t+tau)/k - v_ref) over mask (parabolic refinement)."""
    rms = []
    for tau in TAUS:
        w = interp_gap(t_ref[mask] + tau, t_w, v_w) / k
        e = w - v_ref[mask]
        e = e[np.isfinite(e)]
        rms.append(np.sqrt(np.mean(e * e)) if len(e) > 200 else np.nan)
    rms = np.array(rms)
    if not np.any(np.isfinite(rms)):
        return np.nan, np.nan
    i = int(np.nanargmin(rms))
    if 0 < i < len(TAUS) - 1 and np.all(np.isfinite(rms[i - 1:i + 2])):
        y0, y1, y2 = rms[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
        return TAUS[i] + off * (TAUS[1] - TAUS[0]), y1
    return TAUS[i], rms[i]


def smooth_diff(t, v, half=0.5):
    """Centered slope of v over +-half seconds (for the dynamic-sample mask only)."""
    j0 = np.searchsorted(t, t - half)
    j1 = np.clip(np.searchsorted(t, t + half) - 1, 0, len(t) - 1)
    dt = t[j1] - t[j0]
    return np.where(dt > 0.5, (v[j1] - v[j0]) / np.maximum(dt, 1e-9), 0.0)


def analyse(bag):
    d = load(bag)
    f, r, vm, fx = d["front"], d["rear"], d["velm"], d["fixm"]
    if len(vm) < 1000 or len(f) < 1000:
        return None
    res = {"bag": bag}
    for base, col in (("h", 1), ("b", 0)):
        gv = glitch_free(vm)
        tv, vx, vy = mono(vm[gv, col], vm[gv, 2], vm[gv, 3])
        vg = np.hypot(vx, vy)
        a = smooth_diff(tv, vg)
        ws = {}
        for name, arr in (("front", f), ("rear", r)):
            g = glitch_free(arr)
            tw, vw = mono(arr[g, col], arr[g, 2])
            ws[name] = (tw, vw)
        F = interp_gap(tv, *ws["front"])
        R = interp_gap(tv, *ws["rear"])
        M = 0.5 * (F + R)
        clean = (vg > 3.0) & np.isfinite(M) & (np.abs(F - R) < 0.02 * M) & (np.abs(a) < 0.1)
        if base == "h":
            ratio = M[clean] / vg[clean]
            ok = (ratio > 3.3) & (ratio < 3.9)
            res["n_clean"] = int(ok.sum())
            res["k_front"] = float(np.median(F[clean][ok] / vg[clean][ok]))
            res["k_rear"] = float(np.median(R[clean][ok] / vg[clean][ok]))
            res["k_mean"] = float(np.median(ratio[ok]))
            dtv = np.r_[np.diff(tv), 0.0]
            cm = np.zeros(len(tv), bool)
            cm[np.flatnonzero(clean)[ok]] = True
            cm &= dtv < 0.15
            res["kd_mean"] = float(np.sum(M[cm] * dtv[cm]) / np.sum(vg[cm] * dtv[cm]))
            k = res["k_mean"]
        dyn = (vg > 1.0) & (np.abs(a) > 0.25) & np.isfinite(M) & (np.abs(F - R) < 0.05 * M + 0.5)
        tm, vmn = mono(tv, np.zeros_like(tv))  # placeholder to keep names aligned
        # wheel mean series on its own stamps (front & rear share stamps; average where both exist)
        tf, vf = ws["front"]
        Rf = interp_gap(tf, *ws["rear"])
        wmean = np.where(np.isfinite(Rf), 0.5 * (vf + Rf), vf)
        lag, rms = best_lag(tv, vg, tf, wmean, k, dyn)
        res[f"lag_{base}{base}"] = lag
        res[f"rms_{base}{base}"] = rms
        if base == "h":
            res["n_dyn"] = int(dyn.sum())
            # cross time bases: wheel header vs vel bag
    # position-derivative based lags (header time, status-2 fixes only)
    gf = glitch_free(fx) & (fx[:, 5] == 2)
    if gf.sum() > 2000:
        tfx, la, lo, al = mono(fx[gf, 1], fx[gf, 2], fx[gf, 3], fx[gf, 4])
        p = enu(la, lo, al, la[0], lo[0], al[0])
        ok = (np.abs(tfx[2:] - tfx[:-2] - 0.2) < 0.02)
        tc = tfx[1:-1][ok]
        vp = np.hypot(*(p[2:, :2] - p[:-2, :2])[ok].T) / (tfx[2:] - tfx[:-2])[ok]
        a = smooth_diff(tc, vp)
        dyn = (vp > 1.0) & (np.abs(a) > 0.25) & (vp < 25)
        tf, vf = mono(f[glitch_free(f), 1], f[glitch_free(f), 2])
        res["lag_pos"], res["rms_pos"] = best_lag(tc, vp, tf, vf, res["k_front"], dyn)
        gv = glitch_free(vm)
        tv, vx, vy = mono(vm[gv, 1], vm[gv, 2], vm[gv, 3])
        vg = np.hypot(vx, vy)
        res["lag_vel_pos"], _ = best_lag(tc, vp, tv, vg * 1.0, 1.0, dyn)
    return res


def main():
    meta = bag_meta()
    bags = [b for b, m in meta.items() if m["split"] in ("train", "val")]
    rows = []
    for b in sorted(bags):
        try:
            r = analyse(b)
        except Exception as e:  # keep going
            print(b, "ERR", e)
            continue
        if r is None:
            continue
        r.update({k: meta[b][k] for k in ("vehicle", "date", "hhmm", "split")})
        rows.append(r)
        print(b, r["vehicle"], r["date"], r["hhmm"], f"k={r['k_mean']:.4f} kd={r['kd_mean']:.4f} "
              f"F/R={r['k_front']:.4f}/{r['k_rear']:.4f} lag_hh={r['lag_hh']*1e3:+.1f}ms lag_bb={r['lag_bb']*1e3:+.1f}ms "
              f"lag_pos={r.get('lag_pos', np.nan)*1e3:+.1f}ms vel_vs_pos={r.get('lag_vel_pos', np.nan)*1e3:+.1f}ms "
              f"rms_hh={r['rms_hh']:.3f} rms_bb={r['rms_bb']:.3f}", flush=True)
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "scale_lag_per_bag.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # summaries
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["vehicle"], r["date"])].append(r["k_mean"])
    print("\nk by vehicle/date: median [min..max] (n)")
    for key in sorted(g):
        v = np.array(g[key])
        print(key, f"{np.median(v):.4f} [{v.min():.4f}..{v.max():.4f}] n={len(v)}")
    allk = np.array([r["k_mean"] for r in rows])
    print("fleet median k", np.median(allk), "range", allk.min(), allk.max())
    for key in ("lag_hh", "lag_bb", "lag_pos", "lag_vel_pos"):
        v = np.array([r.get(key, np.nan) for r in rows]) * 1e3
        v = v[np.isfinite(v)]
        print(f"{key}: median {np.median(v):+.1f} ms  IQR [{np.percentile(v,25):+.1f}, {np.percentile(v,75):+.1f}]  "
              f"p5-p95 [{np.percentile(v,5):+.1f}, {np.percentile(v,95):+.1f}]  n={len(v)}")
    for key in ("rms_hh", "rms_bb"):
        v = np.array([r[key] for r in rows])
        print(key, "median", np.nanmedian(v))


if __name__ == "__main__":
    main()
