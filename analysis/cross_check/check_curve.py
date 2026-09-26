"""Cross-check 4: map accuracy + wheel-vs-map distance in curves (resolves 0.5|k| capped vs 0.41|k| uncapped).

For every train+val run with RTK master fixes:
  * project status-2 master fixes (glitch-free header time) onto the train-only map 'main' edge -> s_map(t),
    cross-track d (map accuracy check, val runs only, reported separately)
  * raw wheel distance D(t) = cumulative trapezoid of mean(front, rear)[km/h]/3.6 over header time,
    evaluated at t_fix + 0.045 s (GNSS fix stamps lead the wheel stamps by ~45 ms)
  * 30 m windows of map arc length -> ratio r = ds_map / dD; per-run straight scale a = median r where
    max|k| < 0.001; excess = r / a - 1 as a function of the window's mean |k|
Compares the excess with the two published curve models:
  wheel_anomalies: 1 + min(0.5|k|, 0.0095);  map_build: 1 + 0.41|k| (front) / 0.423|k| (rear)
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\MosTransHack\analysis\cross_check")
sys.path.insert(0, r"C:\MosTransHack\analysis\map_build")
from xc_common import OUT, bag_meta, glitch_free, load, mono  # noqa: E402
from track_map import TrackMap, geodetic_to_enu  # noqa: E402

TM = TrackMap(r"C:\MosTransHack\analysis\map_build\map_train")
MAIN = TM.edges["main"]
ORIGIN = (55.8028, 37.424, 160.0)


def run(bag, win=30.0):
    d = load(bag)
    fx, vm, f, r = d["fixm"], d["velm"], d["front"], d["rear"]
    g = glitch_free(fx) & (fx[:, 5] == 2)
    if g.sum() < 3000:
        return None
    t, la, lo, al = mono(fx[g, 1], fx[g, 2], fx[g, 3], fx[g, 4])
    x, y, z = geodetic_to_enu(la, lo, al, ORIGIN)
    gv = glitch_free(vm)
    tv, vx, vy = mono(vm[gv, 1], vm[gv, 2], vm[gv, 3])
    ve = np.interp(t, tv, vx)
    vn = np.interp(t, tv, vy)
    spd = np.hypot(ve, vn)
    hd = np.where(spd > 0.5, np.arctan2(vn, ve), np.nan)
    # decimate to 2 Hz for projection speed, keep moving samples + some stationary
    sel = np.arange(0, len(t), 5)
    s, dlat, dist, _ = MAIN.project(x[sel], y[sel], heading=hd[sel], max_d=8.0)
    ok = np.isfinite(s)
    ts, s, dlat = t[sel][ok], s[ok], dlat[ok]
    if len(s) < 500:
        return None
    # unwrap the cycle
    L = MAIN.length
    s = np.unwrap(s * 2 * np.pi / L) * L / (2 * np.pi)
    # enforce monotone (tram never reverses); drop points that go backwards > 1 m (mis-projection)
    keep = np.r_[True, np.diff(s) > -1.0]
    ts, s, dlat = ts[keep], s[keep], dlat[keep]
    # wheel distance on header time
    gf, gr = glitch_free(f), glitch_free(r)
    tf, vf = mono(f[gf, 1], f[gf, 2])
    trr, vr = mono(r[gr, 1], r[gr, 2])
    vr_on_f = np.interp(tf, trr, vr)
    both = np.abs(vf - vr_on_f) < 0.03 * np.maximum(vf, 1) + 0.5
    vw = np.where(both, 0.5 * (vf + vr_on_f), vf) / 3.6
    Dw = np.r_[0.0, np.cumsum(0.5 * (vw[1:] + vw[:-1]) * np.diff(tf))]
    Df = np.r_[0.0, np.cumsum(0.5 * (vf[1:] + vf[:-1]) * np.diff(tf))] / 3.6
    Dr_t = np.r_[0.0, np.cumsum(0.5 * (vr[1:] + vr[:-1]) * np.diff(trr))] / 3.6
    D = np.interp(ts + 0.045, tf, Dw)
    DF = np.interp(ts + 0.045, tf, Df)
    DR = np.interp(ts + 0.045, trr, Dr_t)
    # windows of map arc length
    edges = np.arange(s[0] + 5, s[-1] - 5, win)
    if len(edges) < 10:
        return None
    ti = np.interp(edges, s, ts)
    Dm = np.interp(ti, ts, D)
    DFm = np.interp(ti, ts, DF)
    DRm = np.interp(ti, ts, DR)
    out = []
    for i in range(len(edges) - 1):
        ds = win
        dD = Dm[i + 1] - Dm[i]
        if dD < 0.5 * win:
            continue
        # time gap check: skip windows spanning GNSS gaps
        j0, j1 = np.searchsorted(ts, ti[i]), np.searchsorted(ts, ti[i + 1])
        if j1 - j0 < 2 or np.max(np.diff(ts[max(j0 - 1, 0):j1 + 1])) > 2.0:
            continue
        sm = 0.5 * (edges[i] + edges[i + 1]) % L
        ss = np.linspace(edges[i], edges[i + 1], 31) % L
        kk = np.abs(MAIN.curvature(ss))
        out.append((sm, ds / dD, ds / (DFm[i + 1] - DFm[i]), ds / (DRm[i + 1] - DRm[i]), kk.mean(), kk.max(),
                    (ti[i + 1] - ti[i])))
    out = np.array(out)
    straight = out[:, 5] < 0.001
    a = np.median(out[straight, 1])
    aF = np.median(out[straight, 2])
    aR = np.median(out[straight, 3])
    return dict(bag=bag, w=out, a=a, aF=aF, aR=aR, dlat=dlat)


def main():
    meta = bag_meta()
    bags = sorted(b for b, m in meta.items() if m["split"] in ("train", "val"))
    allw, dl_val = [], []
    for b in bags:
        try:
            res = run(b)
        except Exception as e:
            print(b, "ERR", e)
            continue
        if res is None:
            continue
        w = res["w"]
        exc = w[:, 1] / res["a"] - 1
        excF = w[:, 2] / res["aF"] - 1
        excR = w[:, 3] / res["aR"] - 1
        allw.append(np.c_[w[:, 4], w[:, 5], exc, excF, excR, np.full(len(w), meta[b]["vehicle"] == "30639"),
                          w[:, 0]])
        if meta[b]["split"] == "val":
            dl_val.append(res["dlat"])
        print(b, meta[b]["vehicle"], meta[b]["date"], f"a={res['a']:.5f} (k_equiv={3.6/res['a']:.4f}) nwin={len(w)}", flush=True)
    A = np.vstack(allw)
    np.save(OUT / "curve_windows.npy", A)
    print("\nexcess of map distance over wheel distance vs mean|k| of 30 m windows (median, IQR, n)")
    bins = [0, 0.001, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04, 0.07]
    for veh_name, vm_ in (("all", np.ones(len(A), bool)), ("30618", A[:, 5] == 0), ("30639", A[:, 5] == 1)):
        print(f"-- vehicle {veh_name}")
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = vm_ & (A[:, 0] >= lo) & (A[:, 0] < hi)
            if m.sum() < 5:
                continue
            e = A[m, 2]
            kmid = np.median(A[m, 0])
            print(f" |k| {lo:.3f}-{hi:.3f} (med {kmid:.4f}, R~{1/max(kmid,1e-9):6.0f} m): excess {np.median(e)*100:+.3f}% "
                  f"IQR [{np.percentile(e,25)*100:+.3f},{np.percentile(e,75)*100:+.3f}] n={m.sum():5d} | "
                  f"front {np.median(A[m,3])*100:+.3f}% rear {np.median(A[m,4])*100:+.3f}% | model 0.41k {0.41*kmid*100:.3f}%  "
                  f"capped {min(0.5*kmid,0.0095)*100:.3f}%")
    # least squares fit excess = c*|k| on curved windows
    m = A[:, 0] > 0.001
    c = np.sum(A[m, 0] * A[m, 2]) / np.sum(A[m, 0] ** 2)
    print(f"LS fit excess = c*|k|: c = {c:.3f} m (all curved windows)")
    for lo in (0.02, 0.03):
        mm = A[:, 0] > lo
        print(f" windows with mean|k|>{lo}: median excess {np.median(A[mm,2])*100:.2f}% vs 0.41k {np.median(0.41*A[mm,0])*100:.2f}% "
              f"vs capped {np.median(np.minimum(0.5*A[mm,0],0.0095))*100:.2f}%  n={mm.sum()}")
    dl = np.abs(np.concatenate(dl_val))
    print(f"\nVAL RTK cross-track to train-only map (2 Hz, heading-gated main only): n={len(dl)} p50 {np.median(dl)*100:.2f} cm "
          f"p68 {np.percentile(dl,68)*100:.2f} cm p90 {np.percentile(dl,90)*100:.1f} cm p95 {np.percentile(dl,95)*100:.1f} cm "
          f"p99 {np.percentile(dl,99)*100:.0f} cm; frac<10cm {np.mean(dl<0.1):.3f} frac<1m {np.mean(dl<1):.3f}")


if __name__ == "__main__":
    main()
