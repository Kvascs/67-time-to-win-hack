"""Cross-check 7: smaller decision-critical claims (each function prints its result).

  start_of_run()      first wheel motion vs bag start; GNSS speed inside a 5 s bag-time window
  clock_episodes()    +-1 s header-clock episodes per topic (GNSS vs wheel/cmd)
  stamps()            age of newest wheel sample at cmd stamps, shared front/rear stamps,
                      distinct wheel stamps per second, backward stamps in the wheel+cmd union
  within_run_scale()  wheel scale drift inside a run (last quarter vs first quarter)
  detour()            westbound detour usage by date (offset from 'main' at s 1200-1700 m)
  grade_agreement()   traction_id grade profile vs map_build edge grade
Run: python check_misc.py [function ...]
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\MosTransHack\analysis\cross_check")
sys.path.insert(0, r"C:\MosTransHack\analysis\map_build")
sys.path.insert(0, r"C:\MosTransHack\analysis\traction_id")
from xc_common import bag_meta, glitch_free, interp_gap, load, mono  # noqa: E402

ORIGIN = (55.8028, 37.424, 160.0)


def _t0(d):
    return min(d[k][0, 0] for k in d if len(d[k]))


def start_of_run():
    meta = bag_meta()
    tm, rows = [], []
    for b, m in sorted(meta.items()):
        if m["split"] not in ("train", "val", "no_gnss_long"):
            continue
        d = load(b)
        t0 = _t0(d)
        first = []
        for k in ("front", "rear"):
            j = np.flatnonzero(d[k][:, 2] > 1.0)
            first.append(d[k][j[0], 0] - t0 if len(j) else np.inf)
        tm.append(min(first))
        vm = d["velm"]
        if len(vm) and m["split"] == "val":
            sel = vm[:, 0] - t0 <= 5.0
            rows.append((b, round(tm[-1], 1), round(float(np.hypot(vm[sel, 2], vm[sel, 3]).max()), 2)))
    tm = np.array(tm)
    print(f"long bags n={len(tm)}: first motion (>1 km/h) after bag start min {tm.min():.1f} s, "
          f"p10 {np.percentile(tm, 10):.1f}, median {np.median(tm):.1f}, max {tm.max():.1f}; "
          f"moving within 5 s: {np.sum(tm <= 5)}, within 10 s: {np.sum(tm <= 10)}")
    print("val (bag, first motion s, max GNSS speed in first 5 s bag time):", rows)


def clock_episodes(bags=("30639_3b3d9eb8", "30618_40ffd323", "30618_2255aade")):
    for bag in bags:
        d = load(bag)
        t0 = _t0(d)
        print("==", bag)
        for k in ("front", "cmd", "fixm", "velm"):
            a = d[k]
            if not len(a):
                continue
            dev = (a[:, 0] - a[:, 1]) - np.median(a[:, 0] - a[:, 1])
            idx = np.flatnonzero(np.abs(dev) > 0.5)
            eps = []
            if len(idx):
                br = np.flatnonzero(np.diff(idx) > 50)
                for s, e in zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]):
                    eps.append((round(a[s, 0] - t0, 1), round(a[e, 0] - t0, 1), round(float(np.median(dev[s:e + 1])), 2)))
            print(f"  {k:5s} episodes (start s, end s, bag-hdr deviation s): {eps[:5]}")


def stamps(bags=("30618_e3d94878", "30618_01f73500", "30639_9f0b519f", "30618_2366c74a", "30639_2b4a6347",
                 "30618_b95ca60a")):
    ages, same, rates, back = [], [], [], []
    for b in bags:
        d = load(b)
        f, r, c = d["front"], d["rear"], d["cmd"]
        f, c = f[glitch_free(f)], c[glitch_free(c)]
        j = np.searchsorted(f[:, 0], c[:, 0], side="right") - 1
        ok = j >= 0
        ages.append(c[ok, 1] - np.maximum.accumulate(f[:, 1])[j[ok]])
        rs = set(np.round(r[:, 1], 4))
        same.append(np.mean([round(x, 4) in rs for x in f[:, 1]]))
        rates.append(len(np.unique(np.round(f[:, 1], 4))) / (f[-1, 1] - f[0, 1]))
        o = np.argsort(np.r_[f[:, 0], c[:, 0]], kind="stable")
        back.append(np.mean(np.diff(np.r_[f[:, 1], c[:, 1]][o]) < 0))
    a = np.concatenate(ages) * 1e3
    print(f"newest wheel sample age at cmd stamps: median {np.median(a):.0f} ms, p5 {np.percentile(a, 5):.0f}, "
          f"p95 {np.percentile(a, 95):.0f}, p99 {np.percentile(a, 99):.0f}")
    print("front stamps present in rear:", np.round(same, 4))
    print("distinct wheel stamps / s:", np.round(rates, 2))
    print("backward steps in wheel+cmd union (arrival order):", np.round(back, 3))


def within_run_scale():
    meta = bag_meta()
    out = []
    for b, m in sorted(meta.items()):
        if m["split"] not in ("train", "val"):
            continue
        d = load(b)
        vm, f, r = d["velm"], d["front"], d["rear"]
        if len(vm) < 1000:
            continue
        tv, vx, vy = mono(vm[glitch_free(vm), 1], vm[glitch_free(vm), 2], vm[glitch_free(vm), 3])
        vg = np.hypot(vx, vy)
        F = interp_gap(tv, *mono(f[glitch_free(f), 1], f[glitch_free(f), 2]))
        R = interp_gap(tv, *mono(r[glitch_free(r), 1], r[glitch_free(r), 2]))
        M = 0.5 * (F + R)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = M / vg
        ok = (vg > 3) & np.isfinite(M) & (np.abs(F - R) < 0.02 * M) & (ratio > 3.3) & (ratio < 3.9)
        idx = np.flatnonzero(ok)
        if len(idx) < 400:
            continue
        ks = [np.median(ratio[i]) for i in np.array_split(idx, 4)]
        out.append((ks[-1] / ks[0] - 1) * 100)
    out = np.array(out)
    print(f"k(last quarter)/k(first quarter)-1: median {np.median(out):+.3f}%, p5 {np.percentile(out, 5):+.3f}%, "
          f"p95 {np.percentile(out, 95):+.3f}%, max|.| {np.abs(out).max():.3f}% (n={len(out)})")


def detour():
    from track_map import TrackMap, geodetic_to_enu
    main = TrackMap(r"C:\MosTransHack\analysis\map_build\map").edges["main"]
    meta = bag_meta()
    res = {}
    for b, m in sorted(meta.items()):
        if m["split"] not in ("train", "val"):
            continue
        d = load(b)
        fx, vm = d["fixm"], d["velm"]
        g = glitch_free(fx) & (fx[:, 5] == 2)
        if g.sum() < 2000:
            continue
        t, la, lo, al = mono(fx[g, 1], fx[g, 2], fx[g, 3], fx[g, 4])
        sel = np.arange(0, len(t), 10)
        x, y, _ = geodetic_to_enu(la[sel], lo[sel], al[sel], ORIGIN)
        tv, vx, vy = mono(vm[glitch_free(vm), 1], vm[glitch_free(vm), 2], vm[glitch_free(vm), 3])
        hd = np.arctan2(np.interp(t[sel], tv, vy), np.interp(t[sel], tv, vx))
        s, dl, _, _ = main.project(x, y, heading=hd, max_d=20)
        inr = (s > 1200) & (s < 1700) & np.isfinite(dl)
        if inr.sum() >= 20:
            res.setdefault(m["date"], []).append(round(float(np.median(dl[inr])), 2))
    for k, v in sorted(res.items()):
        print(k, "offset from main at s 1200-1700 m (+ left):", v)


def grade_agreement():
    from pyproj import Transformer
    from track_map import TrackMap, enu_to_geodetic
    from traction_model import Track, TractionModel
    m = TrackMap(r"C:\MosTransHack\analysis\map_build\map").edges["main"]
    tr, T = Track(), TractionModel()
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)
    ss = np.arange(0, m.length, 10.0)
    x, y, z, _ = m.pose(ss)
    g_map = m.grade_at(ss)
    lat, lon, _ = enu_to_geodetic(x, y, z, ORIGIN)
    E, N = fwd.transform(lon, lat)
    sc = np.array([tr.project(e, n) for e, n in zip(E, N)])
    g_tid = np.array([T.grade.at(s, dd) for s, dd in zip(sc, np.sign(np.gradient(sc)))])
    dg = g_map - g_tid
    print(f"grade map_build - traction_id: median {np.median(dg):+.4f}, RMS {np.sqrt(np.mean(dg**2)):.4f}, "
          f"p95|d| {np.percentile(np.abs(dg), 95):.4f}, corr {np.corrcoef(g_map, g_tid)[0, 1]:.4f}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["start_of_run", "clock_episodes", "stamps", "within_run_scale", "detour", "grade_agreement"]
    for n in names:
        print(f"\n##### {n}")
        globals()[n]()
