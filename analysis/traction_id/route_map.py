"""Along-track coordinate and track-grade profile for the single tram line.

* Centerline: an eastbound clean pass (30618_7b4d83f4), resampled at 1 m and smoothed.
  s = 0 at the western terminus, increasing eastwards (towards the eastern loop).
* project(x, y) -> s [m], lateral offset [m] via a KD-tree on the 1 m polyline.
* Altitude profile h(s): robust sparse least squares over all train bags,
      alt_i = h(s_i) + b_seg(i) + e_i
  with one offset per contiguous RTK segment (<= 300 s long) to absorb per-run and slowly
  drifting altitude biases, a 2nd-difference smoothness penalty on h, and Huber IRLS.
* grade(s) = dh/ds (positive = uphill when moving towards +s). The grade felt by the tram is
  grade(s) * dir, dir = +1 eastbound / -1 westbound.

Exported: grade_profile.csv (s, h, grade) at 5 m resolution for the C++ port.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from tid_data import OUT, load_grid, list_bags

REF_BAG = "30618_7b4d83f4"
DS = 1.0
H_STEP = 5.0  # altitude knot spacing [m]


@lru_cache(maxsize=1)
def centerline() -> np.ndarray:
    p = OUT / "centerline.csv"
    if p.exists():
        return np.loadtxt(p, delimiter=",", skiprows=1)
    df = load_grid(REF_BAG)
    m = (df.vg > 0.5) & np.isfinite(df.x_m) & np.isfinite(df.y_m)
    x = df.x_m[m].to_numpy()
    y = df.y_m[m].to_numpy()
    # average master & rover when both present (antenna lever arm cancels on straight track)
    xr, yr = df.x_r[m].to_numpy(), df.y_r[m].to_numpy()
    both = np.isfinite(xr) & np.isfinite(yr)
    x = np.where(both, 0.5 * (x + xr), x)
    y = np.where(both, 0.5 * (y + yr), y)
    seg = np.hypot(np.diff(x), np.diff(y))
    keep = np.r_[True, seg > 0.05]
    x, y = x[keep], y[keep]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    sg = np.arange(0, s[-1], DS)
    xs = np.interp(sg, s, x)
    ys = np.interp(sg, s, y)
    xs = savgol_filter(xs, 21, 2)
    ys = savgol_filter(ys, 21, 2)
    # re-parameterise by arc length of the smoothed curve
    s2 = np.r_[0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    sg2 = np.arange(0, s2[-1], DS)
    cl = np.c_[sg2, np.interp(sg2, s2, xs), np.interp(sg2, s2, ys)]
    np.savetxt(p, cl, delimiter=",", header="s,x,y", comments="", fmt="%.3f")
    return cl


@lru_cache(maxsize=1)
def _tree():
    cl = centerline()
    return cKDTree(cl[:, 1:3])


def project(x: np.ndarray, y: np.ndarray):
    """Return s [m] and signed lateral offset [m] (left of eastbound direction positive)."""
    cl = centerline()
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fin = np.isfinite(x) & np.isfinite(y)
    s = np.full(x.shape, np.nan)
    lat = np.full(x.shape, np.nan)
    if fin.any():
        dist, idx = _tree().query(np.c_[x[fin], y[fin]])
        i0 = np.clip(idx, 1, len(cl) - 2)
        tx = cl[i0 + 1, 1] - cl[i0 - 1, 1]
        ty = cl[i0 + 1, 2] - cl[i0 - 1, 2]
        nrm = np.hypot(tx, ty)
        tx, ty = tx / nrm, ty / nrm
        dx = x[fin] - cl[idx, 1]
        dy = y[fin] - cl[idx, 2]
        along = dx * tx + dy * ty
        s[fin] = cl[idx, 0] + along
        lat[fin] = -dx * ty + dy * tx
    return s, lat


def add_track_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Adds s (along-track), lat, dir (+1 east / -1 west, 0 unknown/standing) to a grid frame."""
    x = df["x_m"].to_numpy()
    y = df["y_m"].to_numpy()
    xr, yr = df["x_r"].to_numpy(), df["y_r"].to_numpy()
    both = np.isfinite(xr) & np.isfinite(yr) & np.isfinite(x)
    xc = np.where(both, 0.5 * (x + xr), x)
    yc = np.where(both, 0.5 * (y + yr), y)
    s, lat = project(xc, yc)
    df["s"] = s
    df["lat"] = lat
    # travel direction from smoothed ds/dt, held through stops
    ss = pd.Series(s).interpolate(limit_area="inside").to_numpy()
    dsdt = np.gradient(ss, df["t"].to_numpy())
    dsdt = pd.Series(dsdt).rolling(41, center=True, min_periods=5).median().to_numpy()
    dirn = np.where(dsdt > 0.3, 1.0, np.where(dsdt < -0.3, -1.0, np.nan))
    dirn = pd.Series(dirn).ffill().bfill().fillna(0).to_numpy()
    df["dir"] = dirn
    return df


def _segments(df: pd.DataFrame, max_len_s: float = 300.0):
    """Contiguous RTK-quality altitude segments -> integer segment id per sample (-1 = unused)."""
    st_m = df["st_m"].to_numpy()
    alt = df["alt_m"].to_numpy()
    alt_r = df["alt_r"].to_numpy()
    ok = (st_m == 2) & np.isfinite(alt) & np.isfinite(df["s"].to_numpy()) & (df["vg"].to_numpy() > 0.5)
    # master/rover altitude consistency (lever arm ~ constant); reject glitches
    dar = alt - alt_r
    med = np.nanmedian(dar[ok]) if ok.any() else 0.0
    ok &= ~(np.isfinite(dar) & (np.abs(dar - med) > 0.5))
    t = df["t"].to_numpy()
    seg = np.full(len(df), -1)
    sid = -1
    last_t = -np.inf
    last_alt = np.nan
    seg_start = -np.inf
    for i in np.flatnonzero(ok):
        new = (t[i] - last_t > 2.0) or (abs(alt[i] - last_alt) > 0.5) or (t[i] - seg_start > max_len_s)
        if new:
            sid += 1
            seg_start = t[i]
        seg[i] = sid
        last_t = t[i]
        last_alt = alt[i]
    return seg


def solve_altitude(bags: list[str], step: float = H_STEP, lam: float = 30.0, decim: int = 10):
    cl = centerline()
    L = cl[-1, 0]
    knots = np.arange(-50.0, L + 50.0 + step, step)
    nk = len(knots)
    rows_s, rows_alt, rows_seg = [], [], []
    seg_off = 0
    for b in bags:
        df = load_grid(b)
        if "s" not in df:
            df = add_track_coords(df)
        seg = _segments(df)
        sel = np.flatnonzero(seg >= 0)[::decim]
        if len(sel) == 0:
            continue
        rows_s.append(df["s"].to_numpy()[sel])
        rows_alt.append(df["alt_m"].to_numpy()[sel])
        rows_seg.append(seg[sel] + seg_off)
        seg_off += seg.max() + 1
    s = np.concatenate(rows_s)
    alt = np.concatenate(rows_alt)
    sg = np.concatenate(rows_seg)
    # drop tiny segments
    u, cnt = np.unique(sg, return_counts=True)
    good = np.isin(sg, u[cnt >= 20])
    s, alt, sg = s[good], alt[good], sg[good]
    _, sg = np.unique(sg, return_inverse=True)
    nseg = sg.max() + 1
    n = len(s)
    # linear interpolation weights on knots
    j = np.clip(np.searchsorted(knots, s) - 1, 0, nk - 2)
    w = (s - knots[j]) / step
    A_h = sparse.csr_matrix((np.r_[1 - w, w], (np.r_[np.arange(n), np.arange(n)], np.r_[j, j + 1])), shape=(n, nk))
    A_b = sparse.csr_matrix((np.ones(n), (np.arange(n), sg)), shape=(n, nseg))
    A = sparse.hstack([A_h, A_b]).tocsr()
    # smoothness (2nd difference on h) and gauge (mean offset 0 -> anchor h via mean of b)
    D = sparse.diags([np.ones(nk - 2), -2 * np.ones(nk - 2), np.ones(nk - 2)], [0, 1, 2], shape=(nk - 2, nk))
    D = sparse.hstack([D, sparse.csr_matrix((nk - 2, nseg))])
    G = sparse.csr_matrix(np.r_[np.zeros(nk), np.ones(nseg) / nseg][None, :])
    wts = np.ones(n)
    for it in range(8):
        Wa = sparse.diags(wts) @ A
        M = sparse.vstack([Wa, lam * D, 1e3 * G]).tocsr()
        rhs = np.r_[wts * alt, np.zeros(nk - 2), 0.0]
        sol = lsqr(M, rhs, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
        r = alt - A @ sol
        sc = 1.4826 * np.median(np.abs(r)) + 1e-6
        c = 1.5 * sc
        wts = np.sqrt(np.where(np.abs(r) <= c, 1.0, c / np.abs(r)))
    h = sol[:nk]
    # data support per knot
    cnt = np.bincount(j, minlength=nk) + np.bincount(j + 1, minlength=nk)
    resid_mad = 1.4826 * np.median(np.abs(r))
    return knots, h, cnt, resid_mad, dict(n=n, nseg=nseg)


def build_grade_profile(bags: list[str] | None = None, smooth_m: float = 30.0):
    if bags is None:
        bags = list_bags("train")
    knots, h, cnt, mad, info = solve_altitude(bags)
    step = knots[1] - knots[0]
    w = int(round(smooth_m / step)) | 1
    h_s = savgol_filter(h, w, 2)
    grade = savgol_filter(h, w, 2, deriv=1, delta=step)
    prof = pd.DataFrame({"s": knots, "h": h_s, "grade": grade, "support": cnt})
    prof.to_csv(OUT / "grade_profile.csv", index=False, float_format="%.5f")
    return prof, mad, info


@lru_cache(maxsize=1)
def grade_profile() -> pd.DataFrame:
    p = OUT / "grade_profile.csv"
    if not p.exists():
        build_grade_profile()
    return pd.read_csv(p)


def grade_at(s: np.ndarray) -> np.ndarray:
    prof = grade_profile()
    g = np.interp(s, prof["s"].to_numpy(), prof["grade"].to_numpy())
    return np.where(np.isfinite(s), g, np.nan)


def height_at(s: np.ndarray) -> np.ndarray:
    prof = grade_profile()
    return np.interp(s, prof["s"].to_numpy(), prof["h"].to_numpy())


G = 9.80665


def load_full(bag: str, refresh: bool = False) -> pd.DataFrame:
    """Grid + along-track coordinate s, direction, grade felt by the vehicle (gr, +uphill)."""
    p = OUT / "cache" / f"{bag}_full.pkl"
    if p.exists() and not refresh:
        return pd.read_pickle(p)
    df = load_grid(bag)
    attrs = dict(df.attrs)
    df = add_track_coords(df.copy())
    gr = grade_at(df["s"].to_numpy()) * df["dir"].to_numpy()
    # hold the last valid grade through short GNSS gaps (offline convenience only)
    df["gr"] = pd.Series(gr).ffill(limit=100).to_numpy()
    df.attrs.update(attrs)
    df.to_pickle(p)
    return df


def _full_one(b):
    load_full(b, refresh=True)
    return b


if __name__ == "__main__":
    cl = centerline()
    print("centerline length [m]:", cl[-1, 0])
    prof, mad, info = build_grade_profile()
    print("altitude residual MAD [m]:", round(mad, 3), info)
    print(prof.describe())
