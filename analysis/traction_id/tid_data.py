"""Data preparation for traction/braking identification.

Builds, per bag, a uniform 20 Hz grid on the *header-stamp* time base with
    notch        driver controller position (zero-order hold, causal)
    vf, vr       front / rear bogie speed [m/s] (raw km/h divided by a per-bag
                 calibration factor k_f / k_r fitted against GNSS speed)
    vg           GNSS horizontal speed [m/s] (mean of master/rover when both valid)
    ag           zero-phase smoothed GNSS acceleration [m/s^2] (identification target)
    aw           zero-phase smoothed wheel acceleration (mean of bogies)
    x, y, alt    GNSS position (UTM 37N) and altitude (master antenna)
    clean        True where GNSS is valid and both bogies agree with GNSS
The header stamps of wheels, cmd and GNSS are mutually consistent (cross-correlation
lag wheel-vs-GNSS ~0 +-0.03 s), so they are used as the common time base.

Usage:  from tid_data import load_grid, list_bags
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(r"C:\MosTransHack")
NPZ = ROOT / "data" / "npz"
OUT = ROOT / "analysis" / "traction_id"
CACHE = OUT / "cache"
DT = 0.05  # grid step [s]

KMH = 3.6


@lru_cache(maxsize=1)
def splits() -> dict:
    return json.loads((ROOT / "data" / "splits.json").read_text())


def list_bags(split: str) -> list[str]:
    return list(splits()[split])


def vehicle_of(bag: str) -> str:
    return bag.split("_")[0]


def bag_t0(bag: str) -> float:
    for i in splits()["info"]:
        if i["bag"] == bag:
            return float(i["t0"])
    raise KeyError(bag)


def vehicle_group(bag: str) -> str:
    """30618, 30639a (early date ~1777.9e6 s) or 30639b (late date ~1787.7e6 s)."""
    v = vehicle_of(bag)
    if v == "30639":
        return "30639a" if bag_t0(bag) < 1.78e9 else "30639b"
    return v


@lru_cache(maxsize=1)
def _utm():
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)


def to_utm(lat: np.ndarray, lon: np.ndarray):
    return _utm().transform(lon, lat)


def _zoh(t_src: np.ndarray, x_src: np.ndarray, t: np.ndarray, fill=0.0) -> np.ndarray:
    """Causal zero-order hold: value of the last sample with t_src <= t."""
    idx = np.searchsorted(t_src, t, side="right") - 1
    out = np.where(idx >= 0, x_src[np.clip(idx, 0, None)], fill)
    return out


def _age(t_src: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Time since last sample (inf before the first) - causal gap indicator."""
    idx = np.searchsorted(t_src, t, side="right") - 1
    return np.where(idx >= 0, t - t_src[np.clip(idx, 0, None)], np.inf)


def _gap(t_src: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Length of the sampling interval that contains t (non-causal, for masking)."""
    if len(t_src) < 2:
        return np.full_like(t, np.inf)
    idx = np.clip(np.searchsorted(t_src, t, side="right") - 1, 0, len(t_src) - 2)
    g = t_src[idx + 1] - t_src[idx]
    g = np.where((t < t_src[0]) | (t > t_src[-1]), np.inf, g)
    return g


def _dilate(mask_bad: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return mask_bad
    k = np.ones(2 * n + 1)
    return np.convolve(mask_bad.astype(float), k, mode="same") > 0


def _sg_deriv(x: np.ndarray, win_s: float, dt: float = DT) -> np.ndarray:
    w = int(round(win_s / dt)) | 1
    return savgol_filter(x, w, 2, deriv=1, delta=dt, mode="interp")


def _sg_smooth(x: np.ndarray, win_s: float, dt: float = DT) -> np.ndarray:
    w = int(round(win_s / dt)) | 1
    return savgol_filter(x, w, 2, deriv=0, mode="interp")


def _sorted_unique(t: np.ndarray, *cols):
    o = np.argsort(t, kind="stable")
    t = t[o]
    keep = np.r_[True, np.diff(t) > 1e-6]
    return (t[keep],) + tuple(c[o][keep] for c in cols)


def build_grid(bag: str, dt: float = DT, acc_win: float = 1.0) -> pd.DataFrame:
    d = np.load(NPZ / f"{bag}.npz")
    cmd = d["vehicle__driver_position_cmd"]
    fr = d["vehicle__front_bogie_velocity"]
    rr = d["vehicle__rear_bogie_velocity"]
    t_c, n_c = _sorted_unique(cmd[:, 1], cmd[:, 2])
    t_f, v_f = _sorted_unique(fr[:, 1], fr[:, 2])
    t_r, v_r = _sorted_unique(rr[:, 1], rr[:, 2])
    tb_f = fr[:, 0]
    # skip the start-up burst of buffered messages (header far behind bag time)
    t_start = max(t_c[0], t_f[0], t_r[0])
    lat_f = fr[:, 0] - fr[:, 1]
    burst = np.where(lat_f > 0.5)[0]
    if len(burst):
        t_start = max(t_start, fr[burst[-1], 1] + 0.1)
    t_end = min(t_c[-1], t_f[-1], t_r[-1])
    t = np.arange(np.ceil(t_start / dt) * dt, t_end, dt)
    df = pd.DataFrame({"t": t})
    df["notch"] = _zoh(t_c, n_c, t).astype(np.int8)
    df["vf_raw"] = np.interp(t, t_f, v_f)
    df["vr_raw"] = np.interp(t, t_r, v_r)
    df["gap_f"] = _gap(t_f, t)
    df["gap_r"] = _gap(t_r, t)
    df["gap_c"] = _gap(t_c, t)

    # --- GNSS -----------------------------------------------------------------
    have_g = False
    vg_list = []
    for ant in ("master", "rover"):
        key = f"sensing__gnss__{ant}__vel"
        if key in d.files and len(d[key]) > 10:
            g = d[key]
            tg, vx, vy = _sorted_unique(g[:, 1], g[:, 2], g[:, 3])
            sp = np.hypot(vx, vy)
            v = np.interp(t, tg, sp)
            gap = _gap(tg, t)
            v[gap > 0.35] = np.nan
            df[f"vg_{ant[0]}"] = v
            vg_list.append(v)
            have_g = True
        else:
            df[f"vg_{ant[0]}"] = np.nan
    if have_g:
        vm, vro = df["vg_m"].to_numpy(), df["vg_r"].to_numpy()
        both = np.isfinite(vm) & np.isfinite(vro)
        vg = np.where(both, 0.5 * (vm + vro), np.where(np.isfinite(vm), vm, vro))
        disagree = both & (np.abs(vm - vro) > 0.15)
        vg[disagree] = np.nan
        df["vg"] = vg
    else:
        df["vg"] = np.nan

    for ant in ("master", "rover"):
        key = f"sensing__gnss__{ant}__fix"
        a = ant[0]
        if key in d.files and len(d[key]) > 10:
            f = d[key]
            tf, lat, lon, alt, st = _sorted_unique(f[:, 1], f[:, 2], f[:, 3], f[:, 4], f[:, 5])
            ok = np.isfinite(lat) & (np.abs(lat) > 1)
            tf, lat, lon, alt, st = tf[ok], lat[ok], lon[ok], alt[ok], st[ok]
            x, y = to_utm(lat, lon)
            gap = _gap(tf, t)
            for name, arr in (("x", x), ("y", y), ("alt", alt), ("st", st)):
                col = np.interp(t, tf, arr)
                col[gap > 0.35] = np.nan
                df[f"{name}_{a}"] = col
        else:
            for name in ("x", "y", "alt", "st"):
                df[f"{name}_{a}"] = np.nan

    # --- per-bag wheel calibration against GNSS (offline) ----------------------
    vg = df["vg"].to_numpy()
    k_f = k_r = np.nan
    if np.isfinite(vg).sum() > 200:
        ag_tmp = np.full_like(vg, np.nan)
        fin = np.isfinite(vg)
        ag_tmp[fin] = np.gradient(vg[fin], dt)
        sel = fin & (vg > 3.0)
        if sel.sum() > 200:
            rf = df["vf_raw"].to_numpy()[sel] / vg[sel]
            rr_ = df["vr_raw"].to_numpy()[sel] / vg[sel]
            # robust: median of ratios in the central band
            k_f = float(np.median(rf[(rf > 3.3) & (rf < 3.9)])) if ((rf > 3.3) & (rf < 3.9)).sum() > 100 else np.nan
            k_r = float(np.median(rr_[(rr_ > 3.3) & (rr_ < 3.9)])) if ((rr_ > 3.3) & (rr_ < 3.9)).sum() > 100 else np.nan
    kf_use = k_f if np.isfinite(k_f) else KMH
    kr_use = k_r if np.isfinite(k_r) else KMH
    df["vf"] = df["vf_raw"] / kf_use
    df["vr"] = df["vr_raw"] / kr_use
    df.attrs.update(bag=bag, k_f=k_f, k_r=k_r, vehicle=vehicle_of(bag), group=vehicle_group(bag), dt=dt)

    # --- clean mask ------------------------------------------------------------
    vf, vr = df["vf"].to_numpy(), df["vr"].to_numpy()
    wheel_ok = (df["gap_f"].to_numpy() < 0.25) & (df["gap_r"].to_numpy() < 0.25) & (df["gap_c"].to_numpy() < 0.25)
    if have_g:
        tol = 0.12 + 0.02 * np.nan_to_num(vg)
        g_ok = np.isfinite(vg)
        agree = g_ok & (np.abs(vf - vg) < tol) & (np.abs(vr - vg) < tol)
        bad = ~(agree & wheel_ok)
        clean = ~_dilate(bad, int(round(1.0 / dt)))
    else:
        clean = np.zeros(len(df), bool)
    df["wheel_ok"] = wheel_ok
    df["clean"] = clean

    # --- accelerations (zero-phase, offline) -----------------------------------
    vw = 0.5 * (vf + vr)
    df["vw"] = vw
    df["aw"] = _sg_deriv(vw, acc_win, dt)
    if have_g:
        # fill short NaN runs for smoothing, then re-mask
        s = pd.Series(vg).interpolate(limit=10, limit_area="inside")
        v_fill = s.to_numpy()
        fin = np.isfinite(v_fill)
        ag = np.full(len(df), np.nan)
        vgs = np.full(len(df), np.nan)
        # process contiguous finite runs
        idx = np.flatnonzero(fin)
        if len(idx):
            breaks = np.flatnonzero(np.diff(idx) > 1)
            starts = np.r_[idx[0], idx[breaks + 1]]
            ends = np.r_[idx[breaks], idx[-1]] + 1
            w = int(round(acc_win / dt)) | 1
            for s0, e0 in zip(starts, ends):
                if e0 - s0 > w + 2:
                    ag[s0:e0] = _sg_deriv(v_fill[s0:e0], acc_win, dt)
                    vgs[s0:e0] = _sg_smooth(v_fill[s0:e0], acc_win, dt)
        df["ag"] = ag
        df["vgs"] = vgs
    else:
        df["ag"] = np.nan
        df["vgs"] = np.nan
    df["tr"] = df["t"] - df["t"].iloc[0]
    return df


def load_grid(bag: str, refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{bag}.pkl"
    if p.exists() and not refresh:
        return pd.read_pickle(p)
    df = build_grid(bag)
    df.to_pickle(p)
    return df


def build_all(bags: list[str], refresh: bool = False, workers: int = 6):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for b, info in zip(bags, ex.map(_build_one, bags, [refresh] * len(bags))):
            print(b, info, flush=True)


def _build_one(bag: str, refresh: bool = False):
    df = load_grid(bag, refresh=refresh)
    return dict(n=len(df), k_f=round(df.attrs["k_f"], 4) if np.isfinite(df.attrs["k_f"]) else None,
                k_r=round(df.attrs["k_r"], 4) if np.isfinite(df.attrs["k_r"]) else None,
                clean=round(float(df["clean"].mean()), 3))


if __name__ == "__main__":
    import sys
    refresh = "--refresh" in sys.argv
    bags = list_bags("train") + list_bags("val")
    build_all(bags, refresh=refresh)
