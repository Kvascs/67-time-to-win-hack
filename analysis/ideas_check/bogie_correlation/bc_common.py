"""Shared helpers for the bogie front/rear lag test (numpy only).

Column layout of data/npz/<bag>.npz:
  front/rear bogie velocity : [t_bag, t_hdr, speed_kmh]
  gnss master vel           : [t_bag, t_hdr, vx_e, vy_n, vz_u, wz]   (wz is always 0 -> heading rate from vx, vy)
  gnss master fix           : [t_bag, t_hdr, lat, lon, alt, status, cov...]

Scale convention (same as analysis/cross_check and k_observability.py):
  wheel_kmh = 3.6 * 1.00037 * (1 + k) * v_true   ->   k = wheel_kmh / (3.6 * 1.00037 * v_gnss) - 1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\MosTransHack")
NPZ = ROOT / "data" / "npz"
HERE = Path(__file__).resolve().parent
KMH = 3.6 * 1.00037          # km/h per (m/s) at k = 0
BASE = 7.55                  # bogie pivot spacing, m (organisers)

K_FRONT = "vehicle__front_bogie_velocity"
K_REAR = "vehicle__rear_bogie_velocity"
K_VEL = "sensing__gnss__master__vel"
K_FIX = "sensing__gnss__master__fix"


def splits():
    return json.load(open(ROOT / "data" / "splits.json", encoding="utf-8"))


def glitch_free(a: np.ndarray, tol: float = 0.25) -> np.ndarray:
    """Rows whose (t_bag - t_hdr) is within tol of the bag median (drops start-up burst, +-1 s clock episodes)."""
    if len(a) == 0:
        return np.zeros(0, bool)
    off = a[:, 0] - a[:, 1]
    return np.abs(off - np.median(off)) < tol


def mono(t, *cols):
    o = np.argsort(t, kind="stable")
    t = t[o]
    keep = np.r_[True, np.diff(t) > 0]
    return (t[keep],) + tuple(c[o][keep] for c in cols)


def interp_gap(tq, t, v, max_gap=0.35):
    """Linear interpolation; NaN where the bracketing samples are more than max_gap apart or outside."""
    out = np.interp(tq, t, v)
    j = np.searchsorted(t, tq)
    j0 = np.clip(j - 1, 0, len(t) - 1)
    j1 = np.clip(j, 0, len(t) - 1)
    bad = (t[j1] - t[j0] > max_gap) | (tq < t[0]) | (tq > t[-1])
    out = out.astype(float)
    out[bad] = np.nan
    return out


def load_bag(bag: str) -> dict:
    """Joint front/rear series on shared header stamps + GNSS speed/heading-rate at those stamps.

    Returns dict with
      t   : header stamps (s, absolute) of samples present in BOTH front and rear (glitch-free)
      F,R : bogie speeds in km/h
      vg  : GNSS master horizontal Doppler speed (m/s) interpolated to t (NaN across GNSS gaps > 0.35 s)
      ag  : GNSS acceleration (m/s^2, centred difference over +-0.5 s)
      kap : path curvature from GNSS heading rate / speed (1/m, centred over +-1 s), NaN if slow
      n_front, n_rear, n_joint : sample counts
    """
    z = np.load(NPZ / f"{bag}.npz")
    f = np.asarray(z[K_FRONT], float)
    r = np.asarray(z[K_REAR], float)
    f = f[glitch_free(f)]
    r = r[glitch_free(r)]
    tf, vf = mono(f[:, 1], f[:, 2])
    tr, vr = mono(r[:, 1], r[:, 2])
    t, i_f, i_r = np.intersect1d(tf, tr, assume_unique=True, return_indices=True)
    out = {"bag": bag, "t": t, "F": vf[i_f], "R": vr[i_r],
           "n_front": len(tf), "n_rear": len(tr), "n_joint": len(t),
           "tf_all": tf, "vf_all": vf, "tr_all": tr, "vr_all": vr}
    vm = np.asarray(z[K_VEL], float) if K_VEL in z.files else np.zeros((0, 6))
    if len(vm) > 100:
        vm = vm[glitch_free(vm)]
        tv, vx, vy = mono(vm[:, 1], vm[:, 2], vm[:, 3])
        spd = np.hypot(vx, vy)
        psi = np.unwrap(np.arctan2(vy, vx))
        out["tv"], out["spd"] = tv, spd
        out["vg"] = interp_gap(t, tv, spd)
        out["ag"] = centred_slope(t, out["vg"], 0.5)
        psi_t = interp_gap(t, tv, psi)
        rate = centred_slope(t, psi_t, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["kap"] = np.where(out["vg"] > 2.0, rate / out["vg"], np.nan)
    else:
        n = len(t)
        out["vg"] = out["ag"] = out["kap"] = np.full(n, np.nan)
    return out


def centred_slope(t, v, half):
    """(v(t+half) - v(t-half)) / (2 half) with linear interpolation on the sample's own series."""
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.full(len(t), np.nan)
    tt, vv = t[ok], v[ok]
    a = interp_gap(t + half, tt, vv, 0.35)
    b = interp_gap(t - half, tt, vv, 0.35)
    return (a - b) / (2 * half)


def true_k(d: dict) -> dict:
    """Per-bag wheel scale per bogie on straight, steady, moving samples (median of wheel/GNSS - 1)."""
    F, R, vg, ag, kap = d["F"], d["R"], d["vg"], d["ag"], d["kap"]
    with np.errstate(invalid="ignore"):
        M = 0.5 * (F + R)
        m = (vg > 3.0) & (np.abs(ag) < 0.1) & (np.abs(kap) < 1.0 / 1000.0) & (np.abs(F - R) < 0.02 * M)
        rf = F[m] / (KMH * vg[m]) - 1.0
        rr = R[m] / (KMH * vg[m]) - 1.0
        ok = (np.abs(rf) < 0.1) & (np.abs(rr) < 0.1)
    if ok.sum() < 50:
        return {"k_front": np.nan, "k_rear": np.nan, "k_mean": np.nan, "n_k": int(ok.sum())}
    return {"k_front": float(np.median(rf[ok])), "k_rear": float(np.median(rr[ok])),
            "k_mean": float(np.median(0.5 * (rf[ok] + rr[ok]))), "n_k": int(ok.sum())}


def snap_times(t, period=0.1, block=20.0):
    """Stamps -> the underlying regular 10 Hz ticks (header stamps jitter by ~13 ms around a 0.1000 s grid).
    Piecewise (20 s blocks) least-squares fit t = a + b * round((t - t_block) / period); raw stamp kept where the
    fitted tick is more than 40 ms away."""
    out = t.copy()
    i = 0
    n = len(t)
    while i < n:
        j = np.searchsorted(t, t[i] + block)
        seg = t[i:j]
        if len(seg) >= 20:
            k = np.round((seg - seg[0]) / period)
            p = np.polyfit(k, seg, 1)
            r = seg - np.polyval(p, k)
            ok = np.abs(r) < 0.04
            if ok.sum() >= 20:
                p = np.polyfit(k[ok], seg[ok], 1)
                fit = np.polyval(p, k)
                out[i:j] = np.where(np.abs(seg - fit) < 0.04, fit, seg)
        i = max(j, i + 1)
    return out
