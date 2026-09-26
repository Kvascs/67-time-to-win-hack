"""Shared helpers for the independent cross-checks (numpy only, no dependency on other agents' code).

Column layout of data/npz/<bag>.npz (rows sorted by bag receive time):
  wheels / cmd : [t_bag, t_hdr, value]      (wheel value in km/h)
  fix          : [t_bag, t_hdr, lat, lon, alt, status, cov_xx, cov_yy, cov_zz]
  vel          : [t_bag, t_hdr, vx_e, vy_n, vz_u, wz]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\MosTransHack")
NPZ = ROOT / "data" / "npz"
OUT = ROOT / "analysis" / "cross_check"

KEYS = {
    "front": "vehicle__front_bogie_velocity",
    "rear": "vehicle__rear_bogie_velocity",
    "cmd": "vehicle__driver_position_cmd",
    "fixm": "sensing__gnss__master__fix",
    "fixr": "sensing__gnss__rover__fix",
    "velm": "sensing__gnss__master__vel",
    "velr": "sensing__gnss__rover__vel",
}

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def splits():
    return json.load(open(ROOT / "data" / "splits.json", encoding="utf-8"))


def bag_meta():
    """bag -> dict(vehicle, date (Moscow local), split, t0, dur)."""
    import datetime as dt
    s = splits()
    sp = {}
    for k in ("train", "val", "no_gnss_long", "short"):
        for b in s[k]:
            sp[b] = k
    out = {}
    for r in s["info"]:
        t0 = r["t0"]
        loc = dt.datetime.fromtimestamp(t0, dt.timezone(dt.timedelta(hours=3)))
        out[r["bag"]] = dict(vehicle=r["vehicle"], date=loc.strftime("%Y-%m-%d"), hhmm=loc.strftime("%H:%M"),
                             split=sp.get(r["bag"], "?"), t0=t0, dur=r["dur"])
    return out


def load(bag: str) -> dict:
    z = np.load(NPZ / f"{bag}.npz")
    d = {}
    for short, key in KEYS.items():
        a = np.asarray(z[key], float) if key in z.files else np.zeros((0, 3))
        d[short] = a
    return d


def ecef(lat, lon, alt):
    lat = np.radians(np.asarray(lat, float))
    lon = np.radians(np.asarray(lon, float))
    alt = np.asarray(alt, float)
    s, c = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    return np.stack([(n + alt) * c * np.cos(lon), (n + alt) * c * np.sin(lon), (n * (1 - WGS84_E2) + alt) * s], -1)


def enu(lat, lon, alt, lat0, lon0, alt0):
    """Exact WGS84 ENU of points relative to (lat0, lon0, alt0)."""
    p = ecef(lat, lon, alt) - ecef(lat0, lon0, alt0)
    la, lo = np.radians(lat0), np.radians(lon0)
    R = np.array([[-np.sin(lo), np.cos(lo), 0.0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return p @ R.T


def glitch_free(a: np.ndarray, tol: float = 0.25) -> np.ndarray:
    """Mask of rows whose (t_bag - t_hdr) latency is within tol of the bag-wide median
    (drops the start-up burst and +-1 s header-clock episodes)."""
    if len(a) == 0:
        return np.zeros(0, bool)
    off = a[:, 0] - a[:, 1]
    return np.abs(off - np.median(off)) < tol


def mono(t: np.ndarray, *cols):
    """Sort by t, drop duplicate / non-increasing stamps."""
    o = np.argsort(t, kind="stable")
    t = t[o]
    keep = np.r_[True, np.diff(t) > 0]
    return (t[keep],) + tuple(c[o][keep] for c in cols)


def interp_gap(tq, t, v, max_gap=0.35):
    """Linear interpolation; NaN where the bracketing samples are more than max_gap apart."""
    out = np.interp(tq, t, v)
    j = np.searchsorted(t, tq)
    j0 = np.clip(j - 1, 0, len(t) - 1)
    j1 = np.clip(j, 0, len(t) - 1)
    bad = (t[j1] - t[j0] > max_gap) | (tq < t[0]) | (tq > t[-1])
    out[bad] = np.nan
    return out


def nearest(tq, t_sorted):
    """(index, |dt|) of the nearest stamp in t_sorted for every tq."""
    j = np.searchsorted(t_sorted, tq)
    j0 = np.clip(j - 1, 0, len(t_sorted) - 1)
    j1 = np.clip(j, 0, len(t_sorted) - 1)
    d0 = np.abs(tq - t_sorted[j0])
    d1 = np.abs(t_sorted[j1] - tq)
    idx = np.where(d1 < d0, j1, j0)
    return idx, np.minimum(d0, d1)
