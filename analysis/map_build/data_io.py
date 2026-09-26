"""Loading of the extracted .npz bags and conversion of GNSS to metric frames.

Frames used throughout map_build:
  * MAP ENU  : local tangent plane (east, north, up) at MAP_ORIGIN (fixed, see below). Horizontal
               distances in this plane equal true ground distances to <1e-6 over the route, so
               arc length s along the map is directly comparable with wheel odometry.
  * UTM 37N  : EPSG:32637 (grid distances are k~0.99972 x true; grid north is rotated by the
               meridian convergence ~-1.28 deg w.r.t. true north here).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import geo

ROOT = Path(r'C:/MosTransHack')
NPZ = ROOT / 'data' / 'npz'
SPLITS = json.loads((ROOT / 'data' / 'splits.json').read_text())
OUT = ROOT / 'analysis' / 'map_build'

# fixed map origin (rounded centre of the GNSS bounding box of all runs)
MAP_ORIGIN = (55.8028, 37.4240, 160.0)  # lat [deg], lon [deg], ellipsoidal/receiver height [m]
UTM_ZONE = 37
EPSG_UTM = 32637


def split(name: str) -> list[str]:
    return list(SPLITS[name])


def vehicle(bag: str) -> str:
    return bag.split('_')[0]


@dataclass
class Gnss:
    t: np.ndarray       # bag receive time [s]
    th: np.ndarray      # header stamp [s]
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    status: np.ndarray
    x: np.ndarray       # MAP ENU east [m]
    y: np.ndarray       # MAP ENU north [m]
    z: np.ndarray       # MAP ENU up [m]

    def __len__(self):
        return len(self.t)

    def subset(self, m):
        return Gnss(*(getattr(self, f)[m] for f in ('t', 'th', 'lat', 'lon', 'alt', 'status', 'x', 'y', 'z')))


@lru_cache(maxsize=256)
def load_npz(bag: str) -> dict:
    z = np.load(NPZ / f'{bag}.npz')
    return {k: z[k] for k in z.files}


def gnss_fix(bag: str, ant: str = 'master', status_ok: tuple = (0, 2)) -> Gnss:
    a = load_npz(bag).get(f'sensing__gnss__{ant}__fix')
    if a is None or len(a) == 0:
        e = np.zeros(0)
        return Gnss(e, e, e, e, e, e, e, e, e)
    a = a[np.isin(a[:, 5], status_ok)]
    # drop exact duplicate consecutive messages (same header stamp)
    keep = np.r_[True, np.diff(a[:, 1]) > 1e-6]
    a = a[keep]
    x, y, z = geo.geodetic_to_enu(a[:, 2], a[:, 3], a[:, 4], *MAP_ORIGIN)
    return Gnss(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], x, y, z)


def gnss_vel(bag: str, ant: str = 'master') -> np.ndarray:
    """[t_bag, t_header, ve, vn, vu, wz] (ENU m/s as published)."""
    a = load_npz(bag).get(f'sensing__gnss__{ant}__vel')
    return a if a is not None else np.zeros((0, 6))


def wheels(bag: str) -> dict:
    z = load_npz(bag)
    return {'front': z['vehicle__front_bogie_velocity'], 'rear': z['vehicle__rear_bogie_velocity'],
            'cmd': z['vehicle__driver_position_cmd']}


def interp_vel(v: np.ndarray, t: np.ndarray, col_t: int = 1, max_gap: float = 0.3):
    """Linear interpolation of GNSS vel (ve, vn) to times t (header stamps). NaN where data gap > max_gap."""
    if len(v) < 2:
        return np.full(len(t), np.nan), np.full(len(t), np.nan)
    tv = v[:, col_t]
    o = np.argsort(tv)
    tv = tv[o]
    ve = np.interp(t, tv, v[o, 2])
    vn = np.interp(t, tv, v[o, 3])
    i = np.clip(np.searchsorted(tv, t), 1, len(tv) - 1)
    gap = np.minimum(np.abs(tv[i] - t), np.abs(tv[i - 1] - t))
    bad = gap > max_gap
    ve[bad] = np.nan
    vn[bad] = np.nan
    return ve, vn
