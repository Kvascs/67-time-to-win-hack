"""Core helpers for traction-model fitting / evaluation.

BagArrays : per-bag numpy arrays on the 20 Hz header-time grid
matched_kernel : FIR kernel K such that  SG-derivative(v) == K * a  (the reference acceleration
                 ag is a zero-phase Savitzky-Golay derivative of GNSS speed; model accelerations are
                 passed through the same K before being compared with ag -> no smoothing bias)
lag1 : causal first-order lag (exact discretisation), numba
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numba as nb
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from route_map import G, load_full
from tid_data import DT, vehicle_group
from tid_pool import auto_flag, time_since_change

ACC_WIN = 1.0  # SG window used for ag in tid_data


@lru_cache(maxsize=4)
def matched_kernel(win_s: float = ACC_WIN, dt: float = DT) -> np.ndarray:
    w = int(round(win_s / dt)) | 1
    n = 4 * w + 1
    imp = np.zeros(n)
    imp[n // 2] = 1.0
    v = np.cumsum(imp) * dt  # speed response to a unit acceleration impulse
    k = savgol_filter(v, w, 2, deriv=1, delta=dt, mode="interp")
    k = k[n // 2 - w // 2 - 1: n // 2 + w // 2 + 2]
    return k / k.sum()


def apply_matched(x: np.ndarray, win_s: float = ACC_WIN) -> np.ndarray:
    """Zero-phase smoothing of a model acceleration so it is comparable with ag (offline only)."""
    k = matched_kernel(win_s)
    if x.ndim == 1:
        return np.convolve(x, k[::-1], mode="same")
    from scipy.ndimage import convolve1d
    return convolve1d(x, k[::-1], axis=0, mode="nearest")


@nb.njit(cache=True)
def lag1(x, alpha, y0=0.0):
    y = np.empty_like(x)
    acc = y0
    for i in range(x.shape[0]):
        acc = acc + alpha * (x[i] - acc)
        y[i] = acc
    return y


@nb.njit(cache=True)
def lag1_asym(x, a_up, a_dn, y0=0.0):
    y = np.empty_like(x)
    acc = y0
    for i in range(x.shape[0]):
        al = a_up if x[i] > acc else a_dn
        acc = acc + al * (x[i] - acc)
        y[i] = acc
    return y


def delay_int(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return x.copy()
    return np.r_[np.full(k, x[0]), x[:-k]]


@dataclass
class BagArrays:
    bag: str
    group: str
    t: np.ndarray
    u: np.ndarray        # notch (int)
    v: np.ndarray        # reference speed (GNSS, NaN where invalid)
    a: np.ndarray        # reference acceleration ag (SG 1 s)
    vw: np.ndarray       # wheel speed (calibrated, mean of bogies)
    gr: np.ndarray       # grade felt (+ uphill), NaN if unknown
    s: np.ndarray        # along-track position [m]
    dirn: np.ndarray     # +1 east / -1 west
    clean: np.ndarray
    auto: np.ndarray
    tsc: np.ndarray
    fit: np.ndarray      # samples usable for fitting / scoring accelerations

    @property
    def n(self):
        return len(self.t)


def bag_arrays(bag: str) -> BagArrays:
    df = load_full(bag)
    t = df["t"].to_numpy()
    u = df["notch"].to_numpy().astype(np.int64)
    v = df["vgs"].to_numpy()  # SG-smoothed GNSS speed (same window) - for state / speed reference
    a = df["ag"].to_numpy()
    vw = df["vw"].to_numpy()
    gr = df["gr"].to_numpy()
    tsc, _, _ = time_since_change(t, u)
    a_flag = np.where(np.isfinite(a), a, df["aw"].to_numpy())
    auto = auto_flag(t, u, a_flag, vw)
    clean = df["clean"].to_numpy()
    fit = clean & ~auto & np.isfinite(a) & np.isfinite(v) & np.isfinite(gr)
    return BagArrays(bag, vehicle_group(bag), t, u, v, a, vw, gr, df["s"].to_numpy(),
                     df["dir"].to_numpy(), clean, auto, tsc, fit)


_CACHE: dict[str, BagArrays] = {}


def get_arrays(bag: str) -> BagArrays:
    if bag not in _CACHE:
        _CACHE[bag] = bag_arrays(bag)
    return _CACHE[bag]
