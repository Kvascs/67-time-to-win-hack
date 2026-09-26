"""Pooled identification dataset (train / val) with notch-history features and data flags.

Flags
  clean    : GNSS valid and both bogies agree with GNSS (from tid_data)
  auto     : "notch not in control" episodes (offline flag): a constant notch held > 3 s while the
             measured acceleration wanders (std > 0.12 or range > 0.6 m/s^2 after the first 2 s).
             Typical signature: handle parked at -8 (or -15 / 0) while the tram is driven by
             automation - acceleration from -1.7 to +1.4 m/s^2 with no notch change.
  settled  : notch unchanged for >= 2.5 s (for steady-state maps)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from route_map import G, load_full
from tid_data import OUT, list_bags, vehicle_group

COLS = ["bag", "group", "t", "tr", "notch", "tsc", "prev_notch", "vg", "vgs", "ag", "aw", "vw", "vf", "vr",
        "gr", "s", "dir", "clean", "auto", "settled", "wheel_ok"]


def time_since_change(t: np.ndarray, n: np.ndarray):
    ch = np.r_[True, np.diff(n) != 0]
    grp = np.cumsum(ch)
    first = pd.Series(t).groupby(grp).transform("first").to_numpy()
    tsc = t - first
    starts = np.flatnonzero(ch)
    prev = np.zeros(len(n), dtype=np.int16)
    prev_vals = np.r_[n[0], n[starts[1:] - 1]]
    prev = prev_vals[grp - 1]
    return tsc, prev, grp


def auto_flag(t, n, a, v, min_hold=3.0, skip=2.0, std_thr=0.12, rng_thr=0.6, pad=1.0):
    """Offline flag of constant-notch holds whose acceleration is not explained by a constant command."""
    flag = np.zeros(len(n), bool)
    ch = np.r_[True, np.diff(n) != 0]
    st = np.flatnonzero(ch)
    en = np.r_[st[1:], len(n)]
    dt = np.median(np.diff(t))
    k_skip = int(round(skip / dt))
    for s0, e0 in zip(st, en):
        if t[e0 - 1] - t[s0] < min_hold:
            continue
        i2 = s0 + k_skip
        if e0 - i2 < 10:
            continue
        aa = a[i2:e0]
        vv = v[i2:e0]
        aa = aa[np.isfinite(aa) & (vv > 0.3)]
        if len(aa) < 10:
            continue
        if aa.std() > std_thr or (aa.max() - aa.min()) > rng_thr:
            p = int(round(pad / dt))
            flag[max(0, s0 - p):min(len(n), e0 + p)] = True
    return flag


def build_pool(split: str, refresh: bool = False) -> pd.DataFrame:
    p = OUT / "cache" / f"pool_{split}.pkl"
    if p.exists() and not refresh:
        return pd.read_pickle(p)
    out = []
    for b in list_bags(split):
        df = load_full(b)
        t = df["t"].to_numpy()
        n = df["notch"].to_numpy().astype(np.int16)
        tsc, prev, _ = time_since_change(t, n)
        df["tsc"] = tsc
        df["prev_notch"] = prev
        a_for_flag = np.where(np.isfinite(df["ag"].to_numpy()), df["ag"].to_numpy(), df["aw"].to_numpy())
        df["auto"] = auto_flag(t, n, a_for_flag, df["vw"].to_numpy())
        df["settled"] = tsc >= 2.5
        df["bag"] = b
        df["group"] = vehicle_group(b)
        out.append(df[COLS])
    P = pd.concat(out, ignore_index=True)
    P["bag"] = P["bag"].astype("category")
    P["group"] = P["group"].astype("category")
    P.to_pickle(p)
    return P


if __name__ == "__main__":
    for sp in ("train", "val"):
        P = build_pool(sp, refresh=True)
        mv = P.vg > 0.3
        print(sp, len(P), "clean %.3f" % P.clean.mean(), "auto(moving) %.4f" % P.auto[mv].mean(),
              "auto seconds %.0f" % (P.auto.sum() * 0.05))
