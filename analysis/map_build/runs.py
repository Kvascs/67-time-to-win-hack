"""Per-run GNSS preprocessing for map building.

For every bag: master fixes in MAP ENU, matched rover fixes, master->rover baseline (length and
orientation), Doppler velocity, travel heading and quality flags.

Quality model (verified against the final map, train+val):
  * status == 2 (GBAS = RTK): |d| to map median 0.9 cm, p90 0.20 m;  status == 0: median 0.43 m, p90 1.4 m
    (common-mode errors of up to ~10 m seen with a perfect master-rover baseline) -> RTK required for geometry;
  * baseline check  |L - L0| < 0.25 m, L0 = run median of the master-rover distance (~12.43-12.45 m);
  * jump check      |dp - v*dt| < 0.35 m between consecutive fixes (Doppler-predicted displacement);
  * both antennas matched in time (|dt| < 20 ms).
  good   = RTK & baseline & jump  (geometry)
  usable = baseline & jump        (timing / stops, where decimetres do not matter)
Travel heading psi: Doppler heading when speed > 0.5 m/s, else baseline orientation (the tram is
unidirectional -- rover is ahead of master in >99% of moving samples in every run).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import data_io as D

CACHE = D.OUT / 'cache'


@dataclass
class Run:
    bag: str
    th: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    status: np.ndarray
    rx: np.ndarray
    ry: np.ndarray
    rz: np.ndarray
    L: np.ndarray
    psi_b: np.ndarray
    ve: np.ndarray
    vn: np.ndarray
    speed: np.ndarray
    psi: np.ndarray
    good_base: np.ndarray
    good_jump: np.ndarray
    good: np.ndarray
    L0: float
    usable: np.ndarray = field(default=None)
    wheel_f: np.ndarray = field(default=None)
    wheel_r: np.ndarray = field(default=None)

    def __len__(self):
        return len(self.th)


def _match(t_ref, t_other, tol=0.02):
    j = np.clip(np.searchsorted(t_other, t_ref), 1, max(len(t_other) - 1, 1))
    j = np.where(np.abs(t_other[j] - t_ref) < np.abs(t_other[j - 1] - t_ref), j, j - 1)
    ok = np.abs(t_other[j] - t_ref) < tol
    return j, ok


def build_run(bag: str) -> Run | None:
    gm = D.gnss_fix(bag, 'master')
    if len(gm) < 5:
        return None
    gr = D.gnss_fix(bag, 'rover')
    n = len(gm)
    rx = np.full(n, np.nan); ry = np.full(n, np.nan); rz = np.full(n, np.nan)
    if len(gr) > 2:
        j, ok = _match(gm.th, gr.th)
        rx[ok], ry[ok], rz[ok] = gr.x[j[ok]], gr.y[j[ok]], gr.z[j[ok]]
    L = np.hypot(rx - gm.x, ry - gm.y)
    psi_b = np.arctan2(ry - gm.y, rx - gm.x)
    L0 = float(np.nanmedian(L)) if np.isfinite(L).any() else np.nan
    v = D.gnss_vel(bag, 'master')
    # vel header stamps lag the position epoch by ~40 ms (measured); align to fix epochs
    ve, vn = D.interp_vel(v, gm.th + 0.04)
    speed = np.hypot(ve, vn)
    psi_v = np.arctan2(vn, ve)
    psi = np.where(np.isfinite(speed) & (speed > 0.5), psi_v, psi_b)
    good_base = np.isfinite(L) & (np.abs(L - L0) < 0.25)
    # jump check: displacement vs Doppler-predicted displacement (trapezoid)
    dt = np.diff(gm.th)
    vem = 0.5 * (ve[1:] + ve[:-1]); vnm = 0.5 * (vn[1:] + vn[:-1])
    ex = np.diff(gm.x) - vem * dt
    ey = np.diff(gm.y) - vnm * dt
    jump = np.hypot(ex, ey)
    bad_step = ~(jump < 0.35) | (dt > 0.5)
    good_jump = np.ones(n, bool)
    good_jump[1:] &= ~bad_step
    good_jump[:-1] &= ~bad_step
    usable = good_base & good_jump & np.isfinite(psi)
    good = usable & (gm.status == 2)
    w = D.wheels(bag)
    wf = np.interp(gm.t, w['front'][:, 0], w['front'][:, 2]) / 3.6 if len(w['front']) else None
    wr = np.interp(gm.t, w['rear'][:, 0], w['rear'][:, 2]) / 3.6 if len(w['rear']) else None
    return Run(bag, gm.th, gm.t, gm.x, gm.y, gm.z, gm.lat, gm.lon, gm.alt, gm.status, rx, ry, rz, L, psi_b,
               ve, vn, speed, psi, good_base, good_jump, good, L0, usable, wf, wr)


def load_runs(bags: list[str], use_cache: bool = True) -> dict[str, Run]:
    CACHE.mkdir(exist_ok=True)
    out = {}
    for b in bags:
        f = CACHE / f'run_{b}.pkl'
        r = None
        if use_cache and f.exists():
            d = pickle.loads(f.read_bytes())
            r = None if d is None else Run(**d)
        else:
            r = build_run(b)
            f.write_bytes(pickle.dumps(None if r is None else dict(r.__dict__)))
        if r is not None:
            out[b] = r
    return out


if __name__ == '__main__':
    import sys
    bags = D.split('train') + D.split('val') + D.split('short')
    runs = load_runs(bags, use_cache=False)
    for b, r in runs.items():
        mov = r.speed > 0.3
        rev = mov & (np.abs(np.angle(np.exp(1j * (np.arctan2(r.vn, r.ve) - r.psi_b)))) > np.pi / 2) & np.isfinite(r.psi_b)
        print(f'{b}: n={len(r)} good={r.good.mean():.3f} base={r.good_base.mean():.3f} jump={r.good_jump.mean():.3f} '
              f'L0={r.L0:.3f} reverse_moving={rev.sum()}')
