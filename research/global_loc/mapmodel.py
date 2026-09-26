"""Map-side quantities for GNSS-free global localisation on the closed main cycle.

Wheel-distance coordinate u: the estimator without a map anchor cannot apply the curve correction of the
bogie speeds, so its odometer runs on the raw wheel distance. The core corrects a reading by
(1 + rho'(s)), rho' = min(a|c|, sat) + b c at each bogie (c = map curvature), hence the odometer measures
    ds_rel = (1 + kappa) ds / (1 + rho'(s))      (kappa = wheel scale error of the run)
and u(s) = int_0^s ds' / (1 + rho'(s')) turns the map into the odometer's own metric:
    s_rel(t) = (1 + kappa) [u(s(t)) - u(s(0))].
Everything in the filter lives in u; u -> s only for output.
"""
from __future__ import annotations

import numpy as np

import common as C

# curve correction of the core (params.hpp): wheel_curv_abs, wheel_curv_signed, wheel_curv_sat
CURV_ABS, CURV_SIGNED, CURV_SAT = 0.415, -0.054, 0.009
FRONT_BOGIE, REAR_BOGIE = 9.9, 2.35          # ahead of antenna 1, m
BODY_REAR, BODY_FRONT = -2.1, 14.4           # car body extent relative to antenna 1, m


class MapModel:
    def __init__(self, m: C.TrackMap | None = None, ds: float = 0.5):
        m = m or C.load_map()
        self.m = m
        self.L = m.length
        self.ds = ds
        sg = np.arange(0.0, self.L, ds)
        self.sg = sg

        def corr(sq):
            c = m.interp(m.curv, sq)
            return np.minimum(CURV_ABS * np.abs(c), CURV_SAT) + CURV_SIGNED * c

        rho = 0.5 * (corr(sg + FRONT_BOGIE) + corr(sg + REAR_BOGIE))
        du = ds / (1.0 + rho)
        self.ug = np.r_[0.0, np.cumsum(du)[:-1]]     # u at sg
        self.LU = float(np.sum(du))                  # cycle length in u
        # grade averaged over the car body (as the core does for its grade term)
        offs = np.linspace(BODY_REAR, BODY_FRONT, 5)
        self.grade_body = np.mean([m.interp(m.grade, sg + o) for o in offs], axis=0)

    # s <-> u (periodic)
    def u_of_s(self, s):
        s = np.asarray(s, float)
        n = np.floor(s / self.L)
        sw = s - n * self.L
        return n * self.LU + np.interp(sw, np.r_[self.sg, self.L], np.r_[self.ug, self.LU])

    def s_of_u(self, u):
        u = np.asarray(u, float)
        n = np.floor(u / self.LU)
        uw = u - n * self.LU
        return n * self.L + np.interp(uw, np.r_[self.ug, self.LU], np.r_[self.sg, self.L])

    def wrap_u(self, u):
        return np.mod(u, self.LU)

    def wrap_s(self, s):
        return np.mod(s, self.L)

    # map profiles sampled on a regular u grid (for fast per-cell lookup)
    def profile_on_u(self, values_on_sg: np.ndarray, du: float):
        ugrid = np.arange(0.0, self.LU, du)
        sq = self.s_of_u(ugrid)
        return ugrid, np.interp(sq, np.r_[self.sg, self.L], np.r_[values_on_sg, values_on_sg[0]])
