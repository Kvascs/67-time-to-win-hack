"""Shared helpers: ratio map y = log(v_front/v_rear) per 1 m of master-antenna arc, local matcher.

Reuses the previous study (analysis/ideas_check/curvature_signature: common.py, regress.py, samples.pkl).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / 'curvature_signature'
sys.path.insert(0, str(PREV))
import common as C  # noqa: E402
import regress as R  # noqa: E402

ROOT = C.ROOT
VALMAP = ROOT / 'analysis' / 'validation_maps' / 'track_map.csv'
REPLAY_VAL = ROOT / 'build_core' / 'replay_tmp' / 'bl_h_dly'
REPLAY_CHK = ROOT / 'build_core' / 'replay_tmp' / 'checker'
TAU = 5.39            # integrated autocorrelation factor of y at 10 Hz (regress_v1.5)
VMIN = 1.5            # both bogies faster than this [m/s]
VGRID = np.arange(1.5, 16.01, 0.5)
# estimator flag bits that make a wheel sample unusable (slip/slide/dropout/stuck/invalid, bits 0..10)
BAD_FLAGS = (1 << 11) - 1


def load_valmap():
    m = pd.read_csv(VALMAP, comment='#')
    return C.Polyline(m.x.values, m.y.values, m.s.values, m.curvature.values, cyclic=True)


def sigma_v_table(S):
    """Straight-track std of y by speed (0.5 m/s classes) -> (centres, sigma)."""
    st = S[S.kmax < 0.003]
    sv = np.array([st.y[(st.v >= a) & (st.v < a + 0.5)].std() if ((st.v >= a) & (st.v < a + 0.5)).sum() > 200 else np.nan
                   for a in VGRID])
    ok = np.isfinite(sv)
    return VGRID[ok] + 0.25, sv[ok]


def sigma_v(v, tab):
    return np.interp(v, tab[0], tab[1])


class RatioMap:
    """Per 1 m bin (bin i covers arc [i, i+1)) mean/std of z = y / sigma_v(v); linear interpolation between
    bin centres, cyclic."""

    def __init__(self, n, s1, s2, L, sv_tab, min_n=10, sd_floor=0.5, smooth=0):
        self.L = L
        self.nb = len(n)
        self.sv_tab = sv_tab
        n = n.astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = s1 / n
            var = (s2 - n * mu * mu) / np.maximum(n - 1, 1)
        sd = np.sqrt(np.maximum(var, 0))
        few = (n < min_n) | ~np.isfinite(mu) | ~np.isfinite(sd)
        mu[few] = 0.0
        sd[few] = 1.0
        if smooth:
            k = np.array([0.25, 0.5, 0.25])
            mu = np.convolve(np.r_[mu[-1], mu, mu[0]], k, 'valid')
        self.mu = mu
        self.sd = np.maximum(sd, sd_floor)
        self.few = few
        self.n = n
        # cyclic interpolation tables at bin centres
        self.xc = np.r_[-0.5, np.arange(self.nb) + 0.5, self.nb + 0.5]
        self.muc = np.r_[mu[-1], mu, mu[0]]
        self.sdc = np.r_[self.sd[-1], self.sd, self.sd[0]]

    def at(self, s):
        u = np.mod(s, self.L) * (self.nb / self.L)
        return np.interp(u, self.xc, self.muc), np.interp(u, self.xc, self.sdc)


def bin_sums(s, z, nb, L):
    b = np.floor(np.mod(s, L) * (nb / L)).astype(int) % nb
    return (np.bincount(b, minlength=nb).astype(float), np.bincount(b, weights=z, minlength=nb),
            np.bincount(b, weights=z * z, minlength=nb))


def match(rmap, y, sv, rel, s_now, grid, b_fixed=None, tau=TAU):
    """Log-likelihood of the window over candidate corrections grid (K) of the current position s_now.

    y: ratio samples (offset NOT removed if b_fixed is None: a free offset is fitted per candidate),
    sv: sigma_v(v) per sample, rel: sample arc relative to now (<= 0).
    Model y_i = b + sv_i * (mu_z(p) + sd_z(p) * eps)."""
    P = s_now + rel[None, :] + grid[:, None]
    mu, sd = rmap.at(P)
    u = y / sv
    c = 1.0 / sv
    w = 1.0 / (sd * sd)
    if b_fixed is None:
        b = np.sum(w * c * (u - mu), 1) / np.sum(w * c * c, 1)
    else:
        b = np.full(len(grid), b_fixed)
    r = u[None, :] - b[:, None] * c[None, :] - mu
    ll = np.sum(-0.5 * w * r * r - np.log(sd), 1) / tau
    return ll, b


def peak_stats(grid, ll, excl=2.0):
    """argmax (parabola refined), curvature sigma, posterior sigma, margin to the best LL farther than excl m."""
    k = int(np.argmax(ll))
    step = grid[1] - grid[0]
    d_hat = grid[k]
    if 0 < k < len(grid) - 1:
        a, b0, c = ll[k - 1], ll[k], ll[k + 1]
        den = a - 2 * b0 + c
        if den < 0:
            d_hat = grid[k] + 0.5 * step * (a - c) / den
    # curvature over +-0.5 m
    m = np.abs(grid - grid[k]) <= 0.5 + 1e-9
    if m.sum() >= 3:
        cf = np.polyfit(grid[m] - grid[k], ll[m], 2)
        curv = -2 * cf[0]
    else:
        curv = np.nan
    sig_c = 1 / np.sqrt(curv) if curv > 0 else np.inf
    p = np.exp(ll - ll[k])
    p /= p.sum()
    mean = np.sum(p * grid)
    sig_p = np.sqrt(np.sum(p * (grid - mean) ** 2))
    far = np.abs(grid - grid[k]) > excl
    margin = ll[k] - ll[far].max() if far.any() else np.inf
    # second-best local maximum (grid ends count as maxima)
    lm = np.r_[ll[0] >= ll[1], (ll[1:-1] >= ll[:-2]) & (ll[1:-1] >= ll[2:]), ll[-1] >= ll[-2]]
    lmf = lm & far
    margin_lm = ll[k] - ll[lmf].max() if lmf.any() else np.inf
    edge = k == 0 or k == len(grid) - 1
    return d_hat, sig_c, sig_p, margin, margin_lm, edge
