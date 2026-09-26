"""GNSS-free global localisation on the known closed track: recursive grid (histogram) filter.

State (static per run, plus a random walk of the odometry error):
    u0     start position in the wheel-distance coordinate u of the main cycle (period LU), grid step du
    kappa  wheel scale error of the run, rows kappa_j (grid step k_step)
Current position of hypothesis (u0_i, kappa_j) after the odometer (integrated estimator speed) reads r:
    u = u0_i + r / (1 + kappa_j)   (mod LU)   ->   map arc length s = s_of_u(u)
Cells never move (start-offset coordinates): every cue is a likelihood multiplied into the cells, and the
odometry random walk is a blur along u0 (equivalent to process noise on the current position). For row j
the current position of cell i is cell i + a_j with a_j = r c_j / du, so every map lookup of a row is a
contiguous slice of a periodic table (cheap, and a plain loop in C++).

Cues (all causal, each applied when it becomes known):
  stop      standstill >= dwell at odometer r:  L(u) = 1 + sum_m p_m N(u; u_m, sd_m^2) / lam
  pass      landmarks passed without a stop:    prod (1 - p_m)      (negative information)
  cutoff    notch >= 4 -> 0 at speed:           L(u) = 1 + sum_c q_c N(u; u_c, sd_c^2) / lam_c
  cutpass   cut-off place passed with notch >= 4 and no cut-off near it: (1 - q_c)
  speed     max speed over the last interval above the train envelope + margin: v_eps
  grade     mean disturbance d over the interval vs the train d-profile, per-cell bias Kalman filter
Confidence: posterior mass of the current position within +-win of its mode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d

from mapmodel import MapModel

SQ2PI = np.sqrt(2 * np.pi)


@dataclass
class GLParams:
    du: float = 0.5                  # grid step of u0, m
    k_min: float = -0.025            # wheel scale error range and step
    k_max: float = 0.025
    k_step: float = 0.001
    k_prior_sd: float = 0.012
    q_x: float = 0.004               # odometry random walk, m^2 per m travelled
    jump_rate: float = 1.0e-4        # odometry jumps (slip, bad bridging) per m travelled
    jump_hw: float = 20.0            # half-width of a jump, m (box kernel)
    floor: float = 1e-9              # uniform re-seeding per update (recovery from a wrong mode)
    checkpoint: float = 50.0         # m of travel between moving updates
    # cue switches
    use_stop: bool = True
    use_pass: bool = True
    use_cut: bool = True
    use_cutpass: bool = True
    use_speed: bool = True
    use_grade: bool = False
    use_start: bool = False          # prior: runs that start at rest start where train runs started
    # stop model
    stop_dwell: float = 1.5
    lm_sd_extra: float = 0.5         # added (in quadrature) to the landmark spread, m
    stop_rate_mult: float = 1.5      # conservative multiplier on the train random-stop rate
    guard: float = 3.0               # a stop within guard of a landmark 'explains' it (no pass penalty)
    p_max: float = 0.9               # cap on probabilities used for negative information
    # cut-off model
    cut_sd_min: float = 1.5
    cut_rate_min: float = 5e-5       # random cut-offs per m (conservative)
    cut_notch: int = 4
    cut_vmin: float = 2.0
    # speed model
    v_margin: float = 1.5            # m/s above the train envelope before a hypothesis is penalised
    v_eps: float = 0.05
    # grade (TERCOM) model
    g_sigma: float = 0.15            # m/s^2, per-checkpoint residual
    g_bias_sd: float = 0.08          # prior sd of the per-hypothesis bias
    g_bias_q: float = 2e-6           # bias random walk, (m/s^2)^2 per m
    g_temper: float = 0.5            # likelihood exponent (residuals are autocorrelated)
    g_vmin: float = 3.0
    # start prior
    start_w: float = 0.5
    start_sd: float = 5.0
    # confidence
    win: float = 10.0                # +-m window around the mode
    p_fix: float = 0.99
    min_cues: int = 2                # positive cues (stops / cut-offs) required before a fix


@dataclass
class Calib:
    lm_s: np.ndarray
    lm_sigma: np.ndarray
    lm_p: np.ndarray
    stop_rate: float
    cut_s: np.ndarray
    cut_sigma: np.ndarray
    cut_q: np.ndarray
    cut_rate: float
    vmax_s: np.ndarray
    vmax: np.ndarray
    dprof_s: np.ndarray
    dprof: np.ndarray
    start_s: np.ndarray

    @staticmethod
    def load(path):
        z = np.load(path)
        return Calib(**{k: (z[k] if z[k].ndim else float(z[k])) for k in Calib.__dataclass_fields__})


class GlobalLocalizer:
    SUB = 10  # sub-cells per cell for the stop / cut-off likelihood tables (du / SUB resolution)

    def __init__(self, mm: MapModel, cal: Calib, p: GLParams = GLParams()):
        self.mm, self.cal, self.p = mm, cal, p
        LU = mm.LU
        self.nx = nx = int(np.round(LU / p.du))
        self.du = du = LU / nx                      # exact period in cells
        self.kap = np.arange(p.k_min, p.k_max + 1e-12, p.k_step)
        self.c = 1.0 / (1.0 + self.kap)
        prior_k = np.exp(-0.5 * (self.kap / p.k_prior_sd) ** 2)
        self.G = np.repeat((prior_k / prior_k.sum())[:, None], nx, axis=1) / nx
        self.bias = np.zeros_like(self.G) if p.use_grade else None
        self.Pb = p.g_bias_sd ** 2
        ucell = np.arange(nx) * du
        # ---- periodic tables, stored over 3 laps so that any row slice [a, a + nx) is contiguous ----
        fu = np.arange(nx * self.SUB) * du / self.SUB
        lm_u = mm.u_of_s(cal.lm_s)
        lm_sd = np.hypot(cal.lm_sigma, p.lm_sd_extra)
        self.stop_tab = self._tile(self._bump(fu, lm_u, lm_sd, cal.lm_p, cal.stop_rate * p.stop_rate_mult))
        cut_sd = np.maximum(cal.cut_sigma, p.cut_sd_min)
        self.cut_u, self.cut_q = mm.u_of_s(cal.cut_s), np.minimum(cal.cut_q, p.p_max)
        self.cut_tab = self._tile(self._bump(fu, self.cut_u, cut_sd, cal.cut_q, max(cal.cut_rate, p.cut_rate_min)))
        # cumulative log(1 - p) of landmarks along u (3 laps), for pass-through
        lp = np.zeros(3 * nx)
        for um, pm in zip(lm_u, np.minimum(cal.lm_p, p.p_max)):
            k = int(np.round(um / du)) % nx
            lp[[k, k + nx, k + 2 * nx]] += np.log1p(-pm)
        self.pass_cum = np.cumsum(lp)
        self.pass_lap = float(self.pass_cum[nx - 1])
        # speed envelope (max over the interval behind the current position) and d profile
        vm = np.interp(mm.s_of_u(ucell), mm.sg, np.interp(mm.sg, cal.vmax_s, cal.vmax))
        n = max(1, int(1.2 * p.checkpoint / du))
        self.v_tab = self._tile(maximum_filter1d(vm, size=n, origin=(n - 1) // 2, mode='wrap'))
        dp = np.interp(mm.wrap_s(mm.s_of_u(ucell)), cal.dprof_s, cal.dprof, period=mm.L)
        n = max(1, int(p.checkpoint / du))
        self.d_tab = self._tile(uniform_filter1d(dp, size=n, origin=(n - 1) // 2, mode='wrap'))
        self.start_mix = None
        if p.use_start:
            su = mm.u_of_s(cal.start_s)
            pr = np.zeros(nx)
            for x in su:
                d = (ucell - x + 0.5 * LU) % LU - 0.5 * LU
                pr += np.exp(-0.5 * (d / p.start_sd) ** 2)
            self.start_mix = p.start_w * pr / pr.sum() + (1 - p.start_w) / nx
        # ---- bookkeeping ----
        self.r_blur = 0.0
        self.var_acc = 0.0
        self.eps_acc = 0.0
        self.r_pass, self.off_pass = 0.0, 0.0       # pass-through counted up to u(r_pass) + off_pass
        self.r_cp = 0.0
        self.n_pos = 0
        self.cut_events = []

    # ------------------------------------------------------------------ tables
    @staticmethod
    def _tile(t):
        return np.concatenate([t, t, t])

    def _bump(self, f, cu, sd, w, lam):
        """1 + sum_i w_i N(u; cu_i, sd_i^2) / lam on a fine periodic grid, averaged over one cell."""
        LU = self.mm.LU
        tab = np.full(len(f), lam)
        for ui, si, wi in zip(cu, sd, w):
            d = (f - ui + 0.5 * LU) % LU - 0.5 * LU
            tab += wi * np.exp(-0.5 * (d / si) ** 2) / (SQ2PI * si)
        return uniform_filter1d(tab, size=self.SUB, mode='wrap') / lam

    # ------------------------------------------------------------------ row geometry
    def _shift(self, r, off=0.0):
        """Per-row shift of the current position (in cells) after odometer r, plus offset off (m)."""
        return (r * self.c + off) / self.du

    def _rows(self, tab, r, fine=False):
        """Table value at the current position of every cell (rows x cells), by row slices."""
        nx = self.nx
        out = np.empty_like(self.G)
        if fine:
            a = np.round(self._shift(r) * self.SUB).astype(np.int64) % (nx * self.SUB)
            for j, aj in enumerate(a):
                out[j] = tab[aj:aj + nx * self.SUB:self.SUB]
        else:
            a = np.round(self._shift(r)).astype(np.int64) % nx
            for j, aj in enumerate(a):
                out[j] = tab[aj:aj + nx]
        return out

    def _norm(self):
        self.G /= self.G.sum()

    # ------------------------------------------------------------------ process noise
    def _blur(self, r, force_jump=False):
        p = self.p
        dist = max(0.0, r - self.r_blur)
        self.r_blur = r
        self.var_acc += p.q_x * dist
        self.eps_acc += p.jump_rate * dist
        if self.var_acc >= self.du ** 2:
            self.G = gaussian_filter1d(self.G, np.sqrt(self.var_acc) / self.du, axis=1, mode='wrap', truncate=3.0)
            self.var_acc = 0.0
        if self.eps_acc > 0 and (force_jump or self.eps_acc > 0.01):
            eps = min(0.5, self.eps_acc)
            w = int(p.jump_hw / self.du) * 2 + 1
            self.G = (1 - eps) * self.G + eps * uniform_filter1d(self.G, size=w, axis=1, mode='wrap')
            self.eps_acc = 0.0
        if p.floor > 0:
            self.G += p.floor * self.G.sum(axis=1, keepdims=True) / self.nx
        if self.bias is not None:
            self.Pb += p.g_bias_q * dist

    # ------------------------------------------------------------------ negative information
    def _pass_through(self, r_to, off_to):
        """Landmarks whose position lies in (u(r_pass)+off_pass, u(r_to)+off_to] were passed without a stop."""
        nx = self.nx
        a = np.round(self._shift(self.r_pass, self.off_pass)).astype(np.int64)
        b = np.round(self._shift(r_to, off_to)).astype(np.int64)
        for j in range(len(self.c)):
            if b[j] <= a[j]:
                continue
            laps = (b[j] - a[j]) // nx
            bj = b[j] - laps * nx
            a0 = a[j] % nx
            b0 = a0 + (bj - a[j])
            # cells i: sum of log(1-p) over landmark cells in (i + a, i + b]
            dl = self.pass_cum[b0:b0 + nx] - self.pass_cum[a0:a0 + nx] + laps * self.pass_lap
            self.G[j] *= np.exp(dl)

    def _cut_pass(self, r0, r1, notch_fn):
        """Cut-off places passed in (r0, r1] with notch >= cut_notch just before and no cut-off near them."""
        p, nx, du = self.p, self.nx, self.du
        a = np.floor(self._shift(r0)).astype(np.int64)
        b = np.floor(self._shift(r1)).astype(np.int64)
        ev = np.array(self.cut_events)
        for uc, q in zip(self.cut_u, self.cut_q):
            uci = uc / du
            for j in range(len(self.c)):
                if b[j] <= a[j]:
                    continue
                k = np.arange(a[j] + 1, b[j] + 1)            # passage when cell index + k hits the place
                cells = np.floor(uci - k).astype(np.int64) % nx
                r_pass = (k * du) / self.c[j]
                high = notch_fn(r_pass - 8.0) >= p.cut_notch
                if len(ev):
                    high &= np.min(np.abs(r_pass[:, None] - ev[None, :]), axis=1) > 15.0
                self.G[j, cells[high]] *= (1.0 - q)

    # ------------------------------------------------------------------ cues
    def apply_start_prior(self):
        """The run begins at rest: runs that start at rest start where train runs started (optional)."""
        if self.start_mix is not None:
            self.G *= self.start_mix[None, :] * self.nx
            self._norm()
            self.start_mix = None

    def on_stop(self, r):
        p = self.p
        self._blur(r, force_jump=True)
        if p.use_stop:
            if p.use_pass:
                self._pass_through(r, -p.guard)
            self.G *= self._rows(self.stop_tab, r, fine=True)
            self.n_pos += 1
        self.r_pass, self.off_pass = r, p.guard
        self._norm()

    def on_cutoff(self, r):
        self.cut_events.append(r)
        if not self.p.use_cut:
            return
        self._blur(r)
        self.G *= self._rows(self.cut_tab, r, fine=True)
        self.n_pos += 1
        self._norm()

    def on_checkpoint(self, r, vmax_obs, d_obs, notch_fn):
        """Moving update every `checkpoint` m: pass-through, cut-off passes, speed envelope, grade."""
        p = self.p
        self._blur(r)
        if p.use_stop and p.use_pass:
            self._pass_through(r, -p.guard)
            self.r_pass, self.off_pass = r, -p.guard
        if p.use_cut and p.use_cutpass:
            self._cut_pass(self.r_cp, r, notch_fn)
        if p.use_speed and np.isfinite(vmax_obs):
            vt = self._rows(self.v_tab, r)
            self.G[vmax_obs > vt + p.v_margin] *= p.v_eps
        if p.use_grade and np.isfinite(d_obs):
            nu = d_obs - self._rows(self.d_tab, r) - self.bias
            S = self.Pb + p.g_sigma ** 2
            self.G *= np.exp(-0.5 * p.g_temper * nu * nu / S)
            self.bias += (self.Pb / S) * nu
            self.Pb -= self.Pb ** 2 / S
        self.r_cp = r
        self._norm()

    # ------------------------------------------------------------------ estimate
    def estimate(self, r, want_kappa=True):
        """Marginal of the current position: mode window, confidence mass within +-win, mean, kappa."""
        p, nx, du, LU = self.p, self.nx, self.du, self.mm.LU
        a = np.round(self._shift(r)).astype(np.int64) % nx
        H2 = np.zeros(2 * nx)
        for j, aj in enumerate(a):
            H2[aj:aj + nx] += self.G[j]
        H = H2[:nx] + H2[nx:]
        H /= H.sum()
        w = int(round(p.win / du))
        cs = np.r_[0.0, np.cumsum(np.r_[H, H[:2 * w + 1]])]
        win_mass = cs[2 * w + 1:2 * w + 1 + nx] - cs[:nx]            # mass of cells [i, i + 2w]
        i0 = int(np.argmax(win_mass))
        conf = float(win_mass[i0])
        idx = (i0 + np.arange(2 * w + 1)) % nx
        pw = H[idx]
        u_mean = ((i0 + np.sum(pw * np.arange(2 * w + 1)) / max(pw.sum(), 1e-300)) * du) % LU
        k_mean = np.nan
        if want_kappa:
            kw = np.zeros(len(self.kap))
            G2 = np.concatenate([self.G, self.G], axis=1)
            for j, aj in enumerate(a):
                lo = (i0 - aj) % nx
                kw[j] = G2[j, lo:lo + 2 * w + 1].sum()
            k_mean = float(np.sum(kw * self.kap) / max(kw.sum(), 1e-300))
        return dict(conf=conf, u=float(u_mean), s=float(self.mm.wrap_s(self.mm.s_of_u(u_mean))), kappa=k_mean,
                    n_pos=self.n_pos)
