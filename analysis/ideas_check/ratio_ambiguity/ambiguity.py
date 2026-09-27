"""Can the bogie-ratio map predict its own unreliable places (without held-out runs)?

From the TRAIN-only map alone (analysis/validation_maps/ratio_map.csv: per 1 m bin mu, sd of z = y / sigma_v(v),
y = log(v_front/v_rear); rel = the leave-one-out reliability flag) compute for a 200 m window ending at s:

  window samples: 10 Hz at a typical V_TYP = 8 m/s -> every 0.8 m (251 samples), weight w = 1/sd^2 in z units
  (exactly the estimator's likelihood, ratioStep: in z units the per-sample noise sigma_v is already divided out, so the
  typical speed only sets the sample density), likelihood tempered by TAU = 5.39.
  1) Fisher information of a shift g with the free offset b profiled out:
       I(s) = sum_j w_j (mu'(s+r_j) - wmean mu')^2 / TAU,   sigma_crb = I^-1/2
     mu' = central difference on the 1 m grid (raw) and after a Gaussian smoothing (sd 1.5 bins) of mu.
  2) ambiguity chi(s, D) = sum w mu~(s+r) mu~(s+r+D) / sum w mu~(s+r)^2 (mu~ = mu minus its weighted window mean),
       a(s) = max over 2 m <= |D| <= 8 m (0.1 m grid), argmax D kept.
  3) bonus, the noise-free expected log-likelihood profile of the estimator's matcher (candidate sd, free b, log sd):
       sigma_curv (quadratic fit over +-0.5 m, as the estimator), m_exp = expected margin of the peak over any other
       local maximum > 2 m away (the gate wants > 3).
  4) periodicity: chi on integer shifts 2..40 m, histogram of the argmax |D|; periodogram of mu.
Evaluation: 25 m cells of reliab_cells.csv (LOO over train: cells with failures, and the flag widened by +-1 cell),
VAL triggers of reliab_val.csv (false fixes: accepted by the base gate, plausible truth |e_filt| < 3 m,
|e_corr| > |e_filt| + 0.5 m). AUC = Mann-Whitney (higher score = more unreliable).

  python analysis/ideas_check/ratio_ambiguity/ambiguity.py
Output: ambiguity.txt, ambiguity_per_m.csv (next to this script)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAP = ROOT / 'analysis' / 'validation_maps' / 'ratio_map.csv'
RM = ROOT / 'analysis' / 'ideas_check' / 'ratio_map_correction'

V_TYP, RATE, TAU, W = 8.0, 10.0, 5.39, 200.0
DS = V_TYP / RATE
G = np.round(np.arange(-8.0, 8.0001, 0.1), 10)
EXCL, MARGIN, SIGMAX = 2.0, 3.0, 0.8
CELL = 25.0
FAIL = 0.5
PLACES = [(5875, 5976), (9550, 9640), (10250, 10300)]


def load_map():
    L = None
    for line in MAP.read_text(encoding='utf-8').splitlines():
        if line.startswith('# L='):
            L = float(line.split('=')[1])
    m = pd.read_csv(MAP, comment='#')
    return L, m.mu.to_numpy(float), m.sd.to_numpy(float), m.rel.to_numpy().astype(bool)


class Interp:
    """Linear between bin centres, cyclic (as Estimator::ratioStep mapAt)."""

    def __init__(self, L, nb):
        self.L, self.nb = L, nb

    def idx(self, q):
        u = np.mod(q, self.L) * (self.nb / self.L) - 0.5
        fl = np.floor(u)
        w = u - fl
        i0 = fl.astype(np.int64) % self.nb
        return i0, (i0 + 1) % self.nb, w

    @staticmethod
    def take(tab, ix):
        i0, i1, w = ix
        return tab[i0] + w * (tab[i1] - tab[i0])


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, int(n1), int(n0)
    r = rankdata(score)
    A = (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)  # Hanley-McNeil SE
    se = np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0))
    return A, se, int(n1), int(n0)


def compute(L, mu, sd):
    nb = len(mu)
    ip = Interp(L, nb)
    h = L / nb
    # derivatives on the bin grid (per metre): raw central difference and after Gaussian smoothing (sd 1.5 bins)
    dmu = (np.roll(mu, -1) - np.roll(mu, 1)) / (2 * h)
    k = np.arange(-6, 7)
    gk = np.exp(-0.5 * (k / 1.5) ** 2)
    gk /= gk.sum()
    mus = np.convolve(np.r_[mu[-6:], mu, mu[:6]], gk, 'valid')
    dmus = (np.roll(mus, -1) - np.roll(mus, 1)) / (2 * h)
    r = -np.arange(0.0, W + 1e-9, DS)[::-1]  # 251 samples, the last at the window end
    S = np.arange(nb) * h  # evaluation points: window end at each bin start
    far_g = np.abs(G) >= EXCL - 1e-9
    # quadratic fit over +-0.5 m (11 grid points): coefficient of x^2 as a fixed linear filter
    xq = np.round(np.arange(-5, 6) * 0.1, 10)
    pinv = np.linalg.pinv(np.vander(xq, 3))  # rows: x^2, x, 1
    out = {k_: np.full(nb, np.nan) for k_ in ('sigma_crb', 'sigma_crb_s', 'amb', 'amb_arg', 'sigma_curv', 'm_exp',
                                             'bias_exp', 'amb_ext', 'amb_ext_arg', 'amb_lm', 'amb_lm_arg')}
    DE = np.r_[-np.arange(40, 1, -1), np.arange(2, 41)].astype(float)  # integer shifts for periodicity
    C = 96
    for c0 in range(0, nb, C):
        s = S[c0:c0 + C]
        P = s[:, None] + r[None, :]  # (c, J)
        ix0 = ip.idx(P)
        m0 = ip.take(mu, ix0)
        sd0 = ip.take(sd, ix0)
        w0 = 1.0 / sd0 ** 2
        sw0 = w0.sum(1, keepdims=True)
        # 1) Fisher information with the free offset profiled out
        for key, dtab in (('sigma_crb', dmu), ('sigma_crb_s', dmus)):
            d = ip.take(dtab, ix0)
            dm = (w0 * d).sum(1, keepdims=True) / sw0
            I = (w0 * (d - dm) ** 2).sum(1) / TAU
            out[key][c0:c0 + len(s)] = 1.0 / np.sqrt(I)
        # 2) ambiguity chi over the 0.1 m grid
        m0t = m0 - (w0 * m0).sum(1, keepdims=True) / sw0
        e0 = (w0 * m0t * m0t).sum(1)
        Pg = P[:, None, :] + G[None, :, None]  # (c, G, J)
        ixg = ip.idx(Pg)
        mg = ip.take(mu, ixg)
        mgt = mg - (w0[:, None, :] * mg).sum(2, keepdims=True) / sw0[:, :, None]
        chi = (w0[:, None, :] * m0t[:, None, :] * mgt).sum(2) / e0[:, None]
        chf = np.where(far_g[None, :], chi, -np.inf)
        ka = np.argmax(chf, 1)
        out['amb'][c0:c0 + len(s)] = chf[np.arange(len(s)), ka]
        out['amb_arg'][c0:c0 + len(s)] = G[ka]
        # genuine side peaks only: interior local maxima of chi at |D| > 2 m (a(s) above is mostly the main-lobe edge)
        clm = np.zeros_like(chi, bool)
        clm[:, 1:-1] = (chi[:, 1:-1] >= chi[:, :-2]) & (chi[:, 1:-1] >= chi[:, 2:])
        cs = np.where(clm & (np.abs(G) > EXCL + 1e-9)[None, :], chi, -np.inf)
        kl = np.argmax(cs, 1)
        vl = cs[np.arange(len(s)), kl]
        out['amb_lm'][c0:c0 + len(s)] = np.where(np.isfinite(vl), vl, -1.0)
        out['amb_lm_arg'][c0:c0 + len(s)] = np.where(np.isfinite(vl), G[kl], np.nan)
        # 3) expected LL profile of the matcher (noise-free data mu(P) + noise sd0, candidate map at P + g)
        sdg = ip.take(sd, ixg)
        wg = 1.0 / sdg ** 2
        dlt = m0[:, None, :] - mg
        b = (wg * dlt).sum(2, keepdims=True) / wg.sum(2, keepdims=True)
        ll = (-0.5 * (wg * ((dlt - b) ** 2 + sd0[:, None, :] ** 2)).sum(2) - np.log(sdg).sum(2)) / TAU
        kb = np.argmax(ll, 1)
        rows = np.arange(len(s))
        lm = np.ones_like(ll, bool)
        lm[:, 1:-1] = (ll[:, 1:-1] >= ll[:, :-2]) & (ll[:, 1:-1] >= ll[:, 2:])
        lm[:, 0] = ll[:, 0] >= ll[:, 1]
        lm[:, -1] = ll[:, -1] >= ll[:, -2]
        far = np.abs(G[None, :] - G[kb][:, None]) > EXCL + 1e-9
        sec = np.where(lm & far, ll, -np.inf).max(1)
        out['m_exp'][c0:c0 + len(s)] = ll[rows, kb] - sec
        out['bias_exp'][c0:c0 + len(s)] = G[kb]
        okq = (kb >= 5) & (kb <= len(G) - 6)
        kk = np.clip(kb, 5, len(G) - 6)
        win = ll[rows[:, None], kk[:, None] + np.arange(-5, 6)[None, :]]
        a2 = win @ pinv[0]
        sc = np.where((a2 < 0) & okq, 1.0 / np.sqrt(np.maximum(-2 * a2, 1e-30)), np.inf)
        out['sigma_curv'][c0:c0 + len(s)] = sc
        # 4) chi on integer shifts 2..40 m (periodicity)
        Pe = P[:, None, :] + DE[None, :, None]
        me = ip.take(mu, ip.idx(Pe))
        met = me - (w0[:, None, :] * me).sum(2, keepdims=True) / sw0[:, :, None]
        che = (w0[:, None, :] * m0t[:, None, :] * met).sum(2) / e0[:, None]
        ke = np.argmax(che, 1)
        out['amb_ext'][c0:c0 + len(s)] = che[rows, ke]
        out['amb_ext_arg'][c0:c0 + len(s)] = DE[ke]
    out['s'] = S
    return pd.DataFrame(out)


def cyc_at(S, vals, q, L):
    """Value at arbitrary arc q (nearest-lower grid point, cyclic)."""
    h = S[1] - S[0]
    return vals[(np.floor(np.mod(q, L) / h).astype(int)) % len(S)]


def main():
    L, mu, sd, rel = load_map()
    nb = len(mu)
    D = compute(L, mu, sd)
    D['rel'] = rel.astype(int)
    D[['s', 'sigma_crb', 'amb', 'amb_lm', 'amb_lm_arg', 'sigma_crb_s', 'amb_arg', 'sigma_curv', 'm_exp', 'bias_exp', 'amb_ext', 'amb_ext_arg',
       'rel']].to_csv(HERE / 'ambiguity_per_m.csv', index=False, float_format='%.5g')
    lines = []
    P = lines.append
    P(f'TRAIN-only ratio map {MAP.relative_to(ROOT)}: {nb} bins, L {L:.1f} m; window {W:.0f} m, samples every {DS:.1f} m '
      f'(10 Hz at {V_TYP:.0f} m/s), w = 1/sd^2 (z units, as the estimator), tau {TAU}')
    sc, am = D.sigma_crb.to_numpy(), D.amb.to_numpy()
    q = lambda x: ' / '.join(f'{v:.3g}' for v in np.nanpercentile(x, [10, 50, 90]))  # noqa: E731
    P('')
    P('== map-only quantities over the cycle (p10 / p50 / p90)')
    P(f'sigma_crb (raw mu\')      {q(sc)} m;   smoothed mu\' {q(D.sigma_crb_s)} m')
    P(f'sigma_curv (exp. LL fit)  {q(D.sigma_curv[np.isfinite(D.sigma_curv)])} m (inf at {np.mean(~np.isfinite(D.sigma_curv)) * 100:.1f} %)')
    P(f'side-peak a_lm (interior local max of chi at |D|>2 m; -1 = none): {q(D.amb_lm)}; none at {np.mean(D.amb_lm < -0.5) * 100:.1f} %;'
      f' a_lm > 0.5 at {np.mean(D.amb_lm > 0.5) * 100:.1f} %')
    P(f'amb a(s)                  {q(am)};   expected margin m_exp {q(D.m_exp)} (gate wants > {MARGIN})')
    P(f'|expected peak bias|      {q(np.abs(D.bias_exp))} m')
    fa, fb = sc < 0.3, am < 0.8
    P(f'fraction of cycle: sigma_crb < 0.3 m {fa.mean() * 100:.1f} %, a < 0.8 {fb.mean() * 100:.1f} %, both {np.mean(fa & fb) * 100:.1f} %;'
      f' smoothed sigma_crb < 0.3 {np.mean(D.sigma_crb_s < 0.3) * 100:.1f} %; m_exp > 3 {np.mean(D.m_exp > 3) * 100:.1f} %')
    P(f'LOO rel flag: reliable {rel.mean() * 100:.1f} % of bins; among reliable bins a < 0.8 {np.mean(fb[rel]) * 100:.1f} %, '
      f'among unreliable {np.mean(fb[~rel]) * 100:.1f} %; sigma_crb < 0.3: {np.mean(fa[rel]) * 100:.1f} % vs {np.mean(fa[~rel]) * 100:.1f} %')

    # ---- 25 m cells
    cells = pd.read_csv(RM / 'reliab_cells.csv')
    nc = len(cells)
    ci = (D.s.to_numpy() // CELL).astype(int) % nc
    feats = {}
    for name, x, agg in (('a mean', am, 'mean'), ('a max', am, 'max'), ('a_lm max', D.amb_lm.to_numpy(), 'max'), ('sigma_crb median', sc, 'median'),
                         ('sigma_crb max', sc, 'max'), ('sigma_crb_s median', D.sigma_crb_s.to_numpy(), 'median'),
                         ('sigma_curv median', D.sigma_curv.to_numpy(), 'median'),
                         ('-m_exp (min margin)', -D.m_exp.to_numpy(), 'max'), ('-m_exp median', -D.m_exp.to_numpy(), 'median')):
        feats[name] = pd.Series(x).groupby(ci).agg(agg).reindex(range(nc)).to_numpy()
    has = cells.n.to_numpy() > 0
    fail = cells.fails.to_numpy() > 0
    unrel = cells.unreliable.to_numpy().astype(bool)
    P('')
    P(f'== 25 m cells (reliab_cells.csv, LOO over TRAIN): {nc} cells, with triggers {has.sum()}, with failures {fail.sum()}, '
      f'unreliable (widened +-1) {unrel.sum()}')
    P(f'{"score (higher = worse)":24s} | AUC fail-cells (n>0)  | AUC unreliable (n>0)  | AUC unreliable (all)')
    for name, x in feats.items():
        a1 = auc(x[has], fail[has])
        a2 = auc(x[has], unrel[has])
        a3 = auc(x, unrel)
        P(f'{name:24s} | {a1[0]:.3f} +-{a1[1]:.3f} ({a1[2]}/{a1[3]}) | {a2[0]:.3f} +-{a2[1]:.3f} ({a2[2]}/{a2[3]}) | {a3[0]:.3f}')
    fr = cells.fails.to_numpy() / np.maximum(cells.n.to_numpy(), 1)
    for name in ('a mean', 'sigma_crb median', '-m_exp (min margin)'):
        rho = spearmanr(feats[name][has], fr[has]).correlation
        P(f'Spearman({name}, fail rate per cell, n>0): {rho:+.3f}')

    # ---- VAL triggers (train map), false fixes
    V = pd.read_csv(RM / 'reliab_val.csv')
    S = D.s.to_numpy()
    for col, src in (('a', am), ('sig', sc), ('sig_s', D.sigma_crb_s.to_numpy()), ('m_exp', D.m_exp.to_numpy()),
                     ('rel', rel.astype(float)), ('amb_arg', D.amb_arg.to_numpy()), ('a_lm', D.amb_lm.to_numpy())):
        V[col] = cyc_at(S, src, V.s_mod.to_numpy(), L)
    pl = V.e_filt.abs() < 3.0
    B = V[V.base & pl].copy()
    B['false'] = B.e_corr.abs() > B.e_filt.abs() + FAIL
    B['big'] = B.e_corr.abs() > 1.5
    P('')
    P(f'== VAL triggers (reliab_val.csv = sim_correct triggers, train map): {len(V)}, base-accepted with plausible truth {len(B)}, '
      f'false fixes (|e_corr| > |e_filt| + {FAIL}) {int(B["false"].sum())}, |e_corr| > 1.5 m {int(B.big.sum())}')
    P(f'{"score (higher = worse)":30s} | AUC false fix         | AUC |e_corr|>1.5   | Spearman(score, |e_corr|)')
    scores = {'a(s) map-only': B.a, 'a_lm side peak map-only': B.a_lm, 'sigma_crb map-only': B.sig, 'sigma_crb smoothed map-only': B.sig_s,
              '-m_exp map-only': -B.m_exp, 'LOO unreliable flag (1-rel)': 1 - B.rel,
              'observed sig_c (online)': B.sig_c, '-observed margin_lm (online)': -B.margin_lm}
    for name, x in scores.items():
        a1, a2 = auc(x, B['false']), auc(x, B.big)
        rho = spearmanr(x, B.e_corr.abs(), nan_policy='omit').correlation
        P(f'{name:30s} | {a1[0]:.3f} +-{a1[1]:.3f} ({a1[2]}/{a1[3]}) | {a2[0]:.3f} +-{a2[1]:.3f} ({a2[2]}) | {rho:+.3f}')
    rho = spearmanr(B.sig, B.sig_c).correlation
    ratio = np.median(B.sig_c / B.sig * np.sqrt(np.clip(B.n if 'n' in B else 251, 1, None) / 251)) if 'n' in B else np.median(B.sig_c / B.sig)
    P(f'sanity: Spearman(map sigma_crb, observed sig_c) {rho:+.3f}, median observed/map ratio {np.median(B.sig_c / B.sig):.2f} '
      f'(observed windows carry ~237 samples at 1.2/m vs 251 assumed)')
    # same exclusion budget as the LOO map: exclude the worst 34 % of the cycle by each map-only score
    frac = 1 - rel.mean()
    P(f'-- exclusion at the LOO budget ({frac * 100:.0f} % of the cycle excluded): false fixes kept / accepted fixes kept')
    for name, x_all, xv in (('LOO rel flag', (~rel).astype(float), 1 - B.rel), ('a(s)', am, B.a), ('sigma_crb', sc, B.sig),
                            ('-m_exp', -D.m_exp.to_numpy(), -B.m_exp)):
        if name == 'LOO rel flag':
            keep = xv < 0.5
        else:
            thr = np.nanquantile(x_all, 1 - frac)
            keep = xv < thr
        P(f'  {name:14s}: false fixes kept {int((keep & B["false"]).sum()):2d} of {int(B["false"].sum())}, '
          f'|e_corr|>1.5 kept {int((keep & B.big).sum()):2d} of {int(B.big.sum())}, accepted kept {keep.mean() * 100:.0f} %')
    Fx = B[B['false']].sort_values('s_mod')
    P('false fixes (s_mod, e_filt -> e_corr, sig_c, margin_lm | map: a, sigma_crb, m_exp, LOO rel):')
    for _, r_ in Fx.iterrows():
        P(f'  {r_.s_mod:8.1f} {r_.bag}: {r_.e_filt:+.2f} -> {r_.e_corr:+.2f} sig_c {r_.sig_c:.2f} margin {r_.margin_lm:6.1f} | '
          f'a {r_.a:.2f} (argD {r_.amb_arg:+.1f}) sig {r_.sig:.3f} m_exp {r_.m_exp:6.1f} rel {int(r_.rel)}')

    # ---- named places
    P('')
    P('== teammate\'s places (cycle percentile in brackets; high a / sigma, low m_exp = ambiguous)')
    pct = lambda x, v: np.mean(x <= v) * 100  # noqa: E731
    for a0, a1_ in PLACES:
        m = (S >= a0) & (S < a1_)
        P(f'{a0}-{a1_} m: a mean {am[m].mean():.2f} [{pct(am, am[m].mean()):.0f}], max {am[m].max():.2f}; sigma_crb median '
          f'{np.median(sc[m]):.3f} [{pct(sc, np.median(sc[m])):.0f}]; m_exp median {np.median(D.m_exp[m]):.1f} '
          f'[{pct(D.m_exp.to_numpy(), np.median(D.m_exp[m])):.0f}]; LOO rel {rel[m].mean() * 100:.0f} %')

    # ---- periodicity
    P('')
    P('== periodicity of side peaks')
    h1 = np.histogram(np.abs(D.amb_arg), bins=np.arange(2.0, 8.01, 0.5))[0]
    P('argmax |D| of chi within 2..8 m (0.5 m bins from 2): ' + ' '.join(str(v) for v in h1) + '  (2.0 m = main-lobe edge)')
    h2 = np.histogram(np.abs(D.amb_lm_arg.dropna()), bins=np.arange(2.0, 8.01, 0.5))[0]
    P('genuine side-peak |D| within 2..8 m (0.5 m bins): ' + ' '.join(str(v) for v in h2))
    ae = np.abs(D.amb_ext_arg.to_numpy()).astype(int)
    big = D.amb_ext.to_numpy() > 0.5
    hist = np.bincount(ae, minlength=41)[2:41]
    hb = np.bincount(ae[big], minlength=41)[2:41]
    P('argmax |D| of chi within 2..40 m (integer shifts), counts per |D| = 2..40:')
    P('  all bins : ' + ' '.join(f'{d}:{c}' for d, c in zip(range(2, 41), hist)))
    P('  chi > 0.5: ' + ' '.join(f'{d}:{c}' for d, c in zip(range(2, 41), hb)))
    for per in (12.5, 25.0):
        near = np.abs(ae - per) <= 1.0
        near |= np.abs(ae - 2 * per) <= 1.0 if 2 * per <= 40 else False
        P(f'  share of argmax |D| within 1 m of {per} m (or 2x): {near.mean() * 100:.1f} % (uniform over 2..40 m: '
          f'{(6 if per == 12.5 else 3) / 39 * 100:.1f} %)')
    top = np.argsort(hist)[::-1][:5] + 2
    P(f'  most frequent |D|: {", ".join(f"{d} m ({hist[d - 2]})" for d in top)}')
    # global autocorrelation / periodogram of mu (weighted by 1/sd^2 would be dominated by quiet track; plain here)
    x = mu - mu.mean()
    f = np.fft.rfft(x)
    ac = np.fft.irfft(np.abs(f) ** 2, n=len(x))
    ac /= ac[0]
    lags = np.arange(1, 61)
    P('autocorrelation of mu over the cycle at lags 1..40 m: ' + ' '.join(f'{l}:{ac[l]:+.2f}' for l in lags[:40]))
    loc = [l for l in range(3, 60) if ac[l] > ac[l - 1] and ac[l] > ac[l + 1] and ac[l] > 0.02]
    P(f'  local maxima of the autocorrelation (> 0.02) at lags: {loc}')
    per = len(x) / np.arange(1, len(f))
    pw = np.abs(f[1:]) ** 2
    band = (per >= 4) & (per <= 60)
    order = np.argsort(pw[band])[::-1][:8]
    P('  periodogram peaks (period m, share of the 4-60 m band power): ' +
      ', '.join(f'{per[band][i]:.1f} ({pw[band][i] / pw[band].sum() * 100:.1f} %)' for i in order))
    # a vs sigma relation
    P('')
    P(f'Spearman(a, sigma_crb) over the cycle {spearmanr(am, sc).correlation:+.3f}; Spearman(a, m_exp) {spearmanr(am, D.m_exp).correlation:+.3f}')
    txt = '\n'.join(lines)
    print(txt)
    (HERE / 'ambiguity.txt').write_text(txt + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
