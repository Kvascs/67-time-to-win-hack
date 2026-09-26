"""Step 3: can the bogie speed ratio tell the west_arrival_2 stub from the main line after the
divergence (main s = 5390, see stub_geometry.py)?

u = arc of the FRONT pivot past the divergence point (u = p_antenna + 9.873).
Hypotheses: H_main (path = main map) and H_stub (path = mean RTK trace of the 4 stub runs).
Per-sample model y ~ N(mu_H(u), sigma_H(u)^2):
  kappa-model : mu = M6 regression (kf, kr, |kf|, |kr|, kf^2, kr^2; train, v>1.5) on the path kappa
  chord-model : mu = beta * rigid-car chord prediction on the path geometry (beta from train fit)
  empirical   : mu, sigma = per-2 m-bin mean / pooled std of the OTHER runs of that class (leave-one-out)
  sigma_H     : residual std of the train fit by (max |kappa| at the pivots, speed) class, OR a common sigma
LLR(D) = (1/tau) * sum over samples with 0 <= u <= D of [log N(y; H_main) - log N(y; H_stub)]
(> 0 favours main), tau = integrated residual autocorrelation factor from step 2.
Output: stub_test.txt, stub_llr.csv, fig_stub_test.png
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

import common as C
import regress as R

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

S_DIV = 5390.0
VMIN = 1.0
DS = (20.0, 50.0, 100.0)
_lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def noise_table():
    """Residual std of the M7 fit (train, v>1.5) by kmax class and speed class."""
    S, _ = R.load('train', 1.5)
    S = R.add_features(S)
    cols = ['kf', 'kr', 'akf', 'akr', 'kf2', 'kr2', 'pred', 'a', 'a_v']
    _, _, _, res = R.ols_cluster(S, cols)
    S['res'] = res
    kb = np.array([0, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 1.0])
    vb = np.array([0, 3, 5, 8, 100.0])
    tab = np.full((len(kb) - 1, len(vb) - 1), np.nan)
    for i in range(len(kb) - 1):
        for j in range(len(vb) - 1):
            m = (S.kmax >= kb[i]) & (S.kmax < kb[i + 1]) & (S.v >= vb[j]) & (S.v < vb[j + 1])
            if m.sum() >= 50:
                tab[i, j] = S.res[m].std()
    # fill gaps by the column / row neighbours
    for j in range(tab.shape[1]):
        col = tab[:, j]
        ok = np.isfinite(col)
        tab[:, j] = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok])
    return kb, vb, tab


def sigma_of(kmax, v, kb, vb, tab):
    i = np.clip(np.searchsorted(kb, kmax, side='right') - 1, 0, tab.shape[0] - 1)
    j = np.clip(np.searchsorted(vb, np.maximum(v, 1.5), side='right') - 1, 0, tab.shape[1] - 1)
    return tab[i, j]


def lognorm(y, mu, sig):
    return -0.5 * ((y - mu) / sig) ** 2 - np.log(sig)


def main():
    pl = C.load_main()
    origin = C.map_origin()
    mj = json.load(open(C.HERE / 'model_v1.5.json'))
    tau = mj['tau']
    b6 = np.array(mj['models']['M6 M3+kf^2,kr^2']['beta'])
    b_chord = mj['models']['M5 chord model']['beta'][0]
    say(f'=== Step 3: stub vs main after divergence at main s = {S_DIV} ===')
    say(f'kappa-model M6 beta (kf,kr,|kf|,|kr|,kf2,kr2) = {np.round(b6, 4).tolist()}; chord beta = {b_chord:.3f}; tau = {tau:.2f}')
    kb, vb, tab = noise_table()
    say('noise table sigma [%] (rows kmax bins ' + str(kb.tolist()) + ', cols v bins ' + str(vb.tolist()) + '):')
    for i in range(tab.shape[0]):
        say('   ' + ' '.join(f'{100 * x:5.2f}' for x in tab[i]))
    # ---- paths ----
    sp = pd.read_csv(C.HERE / 'stub_path.csv')     # p relative to the divergence point (p<0: main map)
    stub_pl = C.Polyline(sp.x.values, sp.y.values, sp.p.values, sp.kappa.values, cyclic=False)

    def kappa_path(H, p):
        if H == 'main':
            return pl.k_at(S_DIV + p)
        return np.where(p < 0, pl.k_at(S_DIV + np.minimum(p, 0)), stub_pl.k_at(p))

    def chord_path(H, p):
        if H == 'main':
            return C.chord_pred(pl, S_DIV + p)
        return C.chord_pred(stub_pl, p)

    def kmodel(H, p):
        kf, kr = kappa_path(H, p + C.D_FRONT), kappa_path(H, p + C.D_REAR)
        X = np.c_[kf, kr, np.abs(kf), np.abs(kr), kf ** 2, kr ** 2]
        return X @ b6, np.maximum(np.abs(kf), np.abs(kr))

    # ---- runs ----
    S = pd.read_pickle(C.HERE / 'samples.pkl')
    with np.errstate(divide='ignore', invalid='ignore'):
        S['lr'] = np.log(S.vf / S.vr)
    st = (S.kf.abs() < 0.003) & (S.kr.abs() < 0.003) & (S.v > 3) & np.isfinite(S.lr)
    off = S[st].groupby('bag').lr.median()
    hits = pd.read_csv(C.HERE / 'stub_hits.csv')
    stub_bags = list(hits.bag)
    runs = []
    for bag, g in S.groupby('bag'):
        g = g.reset_index(drop=True)
        if bag in stub_bags:
            d = C.load_bag(bag)
            fx = C.master_fix_enu(d, origin)
            fx = fx[fx.status == 2].drop_duplicates('t').reset_index(drop=True)
            p_fix, _, dist = stub_pl.project(fx.x.values, fx.y.values)
            k = (dist < 1.5) & (p_fix > -140) & (p_fix < stub_pl.s[-1] - 1)
            tq = g.t.values - C.FIX_LEAD_S
            tf, pf = fx.t.values[k], p_fix[k]
            j = np.clip(np.searchsorted(tf, tq), 1, len(tf) - 1)
            okj = (tq >= tf[0]) & (tq <= tf[-1]) & ((tf[j] - tf[j - 1]) < 0.5)
            p = np.where(okj, np.interp(tq, tf, pf), np.nan)
            cls = 'stub'
        else:
            sm = g.sm.values
            if np.sum((sm > S_DIV) & (sm < S_DIV + 140)) < 30:
                continue
            p = np.where((sm > S_DIV - 150) & (sm < S_DIV + 250), sm - S_DIV, np.nan)
            cls = 'main'
        u = p + C.D_FRONT
        y = g.lr.values - off.get(bag, np.nanmedian(g.lr.values))
        m = np.isfinite(u) & np.isfinite(y) & (g.vf.values > VMIN) & (g.vr.values > VMIN) & ~g.bad_ep.values
        m &= np.abs(y) < 0.05
        # keep one pass: the samples of the first contiguous pass through the divergence
        if not m.any():
            continue
        df = pd.DataFrame({'bag': bag, 'cls': cls, 'split': g.split.values[0], 't': g.t.values[m], 'p': p[m], 'u': u[m],
                           'y': y[m], 'v': g.v.values[m], 'a': g.a.values[m]})
        runs.append(df)
    A = pd.concat(runs, ignore_index=True)
    cov = A[(A.u >= 0) & (A.u <= 100)].groupby(['cls', 'bag']).agg(n=('u', 'size'), umax=('u', 'max'),
                                                                     vmed=('v', 'median'))
    say(f'runs with samples at 0<=u<=100: main {int((cov.reset_index().cls == "main").sum())}, '
        f'stub {int((cov.reset_index().cls == "stub").sum())}')
    say(cov.round(2).to_string())
    # ---- model predictions for every sample, both hypotheses ----
    for H in ('main', 'stub'):
        mu, kmx = kmodel(H, A.p.values)
        A[f'mu_k_{H}'] = mu
        A[f'kmax_{H}'] = kmx
        A[f'sig_{H}'] = sigma_of(kmx, A.v.values, kb, vb, tab)
        A[f'mu_c_{H}'] = b_chord * chord_path(H, A.p.values)
    A['sig_eq'] = np.sqrt(0.5 * (A.sig_main ** 2 + A.sig_stub ** 2))
    # ---- empirical templates (2 m bins of u), leave-one-run-out ----
    bins = np.arange(-20, 130.01, 2.0)
    A['bin'] = np.digitize(A.u, bins)
    per = A.groupby(['cls', 'bag', 'bin']).y.agg(['mean', 'count', 'var']).reset_index()

    def template(cls, exclude):
        q = per[(per.cls == cls) & (per.bag != exclude)]
        mu = q.groupby('bin')['mean'].mean()
        # pooled per-sample variance within bin (all samples of other runs of that class)
        w = A[(A.cls == cls) & (A.bag != exclude)]
        sd = w.groupby('bin').y.std()
        return mu, sd

    llr_rows = []
    for bag, g in A.groupby('bag'):
        cls = g.cls.iloc[0]
        tm_main, sd_main = template('main', bag)
        tm_stub, sd_stub = template('stub', bag)
        for D in DS:
            w = g[(g.u >= 0) & (g.u <= D)]
            if len(w) < 5:
                continue
            y = w.y.values
            r = dict(bag=bag, cls=cls, split=w.split.iloc[0], D=D, n=len(w), u_cov=float(w.u.max() - w.u.min()),
                     v_med=float(w.v.median()))
            # kappa-model, equal sigma (mean profile only)
            r['llr_kmean'] = float(np.sum(lognorm(y, w.mu_k_main, w.sig_eq) - lognorm(y, w.mu_k_stub, w.sig_eq)) / tau)
            # kappa-model with class-dependent sigma (mean + roughness)
            r['llr_kfull'] = float(np.sum(lognorm(y, w.mu_k_main, w.sig_main) - lognorm(y, w.mu_k_stub, w.sig_stub)) / tau)
            # chord model, equal sigma
            r['llr_chord'] = float(np.sum(lognorm(y, w.mu_c_main, w.sig_eq) - lognorm(y, w.mu_c_stub, w.sig_eq)) / tau)
            # roughness only (both means zero, class-dependent sigma)
            r['llr_rough'] = float(np.sum(lognorm(y, 0, w.sig_main) - lognorm(y, 0, w.sig_stub)) / tau)
            # empirical templates (leave-one-out); bins without template -> skipped
            mm, sm_ = tm_main.reindex(w.bin).values, sd_main.reindex(w.bin).values
            ms, ss_ = tm_stub.reindex(w.bin).values, sd_stub.reindex(w.bin).values
            ok = np.isfinite(mm) & np.isfinite(ms) & np.isfinite(sm_) & np.isfinite(ss_)
            sm_ = np.maximum(sm_, 0.001)
            ss_ = np.maximum(ss_, 0.001)
            r['llr_emp'] = float(np.sum(lognorm(y[ok], mm[ok], sm_[ok]) - lognorm(y[ok], ms[ok], ss_[ok])) / tau)
            r['emp_n'] = int(ok.sum())
            # expected LLR if the model were exact (KL divergences), for the samples of this run
            mu_m, mu_s, s_m, s_s = w.mu_k_main.values, w.mu_k_stub.values, w.sig_main.values, w.sig_stub.values
            kl_ms = np.log(s_s / s_m) + (s_m ** 2 + (mu_m - mu_s) ** 2) / (2 * s_s ** 2) - 0.5
            kl_sm = np.log(s_m / s_s) + (s_s ** 2 + (mu_m - mu_s) ** 2) / (2 * s_m ** 2) - 0.5
            r['exp_llr_if_main'] = float(np.sum(kl_ms) / tau)
            r['exp_llr_if_stub'] = float(-np.sum(kl_sm) / tau)
            se = w.sig_eq.values
            r['exp_llr_kmean_if_main'] = float(np.sum((mu_m - mu_s) ** 2 / (2 * se ** 2)) / tau)
            r['rms_y'] = float(np.sqrt(np.mean(y ** 2)))
            llr_rows.append(r)
    Lr = pd.DataFrame(llr_rows)
    Lr.to_csv(C.HERE / 'stub_llr.csv', index=False)
    say('--- LLR = log p(ratio | main) - log p(ratio | stub), > 0 favours main; tau-corrected ---')
    for D in DS:
        q = Lr[Lr.D == D]
        say(f'D = {D:.0f} m of front-pivot travel after the divergence: runs main {int((q.cls == "main").sum())}, stub {int((q.cls == "stub").sum())}')
        for col, nm in (('llr_kmean', 'kappa-model mean only'), ('llr_kfull', 'kappa-model mean+sigma'),
                        ('llr_chord', 'chord model mean'), ('llr_rough', 'roughness only'),
                        ('llr_emp', 'empirical templates LOO')):
            qm, qs = q[q.cls == 'main'][col], q[q.cls == 'stub'][col]
            acc_m = np.mean(qm > 0) if len(qm) else np.nan
            acc_s = np.mean(qs < 0) if len(qs) else np.nan
            dec_m = np.mean(qm > np.log(20)) if len(qm) else np.nan
            dec_s = np.mean(qs < -np.log(20)) if len(qs) else np.nan
            say(f'   {nm:26s} main: median {qm.median():+7.2f} [min {qm.min():+7.2f}] correct {acc_m:.2f} decisive(>ln20) {dec_m:.2f} | '
                f'stub: ' + ' '.join(f'{x:+7.2f}' for x in qs) + f'  correct {acc_s:.2f} decisive {dec_s:.2f}')
        say(f'   expected LLR if model exact: main runs median {q[q.cls == "main"].exp_llr_if_main.median():+.2f}, '
            f'stub runs median {q[q.cls == "stub"].exp_llr_if_stub.median():+.2f}; mean-only part (main) '
            f'{q[q.cls == "main"].exp_llr_kmean_if_main.median():.3f}')
        say(f'   rms y: main median {100 * q[q.cls == "main"].rms_y.median():.2f}%, stub ' +
            ' '.join(f'{100 * x:.2f}%' for x in q[q.cls == 'stub'].rms_y))
    # ---- figure ----
    ug = np.arange(-10, 121, 0.5)
    fig, ax = plt.subplots(4, 1, figsize=(15, 17), sharex=True)
    for H, c in (('main', 'k'), ('stub', 'r')):
        ax[0].plot(ug, kappa_path(H, ug - C.D_FRONT + C.D_FRONT - C.D_FRONT + 0) if False else kappa_path(H, ug), c + '-',
                   label=f'kappa {H} path at arc u after divergence')
        mu, _ = kmodel(H, ug - C.D_FRONT)
        ax[1].plot(ug, 100 * mu, c + '-', lw=2, label=f'kappa-model mean, {H}')
        ax[1].plot(ug, 100 * b_chord * chord_path(H, ug - C.D_FRONT), c + '--', label=f'chord model, {H}')
        kmx = np.maximum(np.abs(kappa_path(H, ug)), np.abs(kappa_path(H, ug - (C.D_FRONT - C.D_REAR))))
        ax[1].fill_between(ug, -100 * sigma_of(kmx, np.full(len(ug), 3.0), kb, vb, tab),
                           100 * sigma_of(kmx, np.full(len(ug), 3.0), kb, vb, tab), color=c, alpha=0.08)
    ax[0].set_ylabel('kappa [1/m]'); ax[0].legend(); ax[0].grid()
    ax[0].set_title('path curvature vs arc after the divergence (main s 5390); stub = mean RTK trace of 4 stub runs')
    ax[1].set_ylabel('predicted y [%]'); ax[1].legend(); ax[1].grid()
    ax[1].set_title('predicted log(vf/vr) vs FRONT-pivot arc u; shaded = +-1 sigma noise (v=3 m/s)')
    for cls, c, axi in (('main', 'k', ax[2]), ('stub', 'r', ax[3])):
        for bag, g in A[A.cls == cls].groupby('bag'):
            axi.plot(g.u, 100 * g.y, '.', ms=2, alpha=0.5)
        axi.set_ylim(-4, 4); axi.grid(); axi.set_ylabel('measured y [%]')
        axi.set_title(f'measured y = log(vf/vr) - bag offset, {cls} runs ({A[A.cls == cls].bag.nunique()}), v>{VMIN} m/s')
    ax[3].set_xlabel('u = front-pivot arc after divergence [m]')
    ax[3].set_xlim(-10, 120)
    plt.tight_layout()
    plt.savefig(C.HERE / 'fig_stub_test.png', dpi=70)
    (C.HERE / 'stub_test.txt').write_text('\n'.join(_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
