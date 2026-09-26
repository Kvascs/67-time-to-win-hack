"""Step 2: regress y = log(v_front/v_rear) (minus per-bag offset) on curvature at the pivots.

Selection (train bags): RTK arc s valid, both bogies > VMIN, no known wheel anomaly episode (+-5 s),
arc rate consistent with wheel speed (|ds/dt / v - 1| < 0.1), no obvious slip (|y| < 5 %).
Bag fixed effects (within transform); cluster-robust (by bag) standard errors.
Also: noise level and autocorrelation, ROC of sharp-curve detection from the ratio alone.

Usage: python regress.py [vmin]      (default vmin 3.0 m/s)
Output: regress_<vmin>.txt (all numbers), roc_<vmin>.csv, fig_regress_<vmin>.png, model_<vmin>.json
"""
from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np
import pandas as pd

import common as C

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

VMIN = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
SLIP = 0.05
K_CURVE = 0.02
OUT = C.HERE / f'regress_v{VMIN:g}.txt'
_lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def load(split='train', vmin=VMIN):
    S = pd.read_pickle(C.HERE / 'samples.pkl')
    S = S[S.split == split].copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        S['lr'] = np.log(S.vf / S.vr)
    n0 = len(S)
    base = np.isfinite(S.kf) & np.isfinite(S.pred) & np.isfinite(S.a) & np.isfinite(S.lr)
    mv = base & (S.vf > vmin) & (S.vr > vmin)
    ep = mv & ~S.bad_ep
    cons = ep & (np.abs(S.dsdt / S.v - 1) < 0.1)
    S = S[cons].copy()
    # per-bag offset: median on straight track (both pivots |k| < 0.003), fallback all samples
    st = (S.kf.abs() < 0.003) & (S.kr.abs() < 0.003)
    off = S[st].groupby('bag').lr.median()
    off_all = S.groupby('bag').lr.median()
    S['off'] = S.bag.map(off).fillna(S.bag.map(off_all))
    S['y'] = S.lr - S.off
    slip = np.abs(S.y) >= SLIP
    info = dict(n_all=n0, n_s_valid=int(base.sum()), n_moving=int(mv.sum()), n_no_episode=int(ep.sum()),
                n_consistent=int(cons.sum()), n_slip_removed=int(slip.sum()), n_used=int((~slip).sum()),
                n_bags=int(S.bag.nunique()))
    S = S[~slip].copy()
    return S, info


def add_features(S):
    S['akf'] = S.kf.abs()
    S['akr'] = S.kr.abs()
    S['q'] = S.kf ** 2 - S.kr ** 2
    S['kf2'] = S.kf ** 2
    S['kr2'] = S.kr ** 2
    S['a_v'] = S.a / S.v
    S['kmax'] = np.maximum(S.akf, S.akr)
    return S


def within(S, cols):
    g = S.groupby('bag')
    X = S[cols].values - g[cols].transform('mean').values
    y = S.y.values - g.y.transform('mean').values
    return X, y


def ols_cluster(S, cols):
    X, y = within(S, cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - np.sum(res ** 2) / np.sum(y ** 2)
    # cluster-robust covariance (by bag)
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((len(cols), len(cols)))
    for b, idx in S.groupby('bag').indices.items():
        u = X[idx].T @ res[idx]
        meat += np.outer(u, u)
    G = S.bag.nunique()
    cov = XtX_inv @ meat @ XtX_inv * G / (G - 1)
    se = np.sqrt(np.diag(cov))
    return beta, se, r2, res


def acf(res_by_bag, maxlag=30):
    num = np.zeros(maxlag + 1)
    den = 0.0
    cnt = np.zeros(maxlag + 1)
    for r in res_by_bag:
        r = r - r.mean()
        den += np.sum(r * r)
        for k in range(maxlag + 1):
            if len(r) > k:
                num[k] += np.sum(r[:len(r) - k] * r[k:])
                cnt[k] += len(r) - k
    n_tot = sum(len(r) for r in res_by_bag)
    return (num / cnt) / (den / n_tot)


def roc(score, label):
    """ROC points and AUC (label bool)."""
    o = np.argsort(-score)
    lab = label[o]
    tp = np.cumsum(lab)
    fp = np.cumsum(~lab)
    tpr = tp / max(lab.sum(), 1)
    fpr = fp / max((~lab).sum(), 1)
    auc = np.trapezoid(np.r_[0, tpr], np.r_[0, fpr])
    return fpr, tpr, auc


def tpr_at(fpr, tpr, f):
    i = np.searchsorted(fpr, f, side='right') - 1
    return float(tpr[max(i, 0)])


def rolling_stat(S, col, half_s, fn):
    """Centred rolling statistic over +-half_s seconds within each bag (contiguous samples)."""
    out = np.full(len(S), np.nan)
    for b, idx in S.groupby('bag').indices.items():
        t = S.t.values[idx]
        v = S[col].values[idx]
        lo = np.searchsorted(t, t - half_s)
        hi = np.searchsorted(t, t + half_s, side='right')
        c1 = np.r_[0, np.cumsum(v)]
        c2 = np.r_[0, np.cumsum(v * v)]
        n = hi - lo
        m = (c1[hi] - c1[lo]) / n
        if fn == 'mean':
            out[idx] = m
        elif fn == 'rms':
            out[idx] = np.sqrt((c2[hi] - c2[lo]) / n)
        elif fn == 'std':
            out[idx] = np.sqrt(np.maximum((c2[hi] - c2[lo]) / n - m * m, 0))
    return out


def zone_table(pl, thr=K_CURVE, merge_gap=5.0, min_len=2.0):
    s = pl.s
    z = np.abs(pl.kappa) > thr
    zones = []
    i = 0
    while i < len(z):
        if z[i]:
            j = i
            while j + 1 < len(z) and z[j + 1]:
                j += 1
            zones.append([s[i], s[j]])
            i = j + 1
        else:
            i += 1
    merged = []
    for a, b in zones:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    # cyclic wrap: first zone starts at 0 and last ends at L -> keep separate (window logic handles wrap)
    return [(a, b) for a, b in merged if b - a >= min_len]


def event_roc(S, pl, zones, half_s=0.5):
    """Per passage of a sharp-curve zone vs equal-length straight windows (antenna s windows)."""
    L = pl.L
    S = S.copy()
    S['ym'] = rolling_stat(S, 'y', half_s, 'mean')
    pos, neg = [], []
    rng = np.random.default_rng(0)
    for b, g in S.groupby('bag'):
        sm = g.s.values  # unwrapped antenna s
        ym = np.abs(g.ym.values)
        for a, e in zones:
            w0, w1 = a - C.D_FRONT - 2.0, e - C.D_REAR + 2.0   # front pivot enters .. rear pivot leaves
            for lap in np.unique(np.floor((sm - w0) / L)):
                lo, hi = w0 + lap * L, w1 + lap * L
                m = (sm >= lo) & (sm <= hi)
                if m.sum() >= 5 and (np.max(sm[m]) - np.min(sm[m])) > 0.6 * (hi - lo):
                    pos.append(dict(bag=b, zone=a, n=int(m.sum()), stat=float(np.max(ym[m])),
                                    rms=float(np.sqrt(np.mean(g.y.values[m] ** 2)))))
        # null windows: random starts, window length 30 m, both pivots straight (|k|<0.005) all along
        kmax = np.maximum(np.abs(pl.k_at(sm + C.D_FRONT)), np.abs(pl.k_at(sm + C.D_REAR)))
        if len(sm) < 50:
            continue
        for _ in range(40):
            i0 = rng.integers(0, len(sm) - 10)
            lo = sm[i0]
            m = (sm >= lo) & (sm <= lo + 30.0)
            if m.sum() >= 5 and np.all(kmax[m] < 0.005) and (np.max(sm[m]) - lo) > 18:
                neg.append(dict(bag=b, n=int(m.sum()), stat=float(np.max(ym[m])),
                                rms=float(np.sqrt(np.mean(g.y.values[m] ** 2)))))
    return pd.DataFrame(pos), pd.DataFrame(neg)


def main():
    S, info = load()
    S = add_features(S)
    say(f'=== Step 2 regression, train bags, vmin={VMIN} m/s ===')
    say('selection:', json.dumps(info))
    say(f'samples with max(|kf|,|kr|) > {K_CURVE}: {(S.kmax > K_CURVE).sum()}  '
        f'(both > {K_CURVE}: {((S.akf > K_CURVE) & (S.akr > K_CURVE)).sum()})')
    off = S.groupby('bag').off.first()
    say(f'per-bag offset log(vf/vr) on straight: median {off.median():+.5f}, std {off.std():.5f}, '
        f'min {off.min():+.5f}, max {off.max():+.5f}')
    models = {
        'M1 linear kf,kr': ['kf', 'kr'],
        'M2 abs |kf|,|kr|': ['akf', 'akr'],
        'M3 linear+abs': ['kf', 'kr', 'akf', 'akr'],
        'M4 quad kf^2-kr^2': ['q'],
        'M5 chord model': ['pred'],
        'M6 M3+kf^2,kr^2': ['kf', 'kr', 'akf', 'akr', 'kf2', 'kr2'],
        'N  nuisance a, a/v': ['a', 'a_v'],
        'M7 M6+chord+nuis': ['kf', 'kr', 'akf', 'akr', 'kf2', 'kr2', 'pred', 'a', 'a_v'],
    }
    yw = within(S, ['kf'])[1]
    say(f'y (within) std = {yw.std():.5f}  robust std = {1.4826 * np.median(np.abs(yw - np.median(yw))):.5f}  n = {len(yw)}')
    results = {}
    for name, cols in models.items():
        beta, se, r2, res = ols_cluster(S, cols)
        results[name] = dict(cols=cols, beta=beta.tolist(), se=se.tolist(), r2=float(r2),
                             res_std=float(res.std()))
        coef = ', '.join(f'{c}={b:+.4g}(+-{s:.2g})' for c, b, s in zip(cols, beta, se))
        say(f'{name:22s} within R2={r2:.4f}  res std={res.std():.5f}  {coef}')
    # separate curve subset: only samples with a bogie in a sharp curve
    cs = S[S.kmax > K_CURVE]
    say(f'--- subset max(|kf|,|kr|) > {K_CURVE} (n={len(cs)}) ---')
    for name in ('M3 linear+abs', 'M4 quad kf^2-kr^2', 'M5 chord model', 'M7 M6+chord+nuis'):
        beta, se, r2, res = ols_cluster(cs, models[name])
        coef = ', '.join(f'{c}={b:+.4g}(+-{s:.2g})' for c, b, s in zip(models[name], beta, se))
        say(f'{name:22s} within R2={r2:.4f}  res std={res.std():.5f}  {coef}')
    # main model for later steps: M7 on all samples
    beta, se, r2, res = ols_cluster(S, models['M7 M6+chord+nuis'])
    S['res'] = res
    # noise levels
    say('--- residual noise (M7) ---')
    for lo, hi, nm in [(0, 0.003, 'straight both<0.003'), (0.003, 0.01, 'mild 0.003-0.01'),
                       (0.01, 0.02, '0.01-0.02'), (0.02, 1, 'sharp >0.02')]:
        m = (S.kmax >= lo) & (S.kmax < hi)
        r = S.res[m]
        say(f'  max|k| {nm:20s} n={m.sum():7d} std={r.std():.5f} robust={1.4826 * np.median(np.abs(r - r.median())):.5f}')
    for lo, hi in [(3, 5), (5, 8), (8, 12), (12, 30)] if VMIN >= 3 else [(1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 30)]:
        m = (S.v >= lo) & (S.v < hi) & (S.kmax < 0.003)
        r = S.res[m]
        say(f'  straight, v {lo}-{hi} m/s: n={m.sum():7d} std={r.std():.5f} robust={1.4826 * np.median(np.abs(r - r.median())):.5f}')
    by_bag = [S.res.values[idx] for _, idx in S.groupby('bag').indices.items()]
    rho = acf(by_bag, 30)
    dt_med = np.median(np.diff(S.t.values)[np.diff(S.t.values) > 0])
    say('  residual ACF lag 1..10 samples:', ' '.join(f'{x:.3f}' for x in rho[1:11]),
        f'(median dt {dt_med:.3f} s); lag 20: {rho[20]:.3f}, lag 30: {rho[30]:.3f}')
    tau = 1 + 2 * np.sum(rho[1:31])
    say(f'  integrated autocorrelation factor (1+2*sum rho_1..30) = {tau:.2f} -> effective samples = n/{tau:.2f}')
    straight = S.kmax < 0.003
    rs = [S.res.values[idx][straight.values[idx]] for _, idx in S.groupby('bag').indices.items()]
    rho_s = acf([r for r in rs if len(r) > 40], 30)
    say('  straight-only residual ACF lag 1..10:', ' '.join(f'{x:.3f}' for x in rho_s[1:11]))
    # ---------------- ROC: sharp-curve detection from the ratio alone ----------------
    say('--- ROC: detect max(|kf|,|kr|) > 0.02 from the ratio alone (per sample) ---')
    lab = (S.kmax > K_CURVE).values
    clean = (S.kmax > K_CURVE) | (S.kmax < 0.005)
    rows = []
    S['ym05'] = rolling_stat(S, 'y', 0.5, 'mean')
    S['ym10'] = rolling_stat(S, 'y', 1.0, 'mean')
    S['ysd10'] = rolling_stat(S, 'y', 1.0, 'std')
    S['yrms20'] = rolling_stat(S, 'y', 2.0, 'rms')
    S['fit'] = S.y - S.res
    scores = {'|y| per sample': np.abs(S.y.values), '|mean y| +-0.5 s': np.abs(S.ym05.values),
              '|mean y| +-1 s': np.abs(S.ym10.values), 'std y +-1 s': S.ysd10.values,
              'rms y +-2 s': S.yrms20.values}
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    for nm, sc in scores.items():
        for lbl_nm, msk in (('all negatives', np.ones(len(S), bool)), ('negatives max|k|<0.005', clean.values)):
            fpr, tpr, auc = roc(sc[msk], lab[msk])
            r = dict(score=nm, negatives=lbl_nm, auc=auc, tpr_fpr01=tpr_at(fpr, tpr, 0.01),
                     tpr_fpr05=tpr_at(fpr, tpr, 0.05), tpr_fpr10=tpr_at(fpr, tpr, 0.10),
                     n_pos=int(lab[msk].sum()), n_neg=int((~lab[msk]).sum()))
            rows.append(r)
            say(f'  {nm:18s} [{lbl_nm:22s}] AUC={auc:.3f} TPR@FPR1%={r["tpr_fpr01"]:.3f} '
                f'@5%={r["tpr_fpr05"]:.3f} @10%={r["tpr_fpr10"]:.3f}  (pos {r["n_pos"]}, neg {r["n_neg"]})')
            if lbl_nm == 'all negatives':
                ax[0].plot(fpr, tpr, label=f'{nm} AUC {auc:.2f}')
    # transitions: exactly one pivot in sharp curve -> where the kinematic model predicts signal
    trans = ((S.akf > K_CURVE) ^ (S.akr > K_CURVE)).values
    for nm in ('|y| per sample', '|mean y| +-0.5 s'):
        fpr, tpr, auc = roc(scores[nm], trans)
        say(f'  transition label (one pivot in, other out), {nm}: AUC={auc:.3f} TPR@FPR5%={tpr_at(fpr, tpr, 0.05):.3f}')
    pd.DataFrame(rows).to_csv(C.HERE / f'roc_v{VMIN:g}.csv', index=False)
    ax[0].plot([0, 1], [0, 1], 'k:')
    ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR'); ax[0].legend(fontsize=8)
    ax[0].set_title(f'per-sample ROC, label max(|kf|,|kr|)>0.02, v>{VMIN}')
    # binned y vs chord prediction and vs model fit
    for axi, col, nm in ((ax[1], 'pred', 'chord-model prediction'), (ax[2], 'fit', 'M7 fitted value')):
        x = S[col].values
        edges = np.quantile(x, np.linspace(0, 1, 41))
        edges = np.unique(edges)
        idx = np.clip(np.searchsorted(edges, x) - 1, 0, len(edges) - 2)
        bx = [np.mean(x[idx == i]) for i in range(len(edges) - 1)]
        by = [np.mean(S.y.values[idx == i]) for i in range(len(edges) - 1)]
        bs = [np.std(S.y.values[idx == i]) / np.sqrt(max((idx == i).sum() / tau, 1)) for i in range(len(edges) - 1)]
        axi.errorbar(100 * np.array(bx), 100 * np.array(by), yerr=100 * np.array(bs), fmt='o')
        lim = 100 * max(np.abs(bx).max(), 0.002)
        axi.plot([-lim, lim], [-lim, lim], 'k:')
        axi.set_xlabel(f'{nm} [%]'); axi.set_ylabel('mean y [%] (quantile bins)'); axi.grid()
        axi.set_title(f'y vs {nm}')
    plt.tight_layout()
    plt.savefig(C.HERE / f'fig_regress_v{VMIN:g}.png', dpi=75)
    # ---------------- event-level ROC ----------------
    pl = C.load_main()
    zones = zone_table(pl)
    say(f'--- event level: {len(zones)} sharp-curve zones (|k|>0.02, merged gaps<=5 m) ---')
    pos, neg = event_roc(S, pl, zones)
    if len(pos) and len(neg):
        for stat in ('stat', 'rms'):
            thr01 = np.quantile(neg[stat], 0.99)
            thr05 = np.quantile(neg[stat], 0.95)
            p1 = np.mean(pos[stat] > thr01)
            p5 = np.mean(pos[stat] > thr05)
            sc = np.r_[pos[stat].values, neg[stat].values]
            lb = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
            _, _, auc = roc(sc, lb)
            nm = 'max |mean y over +-0.5 s|' if stat == 'stat' else 'rms y over window'
            say(f'  {nm:26s}: passages {len(pos)}, straight 30 m windows {len(neg)}: AUC={auc:.3f} '
                f'P(detect)@FA1%={p1:.3f} @FA5%={p5:.3f}  (thr {100 * thr01:.2f}% / {100 * thr05:.2f}%)')
        pz = pos.groupby('zone').agg(n=('stat', 'size'), med_stat=('stat', 'median'), med_rms=('rms', 'median'))
        pz['det5_rms'] = pos.groupby('zone').rms.apply(lambda r: np.mean(r > np.quantile(neg.rms, 0.95)))
        say('  per zone (antenna s start): passages, median stats, detection@FA5% (rms):')
        for z, r in pz.iterrows():
            say(f'    zone {z:8.1f}: n={int(r.n):3d} med max|ym|={100 * r.med_stat:.2f}% med rms={100 * r.med_rms:.2f}% det={r.det5_rms:.2f}')
        pos.to_csv(C.HERE / f'event_pos_v{VMIN:g}.csv', index=False)
        neg.to_csv(C.HERE / f'event_neg_v{VMIN:g}.csv', index=False)
    json.dump(dict(vmin=VMIN, info=info, models=results, tau=float(tau)),
              open(C.HERE / f'model_v{VMIN:g}.json', 'w'), indent=1)
    OUT.write_text('\n'.join(_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
