"""Robust-PCA (principal component pursuit) version of the learned disturbance field d(s).

Input: residuals_<split>.parquet from collect_residuals.py (per-sample r = dv/dt - model, s_map).
Matrix: M[run, cell] = median of r of that run inside the 10 m cell (exactly learn_dfield's
"one vote per run and bin"), NaN where the run has no sample in the cell.

PCP with missing entries (observed set W):
    min ||L||_* + lam * sum_{W} |S_ij|   s.t.  L + S = M on W        (entries off W are free)
solved by inexact ALM / ADMM. Off W, S is unpenalised, so S = -L there and the L step sees the
previous L: unobserved entries are imputed by the current low-rank estimate (no ad-hoc fill).
Field = per-cell mean of L over all runs (or over observed runs / median), then the same
post-processing as learn_dfield: cells seen by < 3 runs -> 0, (1,2,1)/4 cyclic smoothing.

Outputs (next to this script): dfield_*.csv (learn_dfield format), cv_train.csv, val_offline.csv,
rpca_field.log (stdout).

    python analysis/ideas_check/rpca_dfield/rpca_field.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)  # nanmean/nanmedian of unseen cells

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'model'))
import learn_dfield  # noqa: E402

MAPS = ROOT / 'analysis' / 'validation_maps'
BIN = 10.0
MIN_RUNS = 3
LENGTH = float(pd.read_csv(MAPS / 'track_map.csv', comment='#').s.iloc[-1])
NB = int(np.ceil(LENGTH / BIN))
S_CELL = (np.arange(NB) + 0.5) * BIN


# ----------------------------------------------------------------------------------------------
def run_cell_matrix(df: pd.DataFrame):
    """runs x cells matrix of per-run cell medians (NaN = unobserved) + run names."""
    b = np.clip((df.s_map.to_numpy() / BIN).astype(int), 0, NB - 1)
    per_run = df.assign(b=b).groupby(['bag', 'b']).r.median()
    bags = sorted(df.bag.unique())
    M = np.full((len(bags), NB), np.nan)
    ri = {k: i for i, k in enumerate(bags)}
    for (bag, cell), val in per_run.items():
        M[ri[bag], cell] = val
    return M, bags


def svt(X: np.ndarray, tau: float):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    r = int((s > 0).sum())
    return (U[:, :r] * s[:r]) @ Vt[:r], r


def pcp_masked(M: np.ndarray, lam: float, rho: float = 1.1, tol: float = 1e-7, max_iter: int = 5000):
    """Principal component pursuit on the observed entries (inexact ALM, Lin-Chen-Ma 2010)."""
    W = np.isfinite(M)
    D = np.where(W, M, 0.0)
    n2 = np.linalg.norm(D, 2)
    Y = D / max(n2, np.abs(D).max() / lam)          # dual init
    mu, mu_max = 1.25 / n2, 1.25 / n2 * 1e7
    S = np.zeros_like(D)
    dn = np.linalg.norm(D)
    for it in range(1, max_iter + 1):
        L, rank = svt(D - S + Y / mu, 1.0 / mu)
        T = D - L + Y / mu
        S = np.where(W, np.sign(T) * np.maximum(np.abs(T) - lam / mu, 0.0), T)
        Z = D - L - S                                   # zero off W by construction
        Y += mu * Z
        mu = min(mu * rho, mu_max)
        err = np.linalg.norm(Z) / dn
        if err < tol:
            break
    S = np.where(W, S, 0.0)
    sv = np.linalg.svd(L, compute_uv=False)
    obj = sv.sum() + lam * np.abs(S[W]).sum()
    info = dict(iters=it, err=err, rank=int((sv > 1e-6 * sv[0]).sum()), obj=obj,
                s_frac=float((np.abs(S[W]) > 1e-9).mean()), sv=sv)
    return L, S, info


def postprocess(raw: np.ndarray, runs: np.ndarray) -> np.ndarray:
    """learn_dfield.build_field post-processing: <MIN_RUNS -> 0, (1,2,1)/4 cyclic smoothing."""
    f = np.where(runs >= MIN_RUNS, raw, 0.0)
    f = np.nan_to_num(f)
    return (np.roll(f, 1) + 2 * f + np.roll(f, -1)) / 4.0


def field_median(M: np.ndarray) -> np.ndarray:
    runs = np.isfinite(M).sum(0)
    with np.errstate(all='ignore'):
        raw = np.nanmedian(M, axis=0)
    return postprocess(raw, runs)


def field_pcp(M: np.ndarray, lam: float, how: str = 'mean_all', rho: float = 1.1):
    L, S, info = pcp_masked(M, lam, rho=rho)
    W = np.isfinite(M)
    runs = W.sum(0)
    if how == 'mean_all':
        raw = L.mean(0)
    elif how == 'mean_obs':
        with np.errstate(all='ignore'):
            raw = np.nanmean(np.where(W, L, np.nan), 0)
    elif how == 'median_obs':
        with np.errstate(all='ignore'):
            raw = np.nanmedian(np.where(W, L, np.nan), 0)
    else:
        raise ValueError(how)
    return postprocess(raw, runs), L, S, info


def eval_field(df: pd.DataFrame, field: np.ndarray, block: int = 100) -> dict:
    """Out-of-sample fit of a field to per-sample residuals: RMS of r - f(s) per sample and of
    its mean over blocks of `block` consecutive kept samples (~10 s: model-only-window proxy)."""
    f = np.interp(df.s_map.to_numpy(), S_CELL, field, period=LENGTH)
    e = df.r.to_numpy() - f
    out = {'rms': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e)))}
    bm = []
    for _, g in pd.DataFrame({'bag': df.bag.to_numpy(), 'e': e}).groupby('bag', sort=False):
        x = g.e.to_numpy()
        nblk = len(x) // block
        if nblk:
            bm.append(x[:nblk * block].reshape(nblk, block).mean(1))
    bm = np.concatenate(bm)
    out['blk_rms'] = float(np.sqrt(np.mean(bm ** 2)))
    return out


def write_field(path: Path, field: np.ndarray, runs: np.ndarray, note: str):
    with open(path, 'w', newline='\n') as fh:
        fh.write(f'# {note}\n')
        fh.write('s,d,runs\n')
        for s, d, n in zip(S_CELL, field, runs):
            fh.write(f'{s:.1f},{d:.5f},{int(n)}\n')


def load_field(path: Path) -> np.ndarray:
    f = pd.read_csv(path, comment='#')
    return np.interp(S_CELL, f.s.to_numpy(), f.iloc[:, 1].to_numpy(), period=LENGTH)


# ----------------------------------------------------------------------------------------------
def main():
    tr = pd.read_parquet(HERE / 'residuals_train.parquet')
    va = pd.read_parquet(HERE / 'residuals_val.parquet')
    M, bags = run_cell_matrix(tr)
    m, n = M.shape
    W = np.isfinite(M)
    runs = W.sum(0)
    p = W.mean()
    lam0 = 1.0 / np.sqrt(max(m, n))
    print(f'train matrix {m} runs x {n} cells, observed {p:.1%}, cells with >= {MIN_RUNS} runs '
          f'{(runs >= MIN_RUNS).mean():.1%}; entry std {np.nanstd(M):.4f}, lam0 = 1/sqrt(max(m,n)) = {lam0:.4f}')

    # median field: must reproduce learn_dfield.build_field on the same samples
    f_med = field_median(M)
    ref = learn_dfield.build_field(tr[['bag', 's_map', 'r']].copy(), LENGTH, BIN, MIN_RUNS)
    print(f'median field vs learn_dfield.build_field: max |diff| {np.abs(ref.d.to_numpy() - f_med).max():.2e}')

    # PCP solution sanity: step-size schedule must not change the optimum
    for rho in (1.1, 1.02):
        L, S, info = pcp_masked(M, lam0, rho=rho)
        print(f'  PCP lam0 rho={rho}: iters {info["iters"]}, obj {info["obj"]:.4f}, rank {info["rank"]}, '
              f'S nonzero {info["s_frac"]:.1%}, feas {info["err"]:.1e}')

    # structure of the matrix: spectrum of L at lam0, energy of the common (rank-1) part
    L, S, info = pcp_masked(M, lam0)
    sv = info['sv']
    print('  L singular values (top 8):', np.round(sv[:8], 3).tolist(),
          f'| sum {sv.sum():.3f}; rank-1 energy share {sv[0] ** 2 / (sv ** 2).sum():.1%}')
    obsS = S[W]
    print(f'  S on observed: nonzero {np.mean(np.abs(obsS) > 1e-9):.1%}, |S| p50/p95/max of nonzero '
          f'{np.percentile(np.abs(obsS[np.abs(obsS) > 1e-9]), [50, 95, 100]).round(3).tolist()}; '
          f'|M| p50/p95 {np.nanpercentile(np.abs(M), [50, 95]).round(3).tolist()}')
    # what the low-rank components are: row loadings by vehicle, column profiles vs the median field
    U, sv_, Vt = np.linalg.svd(L, full_matrices=False)
    veh = np.array([b.split('_')[0] for b in bags])
    for k in range(min(3, int((sv_ > 1e-6 * sv_[0]).sum()))):
        u = U[:, k] * np.sign(U[:, k].sum() or 1.0)
        v = Vt[k] * np.sign(U[:, k].sum() or 1.0)
        byv = {x: round(float(u[veh == x].mean()), 3) for x in sorted(set(veh))}
        print(f'  comp {k}: sigma {sv_[k]:.3f}, row loading mean/sd {u.mean():.3f}/{u.std():.3f} '
              f'(1/sqrt(m) = {1 / np.sqrt(m):.3f}), by vehicle {byv}, corr(profile, median field) '
              f'{np.corrcoef(v, f_med)[0, 1]:+.2f}, corr(profile, cell coverage) {np.corrcoef(v, runs)[0, 1]:+.2f}')

    # ---- 5-fold CV over TRAIN runs: pick nothing on val ----
    lam_mults = [0.25, 0.5, 1.0, 1.0 / np.sqrt(p), 2.0, 4.0, 8.0, 16.0, 32.0]
    hows = ['mean_all', 'mean_obs', 'median_obs']
    rng = np.random.default_rng(0)
    perm = rng.permutation(m)
    folds = np.array_split(perm, 5)
    rows = []
    for k, te in enumerate(folds):
        trn = np.setdiff1d(np.arange(m), te)
        te_bags = [bags[i] for i in te]
        dte = tr[tr.bag.isin(te_bags)]
        Mk = M[trn]
        rows.append({'fold': k, 'method': 'zero', 'lam_mult': np.nan, **eval_field(dte, np.zeros(n))})
        rows.append({'fold': k, 'method': 'median', 'lam_mult': np.nan, **eval_field(dte, field_median(Mk))})
        # plain per-cell mean of the per-run medians (= PCP with lam -> inf, mean over observed runs)
        rows.append({'fold': k, 'method': 'mean', 'lam_mult': np.inf,
                     **eval_field(dte, postprocess(np.nanmean(Mk, 0), np.isfinite(Mk).sum(0)))})
        lam_k = 1.0 / np.sqrt(max(Mk.shape))
        for lm in lam_mults:
            Lk, Sk, ik = pcp_masked(Mk, lam_k * lm)
            Wk = np.isfinite(Mk)
            rk = Wk.sum(0)
            with np.errstate(all='ignore'):
                raws = {'mean_all': Lk.mean(0), 'mean_obs': np.nanmean(np.where(Wk, Lk, np.nan), 0),
                        'median_obs': np.nanmedian(np.where(Wk, Lk, np.nan), 0)}
            for h in hows:
                rows.append({'fold': k, 'method': f'pcp_{h}', 'lam_mult': lm, 'rank': ik['rank'],
                             's_frac': ik['s_frac'], **eval_field(dte, postprocess(raws[h], rk))})
    cv = pd.DataFrame(rows)
    cv.to_csv(HERE / 'cv_train.csv', index=False)
    agg = cv.groupby(['method', 'lam_mult'], dropna=False)[['rms', 'blk_rms', 'mae']].mean()
    base = cv[cv.method == 'median'].set_index('fold')
    d = cv.join(base[['rms', 'blk_rms']], on='fold', rsuffix='_med')
    d['d_rms'] = d.rms - d.rms_med
    d['d_blk'] = d.blk_rms - d.blk_rms_med
    dd = d.groupby(['method', 'lam_mult'], dropna=False)[['d_rms', 'd_blk']].agg(['mean', 'std'])
    dd.columns = [f'{a}_{b}' for a, b in dd.columns]
    agg = agg.join(cv.groupby(['method', 'lam_mult'], dropna=False)[['rank', 's_frac']].mean())
    pd.set_option('display.width', 220)
    print('\n5-fold CV on train runs (held-out runs; residual after subtracting the field, m/s^2):')
    print(agg.join(dd).round(5).to_string())

    # CV choice (train only): the PCP setting with the lowest mean 10-s block RMS on held-out runs
    pc = agg.reset_index()
    pc = pc[pc.method.str.startswith('pcp_')].sort_values('blk_rms').iloc[0]
    best = (float(pc.lam_mult), pc.method.replace('pcp_', ''))
    print(f'CV-best PCP setting (train, by blk_rms): lam x{best[0]:g}, field = {best[1]}')

    # ---- final fields on all train runs ----
    fields = {'zero': np.zeros(n), 'median_repo': load_field(MAPS / 'dfield.csv'), 'median_final2': f_med,
              'mean_cells': postprocess(np.nanmean(M, 0), runs)}
    notes = {}
    variants = [('pcp_lam1', 1.0, 'mean_all'), ('pcp_lamp', 1.0 / np.sqrt(p), 'mean_all'),
                ('pcp_lam1_obs', 1.0, 'mean_obs'), ('pcp_lam1_med', 1.0, 'median_obs'),
                ('pcp_lam4', 4.0, 'mean_all'), ('pcp_lam05', 0.5, 'mean_all'), ('pcp_cvbest', *best)]
    for name, lm, how in variants:
        f, L, S, info = field_pcp(M, lam0 * lm, how)
        fields[name] = f
        notes[name] = (f'PCP field (robust PCA on {m} train runs x {n} cells of per-run cell medians, '
                       f'observed-entries ADMM): lam = {lm:.3f}/sqrt(max(m,n)) = {lam0 * lm:.4f}, field = {how} of L; '
                       f'rank(L) {info["rank"]}, S nonzero {info["s_frac"]:.1%}; min {MIN_RUNS} runs, (1,2,1)/4 smoothing')
    write_field(HERE / 'dfield_median_final2.csv', f_med, runs,
                f'median field re-learned with tbo_replay_final2 (learn_dfield.build_field on the same samples): '
                f'{m} train runs; min {MIN_RUNS} runs per bin')
    for name in notes:
        write_field(HERE / f'dfield_{name}.csv', fields[name], runs, notes[name])

    print('\nfields vs median_final2: RMS diff / max |diff| (m/s^2); |d| p50/p95')
    for k, f in fields.items():
        print(f'  {k:14s} {np.sqrt(np.mean((f - f_med) ** 2)):.4f} / {np.abs(f - f_med).max():.4f};  '
              f'{np.percentile(np.abs(f), 50):.4f} / {np.percentile(np.abs(f), 95):.4f}')

    # ---- offline out-of-sample check on VAL (no replays) ----
    rows = []
    vbags = sorted(va.bag.unique())
    for k, f in fields.items():
        r = eval_field(va, f)
        per = {b: eval_field(va[va.bag == b], f) for b in vbags}
        rows.append({'field': k, **r, **{f'rms_{b}': per[b]['rms'] for b in vbags},
                     **{f'blk_{b}': per[b]['blk_rms'] for b in vbags}})
    vo = pd.DataFrame(rows).set_index('field')
    vo.to_csv(HERE / 'val_offline.csv')
    print('\nVAL out-of-sample (17 runs): residual after subtracting the field, m/s^2')
    print(vo[['rms', 'mae', 'blk_rms']].round(5).to_string())
    for pre in ('rms', 'blk'):
        cols = [f'{pre}_{b}' for b in vbags]
        for k in fields:
            diff = vo.loc[k, cols].to_numpy(float) - vo.loc['median_final2', cols].to_numpy(float)
            print(f'  per-bag {pre} diff {k:14s} vs median_final2: mean {diff.mean():+.5f}  sd {diff.std(ddof=1):.5f}  '
                  f'better/worse {int((diff < 0).sum())}/{int((diff > 0).sum())}')


if __name__ == '__main__':
    main()
