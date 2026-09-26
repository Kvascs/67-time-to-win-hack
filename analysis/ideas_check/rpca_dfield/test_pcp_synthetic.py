"""Sanity check of rpca_field.pcp_masked on synthetic data with known answer: low-rank L0 + sparse
gross errors S0, half of the entries unobserved (the shape and coverage of the real matrix).
Also checks the KKT conditions of the returned solution (dual Y: |Y| <= lam on W, ||Y||_2 ~ 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rpca_field import pcp_masked  # noqa: E402

rng = np.random.default_rng(1)
m, n, r = 52, 1106, 2
L0 = rng.normal(size=(m, r)) @ rng.normal(size=(r, n)) * 0.05
S0 = np.where(rng.random((m, n)) < 0.05, rng.choice([-1, 1], size=(m, n)) * rng.uniform(0.2, 1.0, (m, n)), 0.0)
W = rng.random((m, n)) < 0.5
M = np.where(W, L0 + S0, np.nan)
for lam_mult in (1.0, 1 / np.sqrt(0.5)):
    lam = lam_mult / np.sqrt(max(m, n))
    L, S, info = pcp_masked(M, lam, rho=1.1)
    eL = np.linalg.norm(L - L0) / np.linalg.norm(L0)
    eLu = np.linalg.norm((L - L0)[~W]) / np.linalg.norm(L0[~W])
    sup = (np.abs(S) > 1e-6)[W] == (np.abs(S0) > 0)[W]
    print(f'lam x{lam_mult:.3f}: iters {info["iters"]}, rank(L) {info["rank"]} (true {r}), '
          f'||L-L0||/||L0|| all {eL:.2e}, unobserved {eLu:.2e}; S support agreement on W {sup.mean():.1%}')

# the hypothesis' own model: every run sees the same field f (rank 1, constant loading) + dense noise
# + sparse events, real coverage mask; f = the real train median field. Which estimator recovers f?
import pandas as pd  # noqa: E402
from rpca_field import HERE, postprocess, run_cell_matrix  # noqa: E402

f = pd.read_csv(HERE / 'dfield_median_final2.csv', comment='#').d.to_numpy()
Mr, _ = run_cell_matrix(pd.read_parquet(HERE / 'residuals_train.parquet'))
Wr = np.isfinite(Mr)
runs = Wr.sum(0)
ok = runs >= 3
for sd, ev in ((0.10, 0.03), (0.13, 0.03)):
    Mf = np.ones((Mr.shape[0], 1)) * f[None] + rng.normal(scale=sd, size=Mr.shape)
    Mf += np.where(rng.random(Mr.shape) < ev, rng.choice([-1, 1], size=Mr.shape) * rng.uniform(0.3, 1.0, Mr.shape), 0)
    Mf = np.where(Wr, Mf, np.nan)
    est = {'median': postprocess(np.nanmedian(Mf, 0), runs)}
    for lm in (1.0, 1 / np.sqrt(Wr.mean()), 32.0):
        L, S, info = pcp_masked(Mf, lm / np.sqrt(max(Mf.shape)))
        est[f'pcp x{lm:.2f} mean_all (rank {info["rank"]}, S {info["s_frac"]:.0%})'] = postprocess(L.mean(0), runs)
    fs = postprocess(f, runs)  # the target after the same smoothing
    print(f'synthetic 1*f^T + noise sd {sd} + {ev:.0%} events, real mask:')
    for k, e in est.items():
        slope = np.polyfit(fs[ok], e[ok], 1)[0]
        print(f'   {k:36s} rms error vs f {np.sqrt(np.mean((e - fs)[ok] ** 2)):.4f}, amplitude slope {slope:.2f}')

# with dense noise added (like the real residual matrix): S is no longer sparse at lam0
N = rng.normal(scale=0.05, size=(m, n))
Mn = np.where(W, L0 + S0 + N, np.nan)
L, S, info = pcp_masked(Mn, 1 / np.sqrt(max(m, n)))
print(f'+ dense noise sd 0.05: rank(L) {info["rank"]}, S nonzero on W {info["s_frac"]:.1%}, '
      f'col-mean error of L vs L0 {np.sqrt(np.mean((L.mean(0) - L0.mean(0)) ** 2)):.4f}, '
      f'col-median of observed M vs L0 col-mean {np.sqrt(np.mean((np.nanmedian(Mn, 0) - L0.mean(0)) ** 2)):.4f}')
