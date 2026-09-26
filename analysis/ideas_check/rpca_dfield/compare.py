"""Paired comparison of the eval_par results of the field variants (build_core/eval/rpca_*.csv).

    python analysis/ideas_check/rpca_dfield/compare.py none median_repo median_final2 pcp_lam1 pcp_lamp
Writes summary.csv (means) and paired.csv (per-variant paired differences) next to this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT / 'build_core' / 'eval'
METRICS = [('dw', 'win_v_rmse'), ('dw', 'win_ds_abs'), ('clean', 'v_rmse'), ('clean', 'along_rmse')]


def load(mode: str, v: str) -> pd.DataFrame:
    df = pd.read_csv(EVAL / f'rpca_{mode}_{v}.csv')
    if 'error' in df.columns and df['error'].notna().any():
        raise RuntimeError(f'{mode}/{v}: errors in {df[df.error.notna()].bag.tolist()}')
    return df.set_index('bag').sort_index()


def main():
    variants = sys.argv[1:]
    data = {(m, v): load(m, v) for m in ('dw', 'clean') for v in variants}
    rows = []
    for v in variants:
        r = {'variant': v}
        for m, k in METRICS:
            x = data[(m, v)][k]
            r[f'{k}_mean'] = x.mean()
            r[f'{k}_median'] = x.median()
        rows.append(r)
    summ = pd.DataFrame(rows).set_index('variant')
    summ.to_csv(HERE / 'summary.csv')
    pd.set_option('display.width', 250)
    print('VAL means / medians over 17 bags (dw = --dropwin 60:10, window metrics; clean = no drop)')
    print(summ.round(4).to_string())

    rows = []
    for ref in ('none', 'median_repo', 'median_final2'):
        if ref not in variants:
            continue
        for v in variants:
            if v == ref:
                continue
            for m, k in METRICS:
                a, b = data[(m, v)][k], data[(m, ref)][k]
                d = (a - b).dropna()
                try:
                    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                except ValueError:
                    p = np.nan
                rows.append({'variant': v, 'vs': ref, 'metric': k, 'mean_diff': d.mean(),
                             'sd_diff': d.std(ddof=1), 'se_diff': d.std(ddof=1) / np.sqrt(len(d)),
                             'better': int((d < 0).sum()), 'worse': int((d > 0).sum()), 'n': len(d),
                             'wilcoxon_p': p, 'bag_sd_of_metric': b.std(ddof=1)})
    pr = pd.DataFrame(rows)
    pr.to_csv(HERE / 'paired.csv', index=False)
    print('\nPaired per-bag differences (variant - vs; negative = better). sd_diff = bag-to-bag spread of the '
          'difference, se = sd/sqrt(n); bag_sd_of_metric = spread of the metric itself across bags')
    print(pr.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
