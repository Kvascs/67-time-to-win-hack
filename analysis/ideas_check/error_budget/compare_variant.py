"""Error budget: compare a parameter variant with the submitted build on the RTK bags (judge frame).

Per bag 3-D / along RMSE of both runs, (a) on all matched epochs and (b) without the epochs that the base
classification (analyze_rtk.py -> rtk_epochs_all.parquet) marks as reference faults, west-terminal stub,
start branch or parallel track (the categories a wheel-scale or landmark change cannot touch).

    python analysis/ideas_check/error_budget/compare_variant.py _max_dk_1e-3
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SKIP = ('ref_clock', 'ref_jump', 'west_terminal', 'start_off', 'parallel_track')


def per_bag(tag, A):
    rows = []
    summ = pd.read_csv(HERE / f'rtk_summary{tag}.csv')
    summ = summ[summ['error'].isna()]
    for bag, split in zip(summ.bag, summ.split):
        E = pd.read_parquet(HERE / f'rtk_epochs{tag}' / f'{bag}.parquet')
        c = A[A.bag == bag][['t', 'cat']]
        E = E.merge(c, on='t', how='left')
        e2 = E.along ** 2 + E.cross ** 2 + E.dz ** 2
        k = ~E.cat.isin(SKIP)
        rows.append({'bag': bag, 'split': split, 'p3': np.sqrt(e2.mean()), 'along': np.sqrt(np.mean(E.along ** 2)),
                     'p3_clean': np.sqrt(e2[k].mean()), 'along_clean': np.sqrt(np.mean(E.along[k] ** 2))})
    return pd.DataFrame(rows).set_index('bag')


def main():
    tag = sys.argv[1]
    A = pd.read_parquet(HERE / 'rtk_epochs_all.parquet')
    b = per_bag('', A)
    v = per_bag(tag, A)
    b = b.loc[v.index]
    out = []
    for split in ('val', 'train'):
        k = b.split == split
        for m in ('p3', 'along', 'p3_clean', 'along_clean'):
            d = v.loc[k, m] - b.loc[k, m]
            out.append({'split': split, 'metric': m, 'base_mean': b.loc[k, m].mean(), 'var_mean': v.loc[k, m].mean(),
                        'base_median': b.loc[k, m].median(), 'var_median': v.loc[k, m].median(),
                        'better_>5%': int((d < -0.05 * b.loc[k, m]).sum()), 'worse_>5%': int((d > 0.05 * b.loc[k, m]).sum()),
                        'n': int(k.sum())})
    R = pd.DataFrame(out)
    pd.set_option('display.width', 250)
    print(f'variant {tag} vs base')
    print(R.round(3).to_string(index=False))
    D = v[['p3_clean']].join(b[['p3_clean', 'split']], rsuffix='_base')
    D['delta'] = D.p3_clean - D.p3_clean_base
    print('largest changes of the clean 3-D RMSE:')
    print(D.sort_values('delta').iloc[np.r_[0:6, -6:0]].round(3).to_string())
    R.to_csv(HERE / f'compare{tag}.csv', index=False)
    D.to_csv(HERE / f'compare{tag}_bags.csv')


if __name__ == '__main__':
    main()
