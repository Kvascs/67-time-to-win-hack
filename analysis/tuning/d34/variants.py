"""D34 variants: paired comparison with the D33 build (rm3) on val and train, base_link 3-D RMSE per bag.

Inputs are the per-bag tables of tools/replay/eval_base_link.py (build_core/eval/bl_<tag>.csv):
    rm3   D33 as submitted before D34                       (build of commit 983867f)
    uk    ratio_update_k=1 (full k gain)                     --set ratio_update_k=1
    kd1   ratio_update_k=1, ratio_k_dmax=1.0                 (build with the D34 parameters, gain 1)
    kd05  ratio_update_k=1, ratio_k_dmax=0.5
    kg03  ratio_update_k=1, ratio_k_gain=0.3  (= D34 defaults, tag d34 for the default build)
Writes analysis/tuning/d34/variants.csv and prints the table.
"""
from pathlib import Path

import pandas as pd

EV = Path(__file__).resolve().parents[3] / 'build_core' / 'eval'
rows = []
for split in ('val', 'train'):
    a = pd.read_csv(EV / f'bl_rm3_{split}.csv').set_index('bag').p3_rmse
    for tag in ('uk', 'kd1', 'kd05', 'kg03', 'd34'):
        f = EV / f'bl_{tag}_{split}.csv'
        if not f.exists():
            continue
        b = pd.read_csv(f).set_index('bag').p3_rmse
        d = (b - a).dropna()
        rows.append({'split': split, 'variant': tag, 'n': len(d), 'median': round(b.median(), 3), 'mean': round(b.mean(), 3),
                     'd33_median': round(a.median(), 3), 'd33_mean': round(a.mean(), 3),
                     'better': int((d < -0.001).sum()), 'worse': int((d > 0.001).sum()),
                     'paired_median_diff': round(d.median(), 4), 'worst_diff': round(d.max(), 3), 'worst_bag': d.idxmax(),
                     '4d487b0d': round(b.get('30618_4d487b0d', float('nan')), 2)})
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).with_name('variants.csv'), index=False)
print(df.to_string(index=False))
