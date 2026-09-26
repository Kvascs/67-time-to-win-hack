"""Upper bound of what better (retroactive) landmark association could remove on VAL: every gross episode
(|e_along| > 1 m, c_episodes_val.csv) clipped to 1 m, EXCEPT episodes attributed to other causes:
reference glitches (616ec56b false RTK fix ~31 m; spikes > 6 m shorter than 10 s), the F3 track missing in the
train-only map (e3d94878, all episodes; the shipped map has F3), start of run (< 60 s after the first position)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qn_harness as H  # noqa: E402
from c_error_budget import along_series  # noqa: E402

ep = pd.read_csv('c_episodes_val.csv')
other = (ep.bag == '30618_e3d94878') | (ep.t_start < 60) | ((ep.e_max.abs() > 6) & (ep.dur_s < 10)) | \
        ((ep.bag == '30618_616ec56b') & (ep.e_med.abs() > 10))
print('episodes attributed to other causes:\n', ep[other].to_string(index=False))
rows = []
for b in H.splits()['val']:
    if not (H.KEEP / f'{b}.pkl').exists() or pd.read_pickle(H.SCR / 'ref' / f'{b}.pkl')['bl'] is None:
        continue
    t, a, fixes, t0 = along_series(b)
    c_all = np.clip(a, -1, 1)
    c_assoc = a.copy()
    for r in ep[(ep.bag == b) & ~other].itertuples():
        m = (t - t0 >= r.t_start - 1) & (t - t0 <= r.t_start + r.dur_s + 1)
        c_assoc[m] = np.clip(a[m], -1, 1)
    rows.append({'bag': b, 'along': np.sqrt(np.mean(a ** 2)), 'clip_all': np.sqrt(np.mean(c_all ** 2)),
                 'clip_assoc_drift': np.sqrt(np.mean(c_assoc ** 2))})
df = pd.DataFrame(rows)
df.to_csv('c_upper_bound_val.csv', index=False)
print(df.round(3).to_string(index=False))
print('mean', df[['along', 'clip_all', 'clip_assoc_drift']].mean().round(3).to_dict(),
      'median', df[['along', 'clip_all', 'clip_assoc_drift']].median().round(3).to_dict())
