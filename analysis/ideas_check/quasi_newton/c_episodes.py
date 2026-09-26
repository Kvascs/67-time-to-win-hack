"""Gross along-track episodes (|e| > 1 m for >= 3 s) on VAL with their context: which landmark fixes happen inside,
does a fix end the episode, the error sign/size. Uses the outputs kept by c_error_budget.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qn_harness as H  # noqa: E402
from c_error_budget import along_series  # noqa: E402

rows = []
for b in H.splits()['val']:
    if not (H.KEEP / f'{b}.pkl').exists() or pd.read_pickle(H.SCR / 'ref' / f'{b}.pkl')['bl'] is None:
        continue
    t, a, fixes, t0 = along_series(b)
    g = np.abs(a) > 1.0
    edges = np.flatnonzero(np.diff(np.r_[0, g.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        ts, te = t[s], t[e - 1]
        if te - ts < 3.0:
            continue
        inside = fixes[(fixes > ts) & (fixes < te)]
        ended_by_fix = bool(np.any((fixes >= te - 5) & (fixes <= te + 3)))
        rows.append({'bag': b, 't_start': round(ts - t0), 'dur_s': round(te - ts, 1), 'e_med': round(float(np.median(a[s:e])), 2),
                     'e_max': round(float(a[s:e][np.argmax(np.abs(a[s:e]))]), 2), 'fixes_inside': len(inside),
                     'ended_by_fix': ended_by_fix, 'mse_share_of_bag': round(float(np.sum(a[s:e] ** 2) / np.sum(a ** 2)), 3)})
df = pd.DataFrame(rows)
df.to_csv('c_episodes_val.csv', index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False))
print('episodes >= 3 s:', len(df), '| ended by a fix:', int(df.ended_by_fix.sum()), '| with >= 1 fix inside that did not end it:',
      int((df.fixes_inside > 0).sum()))
