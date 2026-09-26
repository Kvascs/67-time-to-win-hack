"""Does the dwell time of a stop tell a known place (platform / signal) from a random stop?
(teammate idea: use dwell as an association feature for stop landmarks)

Input: analysis/map_build/map/stops_events.csv (914 stops of all runs) + stops.csv (clusters, class).
Only mid-run stops (not the first / last stop of a bag). Landmark = cluster used by the estimator
(platform / signal / terminal, seen in >= 3 runs, robust spread <= 1 m).

Result (26.09): medians 20.6 s (platform), 20.6 s (signal), 18.8 s (other); per-bin likelihood
ratio platform/other is 0.6-1.6 over 6-45 s, only extreme dwells (< 6 s or 45-60 s) are informative.
Too weak to pay for deferring the fix to departure -> not used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

MAP = Path(__file__).resolve().parent / 'map_build' / 'map'

e = pd.read_csv(MAP / 'stops_events.csv')
st = pd.read_csv(MAP / 'stops.csv')
e = e.merge(st[['cluster', 'cls', 'n_runs', 's_robust_std']], on='cluster', how='left')
e = e[~e['first'] & ~e['last']]
lm = e.cls.isin(['platform', 'signal', 'terminal']) & (e.n_runs >= 3) & (e.s_robust_std <= 1.0)
e['grp'] = np.where(lm, e.cls, 'other')
print(e.groupby('grp').dwell.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
bins = [0, 3, 6, 10, 15, 20, 30, 45, 60, 120, 1e9]
t = pd.crosstab(pd.cut(e.dwell, bins), e.grp, normalize='columns')
t['LR_platform_vs_other'] = t['platform'] / t['other'].replace(0, np.nan)
print(t.round(3).to_string())

# ---- "SLAM on the line" (teammate idea): learn new stop places online, inside one run ----
# Opportunity = stops at places that are NOT landmarks but repeat within the same run.
# Result (26.09): 93 non-landmark mid-run stops in 46 bags on the main cycle; the same unmapped place
# twice within one run: 3 cases (7 stops) in all runs; 88 % of stops are already at landmarks.
# -> no measurable gain online; the offline map rebuild from new runs (analysis/map_build) is the SLAM.
m = e[e.edge == 'main']
lm_m = m.cls.isin(['platform', 'signal', 'terminal']) & (m.n_runs >= 3) & (m.s_robust_std <= 1.0)
rep = m[~lm_m].groupby(['bag', 'cluster']).size()
print(f'non-landmark stops {int((~lm_m).sum())}; repeated within a run: {int((rep >= 2).sum())} cases; '
      f'share at landmarks {lm_m.mean():.2f}')
