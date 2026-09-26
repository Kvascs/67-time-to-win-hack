"""Error budget: drift-free estimate of the position timing offset on the check bag.

Around every stop the along error is compared between standstill and the few seconds of motion right before
the stop (braking) and right after the departure: over those ~10-40 m the odometry drift is negligible, so the
change of the along error with speed is the timing offset tau (along = ... + tau * v).
    python analysis/ideas_check/error_budget/timing_stops.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for tag in ('', '_lead_0.02'):
    P = pd.read_parquet(HERE / f'pairs_pos{tag}.parquet')
    P = P[P.t < 1270].reset_index(drop=True)
    lm = pd.read_csv(HERE / f'lm_check{tag}.csv')
    fix = lm.t.to_numpy()
    t, v, al = P.t.to_numpy(), P.v_ref.to_numpy(), P.along.to_numpy()
    still = v < 0.01
    m = np.r_[False, still, False]
    d = np.diff(m.astype(int))
    xs, ys = [], []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
        if t[b] - t[a] < 3.0:
            continue
        # before the stop: 6 s of braking, standstill level before any fix at this stop
        f_in = fix[(fix >= t[a]) & (fix <= t[b])]
        t_fix = f_in[0] if len(f_in) else t[b]
        lvl0 = np.median(al[(t >= t[a] + 0.5) & (t < min(t_fix, t[a] + 1.5))]) if t_fix > t[a] + 0.6 else np.nan
        k0 = (t >= t[a] - 6.0) & (t < t[a] - 0.2) & (v > 0.5)
        if np.isfinite(lvl0) and k0.sum() > 20:
            xs.append(v[k0]); ys.append(al[k0] - lvl0)
        # after the departure: standstill level after the fix, 6 s of acceleration
        lvl1 = np.median(al[(t >= max(t[b] - 1.5, t_fix + 1.0)) & (t <= t[b])]) if t[b] > t_fix + 1.5 else np.nan
        k1 = (t > t[b] + 0.2) & (t <= t[b] + 6.0) & (v > 0.5)
        if np.isfinite(lvl1) and k1.sum() > 20:
            xs.append(v[k1]); ys.append(al[k1] - lvl1)
    x, y = np.concatenate(xs), np.concatenate(ys)
    tau = np.sum(x * y) / np.sum(x * x)
    res = y - tau * x
    # bootstrap over windows
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(500):
        idx = rng.integers(0, len(xs), len(xs))
        xx = np.concatenate([xs[i] for i in idx]); yy = np.concatenate([ys[i] for i in idx])
        bs.append(np.sum(xx * yy) / np.sum(xx * xx))
    print(f'[{tag or "base"}] windows {len(xs)}, samples {len(x)}: tau = {tau:+.4f} s '
          f'(bootstrap 90 %: {np.percentile(bs, 5):+.4f} .. {np.percentile(bs, 95):+.4f}), resid RMSE {np.sqrt(np.mean(res**2)):.3f} m')
