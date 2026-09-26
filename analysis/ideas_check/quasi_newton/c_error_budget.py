"""Experiment (c), cheap part: what could a fixed-lag smoother with retroactive landmark association fix at most?

Runs final2 with the defaults on the 17 VAL bags (train-only maps, GNSS only in the first 5 s), keeps the outputs,
and splits the along-track error of the published base_link (vs the RTK two-antenna reference) into
  * "gross" samples |e_along| > 1 m (association / drift / reference failures: the only part a smarter,
    retroactive association could remove),
  * the rest (landmark spread, drift between fixes, reference noise: a causal smoother cannot remove it).
Landmark fixes = rising edges of flag bit 19 (kFlagLandmark); for each fix the along error 2 s before and
3 s after is compared (a fix that leaves |e| > 1 m or makes it worse by > 0.5 m is a suspicious association).

    python c_error_budget.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a_tune  # noqa: E402
import qn_harness as H  # noqa: E402

GROSS = 1.0


def along_series(bag):
    o = pd.read_pickle(H.KEEP / f'{bag}.pkl')
    o = o.drop_duplicates('stamp_ns').sort_values('stamp_ns')
    R = pd.read_pickle(H.SCR / 'ref' / f'{bag}.pkl')
    bl = R['bl']
    op = o[o.pos_valid == 1]
    t_op = op.stamp_ns.to_numpy() * 1e-9
    i, ok = H._nearest(bl.t.to_numpy(), t_op)
    e = op[['x', 'y', 'z']].to_numpy()[i[ok]] - bl[['x', 'y', 'z']].to_numpy()[ok]
    yaw = bl.yaw.to_numpy()[ok]
    along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
    t = bl.t.to_numpy()[ok]
    lm = (o['flags'].to_numpy().astype(np.int64) >> 19) & 1
    ot = o.stamp_ns.to_numpy() * 1e-9
    fixes = ot[np.flatnonzero(np.diff(lm) == 1) + 1]
    return t, along, fixes, t_op[0]


def main():
    bags = H.splits()['val']
    missing = [b for b in bags if not (H.KEEP / f'{b}.pkl').exists()]
    if missing:
        ev = H.Evaluator()
        try:
            ev.evaluate_many(missing, [a_tune.sets_of(np.zeros(len(a_tune.PARAMS)))], keep=True)
        finally:
            ev.close()
    rows, fixrows = [], []
    for b in bags:
        if not (H.KEEP / f'{b}.pkl').exists():
            continue
        R = pd.read_pickle(H.SCR / 'ref' / f'{b}.pkl')
        if R['bl'] is None:
            continue
        t, a, fixes, t0 = along_series(b)
        g = np.abs(a) > GROSS
        mse = np.mean(a ** 2)
        clip = np.clip(a, -GROSS, GROSS)
        # gross episodes: start cause
        starts = np.flatnonzero(np.diff(np.r_[0, g.astype(int)]) == 1)
        cause = {'after_fix': 0, 'start_of_run': 0, 'between_fixes': 0}
        for s in starts:
            ts = t[s]
            if ts - t0 < 60:
                cause['start_of_run'] += 1
            elif len(fixes) and np.any((ts - fixes >= -1.0) & (ts - fixes <= 8.0)):
                cause['after_fix'] += 1
            else:
                cause['between_fixes'] += 1
        for tf in fixes:
            pre = a[(t > tf - 3) & (t < tf - 1)]
            post = a[(t > tf + 2) & (t < tf + 4)]
            if len(pre) and len(post):
                fixrows.append({'bag': b, 't': tf - t0, 'pre': float(np.median(pre)), 'post': float(np.median(post))})
        rows.append({'bag': b, 'along_rmse': np.sqrt(mse), 'along_rmse_clip1m': float(np.sqrt(np.mean(clip ** 2))),
                     'gross_time_share': float(g.mean()), 'gross_mse_share': float(np.sum(a[g] ** 2) / np.sum(a ** 2)),
                     'n_fix': len(fixes), 'n_gross_episodes': len(starts), **{f'ep_{k}': v for k, v in cause.items()}})
    df = pd.DataFrame(rows)
    fx = pd.DataFrame(fixrows)
    df.to_csv(HERE / 'c_error_budget_val.csv', index=False)
    fx.to_csv(HERE / 'c_fixes_val.csv', index=False)
    pd.set_option('display.width', 220)
    print(df.round(3).to_string(index=False))
    print(f'\nVAL {len(df)} bags: along RMSE mean {df.along_rmse.mean():.3f} median {df.along_rmse.median():.3f}; '
          f'if every |e| > {GROSS} m were clipped to {GROSS} m: mean {df.along_rmse_clip1m.mean():.3f} '
          f'median {df.along_rmse_clip1m.median():.3f}')
    print(f'gross time share: mean {df.gross_time_share.mean():.3f}; gross share of along MSE (pooled): '
          f'{np.average(df.gross_mse_share, weights=df.along_rmse ** 2):.3f}')
    print('gross episodes by start:', {c: int(df[c].sum()) for c in df.columns if c.startswith('ep_')})
    if len(fx):
        bad = (np.abs(fx.post) > GROSS) | (np.abs(fx.post) > np.abs(fx.pre) + 0.5)
        print(f'landmark fixes: {len(fx)}; |post| median {np.median(np.abs(fx.post)):.2f} m (pre {np.median(np.abs(fx.pre)):.2f}); '
              f'suspicious (post > {GROSS} m or worse by > 0.5 m): {int(bad.sum())}')
        print(fx[bad].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
