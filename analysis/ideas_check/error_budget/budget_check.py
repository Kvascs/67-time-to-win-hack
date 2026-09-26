"""Error budget, step 4: the judge-metric budget of the check bag 30618_88aea4d9 in one table.

Inputs: pairs_pos.parquet / pairs_vel.parquet / out_check.parquet (run_check.py), check_segments.csv
(analyze_check.py), whatif_check.csv (whatif_check.py). Every number is a share of the squared error of the
judge metric (3-D position MSE, speed MSE). Writes budget_check.csv.

    python analysis/ideas_check/error_budget/budget_check.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
END = 1270.0
TAU_STOPS = 0.0137  # timing_stops.py: drift-free position timing offset (90 % 0.004..0.023 s)


def main():
    P_all = pd.read_parquet(HERE / 'pairs_pos.parquet')
    V_all = pd.read_parquet(HERE / 'pairs_vel.parquet')
    O = pd.read_parquet(HERE / 'out_check.parquet')
    G = pd.read_csv(HERE / 'check_segments.csv')
    N = len(P_all)
    sse_all = np.sum(P_all.dist ** 2)
    P = P_all[P_all.t < END]
    rows = []

    def add(metric, source, sse, note=''):
        rows.append({'metric': metric, 'source': source, 'mse_m2': sse / N if metric.startswith('pos') else sse,
                     'share_full_pct': sse / sse_all * 100 if metric.startswith('pos') else np.nan, 'note': note})

    tail = P_all.t >= END
    add('pos_full', 'stub track missing from the map (t >= 1270 s)', np.sum(P_all.dist[tail] ** 2))
    # t < END: along split by drift segment, cross, z
    seg_cause = {267.195: 'no place fix over 2120 m (s 881-3000), wheel scale k off by ~+0.13 %',
                 582.195: 'residual after the PDA fix at 3000 m (beta0 keeps 9 % of the prior)',
                 845.995: 'fix at a signal stop 2 m short of the signal landmark (-0.93 m) and weak later fixes',
                 916.195: 'fix at a signal stop 2 m short of the signal landmark (-0.93 m) and weak later fixes',
                 0.0: 'start: -0.75 m lost in the first 50 m (east loop, kappa 0.03-0.056)',
                 1179.695: 'approach to the stub: -1.1 m in the curves at s 5340-5375 (kappa 0.02-0.037)'}
    Pt = P.t.to_numpy()
    covered = np.zeros(len(P), bool)
    agg = {}
    for _, g in G.iterrows():
        key = min(seg_cause, key=lambda k: abs(k - g.t_start))
        k = (Pt >= g.t_start) & (Pt < g.t_end)
        if abs(key - g.t_start) < 1.0 or (key == 0.0 and g.t_start < 1.0):
            agg[seg_cause[key]] = agg.get(seg_cause[key], 0.0) + np.sum(P.along[k] ** 2)
            covered |= k
    for src, sse in agg.items():
        add('pos_full', f'along: {src}', sse)
    add('pos_full', 'along: all other segments (short drift between fixes)', np.sum(P.along[~covered] ** 2))
    add('pos_full', 'cross (t < 1270 s)', np.sum(P.cross ** 2))
    add('pos_full', 'z bias +0.108 m (t < 1270 s)', len(P) * P.dz.mean() ** 2)
    add('pos_full', 'z scatter (t < 1270 s)', np.sum((P.dz - P.dz.mean()) ** 2))
    mse_1270 = np.mean(P.dist ** 2)
    # speed
    tv = V_all.t.to_numpy() + V_all.dt_ms.to_numpy() / 1e3
    ev = V_all.ev.to_numpy()
    ev12 = np.interp(tv - 0.12, O.t, O.v) - V_all.v_ref.to_numpy()
    mse_v = np.mean(ev ** 2)
    add('speed', 'reference lag (our speed vs speed delayed by 0.12 s)', mse_v - np.mean(ev12 ** 2))
    acc = V_all.acc_ref.to_numpy()
    vr = V_all.v_ref.to_numpy()
    reg = np.where((vr < 0.05) & (np.abs(acc) < 0.05), 'standstill',
                   np.where(acc > 0.15, 'accel', np.where(acc < -0.15, 'brake', 'cruise')))
    for r in ('accel', 'brake', 'cruise', 'standstill'):
        add('speed', f'after the lag: {r}', np.sum(ev12[reg == r] ** 2) / len(ev12))
    B = pd.DataFrame(rows)
    B['share_1270_pct'] = np.where(B.metric.eq('pos_full') & ~B.source.str.startswith('stub'),
                                   B.mse_m2 * N / len(P) / mse_1270 * 100, np.nan)
    B['share_speed_pct'] = np.where(B.metric.eq('speed'), B.mse_m2 / mse_v * 100, np.nan)
    B.to_csv(HERE / 'budget_check.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 95)
    print(f'position: full-bag MSE {sse_all / N:.3f} m^2 (RMSE {np.sqrt(sse_all / N):.3f}); t < {END:.0f} s MSE {mse_1270:.4f} '
          f'(RMSE {np.sqrt(mse_1270):.3f}); speed MSE {mse_v:.5f} (RMSE {np.sqrt(mse_v):.4f})')
    print(B.round(5).to_string(index=False))
    print(f'timing (inside the along rows): tau {TAU_STOPS:+.4f} s -> mean((tau v)^2) = '
          f'{np.mean((TAU_STOPS * P.v_ref) ** 2):.4f} m^2 = {np.mean((TAU_STOPS * P.v_ref) ** 2) / mse_1270 * 100:.1f} % of t<1270 MSE')
    W = pd.read_csv(HERE / 'whatif_check.csv')
    print(W.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
