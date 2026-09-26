"""Product b: is the estimator's traction gain g a usable passenger-load proxy?

g multiplies the tabulated drive acceleration a*(notch, v) (traction AND electric brake):
realised = g * tabulated. If the drive force per notch did not depend on load, g would be
m_ref / m, i.e. about -3.5 % per tonne of payload on a 27.5 t car. Per run we take the median g over
motion after a 120 s warm-up (g starts at 1 with sigma 0.08 and drifts slowly, q = 2e-5 1/s).

Checks against things that should move with load (local hour, weekday rush hours, dwell time at
platforms = boarding volume) and things that should not (vehicle, direction, wheel scale k,
disturbance d). Spearman correlations with p-values.

outputs: out/load_gain_runs.csv, fig/b_load_gain.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from common import (F_STANDSTILL, OUT, bag_meta, long_bags, load_run, platforms, savefig, segments, style,
                    track_map)

WARMUP_S = 120.0


def dwell_times(o: pd.DataFrame, plats: pd.DataFrame, L: float) -> list[float]:
    """Standstill durations at platforms (not terminals), located by the estimator."""
    t = o.t.to_numpy()
    s = o.s_map.to_numpy()
    fl = o['flags'].to_numpy()
    out = []
    ps = plats[plats.cls == 'platform'].s.to_numpy()
    for i, j in segments((fl & F_STANDSTILL) > 0, t, merge_gap=1.0, min_dur=3.0):
        if not np.isfinite(s[i]) or i == 0 or j >= len(t) - 1:
            continue  # the run starts or ends standing: not a dwell
        dmin = np.min(np.abs((ps - s[i] + L / 2) % L - L / 2))
        if dmin < 20.0:
            out.append(t[j] - t[i])
    return out


def main():
    meta = bag_meta()
    m = track_map()
    plats = platforms()
    rows = []
    for bag in long_bags():
        o = load_run(bag)
        t = o.t.to_numpy()
        mv = (o.v.to_numpy() > 1.0) & (t > WARMUP_S)
        g = o.g.to_numpy()[mv]
        tr = mv & (o.notch.to_numpy() > 0)
        dw = dwell_times(o, plats, m.L)
        dirs = o.dir[mv & (o.dir != '')]
        rows.append({'bag': bag, 'g_med': np.median(g), 'g_p10': np.quantile(g, 0.1), 'g_p90': np.quantile(g, 0.9),
                     'g_trac': np.median(o.g.to_numpy()[tr]), 'g_end': o.g.iloc[-1], 'k_end': o.k.iloc[-1],
                     'd_med': np.median(o.d.to_numpy()[mv]), 'dwell_mean': np.mean(dw) if dw else np.nan,
                     'dwell_n': len(dw), 'dir': dirs.mode().iloc[0] if len(dirs) else ''})
    df = pd.DataFrame(rows).set_index('bag').join(meta[['vehicle', 'date', 'hour', 'local_start', 'has_gnss']])
    df['peak'] = df.hour.between(7, 10) | df.hour.between(16.5, 19.5)
    df.to_csv(OUT / 'load_gain_runs.csv', float_format='%.5f')

    print(f'runs {len(df)}; g_med over runs: median {df.g_med.median():.3f}, IQR {df.g_med.quantile(0.25):.3f}..'
          f'{df.g_med.quantile(0.75):.3f}, range {df.g_med.min():.3f}..{df.g_med.max():.3f}')
    print(f'within-run spread (p90 - p10) median {(df.g_p90 - df.g_p10).median():.3f}; '
          f'g_trac vs g_med corr {np.corrcoef(df.g_trac, df.g_med)[0, 1]:.3f}')
    iqr = df.g_med.quantile(0.75) - df.g_med.quantile(0.25)
    print(f'IQR read as payload at -3.5 %/t: {iqr / 0.035:.1f} t; full range: {(df.g_med.max() - df.g_med.min()) / 0.035:.1f} t')
    print(df.groupby(['vehicle', 'date']).g_med.agg(['size', 'median', 'min', 'max']).round(3).to_string())
    a, b = df[df.vehicle == '30618'].g_med, df[df.vehicle == '30639'].g_med
    print(f'vehicle 30618 vs 30639: median {a.median():.3f} vs {b.median():.3f}, Mann-Whitney p {mannwhitneyu(a, b).pvalue:.3g}')
    for veh in ('30618', '30639', 'all'):
        q = df if veh == 'all' else df[df.vehicle == veh]
        out = []
        for col in ('hour', 'dwell_mean', 'k_end', 'd_med'):
            z = q[['g_med', col]].dropna()
            if len(z) > 5:
                rho, p = spearmanr(z.g_med, z[col])
                out.append(f'{col}: rho {rho:+.2f} (p {p:.2g}, n {len(z)})')
        pk = q[q.peak].g_med
        op = q[~q.peak].g_med
        if len(pk) > 2 and len(op) > 2:
            out.append(f'peak {pk.median():.3f} (n {len(pk)}) vs off-peak {op.median():.3f} (n {len(op)}), '
                       f'MW p {mannwhitneyu(pk, op).pvalue:.2g}')
        print(f'[{veh}] ' + '; '.join(out))
    # does dwell time explain g beyond the time of day? OLS g ~ 1 + hour + dwell within each vehicle
    for veh in ('30618', '30639'):
        z = df[(df.vehicle == veh)][['g_med', 'hour', 'dwell_mean']].dropna()
        X = np.column_stack([np.ones(len(z)), z.hour, z.dwell_mean])
        beta, res, *_ = np.linalg.lstsq(X, z.g_med.to_numpy(), rcond=None)
        resid = z.g_med.to_numpy() - X @ beta
        s2 = resid @ resid / (len(z) - 3)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        print(f'[{veh}] OLS g ~ hour + dwell (n {len(z)}): per hour {beta[1]:+.4f} (t {beta[1] / se[1]:+.1f}), '
              f'per 10 s dwell {10 * beta[2]:+.4f} (t {beta[2] / se[2]:+.1f})')
    q = df[df.dir != '']
    wb, eb = q[q.dir == 'WB'].g_med, q[q.dir == 'EB'].g_med
    print(f'direction WB {wb.median():.3f} (n {len(wb)}) vs EB {eb.median():.3f} (n {len(eb)}), MW p {mannwhitneyu(wb, eb).pvalue:.2g}')
    # day-level structure: how much of the variance is between (vehicle, date) groups?
    grp = df.groupby(['vehicle', 'date']).g_med
    within = (df.g_med - grp.transform('median')).abs().median()
    print(f'median |g - day median| {within:.3f}; spread of day medians {grp.median().std():.3f}')

    # ---------------- figure ----------------
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cols = {'30618': '#2a78d6', '30639': '#eb6834'}
    for veh, grp_ in df.groupby('vehicle'):
        axes[0].scatter(grp_.hour, grp_.g_med, s=16, color=cols[veh], lw=0, alpha=0.8, label=f'вагон {veh}')
        axes[1].scatter(grp_.dwell_mean, grp_.g_med, s=16, color=cols[veh], lw=0, alpha=0.8, label=f'вагон {veh}')
    for ax in axes:
        ax.axhline(1.0, color='#52514e', lw=0.6)
    for x0, x1 in ((7, 10), (16.5, 19.5)):
        axes[0].axvspan(x0, x1, color='#52514e', alpha=0.06, lw=0)
    axes[0].set_xlabel('местное время старта поездки, ч (серые полосы — часы пик)')
    axes[0].set_ylabel('медиана g за поездку')
    axes[0].set_title('Тяговый коэффициент g по поездкам', loc='left', fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel('средняя стоянка на платформах, с (оценщик)')
    axes[1].set_title('g против времени стоянок (посадка/высадка)', loc='left', fontsize=9)
    fig.tight_layout()
    savefig(fig, 'b_load_gain.png')


if __name__ == '__main__':
    main()
