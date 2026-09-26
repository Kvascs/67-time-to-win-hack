"""Product e: eco-driving indicators per run from the notch, the estimated speed and the traction table.

Tractive work per unit mass (J/kg) = sum over traction notches of (a*(n, v) - a*(0, v)) * v * dt,
where a*(n, v) is the package traction table (level-track acceleration per notch and speed,
running resistance included, so a*(0, v) is the coasting deceleration) and v the estimated speed.
Braking work uses the brake notches the same way. The estimator's g is NOT applied in the main
proxy (if the drive force per notch does not depend on load, energy = force x distance does not
either); the variant with g is reported for comparison. Coasting share = time at notch 0 among
time above 1 m/s.

Runs differ mostly by how fast and how often they stop, so the eco signal is the residual of
tractive work per km after regressing it on mean running speed, stops per km and direction.

outputs: out/eco_runs.csv, fig/e_ecodriving.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (F_STANDSTILL, OUT, TRACTION, bag_meta, long_bags, load_run, ref_aligned, savefig, segments,
                    style)

MASS_T = 27.5  # empty mass, organisers


class Lut:
    def __init__(self, path=TRACTION):
        df = pd.read_csv(path, comment='#')
        self.v = np.array([float(c) for c in df.columns[1:]])
        self.tab = {int(n): df.iloc[i, 1:].to_numpy(float) for i, n in enumerate(df.notch)}

    def __call__(self, notch: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros(len(v))
        n = np.clip(np.nan_to_num(notch), -15, 15).round().astype(int)
        for k in np.unique(n):
            m = n == k
            out[m] = np.interp(v[m], self.v, self.tab[k])
        return out


def work(t, v, notch, g, lut: Lut) -> dict:
    dt = np.clip(np.diff(t, append=t[-1]), 0, 0.5)
    n = np.nan_to_num(notch).round()
    a = lut(n, v)
    a0 = lut(np.zeros_like(n), v)
    trac = np.where(n > 0, np.maximum(a - a0, 0), 0.0)
    brake = np.where(n < 0, np.maximum(a0 - a, 0), 0.0)
    fast = v > 1.0
    return {'W_trac': float(np.sum(trac * v * dt)), 'W_trac_g': float(np.sum(g * trac * v * dt)),
            'W_brake': float(np.sum(brake * v * dt)), 'notch_s': float(np.sum(np.maximum(n, 0) * dt)),
            'dist_km': float(np.sum(v * dt) / 1e3), 't_move': float(np.sum(dt[v > 0.3])),
            'coast_share': float(np.sum(dt[fast & (n == 0)]) / max(np.sum(dt[fast]), 1e-9)),
            'trac_share': float(np.sum(dt[fast & (n > 0)]) / max(np.sum(dt[fast]), 1e-9)),
            'brake_share': float(np.sum(dt[fast & (n < 0)]) / max(np.sum(dt[fast]), 1e-9)),
            'notch_changes': int(np.sum(np.diff(n) != 0))}


def main():
    meta = bag_meta()
    lut = Lut()
    rows = []
    for bag in long_bags():
        o = load_run(bag)
        t, v = o.t.to_numpy(), o.v.to_numpy()
        r = work(t, v, o.notch.to_numpy(), o.g.to_numpy(), lut)
        fl = o['flags'].to_numpy()
        r['stops'] = len([1 for i, j in segments((fl & F_STANDSTILL) > 0, t, merge_gap=1.0, min_dur=3.0)
                          if 0 < i and j < len(t) - 1])
        dirs = o.dir[(v > 1) & (o.dir != '')]
        r['dir'] = dirs.mode().iloc[0] if len(dirs) else ''
        r['bag'] = bag
        if meta.loc[bag, 'has_gnss']:
            q = ref_aligned(bag)
            q = q[np.isfinite(q.notch)]
            rg = work(q.t.to_numpy(), q.v_ref.to_numpy(), q.notch.to_numpy(), np.ones(len(q)), lut)
            r['W_trac_gnss'] = rg['W_trac']
            r['dist_km_gnss'] = rg['dist_km']
        rows.append(r)
    df = pd.DataFrame(rows).set_index('bag').join(meta[['vehicle', 'date', 'hour', 'has_gnss']])
    df['v_run_kmh'] = df.dist_km * 1e3 / df.t_move * 3.6
    df['stops_per_km'] = df.stops / df.dist_km
    df['Wkm'] = df.W_trac / df.dist_km
    df['kwh_km'] = df.Wkm * MASS_T * 1e3 / 3.6e6
    df['brake_over_trac'] = df.W_brake / df.W_trac
    # expected tractive work from speed, stops and direction (ordinary least squares)
    X = np.column_stack([np.ones(len(df)), df.v_run_kmh, df.stops_per_km, (df.dir == 'EB').astype(float),
                         (df.dir == '').astype(float)])
    beta, *_ = np.linalg.lstsq(X, df.Wkm.to_numpy(), rcond=None)
    df['Wkm_expected'] = X @ beta
    df['eco_resid_pct'] = 100 * (df.Wkm / df.Wkm_expected - 1)
    df.to_csv(OUT / 'eco_runs.csv', float_format='%.4f')

    print(f'runs {len(df)}, {df.dist_km.sum():.0f} km')
    print(f'tractive work per km: median {df.Wkm.median():.0f} J/kg/km (IQR {df.Wkm.quantile(0.25):.0f}..{df.Wkm.quantile(0.75):.0f}), '
          f'range {df.Wkm.min():.0f}..{df.Wkm.max():.0f}; = {df.kwh_km.median():.2f} kWh/km at the wheel for {MASS_T} t')
    print(f'p90/p10 of tractive work per km: {df.Wkm.quantile(0.9) / df.Wkm.quantile(0.1):.2f}')
    print(f'coasting share: median {df.coast_share.median():.3f} (IQR {df.coast_share.quantile(0.25):.3f}..{df.coast_share.quantile(0.75):.3f}); '
          f'traction {df.trac_share.median():.3f}, braking {df.brake_share.median():.3f}')
    print(f'braking work / tractive work: median {df.brake_over_trac.median():.2f}')
    ss_tot = ((df.Wkm - df.Wkm.mean()) ** 2).sum()
    ss_res = ((df.Wkm - df.Wkm_expected) ** 2).sum()
    print(f'regression Wkm ~ speed + stops/km + dir: R2 {1 - ss_res / ss_tot:.2f}; coef per km/h {beta[1]:.1f}, '
          f'per stop/km {beta[2]:.1f}, EB {beta[3]:+.1f}; residual std {df.eco_resid_pct.std():.1f} %')
    print(f'residual vs coasting share: corr {np.corrcoef(df.eco_resid_pct, df.coast_share)[0, 1]:+.2f}; '
          f'vs brake/trac ratio {np.corrcoef(df.eco_resid_pct, df.brake_over_trac)[0, 1]:+.2f}')
    print(f'W with g vs without: corr {np.corrcoef(df.W_trac_g, df.W_trac)[0, 1]:.3f}, median ratio {(df.W_trac_g / df.W_trac).median():.3f}; '
          f'notch-seconds vs W: corr {np.corrcoef(df.notch_s / df.dist_km, df.Wkm)[0, 1]:.2f}')
    g = df[np.isfinite(df.get('W_trac_gnss', pd.Series(dtype=float)))]
    print(f'estimator vs GNSS speed in the same proxy: median |dW| {(g.W_trac / g.W_trac_gnss - 1).abs().median() * 100:.2f} %, '
          f'max {(g.W_trac / g.W_trac_gnss - 1).abs().max() * 100:.2f} %, distance {(g.dist_km / g.dist_km_gnss - 1).abs().median() * 100:.2f} % (n {len(g)})')
    print(df.groupby(['vehicle', 'date'])[['Wkm', 'coast_share', 'v_run_kmh', 'eco_resid_pct']].median().round(3).to_string())
    cols = ['vehicle', 'date', 'dir', 'dist_km', 'v_run_kmh', 'stops', 'Wkm', 'coast_share', 'brake_over_trac', 'eco_resid_pct']
    print('best 5 (least work for their speed and stops):')
    print(df.sort_values('eco_resid_pct').head(5)[cols].round(2).to_string())
    print('worst 5:')
    print(df.sort_values('eco_resid_pct').tail(5)[cols].round(2).to_string())
    best = df.eco_resid_pct.quantile(0.25)
    print(f'saving if runs above the best quartile residual reached it: {np.mean(np.maximum(df.eco_resid_pct - best, 0)):.1f} % of work')

    # ---------------- figure ----------------
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cols_ = {'WB': '#2a78d6', 'EB': '#eb6834', '': '#52514e'}
    names = {'WB': 'на запад', 'EB': 'на восток', '': 'без привязки (нет GNSS)'}
    for d, grp in df.groupby('dir'):
        axes[0].scatter(grp.v_run_kmh, grp.Wkm, s=16, color=cols_[d], lw=0, alpha=0.8, label=names[d])
        axes[1].scatter(grp.coast_share * 100, grp.eco_resid_pct, s=16, color=cols_[d], lw=0, alpha=0.8, label=names[d])
    axes[0].set_xlabel('средняя скорость в движении, км/ч')
    axes[0].set_ylabel('тяговая работа, Дж/кг на км')
    axes[0].set_title('Тяговая работа против темпа поездки', loc='left', fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].axhline(0, color='#52514e', lw=0.6)
    axes[1].set_xlabel('доля выбега (позиция 0) при v > 1 м/с, %')
    axes[1].set_ylabel('перерасход относительно ожидаемого, %')
    axes[1].set_title('Эко-остаток против доли выбега', loc='left', fontsize=9)
    fig.tight_layout()
    savefig(fig, 'e_ecodriving.png')


if __name__ == '__main__':
    main()
