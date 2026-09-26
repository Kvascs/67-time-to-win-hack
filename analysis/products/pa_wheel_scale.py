"""Product a: wheel scale (effective wheel diameter) per run, from the estimator's state k.

k = bogie reading / true speed - 1 in the estimator's units (km/h * 1.00037/3.6, curve-corrected).
A smaller (worn) wheel turns faster at the same speed and reads high, so k ~ -dD/D.
The estimator learns k only at stop landmarks and traction cut-off places (Schmidt state), so
it is available for runs anchored on the map (GNSS in the first 5 s), not for the no-GNSS runs.

References (GNSS, whole run): per-bogie median of wheel / GNSS speed - 1 on straight track at
steady speed > 3 m/s; and the per-run k of analysis/cross_check/scale_lag_per_bag.csv
(DATA_FINDINGS, km/h per m/s, curves included), converted to the same units.

Second signal, no GNSS needed: the front/rear difference slip_f - slip_r published by the
estimator (both relative to the same vehicle speed) = relative diameter difference of the bogies.

outputs: out/wheel_scale_runs.csv, fig/a_wheel_scale.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (F_LANDMARK, FRONT_BOGIE_ALONG, NPZ, OUT, REAR_BOGIE_ALONG, ROOT, WHEEL_KMH_TO_MS,
                    accel_centered, bag_meta, long_bags, load_run, ref_aligned, rising_edges, savefig, segments,
                    style, track_map)

STRAIGHT = 0.002  # |curvature| 1/m (R > 500 m) at both bogies


def gnss_scale(bag: str, o: pd.DataFrame) -> dict:
    r = ref_aligned(bag)
    if r is None:
        return {}
    m = track_map()
    d = np.load(NPZ / f'{bag}.npz')
    t0_abs = o.stamp_ns.iloc[0] * 1e-9
    tv = r.t.to_numpy() + t0_abs
    w = {}
    for key, name in (('vehicle__front_bogie_velocity', 'f'), ('vehicle__rear_bogie_velocity', 'r')):
        a = d[key]
        a = a[np.argsort(a[:, 1])]
        w[name] = np.interp(tv, a[:, 1], a[:, 2] * WHEEL_KMH_TO_MS)
    vr = r.v_ref.to_numpy()
    s = r.s_ref.to_numpy()
    ar = accel_centered(r.t.to_numpy(), vr, 1.0)
    cf = np.abs(m.at(np.nan_to_num(s) + FRONT_BOGIE_ALONG, 'curv'))
    cr = np.abs(m.at(np.nan_to_num(s) + REAR_BOGIE_ALONG, 'curv'))
    flags = r['flags'].fillna(0).astype(np.int64).to_numpy()
    ok = (vr > 3.0) & np.isfinite(s) & (cf < STRAIGHT) & (cr < STRAIGHT) & (np.abs(ar) < 0.3) & ((flags & 0xF) == 0)
    if ok.sum() < 200:
        return {'n_ref': int(ok.sum())}
    kf = np.median(w['f'][ok] / vr[ok]) - 1
    kr = np.median(w['r'][ok] / vr[ok]) - 1
    return {'k_ref_front': kf, 'k_ref_rear': kr, 'k_ref': 0.5 * (kf + kr), 'n_ref': int(ok.sum())}


def main():
    meta = bag_meta()
    m = track_map()
    xc = pd.read_csv(ROOT / 'analysis' / 'cross_check' / 'scale_lag_per_bag.csv').set_index('bag')
    rows = []
    for bag in long_bags():
        o = load_run(bag)
        t = o.t.to_numpy()
        fl = o['flags'].to_numpy()
        n_lm = len(segments((fl & F_LANDMARK) > 0, t, merge_gap=1.0))
        k = o.k.to_numpy()
        row = {'bag': bag, 'k_end': float(k[-1]), 'k_mid': float(np.interp(t[-1] / 2, t, k)), 'landmark_fixes': n_lm}
        # front/rear differential from the published slip ratios (nominal mode, steady motion)
        nominal = (o.mu0.to_numpy() > 0.9) & (o.v.to_numpy() > 3.0) & ((fl & 0xF) == 0) & ((fl & (1 << 12)) == 0)
        steady = np.abs(accel_centered(t, o.v.to_numpy(), 1.0)) < 0.3
        diff = (o.slip_f - o.slip_r).to_numpy()
        sel = nominal & steady
        row['fr_diff_all'] = float(np.median(diff[sel])) if sel.sum() > 200 else np.nan
        s = o.s_map.to_numpy()
        if np.isfinite(s).any():
            cf = np.abs(m.at(np.nan_to_num(s) + FRONT_BOGIE_ALONG, 'curv'))
            cr = np.abs(m.at(np.nan_to_num(s) + REAR_BOGIE_ALONG, 'curv'))
            st = sel & np.isfinite(s) & (cf < STRAIGHT) & (cr < STRAIGHT)
            row['fr_diff_straight'] = float(np.median(diff[st])) if st.sum() > 200 else np.nan
        if meta.loc[bag, 'has_gnss']:
            row.update(gnss_scale(bag, o))
        if bag in xc.index:
            row['k_df'] = xc.loc[bag, 'k_mean'] * WHEEL_KMH_TO_MS - 1  # DATA_FINDINGS per-run k, curves included
        rows.append(row)
    df = pd.DataFrame(rows).set_index('bag').join(meta[['vehicle', 'date', 'hour', 'local_start', 'has_gnss']])
    df = df.sort_values(['vehicle', 'local_start'])
    df['k_ref_fr'] = df.k_ref_front - df.k_ref_rear
    df.to_csv(OUT / 'wheel_scale_runs.csv', float_format='%.6f')

    a = df[df.has_gnss & np.isfinite(df.k_ref)]
    e = (a.k_end - a.k_ref) * 100
    print(f'anchored runs with GNSS scale: {len(a)}; landmark fixes per run median {a.landmark_fixes.median():.0f} '
          f'(min {a.landmark_fixes.min()})')
    print(f'k_end - k_ref [% of speed]: median {e.median():+.3f}, MAE {e.abs().mean():.3f}, RMS {np.sqrt((e ** 2).mean()):.3f}, '
          f'p90 |e| {e.abs().quantile(0.9):.3f}, max |e| {e.abs().max():.3f}; corr {np.corrcoef(a.k_end, a.k_ref)[0, 1]:.3f}')
    e0 = a.k_ref * 100
    print(f'error if k were fixed at the prior 0: MAE {e0.abs().mean():.3f} %, RMS {np.sqrt((e0 ** 2).mean()):.3f} %')
    b = a[np.isfinite(a.k_df)]
    print(f'k_ref (straight) - k_df (DATA_FINDINGS, all track): median {(b.k_ref - b.k_df).median() * 100:+.3f} % '
          f'(expected about +0.06 % from curves), corr {np.corrcoef(b.k_ref, b.k_df)[0, 1]:.3f}, n {len(b)}')
    g = df[df.has_gnss].groupby(['vehicle', 'date'])
    summ = pd.DataFrame({'n': g.size(), 'k_end_med_%': g.k_end.median() * 100, 'k_ref_med_%': g.k_ref.median() * 100,
                         'k_end_min_%': g.k_end.min() * 100, 'k_end_max_%': g.k_end.max() * 100,
                         'fr_diff_%': g.fr_diff_straight.median() * 100, 'fr_ref_%': g.k_ref_fr.median() * 100})
    print(summ.round(3).to_string())
    for veh, date in (('30618', '09-03'), ('30639', '05-05')):
        q = df[(df.vehicle == veh) & (df.date == date) & df.has_gnss]
        if len(q) > 2:
            print(f'{veh} {date}: k_ref by hour', [(round(h, 1), round(k * 100, 2)) for h, k in zip(q.hour, q.k_ref)],
                  f'; spearman(hour, k_end) {q.hour.corr(q.k_end, method="spearman"):+.2f}')
    fr = df[np.isfinite(df.fr_diff_straight) & np.isfinite(df.k_ref_fr)]
    d = (fr.fr_diff_straight - fr.k_ref_fr) * 100
    print(f'front-rear difference, estimator (straight) vs GNSS: MAE {d.abs().mean():.3f} %, corr '
          f'{np.corrcoef(fr.fr_diff_straight, fr.k_ref_fr)[0, 1]:.3f}, n {len(fr)}; '
          f'straight vs all-track estimator diff: MAE {(df.fr_diff_straight - df.fr_diff_all).abs().mean() * 100:.3f} %')
    nog = df[~df.has_gnss]
    print(f'no-GNSS runs: front-rear difference (all track) median {nog.fr_diff_all.median() * 100:+.3f} %, '
          f'IQR {nog.fr_diff_all.quantile(0.25) * 100:+.3f}..{nog.fr_diff_all.quantile(0.75) * 100:+.3f} %, n {len(nog)}')

    # ---------------- figure ----------------
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={'width_ratios': [2.2, 1]})
    ax = axes[0]
    x = np.arange(len(df))
    ax.scatter(x, df.k_ref * 100, s=16, color='#eb6834', lw=0, label='GNSS, прямые участки (контроль)')
    ax.scatter(x, df.k_end * 100, s=16, marker='x', color='#2a78d6', lw=1.1, label='оценщик: k в конце поездки')
    ticks, labels = [], []
    for (veh, date), grp in df.groupby(['vehicle', 'date'], sort=False):
        i = [df.index.get_loc(b) for b in grp.index]
        ticks.append(np.mean(i))
        labels.append(f'{veh}\n{date}')
        ax.axvline(max(i) + 0.5, color='#52514e', lw=0.5, alpha=0.4)
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_ylabel('k, % (показание / истинная − 1)')
    ax.set_title('Масштаб колеса по поездкам (порядок — время старта); без GNSS k не калибруется', loc='left', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    ax = axes[1]
    ax.scatter(fr.k_ref_fr * 100, fr.fr_diff_straight * 100, s=14, color='#2a78d6', lw=0)
    lim = [min(fr.k_ref_fr.min(), fr.fr_diff_straight.min()) * 100 - 0.02, max(fr.k_ref_fr.max(), fr.fr_diff_straight.max()) * 100 + 0.02]
    ax.plot(lim, lim, color='#52514e', lw=0.8)
    ax.set_xlabel('перед − зад по GNSS, %')
    ax.set_ylabel('перед − зад по оценщику, %')
    ax.set_title('Разница диаметров тележек', loc='left', fontsize=9)
    fig.tight_layout()
    savefig(fig, 'a_wheel_scale.png')


if __name__ == '__main__':
    main()
