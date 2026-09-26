"""Product d: safety events from the estimator outputs.

1. Unmodelled deceleration (emergency / track brake): the estimator's flag "wheels agree on an
   acceleration the notch does not explain" (IMM maneuver mode > 0.5, not at standstill).
2. Hard braking: acceleration below -1.5 m/s^2 for at least 0.3 s. Acceleration = centred
   difference of the estimated speed over 1 s. The same rule on the GNSS speed gives the check.
3. Overspeed candidates: a pass through a 50 m bin whose maximum speed exceeds the p95 of the
   OTHER passes through that bin by more than 2 km/h (no official limit table is available, so
   the p95 of everyday driving stands in for it). Consecutive bins of one run form one event.
   Needs out/speed_passes.csv from pf_speed_profile.py.

outputs: out/safety_unmodeled.csv, out/safety_hard_braking.csv, out/safety_overspeed.csv,
         fig/d_safety_events.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (F_UNMODELED, OUT, accel_centered, bag_meta, long_bags, load_run, ref_aligned, savefig,
                    segments, style, track_map)

HARD = -1.5          # m/s^2
HARD_MIN_S = 0.3
OVER_MARGIN = 2.0 / 3.6  # m/s above the leave-one-out p95
MATCH_S = 1.5


def hard_events(t, a, v, min_s=HARD_MIN_S):
    ok = np.isfinite(a) & (v > 0.5)
    out = []
    for i, j in segments(ok & (a < HARD), t, merge_gap=1.0, min_dur=min_s):
        k = i + int(np.argmin(a[i:j + 1]))
        out.append((i, j, k))
    return out


def main():
    meta = bag_meta()
    m = track_map()
    unm, hard, hard_ref, runs = [], [], [], []
    for bag in long_bags():
        o = load_run(bag)
        t = o.t.to_numpy()
        v = o.v.to_numpy()
        a = accel_centered(t, v, 1.0)
        fl = o['flags'].to_numpy()
        notch = o.notch.to_numpy()
        s = o.s_map.to_numpy()
        r = ref_aligned(bag) if meta.loc[bag, 'has_gnss'] else None
        if r is not None:
            tr = r.t.to_numpy()
            vr = r.v_ref.to_numpy()
            ar = accel_centered(tr, vr, 1.0)
        # ---- unmodelled deceleration episodes ----
        for i, j in segments((fl & F_UNMODELED) > 0, t, merge_gap=2.0):
            w = slice(max(i - 20, 0), min(j + 40, len(t)))  # +-1-2 s around the flag
            nt = notch[i:j + 1]
            row = {'bag': bag, 't0': t[i], 'dur_s': t[j] - t[i], 'v0_kmh': v[i] * 3.6, 's_map': s[i],
                   'a_min_est': float(np.nanmin(a[w])), 'notch_min': np.nanmin(nt), 'notch_max': np.nanmax(nt)}
            if r is not None:
                mm = (tr >= t[i] - 1) & (tr <= t[j] + 2)
                row['a_min_gnss'] = float(np.nanmin(ar[mm])) if mm.any() else np.nan
            unm.append(row)
        # ---- hard braking ----
        ev = hard_events(t, a, v)
        for i, j, k in ev:
            row = {'bag': bag, 't0': t[i], 'dur_s': t[j] - t[i], 'v0_kmh': v[i] * 3.6, 'a_min': a[k], 's_map': s[k],
                   'notch_min': np.nanmin(notch[i:j + 1]), 'unmodeled_flag': bool(((fl[i:j + 1] & F_UNMODELED) > 0).any())}
            if r is not None:
                mm = (tr >= t[i] - MATCH_S) & (tr <= t[j] + MATCH_S)
                row['a_min_gnss_near'] = float(np.nanmin(ar[mm])) if mm.any() else np.nan
            hard.append(row)
        n_ref = 0
        if r is not None:
            evr = hard_events(tr, ar, vr)
            n_ref = len(evr)
            for i, j, k in evr:
                mm = (t >= tr[i] - MATCH_S) & (t <= tr[j] + MATCH_S)
                hard_ref.append({'bag': bag, 't0': tr[i], 'a_min_gnss': ar[k],
                                 'a_min_est_near': float(np.nanmin(a[mm])) if mm.any() else np.nan,
                                 'matched': bool(any((t[i2] <= tr[j] + MATCH_S) and (t[j2] >= tr[i] - MATCH_S)
                                                     for i2, j2, _ in ev))})
        runs.append({'bag': bag, 'dist_km': float(np.trapezoid(v, t) / 1e3), 'hard': len(ev), 'hard_ref': n_ref,
                     'unmodeled': len(segments((fl & F_UNMODELED) > 0, t, merge_gap=2.0))})
    unm = pd.DataFrame(unm)
    hard = pd.DataFrame(hard)
    hard_ref = pd.DataFrame(hard_ref)
    runs = pd.DataFrame(runs).set_index('bag').join(meta[['vehicle', 'date', 'has_gnss']])
    for df in (unm, hard):
        if len(df):
            ok = np.isfinite(df.s_map)
            df['dir'] = np.where(ok, m.direction(df.s_map.fillna(0)), '')
            df['route_m'] = np.where(ok, m.route_at(df.s_map.fillna(0)), np.nan)
    unm.to_csv(OUT / 'safety_unmodeled.csv', index=False, float_format='%.3f')
    hard.to_csv(OUT / 'safety_hard_braking.csv', index=False, float_format='%.3f')

    km = runs.dist_km.sum()
    print(f'runs {len(runs)}, {km:.0f} km')
    print(f'--- unmodelled deceleration: {len(unm)} episodes in {(runs.unmodeled > 0).sum()} runs')
    if len(unm):
        print(unm.round(2).to_string(index=False))
    print(f'--- hard braking (a < {HARD} m/s^2 >= {HARD_MIN_S} s): {len(hard)} events, {100 * len(hard) / km:.1f} per 100 km, '
          f'in {(runs.hard > 0).sum()}/{len(runs)} runs; median v0 {hard.v0_kmh.median():.0f} km/h, '
          f'median a_min {hard.a_min.median():.2f}, min {hard.a_min.min():.2f} m/s^2')
    print('notch at hard braking (min over event):', hard.notch_min.value_counts().sort_index().to_dict())
    print('hard braking per run by vehicle/date:')
    print(runs.groupby(['vehicle', 'date']).hard.agg(['size', 'sum', 'mean', 'max']).round(2).to_string())
    g = runs[runs.has_gnss]
    he = hard[hard.bag.isin(g.index)]
    if len(hard_ref):
        prec = float((he.a_min_gnss_near < HARD + 0.3).mean())
        rec = float(hard_ref.matched.mean())
        print(f'GNSS check on {len(g)} GNSS runs: estimator events {len(he)}, GNSS events {len(hard_ref)}; '
              f'recall (GNSS events also found by the estimator) {rec:.2f}; '
              f'share of estimator events where GNSS shows a < {HARD + 0.3} within +-{MATCH_S} s: {prec:.2f}')
        d = (he.a_min - he.a_min_gnss_near).dropna()
        print(f'  a_min estimator - GNSS: median {d.median():+.2f}, MAE {d.abs().mean():.2f} m/s^2 (n {len(d)})')
        print('  missed GNSS events (a_min_gnss, a_min_est_near):',
              hard_ref[~hard_ref.matched][['bag', 't0', 'a_min_gnss', 'a_min_est_near']].round(2).values.tolist()[:10])

    # ---- overspeed candidates (needs pf output) ----
    p = pd.read_csv(OUT / 'speed_passes.csv')
    res = []
    for col, tag in (('v_max', 'est'), ('v_max_ref', 'gnss')):
        q = p[np.isfinite(p[col])][['bag', 'bin', col]].copy()
        q['p95_loo'] = np.nan
        for b, grp in q.groupby('bin'):
            vals = grp[col].to_numpy()
            if len(vals) < 10:
                continue
            loo = [np.quantile(np.delete(vals, j), 0.95) for j in range(len(vals))]
            q.loc[grp.index, 'p95_loo'] = loo
        q['excess'] = q[col] - q.p95_loo
        q['flag'] = q.excess > OVER_MARGIN
        q['src'] = tag
        res.append(q)
    est, ref = res
    fl_e = est[est.flag]
    fl_r = ref[ref.flag]
    key_e = set(zip(fl_e.bag, fl_e.bin))
    key_r = set(zip(fl_r.bag, fl_r.bin))
    near = lambda k, ks: any((k[0] == x[0]) and abs(k[1] - x[1]) <= 1 for x in ks)  # noqa: E731
    print(f'--- overspeed candidates (pass max > LOO p95 + {OVER_MARGIN * 3.6:.1f} km/h): estimator {len(key_e)} '
          f'pass-bins, GNSS {len(key_r)}; estimator flags confirmed by GNSS (same or adjacent bin): '
          f'{np.mean([near(k, key_r) for k in key_e]):.2f}; GNSS flags found by estimator: '
          f'{np.mean([near(k, key_e) for k in key_r]):.2f}')
    both = p[np.isfinite(p.v_max) & np.isfinite(p.v_max_ref)]
    for lim in (40.0, 50.0):
        e_, r_ = both.v_max * 3.6 > lim, both.v_max_ref * 3.6 > lim
        print(f'fixed limit {lim:.0f} km/h: passes over the limit by estimator {int(e_.sum())}, by GNSS {int(r_.sum())}, '
              f'both {int((e_ & r_).sum())}; disagreements {int((e_ != r_).sum())} of {len(both)} passes; '
              f'max speed of disagreeing passes within {float((both.v_max[e_ != r_] * 3.6 - lim).abs().max()):.2f} km/h of the limit')
    # merge consecutive bins of one run into events
    ev = []
    for bag, grp in fl_e.sort_values(['bag', 'bin']).groupby('bag'):
        bins = grp.bin.to_numpy()
        exc = grp.excess.to_numpy()
        vm = grp.v_max.to_numpy()
        start = 0
        for i in range(1, len(bins) + 1):
            if i == len(bins) or bins[i] != bins[i - 1] + 1:
                sl = slice(start, i)
                c = (bins[start] + 0.5) * 50.0
                ev.append({'bag': bag, 'bin0': int(bins[start]), 'bins': i - start, 'dir': m.direction(c).item(),
                           'route_m': float(m.route_at(c)), 'v_max_kmh': float(vm[sl].max() * 3.6),
                           'excess_kmh': float(exc[sl].max() * 3.6)})
                start = i
    ev = pd.DataFrame(ev)
    ev = ev.join(meta[['vehicle', 'date', 'local_start']], on='bag')
    ev.to_csv(OUT / 'safety_overspeed.csv', index=False, float_format='%.3f')
    print(f'overspeed events: {len(ev)} in {ev.bag.nunique()} runs of {p.bag.nunique()}; excess median '
          f'{ev.excess_kmh.median():.1f}, max {ev.excess_kmh.max():.1f} km/h')
    print('events per run: ', ev.groupby('bag').size().describe()[['mean', '50%', 'max']].round(2).to_dict())
    print(ev.sort_values('excess_kmh', ascending=False).head(8).round(1).to_string(index=False))
    print('runs with most overspeed events:', ev.groupby('bag').size().sort_values(ascending=False).head(5).to_dict())

    # ---------------- figure ----------------
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    ax.scatter(hard.v0_kmh, hard.a_min, s=10, color='#2a78d6', alpha=0.6, lw=0, label='резкое торможение')
    if len(unm):
        ax.scatter(unm.v0_kmh, unm.a_min_est, s=40, facecolor='none', edgecolor='#eb6834', lw=1.2,
                   label='флаг «неучтённое замедление»')
    ax.axhline(HARD, color='#52514e', lw=0.8)
    ax.set_xlabel('скорость в начале, км/ч')
    ax.set_ylabel('мин. ускорение, м/с²')
    ax.set_title('События торможения (оценщик)', loc='left', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    ax = axes[1]
    d = he.dropna(subset=['a_min_gnss_near'])
    ax.scatter(d.a_min_gnss_near, d.a_min, s=10, color='#2a78d6', alpha=0.6, lw=0)
    lim = [min(d.a_min.min(), d.a_min_gnss_near.min()) - 0.1, -1.2]
    ax.plot(lim, lim, color='#52514e', lw=0.8)
    ax.set_xlabel('мин. ускорение по GNSS, м/с²')
    ax.set_ylabel('мин. ускорение по оценщику, м/с²')
    ax.set_title('Проверка по GNSS: те же события', loc='left', fontsize=9)
    ax = axes[2]
    for dname, col in (('WB', '#2a78d6'), ('EB', '#eb6834')):
        e = ev[ev.dir == dname]
        ax.scatter(e.route_m / 1e3, e.excess_kmh, s=12, color=col, alpha=0.7, lw=0,
                   label='на запад' if dname == 'WB' else 'на восток')
    ax.set_xlabel('расстояние от восточной конечной, км')
    ax.set_ylabel('превышение p95 других проходов, км/ч')
    ax.set_title('Кандидаты на превышение скорости', loc='left', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    savefig(fig, 'd_safety_events.png')


if __name__ == '__main__':
    main()
