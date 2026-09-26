"""Product f: speed profile by location, from the estimator alone.

For every run that the estimator anchored on the map (GNSS in the first 5 s only), the outputs are
cut into 50 m bins of the map arc length (the closed loop separates the two directions). Per pass
(run x bin) we take the maximum and the mean speed; per bin, the median and the p95 over passes.
The same table is computed from the GNSS reference (master fix + vel over the whole run) to show
that the estimator alone reproduces it.

outputs: out/speed_passes.csv (per pass), out/speed_profile.csv (per bin), fig/f_speed_profile.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, bag_meta, long_bags, load_run, platforms, ref_aligned, savefig, style, track_map

BIN = 50.0
KMH = 3.6


def _passes(bag: str, s: np.ndarray, v: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    ok = np.isfinite(s) & np.isfinite(v)
    if ok.sum() < 10:
        return pd.DataFrame()
    s, v, t = s[ok], v[ok], t[ok]
    dt = np.diff(t, append=t[-1])
    dt = np.clip(dt, 0, 0.5)  # gaps do not count as time spent in the bin
    b = np.floor(s / BIN).astype(int)
    df = pd.DataFrame({'bin': b, 'v': v, 'dt': dt, 'vdt': v * dt, 't': t})
    g = df.groupby('bin')
    res = pd.DataFrame({'v_max': g.v.max(), 'v_min': g.v.min(), 'time_s': g.dt.sum(), 'dist': g.vdt.sum(),
                        't_in': g.t.min(), 'n': g.v.size()})
    res['v_mean'] = res.dist / res.time_s.where(res.time_s > 0)
    res = res[res.n >= 3].reset_index()
    res.insert(0, 'bag', bag)
    return res


def build_passes() -> pd.DataFrame:
    meta = bag_meta()
    rows = []
    for bag in long_bags():
        if not meta.loc[bag, 'has_gnss']:
            continue  # without the start fix the run is not anchored on the map
        o = load_run(bag)
        est = _passes(bag, o.s_map.to_numpy(), o.v.to_numpy(), o.t.to_numpy())
        r = ref_aligned(bag)
        ref = _passes(bag, r.s_ref.to_numpy(), r.v_ref.to_numpy(), r.t.to_numpy())
        ref = ref[['bag', 'bin', 'v_max', 'v_mean', 'n']].rename(
            columns={'v_max': 'v_max_ref', 'v_mean': 'v_mean_ref', 'n': 'n_ref'})
        rows.append(est.merge(ref, on=['bag', 'bin'], how='outer'))
    p = pd.concat(rows, ignore_index=True)
    m = track_map()
    c = (p.bin + 0.5) * BIN
    p['dir'] = m.direction(c)
    p['route_m'] = m.route_at(c)
    return p


def profile(p: pd.DataFrame, vcol: str = 'v_max', suffix: str = '') -> pd.DataFrame:
    q = p[np.isfinite(p[vcol])].groupby('bin')[vcol]
    return pd.DataFrame({f'n_pass{suffix}': q.size(), f'{vcol}_med{suffix}': q.median(),
                         f'{vcol}_p95{suffix}': q.quantile(0.95), f'{vcol}_p05{suffix}': q.quantile(0.05)})


def main():
    p = build_passes()
    p.to_csv(OUT / 'speed_passes.csv', index=False, float_format='%.4f')
    m = track_map()
    est = pd.concat([profile(p), profile(p, 'v_mean')[['v_mean_med']]], axis=1)
    ref = profile(p, 'v_max_ref').rename(columns={'n_pass': 'n_pass_ref'})
    prof = est.join(ref, how='outer')
    prof['stop_share'] = p[np.isfinite(p.v_min)].groupby('bin').v_min.apply(lambda x: float((x < 0.1).mean()))
    c = (prof.index.to_numpy() + 0.5) * BIN
    prof['s_center'] = c
    prof['dir'] = m.direction(c)
    prof['route_m'] = m.route_at(c)
    prof['x'], prof['y'] = m.xy_at(c)
    prof['grade_pct'] = 100 * m.at(c, 'grade')
    prof['curv_max'] = [np.abs(m.curv[(m.s >= b * BIN) & (m.s < (b + 1) * BIN)]).max() for b in prof.index]
    prof.index.name = 'bin'
    prof = prof[prof.n_pass >= 10]  # bins passed by fewer than 10 runs (yard, detour ends) are dropped
    prof.to_csv(OUT / 'speed_profile.csv', float_format='%.4f')

    # ---------------- numbers for the doc ----------------
    both = p[np.isfinite(p.v_max) & np.isfinite(p.v_max_ref)]
    dv = (both.v_max - both.v_max_ref) * KMH
    print(f'runs anchored: {p.bag.nunique()}, passes (run x bin): {len(p)}, bins kept: {len(prof)}')
    print(f'per-pass max speed, estimator - GNSS [km/h]: median {dv.median():+.2f}, '
          f'MAE {dv.abs().mean():.2f}, p95 |d| {dv.abs().quantile(0.95):.2f}, n {len(dv)}')
    d95 = (prof.v_max_p95 - prof.v_max_ref_p95) * KMH
    dmed = (prof.v_max_med - prof.v_max_ref_med) * KMH
    print(f'per-bin p95 of max speed, estimator - GNSS [km/h]: median |d| {d95.abs().median():.2f}, '
          f'max |d| {d95.abs().max():.2f}; per-bin median: median |d| {dmed.abs().median():.2f}, max {dmed.abs().max():.2f}')
    for d in ('WB', 'EB'):
        q = prof[prof.dir == d]
        top = q.v_max_p95.idxmax()
        print(f'{d}: bins {len(q)}, passes/bin median {q.n_pass.median():.0f}; top p95 {q.v_max_p95.max() * KMH:.1f} km/h '
              f'at route {q.route_m[top] / 1e3:.2f} km; median of per-pass max over bins {q.v_max_med.median() * KMH:.1f} km/h; '
              f'bins with p95 < 20 km/h: {(q.v_max_p95 * KMH < 20).sum()}, with stop share > 0.5: {(q.stop_share > 0.5).sum()}')
    slow = prof[(prof.v_max_p95 * KMH < 20) & (prof.stop_share < 0.5)].copy()
    slow['p95_kmh'] = slow.v_max_p95 * KMH
    print('slow zones (p95 < 20 km/h, not a stop):')
    print(slow[['dir', 'route_m', 'n_pass', 'p95_kmh', 'curv_max', 'grade_pct']].round(3).to_string())
    fast = prof.sort_values('v_max_p95', ascending=False).head(8).copy()
    fast['p95_kmh'] = fast.v_max_p95 * KMH
    fast['med_kmh'] = fast.v_max_med * KMH
    print('fastest bins:')
    print(fast[['dir', 'route_m', 'n_pass', 'med_kmh', 'p95_kmh', 'grade_pct']].round(2).to_string())

    # ---------------- figure ----------------
    plt = style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    plats = platforms()
    for ax, d, title in zip(axes, ('WB', 'EB'), ('на запад (s 0…5561 м)', 'на восток (s 5561…11054 м)')):
        q = prof[prof.dir == d].sort_values('route_m')
        x = q.route_m / 1e3
        ax.fill_between(x, q.v_max_p05 * KMH, q.v_max_p95 * KMH, color='#2a78d6', alpha=0.12, lw=0,
                        label='p5–p95 макс. скорости прохода (оценщик)')
        ax.plot(x, q.v_max_med * KMH, color='#2a78d6', lw=1.6, label='медиана макс. скорости прохода (оценщик)')
        ax.plot(x, q.v_max_p95 * KMH, color='#0d366b', lw=0.9, label='p95 (оценщик)')
        ax.plot(x, q.v_max_ref_p95 * KMH, color='#eb6834', lw=0.9, alpha=0.9, label='p95 (GNSS, контроль)')
        pl = plats[m.direction(plats.s) == d]
        for r in m.route_at(pl.s) / 1e3:
            ax.axvline(r, color='#52514e', lw=0.5, alpha=0.5)
        ax.set_ylabel('км/ч')
        ax.set_title(f'Профиль скорости, {title}; вертикальные линии — платформы', loc='left', fontsize=9)
        ax.set_ylim(0, None)
    axes[-1].set_xlabel('расстояние от восточной конечной вдоль пути, км')
    axes[0].legend(loc='upper right', fontsize=7, ncol=2, frameon=False)
    savefig(fig, 'f_speed_profile.png')


if __name__ == '__main__':
    main()
