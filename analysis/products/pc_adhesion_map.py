"""Product c: adhesion map. Where along the line do the wheels slip (traction) or slide (braking)?

Two tiers of episodes, both from the estimator's published outputs:
  flag  - the estimator's health flags: per-bogie slip/slide (the bogie is in a "bad" IMM mode and
          reads faster/slower than the vehicle) and model-only (both bogies distrusted, speed
          carried by the traction model). Flags within 2 s form one episode.
  micro - single-bogie disagreement the filter absorbs without raising a flag:
          |front - rear| > 0.3 m/s and > 4 % of the speed, v > 2 m/s, for >= 0.3 s. The culprit
          bogie is the one further from the estimated vehicle speed (1 + k) v; its sign gives
          slip (+) or slide (-). Micro episodes overlapping a flag episode are not counted twice.
The first 5 s of a run and wheel dropouts are ignored. Each episode is placed at the culprit bogie
(front 9.9 m, rear 2.35 m ahead of antenna 1; both -> midpoint) on the map loop, which separates
the two directions. Rates are per pass (runs that drove through the 50 m bin).

GNSS check (GNSS bags only): during a real slip/slide the culprit bogie must disagree with the
ground speed: max |wheel - (1 + k_run) v_gnss| inside the episode vs random windows of the same
length in unflagged motion.

outputs: adhesion_map.csv (per 50 m, requested as a spatial prior for the filter),
         out/adhesion_events.csv, out/adhesion_hotspots.csv, fig/c_adhesion_map.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (F_FRONT_DROPOUT, F_FRONT_SLIDE, F_FRONT_SLIP, F_MODEL_ONLY, F_REAR_DROPOUT, F_REAR_SLIDE,
                    F_REAR_SLIP, FRONT_BOGIE_ALONG, HERE, NPZ, OUT, REAR_BOGIE_ALONG, WHEEL_KMH_TO_MS, bag_meta,
                    long_bags, load_run, platforms, ref_aligned, savefig, segments, style, track_map)

BIN = 50.0
OFFSET = {'front': FRONT_BOGIE_ALONG, 'rear': REAR_BOGIE_ALONG, 'both': 0.5 * (FRONT_BOGIE_ALONG + REAR_BOGIE_ALONG)}


def _row(bag, tier, o, a, b, kind, bogie, model_only=False):
    t, v, s, notch = o.t.to_numpy(), o.v.to_numpy(), o.s_map.to_numpy(), o.notch.to_numpy()
    nt = notch[a:b + 1][np.isfinite(notch[a:b + 1])]
    s_ev = (s[a] + OFFSET[bogie]) % track_map().L if np.isfinite(s[a]) else np.nan
    return {'bag': bag, 'tier': tier, 't0': t[a], 't1': t[b], 'dur_s': t[b] - t[a], 'v0': v[a], 'kind': kind,
            'bogie': bogie, 'model_only': model_only, 'notch_min': nt.min() if len(nt) else np.nan,
            'notch_max': nt.max() if len(nt) else np.nan, 's_ant': s[a], 's_ev': s_ev}


def flag_episodes(bag: str, o: pd.DataFrame) -> list[dict]:
    fl = o['flags'].to_numpy()
    t = o.t.to_numpy()
    both_out = ((fl & F_FRONT_DROPOUT) > 0) & ((fl & F_REAR_DROPOUT) > 0)
    front = (fl & (F_FRONT_SLIP | F_FRONT_SLIDE)) > 0
    rear = (fl & (F_REAR_SLIP | F_REAR_SLIDE)) > 0
    mo = ((fl & F_MODEL_ONLY) > 0) & ~both_out
    rows = []
    for a, b in segments((front | rear | mo) & (t > 5.0), t, merge_gap=2.0):
        sl = slice(a, b + 1)
        n_slip = int(((fl[sl] & (F_FRONT_SLIP | F_REAR_SLIP)) > 0).sum())
        n_slide = int(((fl[sl] & (F_FRONT_SLIDE | F_REAR_SLIDE)) > 0).sum())
        f_any, r_any = bool(front[sl].any()), bool(rear[sl].any())
        bogie = 'both' if (f_any and r_any) or not (f_any or r_any) else ('front' if f_any else 'rear')
        rows.append(_row(bag, 'flag', o, a, b, 'slip' if n_slip >= n_slide else 'slide', bogie, bool(mo[sl].any())))
    return rows


def micro_episodes(bag: str, o: pd.DataFrame) -> list[dict]:
    t, v, k = o.t.to_numpy(), o.v.to_numpy(), o.k.to_numpy()
    wf, wr = o.wf.to_numpy(), o.wr.to_numpy()
    dw = wf - wr
    mask = (t > 5.0) & (v > 2.0) & np.isfinite(dw) & (np.abs(dw) > 0.3) & (np.abs(dw) > 0.04 * v)
    rows = []
    for a, b in segments(mask, t, merge_gap=1.0, min_dur=0.3):
        sl = slice(a, b + 1)
        ref = (1 + k[sl]) * v[sl]
        ef, er = wf[sl] - ref, wr[sl] - ref
        if np.nanmean(np.abs(ef)) >= np.nanmean(np.abs(er)):
            bogie, dev = 'front', ef
        else:
            bogie, dev = 'rear', er
        rows.append(_row(bag, 'micro', o, a, b, 'slip' if np.nanmean(dev) > 0 else 'slide', bogie))
    return rows


def passes_per_bin(o: pd.DataFrame) -> set[int]:
    s = o.s_map.to_numpy()
    s = s[np.isfinite(s)]
    b, n = np.unique(np.floor(s / BIN).astype(int), return_counts=True)
    return set(b[n >= 3].tolist())


def gnss_check(bag: str, ev: pd.DataFrame, rng) -> tuple[list, list]:
    """Max |wheel - (1+k) v_gnss| of the culprit bogie inside each episode, and in random windows."""
    r = ref_aligned(bag)
    if r is None or not len(ev):
        return [np.nan] * len(ev), []
    o = load_run(bag)
    t0_abs = o.stamp_ns.iloc[0] * 1e-9
    d = np.load(NPZ / f'{bag}.npz')
    tv, vr = r.t.to_numpy(), r.v_ref.to_numpy()
    w = {}
    for key, name in (('vehicle__front_bogie_velocity', 'front'), ('vehicle__rear_bogie_velocity', 'rear')):
        a = d[key]
        a = a[np.argsort(a[:, 1])]
        w[name] = np.interp(tv + t0_abs, a[:, 1], a[:, 2] * WHEEL_KMH_TO_MS)
    mov = (vr > 3.0) & ((r['flags'].fillna(0).astype(np.int64).to_numpy() & 0xF) == 0)
    k = np.median(0.5 * (w['front'] + w['rear'])[mov] / vr[mov]) - 1.0  # per-run wheel scale from GNSS
    dev = {'front': np.abs(w['front'] - (1 + k) * vr), 'rear': np.abs(w['rear'] - (1 + k) * vr)}
    dev['both'] = np.maximum(dev['front'], dev['rear'])
    res, base = [], []
    cand = np.flatnonzero(mov)
    for e in ev.itertuples():
        m = (tv >= e.t0 - 0.3) & (tv <= e.t1 + 0.3)
        res.append(float(dev[e.bogie][m].max()) if m.any() else np.nan)
        for _ in range(5):
            i = rng.choice(cand)
            mm = (tv >= tv[i]) & (tv <= tv[i] + e.dur_s + 0.6) & mov
            if mm.any():
                base.append(float(dev[e.bogie][mm].max()))
    return res, base


def main():
    meta = bag_meta()
    m = track_map()
    L = m.L
    rng = np.random.default_rng(0)
    evs, run_rows, base = [], [], {'flag': [], 'micro': []}
    pass_count = np.zeros(int(np.ceil(L / BIN)), int)
    for bag in long_bags():
        o = load_run(bag)
        fe = pd.DataFrame(flag_episodes(bag, o))
        me = pd.DataFrame(micro_episodes(bag, o))
        if len(me):
            me['dup'] = [bool(len(fe)) and bool(((fe.t0 <= e.t1 + 1) & (fe.t1 >= e.t0 - 1)).any()) for e in me.itertuples()]
        if len(fe):
            fe['dup'] = False
        if meta.loc[bag, 'has_gnss']:
            for b in passes_per_bin(o):
                pass_count[b] += 1
            for tier, df in (('flag', fe), ('micro', me)):
                if len(df):
                    c, bl = gnss_check(bag, df, rng)
                    df['gnss_dev'] = c
                    base[tier] += bl
        evs += [x for x in (fe, me) if len(x)]
        run_rows.append({'bag': bag, 'flag': len(fe), 'micro': int((~me.dup).sum()) if len(me) else 0,
                         'dist_km': float(np.trapezoid(o.v, o.t) / 1e3)})
    ev = pd.concat(evs, ignore_index=True)
    ok = np.isfinite(ev.s_ev)
    ev['dir'] = np.where(ok, m.direction(ev.s_ev.fillna(0)), '')
    ev['route_m'] = np.where(ok, m.route_at(ev.s_ev.fillna(0)), np.nan)
    ev['curv_abs'] = np.where(ok, np.abs(m.at(ev.s_ev.fillna(0), 'curv')), np.nan)
    ev['date'] = meta.loc[ev.bag, 'date'].to_numpy()
    ev.to_csv(OUT / 'adhesion_events.csv', index=False, float_format='%.3f')
    runs = pd.DataFrame(run_rows).set_index('bag').join(meta[['vehicle', 'date', 'has_gnss']])
    uniq = ev[~ev.dup]  # flag episodes + micro episodes not already flagged

    # ---------------- per-bin table (spatial prior) ----------------
    loc = uniq[np.isfinite(uniq.s_ev)].copy()
    loc['bin'] = np.floor(loc.s_ev / BIN).astype(int)
    nb = len(pass_count)
    cnt = lambda q: q.groupby('bin').size().reindex(range(nb), fill_value=0).to_numpy()  # noqa: E731
    tab = pd.DataFrame({'s_start': np.arange(nb) * BIN, 's_end': np.minimum((np.arange(nb) + 1) * BIN, L),
                        'passes': pass_count, 'slip_episodes': cnt(loc[loc.kind == 'slip']),
                        'slide_episodes': cnt(loc[loc.kind == 'slide'])})
    tab['rate_per_pass'] = (tab.slip_episodes + tab.slide_episodes) / tab.passes.where(tab.passes > 0)
    tab['flag_episodes'] = cnt(loc[loc.tier == 'flag'])
    tab['model_only_episodes'] = cnt(loc[loc.model_only])
    tab['micro_episodes'] = cnt(loc[loc.tier == 'micro'])
    tab['runs_with_episode'] = loc.groupby('bin').bag.nunique().reindex(range(nb), fill_value=0).to_numpy()
    tab.to_csv(HERE / 'adhesion_map.csv', index=False, float_format='%.4f')

    # ---------------- numbers for the doc ----------------
    km = runs.dist_km.sum()
    print(f'runs: {len(runs)} ({runs.has_gnss.sum()} anchored on the map), {km:.0f} km')
    for tier in ('flag', 'micro'):
        q = ev[ev.tier == tier]
        qu = q[~q.dup]
        print(f'[{tier}] episodes {len(q)} (not already flagged: {len(qu)}); per 100 km {100 * len(qu) / km:.1f}; '
              f'slip {(qu.kind == "slip").sum()}, slide {(qu.kind == "slide").sum()}, model-only {qu.model_only.sum()}; '
              f'bogie {qu.bogie.value_counts().to_dict()}; runs with any {qu.bag.nunique()}; '
              f'median dur {qu.dur_s.median():.2f} s; |curv| > 0.01 share {(qu.curv_abs > 0.01).mean():.2f}')
        tr, br = qu.notch_max > 0, qu.notch_min < 0
        print(f'   context: slips under traction {((qu.kind == "slip") & tr).sum()}/{(qu.kind == "slip").sum()}, '
              f'slides under braking {((qu.kind == "slide") & br).sum()}/{(qu.kind == "slide").sum()}')
        g = qu[np.isfinite(qu.gnss_dev)] if 'gnss_dev' in qu else qu.iloc[:0]
        bb = np.array(base[tier])
        if len(g) and len(bb):
            print(f'   GNSS check: max|wheel-(1+k)v_gnss| > 0.3 m/s in {(g.gnss_dev > 0.3).mean():.2f} of {len(g)} episodes '
                  f'vs {(bb > 0.3).mean():.3f} of {len(bb)} random windows; median {g.gnss_dev.median():.2f} vs {np.median(bb):.3f} m/s')
    print('episodes per run (flag + micro) by vehicle/date:')
    runs['all'] = runs.flag + runs.micro
    print(runs.groupby(['vehicle', 'date'])[['flag', 'micro', 'all']].agg(['sum', 'max']).to_string())
    print('runs with most episodes:', runs['all'].sort_values(ascending=False).head(6).to_dict())
    busy = tab[tab.passes >= 10].copy()
    busy['dir'] = m.direction(busy.s_start + BIN / 2)
    busy['route_m'] = m.route_at(busy.s_start + BIN / 2)
    busy['episodes'] = busy.slip_episodes + busy.slide_episodes
    print(f'bins with >=10 passes: {len(busy)}; with any episode: {(busy.episodes > 0).sum()}; '
          f'with episodes in >=2 runs: {(busy.runs_with_episode >= 2).sum()}; in >=3 runs: {(busy.runs_with_episode >= 3).sum()}')
    top = busy.sort_values(['runs_with_episode', 'rate_per_pass'], ascending=False).head(10).copy()
    plats = platforms()
    top['nearest_platform_m'] = [float(np.min(np.abs((plats.s - (s + BIN / 2) + L / 2) % L - L / 2))) for s in top.s_start]
    top['grade_pct'] = 100 * m.at(top.s_start + BIN / 2, 'grade')
    top['curv_max'] = [np.abs(m.curv[(m.s >= s) & (m.s < s + BIN)]).max() for s in top.s_start]
    ctx = []
    for s in top.s_start:
        e = loc[(loc.s_ev >= s) & (loc.s_ev < s + BIN)]
        ctx.append(f"{(e.kind == 'slip').sum()} slip/{(e.kind == 'slide').sum()} slide, flag {(e.tier == 'flag').sum()}, "
                   f"v0 {e.v0.median() * 3.6:.0f} km/h, {e.bogie.mode().iloc[0] if len(e) else ''}")
    top['context'] = ctx
    top.to_csv(OUT / 'adhesion_hotspots.csv', index=False, float_format='%.4f')
    print('top hotspots (by distinct runs, then rate per pass):')
    print(top[['s_start', 'dir', 'route_m', 'passes', 'slip_episodes', 'slide_episodes', 'runs_with_episode',
               'rate_per_pass', 'nearest_platform_m', 'grade_pct', 'curv_max', 'context']].round(3).to_string(index=False))
    srt = busy.episodes.sort_values(ascending=False)
    print(f'share of located episodes in the 10 worst bins: {srt.head(10).sum() / max(srt.sum(), 1):.2f} '
          f'({len(loc)} located)')

    # ---------------- figure ----------------
    plt = style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    for ax, d, title in zip(axes, ('WB', 'EB'), ('на запад', 'на восток')):
        q = busy[busy.dir == d].sort_values('route_m')
        x = q.route_m / 1e3
        ax.bar(x, q.slip_episodes / q.passes, width=0.045, color='#2a78d6', label='боксование (тяга)')
        ax.bar(x, q.slide_episodes / q.passes, width=0.045, bottom=q.slip_episodes / q.passes, color='#eb6834',
               label='юз (торможение)')
        pl = plats[m.direction(plats.s) == d]
        for r in m.route_at(pl.s) / 1e3:
            ax.axvline(r, color='#52514e', lw=0.5, alpha=0.4)
        ax.set_ylabel('эпизодов / проход')
        ax.set_title(f'Карта сцепления, {title}: эпизоды на проход в бине 50 м '
                     f'({int(q.passes.median())} проходов на бин); линии — платформы', loc='left', fontsize=9)
    axes[0].legend(loc='upper right', fontsize=7, frameon=False)
    axes[-1].set_xlabel('расстояние от восточной конечной вдоль пути, км')
    savefig(fig, 'c_adhesion_map.png')


if __name__ == '__main__':
    main()
