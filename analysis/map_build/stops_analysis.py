"""Stop analysis along the track map: where does the tram stand still (speed < 0.1 m/s for >= 5 s)?

Outputs (in the given map directory and plots/):
  stops_events.csv  -- every detected stop event (run, time, dwell, edge, s, d, GNSS/wheel view)
  stops.csv         -- clustered stop locations with repeatability statistics and class
  plots/stops_hist.png, plots/stops_scatter.png
Classes:
  terminal : layover positions at run start/end (east platform, west yard)
  platform : P(stop | pass) >= 0.5, robust std of s <= 3 m, median dwell >= 8 s  -> along-track landmarks
  signal   : repeated (>= 3 runs) but not platform-like (traffic lights / junctions / queueing)
  random   : everything else
Run: python stops_analysis.py [map_dir]   (default map_train; val statistics reported separately)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import data_io as D
import runs as R
import stops as S
from validate import MapProjector

OUT = D.OUT
PLOTS = OUT / 'plots'


def passes_on_main(mp: MapProjector, r):
    """s-interval of 'main' spanned by a run (one-way trips; s increases in travel direction), unwrapped.

    Uses every fix within 5 m of main (any quality) -- the run covers everything between its first and last
    projected position, even through GNSS outages."""
    E = mp.polys['main']
    L = E.length
    s, d, dist, seg, dpsi = E.project(r.x, r.y, r.psi, max_d=5.0, max_dpsi=np.radians(60))
    idx = np.flatnonzero(np.isfinite(s))
    if len(idx) < 2:
        return []
    su = np.unwrap(s[idx], period=L)
    # robust ends: 1st / 99th percentile of the unwrapped track
    a, b = np.percentile(su, 0.5), np.percentile(su, 99.5)
    return [(a, b)] if b - a > 5 else []


def covered(passes, s, L):
    """Does any pass cover main-s position s (mod L)?"""
    for a, b in passes:
        for k in (-1, 0, 1, 2):
            if a <= s + k * L <= b:
                return True
    return False


def detect_all(mp: MapProjector, runs: dict):
    ev = []
    passes = {}
    for b, r in runs.items():
        passes[b] = passes_on_main(mp, r)
        wst = S.wheel_stops(b)
        for st in S.detect_stops(r):
            s, d, dist, ek = mp.project(np.array([st['x']]), np.array([st['y']]), np.array([st['psi']]), max_d=8.0)
            st['edge'] = mp.ids[ek[0]] if ek[0] >= 0 else ''
            st['s'] = float(s[0])
            st['d'] = float(d[0])
            # wheel view: overlap with a wheel-detected stop
            ov = [(a, c) for a, c in wst if c >= st['t0'] - 0.5 and a <= st['t1'] + 0.5]
            st['wheel_detected'] = len(ov) > 0
            st['wheel_dwell'] = float(sum(min(c, st['t1']) - max(a, st['t0']) for a, c in ov)) if ov else 0.0
            ev.append(st)
    return pd.DataFrame(ev), passes


def cluster(ev: pd.DataFrame, passes: dict, mp: MapProjector, gap=3.0, site_gap=30.0):
    L = mp.polys['main'].length
    rows = []
    ev = ev.copy()
    ev['cluster'] = -1
    cid = 0
    for eid in ev.edge.unique():
        if not eid:
            continue
        sub = ev[ev.edge == eid]
        lab = S.cluster_1d(sub.s.values, gap=gap)
        for l in np.unique(lab):
            g = sub[lab == l]
            ev.loc[g.index, 'cluster'] = cid
            sv = g.s.values
            med = float(np.median(sv))
            nruns = g.bag.nunique()
            if eid == 'main':
                npass = sum(covered(passes[b], med, L) for b in passes)
            else:
                npass = np.nan
            mid = ~(g['first'] | g['last'])
            rows.append(dict(cluster=cid, edge=eid, s_median=med, s_mean=float(np.mean(sv)), s_std=float(np.std(sv)),
                             s_robust_std=float(1.4826 * np.median(np.abs(sv - med))),
                             s_p10=float(np.percentile(sv, 10)), s_p90=float(np.percentile(sv, 90)),
                             x=float(np.median(g.x)), y=float(np.median(g.y)), z=float(np.median(g.z)),
                             n_stops=len(g), n_runs=int(nruns), n_passes=npass,
                             p_stop=float(g[mid].bag.nunique() / npass) if npass and npass > 0 else np.nan,
                             dwell_med=float(np.median(g.dwell)), dwell_p10=float(np.percentile(g.dwell, 10)),
                             dwell_p90=float(np.percentile(g.dwell, 90)), frac_run_start=float(g['first'].mean()),
                             frac_run_end=float(g['last'].mean()), frac_wheel_detected=float(g.wheel_detected.mean()),
                             n_mid=int(mid.sum())))
            cid += 1
    cl = pd.DataFrame(rows)

    def cls(rw):
        if rw.frac_run_start + rw.frac_run_end >= 0.5:
            return 'terminal'
        if rw.n_mid >= 3 and rw.p_stop >= 0.5 and rw.s_robust_std <= 3.0 and rw.dwell_med >= 8:
            return 'platform'
        if rw.n_runs >= 3:
            return 'signal'
        return 'random'
    cl['cls'] = cl.apply(cls, axis=1)
    # stop sites: clusters of the same edge within site_gap of each other (e.g. double-berth platforms)
    cl = cl.sort_values(['edge', 's_median']).reset_index(drop=True)
    site = np.zeros(len(cl), int)
    k = 0
    for i in range(1, len(cl)):
        if cl.edge[i] != cl.edge[i - 1] or cl.s_median[i] - cl.s_median[i - 1] > site_gap:
            k += 1
        site[i] = k
    cl['site'] = site
    return ev, cl


def direction(edge, s):
    if edge != 'main':
        return 'yard' if edge.startswith(('fan', 'west')) else 'WB'
    if 173 <= s <= 5448:
        return 'WB'
    if 5698 <= s <= 10975:
        return 'EB'
    return 'west_loop' if 5448 < s < 5698 else 'east_loop'


def main(map_dir='map_train'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mp = MapProjector(OUT / map_dir)
    L = mp.polys['main'].length
    tr = R.load_runs(D.split('train'))
    va = R.load_runs(D.split('val'))
    ev_tr, pas_tr = detect_all(mp, tr)
    ev_va, pas_va = detect_all(mp, va)
    ev_tr['split'] = 'train'
    ev_va['split'] = 'val'
    ev_all = pd.concat([ev_tr, ev_va], ignore_index=True)
    passes = {**pas_tr, **pas_va}
    ev_all, cl = cluster(ev_all, passes, mp)
    cl['dir'] = [direction(e, s) for e, s in zip(cl.edge, cl.s_median)]
    # train-only stats for landmark std, val check: distance of val stops to the train cluster median
    cl = cl.sort_values(['edge', 's_median']).reset_index(drop=True)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 300)
    cols = ['site', 'cluster', 'edge', 'dir', 'cls', 's_median', 's_robust_std', 's_std', 's_p10', 's_p90', 'n_stops', 'n_runs',
            'n_passes', 'p_stop', 'dwell_med', 'dwell_p10', 'dwell_p90', 'frac_run_start', 'frac_run_end',
            'frac_wheel_detected', 'x', 'y']
    print(cl[cl.n_runs >= 3][cols].round(2).to_string())
    # val consistency for platform clusters: stats of val stop offsets from the train-only median
    rows = []
    for _, c in cl[cl.cls.isin(['platform', 'terminal'])].iterrows():
        g = ev_all[ev_all.cluster == c.cluster]
        gt, gv = g[g.split == 'train'], g[g.split == 'val']
        if len(gt) >= 2 and len(gv) >= 1:
            m = np.median(gt.s)
            rows.append(dict(cluster=c.cluster, cls=c.cls, n_train=len(gt), n_val=len(gv),
                             val_offset_med=float(np.median(gv.s - m)), val_offset_absmax=float(np.max(np.abs(gv.s - m))),
                             val_abs_p90=float(np.percentile(np.abs(gv.s - m), 90))))
    vc = pd.DataFrame(rows)
    print('\nVAL stops vs TRAIN cluster median (platform/terminal):')
    print(vc.round(2).to_string())
    ev_all.to_csv(OUT / map_dir / 'stops_events.csv', index=False)
    cl.to_csv(OUT / map_dir / 'stops.csv', index=False)
    # stats on event detection
    print('\nevents', len(ev_all), 'on main', int((ev_all.edge == 'main').sum()), 'wheel-detected frac',
          round(ev_all.wheel_detected.mean(), 4))
    # plots
    fig, axs = plt.subplots(2, 1, figsize=(22, 9))
    for ax, (nm, a, b) in zip(axs, (('WB (east->west), main s', 0, 5700), ('EB (west->east), main s', 5600, L))):
        m = (ev_all.edge == 'main') & (ev_all.s >= a) & (ev_all.s <= b)
        mm = m & ~(ev_all['first'] | ev_all['last'])
        ax.hist(ev_all.s[mm], bins=np.arange(a, b, 5.0), color='tab:blue', label='mid-run stops')
        ax.hist(ev_all.s[m & (ev_all['first'] | ev_all['last'])], bins=np.arange(a, b, 5.0), color='tab:orange',
                alpha=0.7, label='run start/end (layover)')
        for _, c in cl[(cl.edge == 'main') & (cl.s_median >= a) & (cl.s_median <= b)].iterrows():
            if c.cls == 'platform':
                ax.axvline(c.s_median, color='g', lw=0.8)
                ax.text(c.s_median, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 1, f'P{int(c.cluster)}', color='g', fontsize=7)
        ax.set_title(nm + ' -- stop events (5 m bins); green = platform clusters')
        ax.set_xlabel('s [m]'); ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(PLOTS / 'stops_hist.png', dpi=80, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(22, 6))
    m = ev_all.edge == 'main'
    sc = ax.scatter(ev_all.s[m], ev_all.dwell[m], s=6, c=(ev_all['first'] | ev_all['last'])[m], cmap='coolwarm')
    ax.set_yscale('log'); ax.set_xlabel('s on main [m]'); ax.set_ylabel('dwell [s]'); ax.grid(True, alpha=0.3)
    ax.set_title('stop events: position vs dwell (red = run start/end)')
    fig.savefig(PLOTS / 'stops_scatter.png', dpi=80, bbox_inches='tight')
    plt.close(fig)
    return ev_all, cl, vc


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'map_train')
