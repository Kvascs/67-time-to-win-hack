"""Validation of the track map on held-out (val) runs.

1. cross-track: lateral residual of val master fixes vs the train-only map (all edges, heading-gated),
   split by fix quality (RTK / non-RTK), recording date, and region;
2. along-track consistency: map arc length travelled between fixes vs GNSS Doppler distance and vs
   wheel-odometer distance, globally and as a function of curvature.
Run: python validate.py  -> prints tables, writes plots/val_*.png and cache/val_*.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import data_io as D
import runs as R
from polyline import Polyline

OUT = D.OUT
PLOTS = OUT / 'plots'


class MapProjector:
    """Build-time multi-edge projector (numba) over an exported map directory."""

    def __init__(self, map_dir):
        import track_map as TM
        self.tm = TM.TrackMap(map_dir)
        self.polys = {}
        for eid, E in self.tm.edges.items():
            self.polys[eid] = Polyline(np.column_stack([E.x, E.y]), closed=E.closed)
        self.ids = list(self.polys)

    def project(self, x, y, psi, max_d=15.0, max_dpsi=np.radians(60)):
        best = None
        for k, eid in enumerate(self.ids):
            s, d, dist, seg, dpsi = self.polys[eid].project(x, y, psi, max_d=max_d, max_dpsi=max_dpsi)
            if best is None:
                best = [s, d, dist, np.full(len(s), k if np.isfinite(s).any() else -1)]
                best[3] = np.where(np.isfinite(s), k, -1)
                continue
            better = np.isfinite(dist) & ~(dist >= np.nan_to_num(best[2], nan=np.inf))
            best[0] = np.where(better, s, best[0])
            best[1] = np.where(better, d, best[1])
            best[2] = np.where(better, dist, best[2])
            best[3] = np.where(better, k, best[3])
        return best  # s, d, dist, edge index (-1 none)


def region_of(eid, s):
    if eid != 'main':
        return 'branch'
    if 173 <= s <= 5448 or 5698 <= s <= 10975:
        return 'main_double_track'
    if 5448 < s < 5698:
        return 'west_yard_loop'
    return 'east_loop'


def cross_track(mp: MapProjector, runs: dict):
    info = {i['bag']: i for i in D.SPLITS['info']}
    rows = []
    per_fix = []
    for b, r in runs.items():
        s, d, dist, ek = mp.project(r.x, r.y, r.psi, max_d=15.0)
        day = int(round(info[b]['t0'] / 86400 - 20600))
        for cls, m in (('rtk', r.good), ('usable_nonrtk', r.usable & ~r.good), ('all', np.ones(len(r), bool))):
            mm = m & np.isfinite(d)
            if mm.sum() < 10:
                continue
            a = np.abs(d[mm])
            rows.append(dict(bag=b, day=day, cls=cls, n=int(mm.sum()), frac_proj=float(mm.sum() / max(m.sum(), 1)),
                             p50=np.percentile(a, 50), p90=np.percentile(a, 90), p95=np.percentile(a, 95),
                             p99=np.percentile(a, 99), frac_lt_0p1=float(np.mean(a < 0.1)),
                             frac_lt_0p5=float(np.mean(a < 0.5)), frac_lt_1=float(np.mean(a < 1.0))))
        reg = np.array([region_of(mp.ids[k], ss) if k >= 0 else 'none' for k, ss in zip(ek, s)])
        per_fix.append(pd.DataFrame(dict(bag=b, day=day, d=d, s=s, edge=[mp.ids[k] if k >= 0 else '' for k in ek],
                                         region=reg, rtk=r.good, usable=r.usable, status=r.status)))
    return pd.DataFrame(rows), pd.concat(per_fix, ignore_index=True)


def along_track(mp: MapProjector, runs: dict, win_s=30.0):
    """Per-window comparison of map arc length vs Doppler and wheel distance (main edge, RTK fixes)."""
    E = mp.polys['main']
    L = E.length
    curv_v = mp.tm.edges['main'].curv
    s_v = mp.tm.edges['main'].s
    rows = []
    for b, r in runs.items():
        s, d, dist, seg, dpsi = E.project(r.x, r.y, r.psi, max_d=1.0, max_dpsi=np.radians(45))
        ok = r.good & np.isfinite(s)
        if ok.sum() < 100:
            continue
        idx = np.flatnonzero(ok)
        t = r.th[idx]
        su = np.unwrap(s[idx], period=L)
        # windows of ~win_s seconds of continuous RTK coverage (gaps <= 0.3 s)
        brk = np.flatnonzero((np.diff(t) > 0.3) | (np.abs(np.diff(su)) > 5.0)) + 1
        for chunk in np.split(np.arange(len(idx)), brk):
            if len(chunk) < 20:
                continue
            tt = t[chunk]
            nwin = max(int((tt[-1] - tt[0]) // win_s), 1)
            for w in np.array_split(chunk, nwin):
                if len(w) < 20:
                    continue
                ii = idx[w]
                ds_map = su[w[-1]] - su[w[0]]
                if ds_map < 20.0:
                    continue
                tw = r.th[ii]
                sp = r.speed[ii]
                d_dopp = float(np.trapezoid(np.nan_to_num(sp), tw))
                wf = r.wheel_f[ii] if r.wheel_f is not None else np.full(len(ii), np.nan)
                wr = r.wheel_r[ii] if r.wheel_r is not None else np.full(len(ii), np.nan)
                d_wf = float(np.trapezoid(wf, tw))
                d_wr = float(np.trapezoid(wr, tw))
                chord = float(np.sum(np.hypot(np.diff(r.x[ii]), np.diff(r.y[ii]))))
                sm = np.mod(su[w], L)
                kc = np.interp(sm, s_v, curv_v, period=L)
                rows.append(dict(bag=b, s0=float(np.mod(su[w[0]], L)), ds_map=ds_map, d_dopp=d_dopp, d_chord=chord,
                                 d_wheel_f=d_wf, d_wheel_r=d_wr, mean_abs_curv=float(np.mean(np.abs(kc))),
                                 mean_curv=float(np.mean(kc)), turn=float(np.trapezoid(kc, su[w])),
                                 v_mean=float(np.mean(sp)), dur=float(tw[-1] - tw[0])))
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    val = R.load_runs(D.split('val'))
    mp = MapProjector(OUT / 'map_train')
    ct, pf = cross_track(mp, val)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', 200)
    summ = {}
    print('\n=== cross-track |d| of VAL master fixes vs TRAIN map (heading-gated nearest edge) ===')
    for cls in ('rtk', 'usable_nonrtk', 'all'):
        sel = pf[{'rtk': pf.rtk, 'usable_nonrtk': pf.usable & ~pf.rtk, 'all': np.ones(len(pf), bool)}[cls]]
        a = np.abs(sel.d.dropna())
        summ[cls] = dict(n=int(len(a)), proj=float(sel.d.notna().mean()), p50=float(np.percentile(a, 50)),
                         p68=float(np.percentile(a, 68)), p90=float(np.percentile(a, 90)),
                         p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                         frac_lt_0p1=float(np.mean(a < 0.1)), frac_lt_0p5=float(np.mean(a < 0.5)),
                         frac_lt_1=float(np.mean(a < 1.0)), rms_trim99=float(np.sqrt(np.mean(np.minimum(a, np.percentile(a, 99)) ** 2))))
        print(cls, {k: round(v, 4) if isinstance(v, float) else v for k, v in summ[cls].items()})
    print('\n--- RTK fixes by region ---')
    reg = pf[pf.rtk & pf.d.notna()].groupby('region').d.apply(lambda v: pd.Series(
        dict(n=len(v), p50=np.percentile(np.abs(v), 50), p90=np.percentile(np.abs(v), 90),
             p99=np.percentile(np.abs(v), 99), mean=np.mean(v)))).unstack()
    print(reg.round(4))
    print('\n--- by date (all fixes / RTK fixes) ---')
    byday = pf[pf.d.notna()].groupby(['day']).apply(lambda g: pd.Series(dict(
        n=len(g), frac_rtk=g.rtk.mean(), p50_all=np.percentile(np.abs(g.d), 50), p90_all=np.percentile(np.abs(g.d), 90),
        p50_rtk=np.percentile(np.abs(g.d[g.rtk]), 50) if g.rtk.any() else np.nan,
        p90_rtk=np.percentile(np.abs(g.d[g.rtk]), 90) if g.rtk.any() else np.nan)))
    print(byday.round(4))
    print('\n--- per run (RTK) ---')
    print(ct[ct.cls == 'rtk'].round(4).to_string())
    # plots
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    bins = np.logspace(-3.5, 1.3, 80)
    for cls, m in (('RTK (status 2 + checks)', pf.rtk), ('non-RTK usable', pf.usable & ~pf.rtk)):
        a = np.abs(pf.d[m].dropna())
        axs[0].hist(a, bins=bins, histtype='step', lw=1.5, label=f'{cls}: n={len(a)}, p50={np.median(a):.3f} m', density=True)
    axs[0].set_xscale('log'); axs[0].set_xlabel('|cross-track residual| [m]'); axs[0].legend(); axs[0].grid(True, which='both', alpha=0.3)
    axs[0].set_title('VAL master fixes vs TRAIN-only map')
    a = np.sort(np.abs(pf.d[pf.rtk].dropna()))
    axs[1].plot(a, np.arange(1, len(a) + 1) / len(a), label='RTK')
    a2 = np.sort(np.abs(pf.d[pf.usable & ~pf.rtk].dropna()))
    axs[1].plot(a2, np.arange(1, len(a2) + 1) / len(a2), label='non-RTK')
    axs[1].set_xscale('log'); axs[1].set_xlabel('|d| [m]'); axs[1].set_ylabel('CDF'); axs[1].grid(True, which='both', alpha=0.3); axs[1].legend()
    fig.savefig(PLOTS / 'val_cross_track_hist.png', dpi=90, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(20, 5))
    m = pf.rtk & (pf.edge == 'main')
    ax.scatter(pf.s[m], pf.d[m], s=0.3, c=pf.day[m], cmap='viridis')
    ax.set_ylim(-1.5, 1.5); ax.grid(True); ax.set_xlabel('s on main [m]'); ax.set_ylabel('d [m]')
    ax.set_title('VAL RTK fixes: lateral residual vs s (colour = recording day)')
    fig.savefig(PLOTS / 'val_cross_track_vs_s.png', dpi=80, bbox_inches='tight')
    plt.close(fig)
    # along-track
    at = along_track(mp, val)
    at['r_dopp'] = at.ds_map / at.d_dopp
    at['r_chord'] = at.ds_map / at.d_chord
    at['r_wf'] = at.ds_map / at.d_wheel_f
    at['r_wr'] = at.ds_map / at.d_wheel_r
    print('\n=== along-track consistency (windows ~30 s, RTK, main edge) ===')
    print('windows', len(at), 'total map dist [km]', round(at.ds_map.sum() / 1000, 2))
    tot = dict(map_over_doppler=float(at.ds_map.sum() / at.d_dopp.sum()),
               map_over_chord=float(at.ds_map.sum() / at.d_chord.sum()),
               map_over_wheel_front=float(at.ds_map.sum() / at.d_wheel_f.sum()),
               map_over_wheel_rear=float(at.ds_map.sum() / at.d_wheel_r.sum()),
               window_ratio_dopp_p5_p50_p95=[float(v) for v in np.percentile(at.r_dopp, [5, 50, 95])],
               window_ratio_dopp_mad=float(np.median(np.abs(at.r_dopp - np.median(at.r_dopp)))))
    print(json.dumps(tot, indent=1))
    # curvature dependence of map/doppler and map/wheel
    at['kbin'] = pd.cut(at.mean_abs_curv, [0, 0.001, 0.003, 0.006, 0.01, 0.02, 0.05, 1])
    kt = at.groupby('kbin', observed=True).apply(lambda g: pd.Series(dict(
        n=len(g), dist_km=g.ds_map.sum() / 1000, map_over_dopp=g.ds_map.sum() / g.d_dopp.sum(),
        map_over_wheel_f=g.ds_map.sum() / g.d_wheel_f.sum(), map_over_wheel_r=g.ds_map.sum() / g.d_wheel_r.sum())))
    print(kt.round(5))
    per_run = at.groupby('bag').apply(lambda g: pd.Series(dict(km=g.ds_map.sum() / 1000, map_over_dopp=g.ds_map.sum() / g.d_dopp.sum(),
                                                               map_over_wheel_f=g.ds_map.sum() / g.d_wheel_f.sum())))
    print(per_run.round(5))
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    axs[0].scatter(at.mean_abs_curv, at.r_dopp, s=4)
    axs[0].set_xscale('symlog', linthresh=1e-3); axs[0].set_ylim(0.97, 1.03); axs[0].grid(True)
    axs[0].set_xlabel('mean |curvature| in window [1/m]'); axs[0].set_ylabel('map ds / Doppler distance')
    axs[1].scatter(at.mean_abs_curv, at.r_wf, s=4, label='front'); axs[1].scatter(at.mean_abs_curv, at.r_wr, s=4, label='rear')
    axs[1].set_xscale('symlog', linthresh=1e-3); axs[1].set_ylim(0.95, 1.05); axs[1].grid(True); axs[1].legend()
    axs[1].set_xlabel('mean |curvature| [1/m]'); axs[1].set_ylabel('map ds / wheel distance (wheel km/h / 3.6)')
    fig.savefig(PLOTS / 'val_along_track_ratio.png', dpi=90, bbox_inches='tight')
    plt.close(fig)
    res = dict(cross_track=summ, cross_track_by_region=json.loads(reg.to_json()), along_track=tot,
               along_track_by_curv=json.loads(kt.reset_index().astype({'kbin': str}).to_json(orient='records')))
    (OUT / 'cache' / 'val_results.json').write_text(json.dumps(res, indent=1))
    pf.to_parquet(OUT / 'cache' / 'val_perfix.parquet') if hasattr(pf, 'to_parquet') else None
    at.to_csv(OUT / 'cache' / 'val_along_track_windows.csv', index=False)
    return res


if __name__ == '__main__':
    main()
