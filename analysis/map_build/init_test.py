"""Test of GNSS initial alignment (track + s from the first seconds of GNSS) against the train-only map.

Truth: position at the query time from RTK fixes (status 2 + checks), map-matched with full hindsight.
Cases
  start_T   : real run starts, window [t0, t0+T] for T = 1, 2, 5 s (all fixes incl. non-RTK, rover, Doppler)
  mid       : 25 random start times per run (window 2 s), mostly moving on the double track
  degraded  : run starts with extra noise / common bias on fixes and without rover / Doppler
Metric: position error of the initialised map point vs truth (|dpos|), wrong-track rate (|dpos| > 2 m with
lateral part > 1.5 m), along-track error, heading source, ambiguity flag.
Run: python init_test.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import data_io as D
import runs as R
from validate import MapProjector

OUT = D.OUT


def truth_at(mp: MapProjector, r, tq, s_all, ek_all):
    """(x, y, edge, s) of the tram at time tq from RTK fixes, or None."""
    g = r.good & np.isfinite(s_all)
    near = g & (np.abs(r.th - tq) <= 0.15)
    if near.any():
        k = np.flatnonzero(near)[np.argmin(np.abs(r.th[near] - tq))]
        return r.x[k], r.y[k], mp.ids[ek_all[k]], s_all[k]
    # stationary until first RTK fix?
    after = np.flatnonzero(g & (r.th > tq) & (r.th < tq + 120))
    if len(after) == 0:
        return None
    k = after[0]
    seg = (r.th >= tq) & (r.th <= r.th[k])
    dist = np.trapezoid(np.nan_to_num(r.speed[seg]), r.th[seg]) if seg.sum() > 1 else 0.0
    if dist < 0.3:
        return r.x[k], r.y[k], mp.ids[ek_all[k]], s_all[k]
    return None


def run_case(tm, mp, r, s_all, ek_all, t_from, T, noise=0.0, bias=0.0, use_rover=True, use_vel=True,
             use_status=True, rng=None):
    w = (r.th >= t_from) & (r.th <= t_from + T)
    if w.sum() == 0:
        return None
    idx = np.flatnonzero(w)
    tq = r.th[idx[-1]]
    tru = truth_at(mp, r, tq, s_all, ek_all)
    if tru is None:
        return None
    x = r.x[idx].copy()
    y = r.y[idx].copy()
    if noise > 0 or bias > 0:
        ang = rng.uniform(0, 2 * np.pi)
        x += bias * np.cos(ang) + rng.normal(0, noise, len(idx))
        y += bias * np.sin(ang) + rng.normal(0, noise, len(idx))
    rov = None
    if use_rover:
        ok = np.isfinite(r.rx[idx])
        if ok.any():
            rov = (r.rx[idx] + (x - r.x[idx]), r.ry[idx] + (y - r.y[idx]))  # same common error as master
    vel = (r.ve[idx], r.vn[idx]) if use_vel else None
    res = tm.init_from_map_xy(r.th[idx], x, y, rover_xy=rov, vel=vel,
                              status=r.status[idx] if use_status else None)
    if not res.ok:
        return dict(ok=False, pos_err=np.nan, wrong_track=True, along=np.nan, lateral=np.nan, src=res.heading_src,
                    amb=res.ambiguous, moving=res.moving, t_rel=t_from - r.th[0], edge=None, edge_true=tru[2])
    xi, yi, zi, yawi = tm.edges[res.edge].pose(res.s)
    xt, yt = tru[0], tru[1]
    ex, ey = float(xi) - xt, float(yi) - yt
    # decompose vs track direction at truth
    Et = tm.edges[tru[2]]
    _, _, _, yawt = Et.pose(tru[3])
    along = ex * np.cos(yawt) + ey * np.sin(yawt)
    lateral = -ex * np.sin(yawt) + ey * np.cos(yawt)
    return dict(ok=True, pos_err=float(np.hypot(ex, ey)), along=float(along), lateral=float(lateral),
                wrong_track=bool(abs(lateral) > 1.5), src=res.heading_src, amb=res.ambiguous, moving=res.moving,
                t_rel=float(t_from - r.th[0]), edge=res.edge, edge_true=tru[2], n=res.n_fixes,
                frac_rtk_window=float(np.mean(r.status[idx] == 2)))


def summarize(df, name):
    if len(df) == 0:
        return {}
    e = df.pos_err.fillna(99.0)
    out = dict(case=name, n=len(df), ok=float(df.ok.mean()), wrong_track=float(df.wrong_track.mean()),
               pos_err_p50=float(np.percentile(e, 50)), pos_err_p90=float(np.percentile(e, 90)),
               pos_err_p99=float(np.percentile(e, 99)), frac_lt_0p5=float(np.mean(e < 0.5)),
               frac_lt_2=float(np.mean(e < 2.0)), ambiguous=float(df.amb.mean()),
               src=json.dumps(df.src.value_counts(normalize=True).round(3).to_dict()))
    return out


def main():
    import track_map as TM
    tm = TM.TrackMap(OUT / 'map_train')
    mp = MapProjector(OUT / 'map_train')
    runs = R.load_runs(D.split('val'))
    rng = np.random.default_rng(0)
    rows = {k: [] for k in ('start_T1', 'start_T2', 'start_T5', 'mid_T2', 'mid_T2_norover_novel',
                            'start_T2_noise1m', 'start_T2_bias2m', 'start_T5_bias5m', 'mid_T2_bias2m_norover_novel',
                            'start_T2_nostatus')}
    for b, r in runs.items():
        s_all, d_all, dist_all, ek_all = mp.project(r.x, r.y, r.psi, max_d=3.0)
        t0 = r.th[0]
        for T in (1, 2, 5):
            o = run_case(tm, mp, r, s_all, ek_all, t0, T, rng=rng)
            if o: rows[f'start_T{T}'].append(dict(bag=b, **o))
        o = run_case(tm, mp, r, s_all, ek_all, t0, 2, noise=1.0, rng=rng)
        if o: rows['start_T2_noise1m'].append(dict(bag=b, **o))
        o = run_case(tm, mp, r, s_all, ek_all, t0, 2, bias=2.0, rng=rng)
        if o: rows['start_T2_bias2m'].append(dict(bag=b, **o))
        o = run_case(tm, mp, r, s_all, ek_all, t0, 5, bias=5.0, rng=rng)
        if o: rows['start_T5_bias5m'].append(dict(bag=b, **o))
        o = run_case(tm, mp, r, s_all, ek_all, t0, 2, use_status=False, rng=rng)
        if o: rows['start_T2_nostatus'].append(dict(bag=b, **o))
        for tf in rng.uniform(r.th[0] + 10, r.th[-1] - 10, 25):
            o = run_case(tm, mp, r, s_all, ek_all, tf, 2, rng=rng)
            if o: rows['mid_T2'].append(dict(bag=b, **o))
            o = run_case(tm, mp, r, s_all, ek_all, tf, 2, use_rover=False, use_vel=False, rng=rng)
            if o: rows['mid_T2_norover_novel'].append(dict(bag=b, **o))
            o = run_case(tm, mp, r, s_all, ek_all, tf, 2, bias=2.0, use_rover=False, use_vel=False, rng=rng)
            if o: rows['mid_T2_bias2m_norover_novel'].append(dict(bag=b, **o))
    summ = [summarize(pd.DataFrame(v), k) for k, v in rows.items()]
    sm = pd.DataFrame(summ)
    pd.set_option('display.width', 250)
    print(sm.round(4).to_string())
    st = pd.DataFrame(rows['start_T2'])
    print('\nper-run start (T=2 s):')
    print(st[['bag', 'edge', 'edge_true', 'pos_err', 'along', 'lateral', 'src', 'amb', 'moving', 'frac_rtk_window']].round(3).to_string())
    bad = pd.DataFrame(rows['mid_T2_bias2m_norover_novel'])
    if len(bad):
        print('\nmid-run, 2 m bias, no rover/vel: wrong-track cases by true edge/region:')
        print(bad[bad.wrong_track].groupby('edge_true').size())
    (OUT / 'cache' / 'init_results.json').write_text(json.dumps(summ, indent=1))
    return sm


if __name__ == '__main__':
    main()
