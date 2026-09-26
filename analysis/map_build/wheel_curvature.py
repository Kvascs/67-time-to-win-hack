"""Wheel odometry vs map arc length as a function of track curvature (along-track consistency of the map
and a calibration hint for the estimator).

10 s windows of continuous RTK fixes on 'main' (speed > 1 m/s); cruising filter |a| < 0.15 m/s^2.
Fits  ds_map/ds_x - 1 = c0 + c1*k + c2*|k|  for x in {wheel front, wheel rear, GNSS Doppler}.
Run: python wheel_curvature.py [map_dir]  -> cache/wheel_curv_windows.csv, cache/wheel_curv_fit.json
"""
import json
import sys

import numpy as np
import pandas as pd

import data_io as D
import runs as R
from validate import MapProjector

OUT = D.OUT


def windows(mp, runs, win_n=100):
    E = mp.polys['main']
    L = E.length
    em = mp.tm.edges['main']
    rows = []
    for b, r in runs.items():
        s, d, dist, seg, dpsi = E.project(r.x, r.y, r.psi, max_d=1.0, max_dpsi=np.radians(45))
        ok = r.good & np.isfinite(s) & (r.speed > 1.0)
        idx = np.flatnonzero(ok)
        if len(idx) < win_n:
            continue
        su = np.unwrap(s[idx], period=L)
        t = r.th[idx]
        brk = np.flatnonzero((np.diff(t) > 0.3) | (np.abs(np.diff(su)) > 3.0)) + 1
        for ch in np.split(np.arange(len(idx)), brk):
            for k0 in range(0, len(ch) - win_n + 1, win_n):
                w = ch[k0:k0 + win_n]
                ii = idx[w]
                tw = r.th[ii]
                dsm = su[w[-1]] - su[w[0]]
                if dsm < 10:
                    continue
                sp = r.speed[ii]
                kc = np.interp(np.mod(su[w], L), em.s, em.curv, period=L)
                rows.append(dict(bag=b, veh=b[:5], s0=float(np.mod(su[w[0]], L)), ds=dsm,
                                 dd=np.trapezoid(sp, tw), dwf=np.trapezoid(r.wheel_f[ii], tw),
                                 dwr=np.trapezoid(r.wheel_r[ii], tw), k=float(np.mean(kc)),
                                 acc=float((sp[-1] - sp[0]) / (tw[-1] - tw[0])), v=float(np.mean(sp))))
    return pd.DataFrame(rows)


def fit(df):
    X = np.column_stack([np.ones(len(df)), df.k, np.abs(df.k)])
    sw = np.sqrt(df.ds.values)
    out = {}
    for col in ('dwf', 'dwr', 'dd'):
        y = df.ds / df[col] - 1
        beta = np.linalg.lstsq(X * sw[:, None], y.values * sw, rcond=None)[0]
        res = y.values - X @ beta
        out[col] = dict(c0=float(beta[0]), c_signed=float(beta[1]), c_abs=float(beta[2]),
                        resid_mad=float(np.median(np.abs(res))))
    return out


def main(map_dir='map'):
    mp = MapProjector(OUT / map_dir)
    runs = {**R.load_runs(D.split('train')), **R.load_runs(D.split('val'))}
    df = windows(mp, runs)
    df.to_csv(OUT / 'cache' / 'wheel_curv_windows.csv', index=False)
    cr = df[np.abs(df.acc) < 0.15]
    res = dict(n_windows=int(len(df)), n_cruise=int(len(cr)), km_cruise=float(cr.ds.sum() / 1000), all_cruise=fit(cr))
    for veh in ('30618', '30639'):
        c = cr[cr.veh == veh]
        if len(c) > 50:
            res[f'veh_{veh}'] = fit(c)
    # per-run straight-line scale (|k| < 0.0005): run-to-run variability of the wheel scale
    st = cr[np.abs(cr.k) < 0.0005]
    pr = st.groupby('bag').apply(lambda g: pd.Series(dict(km=g.ds.sum() / 1000, f=g.ds.sum() / g.dwf.sum(),
                                                          r=g.ds.sum() / g.dwr.sum())), include_groups=False)
    pr = pr[pr.km > 0.5]
    res['per_run_straight_scale'] = dict(front_p5_p50_p95=[float(v) for v in np.percentile(pr.f, [5, 50, 95])],
                                         rear_p5_p50_p95=[float(v) for v in np.percentile(pr.r, [5, 50, 95])],
                                         n_runs=int(len(pr)))
    (OUT / 'cache' / 'wheel_curv_fit.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'map')
