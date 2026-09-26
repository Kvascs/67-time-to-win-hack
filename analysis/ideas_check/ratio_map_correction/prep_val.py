"""Step 2a: per VAL bag, align on the paired wheel stamps
  - the ratio sample y = log(vf/vr) and speeds,
  - our estimator's online output (replay bl_h_dly): antenna-1 arc s_map (unwrapped), odometer odo = integral
    of the published speed (published speed lags 0.1 s -> taken at stamp + 0.1 s), s_var, flags, pos_valid,
  - truth: RTK master fixes (status 2, within 1.5 m of the train-only map) projected on the map, at t - 43.5 ms
    (the filter's s at a wheel stamp corresponds to the fix at stamp - 45 ms, see MODEL 6.4 'Upreждение').
Also the truth at every RTK fix for the error evaluation.

Output: prep/<bag>.pkl (wheel-stamp table), prep/<bag>_fix.pkl
Usage: python prep_val.py [bag ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import rm_common as M

C = M.C
OUT = M.HERE / 'prep'
V_DELAY = 0.1     # published speed is that of stamp - 0.1 s
LEAD = 0.0435     # fix positions lead wheel stamps


def load_out(path):
    o = pd.read_csv(path, usecols=['stamp_ns', 'v', 's', 's_var', 'k', 'flags', 'pos_valid', 's_map'])
    o = o.drop_duplicates('stamp_ns', keep='last').sort_values('stamp_ns').reset_index(drop=True)
    o['t'] = o.stamp_ns.values * 1e-9
    dt = np.diff(o.t.values, prepend=o.t.values[0])
    o['odo'] = np.cumsum(np.maximum(o.v.values, 0) * dt)
    return o


def unwrap_valid(sm, L):
    """Unwrap s_map where valid (>= 0); NaN elsewhere; unwrapping continues across gaps."""
    out = np.full(len(sm), np.nan)
    ok = sm >= 0
    if ok.any():
        out[ok] = np.unwrap(sm[ok], period=L)
    return out


def local_project(pl, qx, qy, s_c, half=40.0):
    """Nearest point of the cyclic map within arc s_c +- half (avoids jumps to other parts of the track
    at crossings / parallel branches). Returns s (in [0, L)), distance."""
    n = len(pl.px) - 1
    nseg = int(2 * half) + 2
    i0 = np.searchsorted(pl.ps, np.mod(s_c - half, pl.L)) - 1
    idx = (i0[:, None] + np.arange(nseg)[None, :]) % n
    ax, ay = pl.px[idx], pl.py[idx]
    dx, dy = pl.px[idx + 1] - ax, pl.py[idx + 1] - ay
    l2 = np.maximum(dx * dx + dy * dy, 1e-12)
    tt = np.clip(((qx[:, None] - ax) * dx + (qy[:, None] - ay) * dy) / l2, 0, 1)
    dd = np.hypot(qx[:, None] - ax - tt * dx, qy[:, None] - ay - tt * dy)
    b = np.argmin(dd, 1)
    r = np.arange(len(qx))
    j = idx[r, b]
    s = pl.ps[j] + tt[r, b] * (pl.ps[j + 1] - pl.ps[j])
    return np.mod(s, pl.L), dd[r, b]


def truth_fixes(d, pl, origin):
    fx = C.master_fix_enu(d, origin)
    fx = fx[fx.status == 2].drop_duplicates('t').reset_index(drop=True)
    s, lat, dist = pl.project(fx.x.values, fx.y.values)
    fx['s_glob'] = s
    fx['d_glob'] = dist
    return fx


def prep_bag(bag, out_csv, pl, origin, L, truth='rtk'):
    d = C.load_bag(bag)
    w = C.wheel_pairs(d)
    o = load_out(out_csv)
    t = w.t.values
    ot = o.t.values
    s_unw = unwrap_valid(o.s_map.values, L)
    ok_o = np.isfinite(s_unw)
    df = pd.DataFrame({'t': t, 'vf': w.vf.values, 'vr': w.vr.values})
    df['v'] = 0.5 * (df.vf + df.vr)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['y'] = np.log(df.vf / df.vr)
    # estimator state at the wheel stamp (nearest output at or before the stamp)
    j = np.clip(np.searchsorted(ot, t, side='right') - 1, 0, len(ot) - 1)
    df['s_est'] = np.interp(t, ot[ok_o], s_unw[ok_o]) if ok_o.any() else np.nan
    df['est_ok'] = (o.pos_valid.values[j] == 1) & (o.s_map.values[j] >= 0) & (np.abs(ot[j] - t) < 0.2)
    df['flags'] = o['flags'].values[j]
    df['s_var'] = o.s_var.values[j]
    df['kscale'] = o.k.values[j]
    df['odo'] = np.interp(t + V_DELAY, ot, o.odo.values)
    df['v_est'] = np.interp(t + V_DELAY, ot, o.v.values)
    return df, o


def main():
    pl = M.load_valmap()
    origin = C.map_origin(M.VALMAP)
    L = pl.L
    OUT.mkdir(exist_ok=True)
    bags = sys.argv[1:] or [p.name.replace('_out.csv', '') for p in sorted(M.REPLAY_VAL.glob('*_out.csv'))]
    for bag in bags:
        df, o = prep_bag(bag, M.REPLAY_VAL / f'{bag}_out.csv', pl, origin, L)
        d = C.load_bag(bag)
        fx = truth_fixes(d, pl, origin)
        # estimator at each fix time (+ lead): s_map, flags, odo -> error evaluation at fix epochs
        ot = o.t.values
        s_unw = unwrap_valid(o.s_map.values, L)
        ok_o = np.isfinite(s_unw) & (o.pos_valid.values == 1)
        te = fx.t.values + LEAD
        jj = np.clip(np.searchsorted(ot, te, side='right') - 1, 0, len(ot) - 1)
        fx['s_est'] = np.interp(te, ot[ok_o], s_unw[ok_o])
        fx['est_ok'] = ok_o[jj] & (np.abs(ot[jj] - te) < 0.2)
        fx['odo'] = np.interp(te + V_DELAY, ot, o.odo.values)
        fx['v_est'] = np.interp(te + V_DELAY, ot, o.v.values)
        fx['t_est'] = te
        fx['flags'] = o['flags'].values[jj]
        # truth: fix projected on the map within +-40 m of the estimate (global projection if no estimate)
        sl, dl = local_project(pl, fx.x.values, fx.y.values, np.mod(fx.s_est.values, L))
        fx['s_true'] = np.where(fx.est_ok, sl, fx.s_glob)
        fx['dmain'] = np.where(fx.est_ok, dl, fx.d_glob)
        e = (fx.s_est - fx.s_true + L / 2) % L - L / 2
        fx['err'] = e
        fx['eval'] = fx.est_ok & (fx.dmain <= 0.5)
        fx.to_pickle(OUT / f'{bag}_fix.pkl')
        # truth at wheel stamps (diagnostics): unwrapped truth = s_est - err, interpolated between fixes < 0.5 s apart
        g = fx[fx['eval']]
        tf = g.t.values
        su = (g.s_est - g.err).values
        tq = df.t.values - LEAD
        k = np.clip(np.searchsorted(tf, tq), 1, max(len(tf) - 1, 1))
        gap = tf[k] - tf[k - 1]
        okk = (tq >= tf[k - 1]) & (tq <= tf[k]) & (gap < 0.5) & (np.abs(su[k] - su[k - 1]) < 15 * gap + 1)
        wgt = (tq - tf[k - 1]) / np.maximum(gap, 1e-9)
        df['s_true'] = np.where(okk, su[k - 1] + wgt * (su[k] - su[k - 1]), np.nan)
        df.to_pickle(OUT / f'{bag}.pkl')
        ev = e[fx['eval']]
        print(f'{bag}: wheel pairs {len(df)}, fixes {len(fx)} eval {len(ev)}, raw along err RMSE {np.sqrt(np.mean(ev ** 2)):.3f} m, '
              f'p95 {np.percentile(np.abs(ev), 95):.2f}, max {np.abs(ev).max():.2f}, bias {ev.mean():+.3f}', flush=True)


if __name__ == '__main__':
    main()
