"""How much does the map help along-track?  Dead-reckoning experiment on val runs (wheel data only after init).

For each val run: initialise on 'main' at the first RTK fix (after the run has reached main), then integrate
wheel distance and compare with the RTK truth s(t) (fixes projected on main, unwrapped). Variants:
  A raw        : ds = mean(front, rear) km/h / 3.6 * dt
  B scale      : A * (1 + c0)                       (global straight-track scale from train)
  C curvature  : A * (1 + c0 + c_abs*|k(s_est)|)     (map curvature, calibration from train windows)
  D landmarks  : C + reset s_est to the nearest stop landmark (train map) when the wheels report standstill
                 >= 3 s and a landmark lies within +-gate (gate = 3 m + 1 % of distance since last reset)
  E online     : D + online wheel-scale estimate from consecutive landmark resets (k <- 0.5 k + 0.5 k_seg)
Errors are evaluated only at RTK fix times (truth), not across GNSS gaps.
Metrics per run: along-track RMSE / MAX / end error, end drift % of distance.
NB: no slip handling here (raw wheel speeds) -- this isolates the map's contribution.
Run: python drift_sim.py
"""
import json

import numpy as np
import pandas as pd

import data_io as D
import runs as R
import track_map as TM
from polyline import Polyline

OUT = D.OUT


def wheel_series(bag):
    w = D.wheels(bag)
    f, r = w['front'], w['rear']
    t = f[:, 1]                     # header stamps (measurement epoch), consistent with GNSS header stamps
    o = np.argsort(t)
    f = f[o]
    t = t[o]
    ro = np.argsort(r[:, 1])
    vr = np.interp(t, r[ro, 1], r[ro, 2])
    return t, 0.5 * (f[:, 2] + vr) / 3.6, f[:, 2] / 3.6, vr / 3.6


def simulate(run, tm, cal, landmarks, P, variant):
    E = tm.edges['main']
    L = E.length
    s, d, dist, seg, _ = P.project(run.x, run.y, run.psi, max_d=1.0, max_dpsi=np.radians(45))
    ok = run.good & np.isfinite(s)
    if ok.sum() < 200:
        return None
    k0 = int(np.flatnonzero(ok)[0])
    t_true = run.th[ok]           # GNSS header stamps (position epoch)
    s_true = np.unwrap(s[ok], period=L)
    s_true = s_true - s_true[0] + s[ok][0]
    tw, vw, vf, vr = wheel_series(run.bag)
    m = (tw >= run.th[k0]) & (tw <= t_true[-1])
    tw, vw = tw[m], vw[m]
    if len(tw) < 100:
        return None
    s_est = np.empty(len(tw))
    s_cur = s_true[0]
    last_reset_s = s_cur
    raw_since = 0.0            # curvature-corrected wheel distance since last reset (unscaled by k)
    k_scale = 1.0 + cal['c0']
    stand = 0.0
    applied = []
    for i in range(len(tw)):
        if i > 0:
            dt = tw[i] - tw[i - 1]
            v = 0.5 * (vw[i] + vw[i - 1])
            if variant == 'A':
                scale = 1.0
            elif variant == 'B':
                scale = 1.0 + cal['c0']
            else:
                kk = abs(float(E.sample(s_cur)[4]))
                base = 1.0 + cal['c_abs'] * kk
                scale = base * (k_scale if variant == 'E' else 1.0 + cal['c0'])
                raw_since += v * dt * base
            s_cur += v * dt * scale
            if variant in ('D', 'E'):
                stand = stand + dt if vw[i] < 0.02 else 0.0
                if stand >= 3.0 and (not applied or applied[-1][0] < tw[i] - 30):
                    gate = 3.0 + 0.01 * abs(s_cur - last_reset_s)
                    sm = s_cur % L
                    dd = (landmarks - sm + L / 2) % L - L / 2
                    j = np.flatnonzero(np.abs(dd) <= gate)
                    if len(j) == 1:
                        s_new = s_cur + dd[j[0]]
                        if variant == 'E' and applied and raw_since > 200.0:
                            k_seg = (s_new - last_reset_s) / raw_since
                            if 0.95 < k_seg < 1.05:
                                k_scale = 0.5 * k_scale + 0.5 * k_seg
                        s_cur = s_new
                        last_reset_s = s_cur
                        raw_since = 0.0
                        applied.append((tw[i], dd[j[0]]))
                    elif len(j) == 0 and variant == 'E' and not applied:
                        pass
                if variant == 'E' and not applied and raw_since == 0.0:
                    pass
        s_est[i] = s_cur
    # evaluate at truth epochs only
    s_hat = np.interp(t_true, tw, s_est)
    e = s_hat - s_true
    dist_tr = s_true[-1] - s_true[0]
    return dict(bag=run.bag, variant=variant, dist_km=dist_tr / 1000, rmse=float(np.sqrt(np.mean(e ** 2))),
                max=float(np.max(np.abs(e))), end=float(e[-1]), end_drift_pct=float(100 * abs(e[-1]) / dist_tr),
                n_resets=len(applied), k_final=float(k_scale))


def main():
    tm_tr = TM.TrackMap(OUT / 'map_train')
    E = tm_tr.edges['main']
    P = Polyline(np.column_stack([E.x, E.y]), closed=True)
    wf = json.loads((OUT / 'cache' / 'wheel_curv_fit.json').read_text()) if (OUT / 'cache' / 'wheel_curv_fit.json').exists() else None
    # calibration: mean of front/rear fits (train+val windows; c_abs is physics-like, c0 tiny) -- see REPORT
    if wf:
        a = wf['all_cruise']
        cal = dict(c0=0.5 * (a['dwf']['c0'] + a['dwr']['c0']), c_abs=0.5 * (a['dwf']['c_abs'] + a['dwr']['c_abs']))
    else:
        cal = dict(c0=0.0004, c_abs=0.36)
    lm = np.array([st['s'] for st in tm_tr.stops if st['edge'] == 'main' and st['landmark']])
    rows = []
    for b, r in R.load_runs(D.split('val')).items():
        for v in ('A', 'B', 'C', 'D', 'E'):
            o = simulate(r, tm_tr, cal, lm, P, v)
            if o:
                rows.append(o)
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print('calibration', cal, 'landmarks', len(lm))
    print(df.pivot_table(index='bag', columns='variant', values=['rmse', 'end_drift_pct']).round(3).to_string())
    summ = df.groupby('variant').agg(runs=('bag', 'count'), km=('dist_km', 'sum'), rmse_med=('rmse', 'median'),
                                      rmse_mean=('rmse', 'mean'), max_med=('max', 'median'), max_max=('max', 'max'),
                                      end_abs_med=('end', lambda x: np.median(np.abs(x))),
                                      drift_pct_med=('end_drift_pct', 'median'), drift_pct_max=('end_drift_pct', 'max'),
                                      resets_med=('n_resets', 'median'))
    print(summ.round(3).to_string())
    (OUT / 'cache' / 'drift_sim.json').write_text(json.dumps(dict(cal=cal, summary=json.loads(summ.to_json(orient='index')),
                                                                 per_run=rows), indent=1))
    return df


if __name__ == '__main__':
    main()
