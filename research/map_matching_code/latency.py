import sys, pickle
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
import numpy as np
from common import load

SP = 'C:/MosTransHack/research/map_matching_code/'
C = pickle.load(open(SP + 'cache.pkl', 'rb'))
taus = np.arange(-0.6, 0.61, 0.01)
best_pos = []
best_vel = []
for n, c in C.items():
    ok = np.isfinite(c['Sg']) & (np.abs(np.nan_to_num(c['Eg'], nan=99)) < 1.0)
    t = c['t'][ok]
    s = c['Sg'][ok]
    # speed from GNSS positions projected on map (central difference over 0.4 s)
    tg = np.arange(t[0] + 1, t[-1] - 1, 0.1)
    sg = np.interp(tg, t, s)
    vpos = (np.interp(tg + 0.2, t, s) - np.interp(tg - 0.2, t, s)) / 0.4
    # reject gaps
    gap = np.interp(tg, t[1:], np.diff(t)) > 0.3
    vw = 0.5 * (c['vf'] + c['vr']) / 3.6
    tw = c['tw']
    # use only dynamic parts (accelerating / braking) for lag estimation
    acc = np.gradient(np.interp(tg, tw, vw), tg)
    dyn = (np.abs(acc) > 0.3) & ~gap & (vpos > 0.5)
    if dyn.sum() < 200:
        continue
    err = []
    for tau in taus:
        vw_s = np.interp(tg - tau, tw, vw)  # wheel speed shifted: compare v_pos(t) with v_wheel(t - tau)
        err.append(np.mean((vpos[dyn] - vw_s[dyn]) ** 2))
    best_pos.append(taus[int(np.argmin(err))])
    # same with GNSS velocity topic (header stamps)
    d = load(n)
    mv = d['sensing__gnss__master__vel']
    gs = np.hypot(mv[:, 2], mv[:, 3])
    vg = np.interp(tg, mv[:, 1], gs)
    err2 = []
    for tau in taus:
        vw_s = np.interp(tg - tau, tw, vw)
        err2.append(np.mean((vg[dyn] - vw_s[dyn]) ** 2))
    best_vel.append(taus[int(np.argmin(err2))])
best_pos = np.array(best_pos)
best_vel = np.array(best_vel)
print('runs', len(best_pos))
print('lag of GNSS-position-derived speed vs wheel speed (positive = GNSS position lags wheel): median %.3f s, IQR %.3f..%.3f' % (np.median(best_pos), *np.percentile(best_pos, [25, 75])))
print('lag of GNSS vel topic vs wheel speed: median %.3f s, IQR %.3f..%.3f' % (np.median(best_vel), *np.percentile(best_vel, [25, 75])))
