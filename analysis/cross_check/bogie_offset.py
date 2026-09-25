"""Where are the bogies relative to the master GNSS antenna (along the track)?

Wheel sensors under-read in tight curves; the effect is local to the bogie that carries the
sensor. Fit ratio(t) = wheel speed / master Doppler speed against the map curvature evaluated at
s_master(t) + D for a grid of D and pick the D that explains the curve dips best, separately for
the front and the rear bogie. The difference D_front - D_rear should match the bogie base
(7.55 m per the organisers), which validates the method.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\MosTransHack')
sys.path.insert(0, str(ROOT / 'analysis' / 'map_build'))
from track_map import TrackMap, geodetic_to_enu  # noqa: E402

ORIGIN = (55.8028, 37.424, 160.0)
tm = TrackMap(str(ROOT / 'analysis' / 'map_build' / 'map_train'))
M = tm.edges['main']
ms = np.arange(0.0, M.length, 0.5)
kap = np.array([M.curvature(s) for s in ms])


def kappa_at(s):
    return np.interp(np.mod(s, M.length), ms, np.abs(kap))


splits = json.loads((ROOT / 'data' / 'splits.json').read_text())
rows = {'front': [], 'rear': []}
for b in splits['train'] + splits['val']:
    d = np.load(ROOT / 'data' / 'npz' / f'{b}.npz')
    fx, mv = d['sensing__gnss__master__fix'], d['sensing__gnss__master__vel']
    if len(fx) < 1000:
        continue
    ok = fx[:, 5] == 2
    if ok.mean() < 0.8:
        continue
    fx = fx[ok]
    x, y, _ = geodetic_to_enu(fx[:, 2], fx[:, 3], fx[:, 4], ORIGIN)
    vt, vx, vy = mv[:, 1], mv[:, 2], mv[:, 3]
    hd = np.arctan2(np.interp(fx[:, 1], vt, vy), np.interp(fx[:, 1], vt, vx))
    s, dist, _, _ = M.project(x, y, heading=hd, max_d=3)
    good = np.isfinite(s)
    if good.sum() < 1000:
        continue
    t_s = fx[good, 1] - 0.045            # fixes lead wheel/vel stamps by ~45 ms
    s_u = np.unwrap(s[good] / M.length * 2 * np.pi) / (2 * np.pi) * M.length
    for name, key in (('front', 'vehicle__front_bogie_velocity'), ('rear', 'vehicle__rear_bogie_velocity')):
        w = d[key]
        tw = w[:, 1]
        vw = w[:, 2] / 3.6
        sel = (tw > t_s[0]) & (tw < t_s[-1])
        tw, vw = tw[sel], vw[sel]
        vg = np.hypot(np.interp(tw, vt, vx), np.interp(tw, vt, vy))
        sm = np.interp(tw, t_s, s_u)
        # cruising, moving: stable ratio
        acc = np.gradient(np.convolve(vg, np.ones(5) / 5, 'same'), tw)
        m = (vg > 3.0) & (vw > 3.0) & (np.abs(acc) < 0.2)
        if m.sum() < 200:
            continue
        r = vw[m] / vg[m]
        r = r / np.median(r)                 # per-run scale removed
        rows[name].append((sm[m], r))

res = {}
Ds = np.arange(-25.0, 25.01, 0.5)
for name, lst in rows.items():
    S = np.concatenate([a for a, _ in lst])
    R = np.concatenate([b for _, b in lst])
    keep = np.abs(R - 1) < 0.05
    S, R = S[keep], R[keep]
    best = []
    for D in Ds:
        k = np.minimum(kappa_at(S + D), 0.03)
        A = np.vstack([np.ones_like(k), k]).T
        coef, rss, *_ = np.linalg.lstsq(A, R, rcond=None)
        resid = R - A @ coef
        best.append((float(np.mean(resid ** 2)), D, coef[1]))
    best.sort()
    res[name] = best[0]
    curve = sorted(best, key=lambda x: x[1])
    print(f'{name}: n={len(R)} best D={best[0][1]:+.1f} m (slope {best[0][2]:+.3f} per 1/m); '
          f'mse by D: ' + ' '.join(f'{D:+.0f}:{mse*1e6:.1f}' for mse, D, _ in curve[::4]))
print(f"front - rear = {res['front'][1] - res['rear'][1]:+.1f} m (organisers: bogie base 7.55 m)")
