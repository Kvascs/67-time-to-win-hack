"""Do bogie wheel speeds carry a usable track-curvature signature?  (competitions_prior_art.md, section 10)

If a bogie sensor measured a single wheel, v_wheel/v_true - 1 would swing by about +-(b/2)*kappa
(b ~ 1.52 m gauge, so +-3.8% at R = 20 m). We bin the relative error against GNSS speed by signed curvature.
Result on 30618 bags: all bins within -0.11..-0.35 %  -> no usable signature (axle-average behaviour).
"""
import glob, os, collections
import numpy as np

root = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'npz')
res = []
for f in sorted(glob.glob(os.path.join(root, '30618_*.npz')))[:40]:
    d = np.load(f)
    mv, fr, rr = d['sensing__gnss__master__vel'], d['vehicle__front_bogie_velocity'], d['vehicle__rear_bogie_velocity']
    if len(mv) < 3000:
        continue
    t, vx, vy = mv[:, 0], mv[:, 2], mv[:, 3]
    sp = np.hypot(vx, vy)
    psi = np.unwrap(np.arctan2(vy, vx))
    kappa = np.where(sp > 3, np.gradient(psi, t) / np.maximum(sp, 0.1), np.nan)
    vf = np.interp(t, fr[:, 0], fr[:, 2]) / 3.6
    vr = np.interp(t, rr[:, 0], rr[:, 2]) / 3.6
    m = (sp > 4) & np.isfinite(kappa)
    ef, er, k = vf[m] / sp[m] - 1, vr[m] / sp[m] - 1, kappa[m]
    for lo, hi in [(0, 0.002), (0.002, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.2)]:
        for sgn in (1, -1):
            mm = (sgn * k >= lo) & (sgn * k < hi)
            if mm.sum() > 50:
                res.append((lo, hi, sgn, mm.sum(), np.median(ef[mm]), np.median(er[mm])))
agg = collections.defaultdict(list)
for lo, hi, sgn, n, a, b in res:
    agg[(lo, hi, sgn)].append((n, a, b))
for key in sorted(agg):
    arr = np.array(agg[key])
    w = arr[:, 0]
    print('kappa bin %s  n=%6d  front %+.4f  rear %+.4f' % (key, w.sum(), np.average(arr[:, 1], weights=w), np.average(arr[:, 2], weights=w)))
