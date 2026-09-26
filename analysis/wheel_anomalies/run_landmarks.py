"""Along-track landmarks usable without GNSS (for position/scale correction): repeatable stop locations
(wheels at 0 for >= 8 s) and the fixed locations of abrupt traction cut-offs (notch +>=3 -> 0).
Positions from GNSS master fixes (status 2 preferred), UTM 37N. Output: landmarks.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import anomalies as A  # noqa: E402

TR = Transformer.from_crs('EPSG:4326', 'EPSG:32637', always_xy=True)


def cluster(P, radius):
    used = np.zeros(len(P), bool)
    out = []
    for i in range(len(P)):
        if used[i]:
            continue
        d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
        m = (d < radius) & ~used
        used |= m
        c = P[m]
        cen = np.median(c, 0)
        r = np.hypot(c[:, 0] - cen[0], c[:, 1] - cen[1])
        out.append(dict(n=int(m.sum()), x=float(cen[0]), y=float(cen[1]), r_median=float(np.median(r)),
                        r_p90=float(np.percentile(r, 90))))
    return out


def main():
    stops, cuts = [], []
    for x in A.splits()['info']:
        if x['dur'] < 300:
            continue
        b = A.load_bag(x['bag'])
        f = b.gnss_fix['master']
        if len(f) < 500:
            continue
        o = np.argsort(f.t_hdr)
        th = f.t_hdr[o]
        X, Y = TR.transform(f.v[o, 1], f.v[o, 0])
        st = f.v[o, 3]
        w = np.interp(th, b.front.t_hdr, b.front.v)
        for s, e in A.runs(w == 0):
            if th[e] - th[s] < 8:
                continue
            idx = np.arange(s, e + 1)
            if (st[idx] == 2).sum() > 20:
                idx = idx[st[idx] == 2]
            stops.append((np.median(X[idx]), np.median(Y[idx])))
        c = b.cmd
        oc = np.argsort(c.t_hdr)
        v = c.v[oc]; tc = c.t_hdr[oc]
        for i in np.where((v[:-1] >= 3) & (v[1:] == 0))[0]:
            k = np.argmin(np.abs(th - tc[i]))
            if abs(th[k] - tc[i]) < 0.5:
                cuts.append((X[k], Y[k]))
    rows = []
    for kind, P, rad in (('stop', np.array(stops), 25.0), ('traction_cutoff', np.array(cuts), 40.0)):
        for c in cluster(P, rad):
            c['kind'] = kind
            rows.append(c)
    df = pd.DataFrame(rows).sort_values(['kind', 'n'], ascending=[True, False])
    df.to_csv(HERE / 'landmarks.csv', index=False)
    print(df[df.n >= 10].round(2).to_string())


if __name__ == '__main__':
    main()
