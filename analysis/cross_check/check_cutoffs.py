"""Cross-check 6: abrupt traction cut-offs (notch >= +4 -> 0 in one 50 ms step) as GNSS-free landmarks.
Positions = master RTK fix (header time) projected on the full map 'main' edge; clustered in s."""
import sys
import numpy as np
sys.path.insert(0, r"C:\MosTransHack\analysis\cross_check"); sys.path.insert(0, r"C:\MosTransHack\analysis\map_build")
from xc_common import load, bag_meta, glitch_free, mono
from track_map import TrackMap, geodetic_to_enu
tm = TrackMap(r"C:\MosTransHack\analysis\map_build\map"); M = tm.edges['main']; O = (55.8028, 37.424, 160.0)
meta = bag_meta(); ev = []
for b, m in sorted(meta.items()):
    if m['split'] not in ('train', 'val'):
        continue
    d = load(b); c = d['cmd']; fx = d['fixm']; vm = d['velm']
    g = glitch_free(c); tc, u = mono(c[g, 1], c[g, 2])
    jj = np.flatnonzero((u[:-1] >= 4) & (u[1:] == 0) & (np.diff(tc) < 0.08))
    if not len(jj): continue
    gf = glitch_free(fx) & (fx[:, 5] == 2)
    if gf.sum() < 100: continue
    tf, la, lo, al = mono(fx[gf, 1], fx[gf, 2], fx[gf, 3], fx[gf, 4])
    gv = glitch_free(vm); tv, vx, vy = mono(vm[gv, 1], vm[gv, 2], vm[gv, 3])
    for j in jj:
        t = tc[j + 1]
        k = np.searchsorted(tf, t)
        if k <= 0 or k >= len(tf) or tf[k] - tf[k - 1] > 0.25: continue
        w = (t - tf[k - 1]) / (tf[k] - tf[k - 1])
        x, y, z = geodetic_to_enu(la[k-1] + w*(la[k]-la[k-1]), lo[k-1] + w*(lo[k]-lo[k-1]), al[k], O)
        ve, vn = np.interp(t, tv, vx), np.interp(t, tv, vy)
        s, dd, dist, _ = M.project(np.atleast_1d(x), np.atleast_1d(y), heading=np.atleast_1d(np.arctan2(vn, ve)), max_d=5)
        ev.append((b, t, float(s[0]), float(np.hypot(ve, vn)), int(u[j])))
s = np.array([e[2] for e in ev]); ok = np.isfinite(s); s = s[ok]; ev = [e for e, o in zip(ev, ok) if o]
print('events with RTK position on main:', len(s))
order = np.argsort(s); ss = s[order]
br = np.flatnonzero(np.diff(ss) > 20)
st = np.r_[0, br + 1]; en = np.r_[br, len(ss) - 1]
for a, b2 in zip(st, en):
    grp = ss[a:b2 + 1]
    if len(grp) >= 3:
        med = np.median(grp); mad = 1.4826 * np.median(np.abs(grp - med))
        spd = np.array([ev[i][3] for i in order[a:b2+1]])
        print(f' cluster s={med:8.1f} m  n={len(grp):3d}  robust std={mad:5.2f} m  range={grp.min():.1f}..{grp.max():.1f}  speed med {np.median(spd):.1f} m/s')
print('isolated/small clusters:', sum(1 for a, b2 in zip(st, en) if b2 - a + 1 < 3))
