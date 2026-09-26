"""Robust altitude profile: median ACROSS RUNS of per-run bin medians (stops no longer dominate), then rewrite CSV h/grade."""
import sys, pickle, json
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

SP = 'C:/MosTransHack/research/map_matching_code/'
OUT = 'C:/MosTransHack/research/map_matching_data/'
C = pickle.load(open(SP + 'cache.pkl', 'rb'))
meta = json.load(open(OUT + 'route10_map_meta.json'))
for dirn in ('AB', 'BA'):
    fn = OUT + 'route10_%s_centerline.csv' % dirn
    hdr = open(fn).readline().strip()
    a = np.genfromtxt(fn, delimiter=',', skip_header=1)
    n = len(a)
    per_run = []
    for nm, c in C.items():
        if c['dir'] != dirn or not nm.startswith('30618'):
            continue
        ok = np.isfinite(c['Sg']) & (np.abs(np.nan_to_num(c['Eg'], nan=99)) < 1.0)
        b = np.clip(np.round(c['Sg'][ok]).astype(int), 0, n - 1)
        z = c['z'][ok]
        zr = np.full(n, np.nan)
        order = np.argsort(b)
        bs = b[order]; zs = z[order]
        splits = np.searchsorted(bs, np.arange(n + 1))
        for i in range(n):
            if splits[i + 1] > splits[i]:
                zr[i] = np.median(zs[splits[i]:splits[i + 1]])
        per_run.append(zr)
    Z = np.array(per_run)
    cnt = np.sum(np.isfinite(Z), axis=0)
    zmed = np.nanmedian(Z, axis=0)
    zmed[cnt < 3] = np.nan
    s = np.arange(n)
    good = np.isfinite(zmed)
    zf = np.interp(s, s[good], zmed[good])
    zf = gaussian_filter1d(median_filter(zf, size=9, mode='nearest'), 5)
    grade = np.clip(np.gradient(gaussian_filter1d(zf, 10)), -0.08, 0.08)
    old_h = a[:, 3].copy()
    a[:, 3] = zf
    a[:, 8] = grade
    fmt = ['%.1f', '%.8f', '%.8f', '%.3f', '%.3f', '%.3f', '%.5f', '%.6f', '%.5f', '%.2f', '%.1f', '%.1f', '%.1f', '%.1f']
    np.savetxt(fn, a, delimiter=',', header=hdr, comments='', fmt=fmt)
    meta['edges'][dirn]['max_abs_grade'] = float(np.abs(grade).max())
    meta['edges'][dirn]['p99_abs_grade'] = float(np.percentile(np.abs(grade), 99))
    meta['edges'][dirn]['h_range'] = [float(zf.min()), float(zf.max())]
    print(dirn, 'runs', len(Z), 'max|grade| %.3f p99 %.3f | h %.2f..%.2f | max |h_new-h_old| %.2f m at s=%d' % (
        np.abs(grade).max(), np.percentile(np.abs(grade), 99), zf.min(), zf.max(), np.abs(zf - old_h).max(), int(np.argmax(np.abs(zf - old_h)))))
meta['altitude'] = 'NavSatFix altitude (WGS84 ellipsoidal per ROS spec): median across 30618 runs of per-run 1 m-bin medians, median(9)+gaussian(5 m); grade from gaussian(10 m), clipped to +-8 %'
json.dump(meta, open(OUT + 'route10_map_meta.json', 'w'), indent=2, ensure_ascii=False)
