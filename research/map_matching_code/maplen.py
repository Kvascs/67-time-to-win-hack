import sys, pickle
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
import numpy as np
import evaldr2 as E

W = 50.0
for dirn in ('AB', 'BA'):
    n_s = len(E.M[dirn]['P'])
    grid = np.arange(0, n_s, W)
    R = []
    for nm, c in E.C.items():
        if c['dir'] != dirn:
            continue
        ok = np.isfinite(c['Sg']) & (np.abs(np.nan_to_num(c['Eg'], nan=99)) < 3)
        t = c['t'][ok]
        sg = c['Sg'][ok]
        vw = 0.5 * (c['vf'] + c['vr']) / 3.6
        odo = np.r_[0, np.cumsum(0.5 * (vw[1:] + vw[:-1]) * np.diff(c['tw']))]
        od = np.interp(t, c['tw'], odo)
        mono = np.maximum.accumulate(sg)
        keep = np.r_[True, np.diff(mono) > 0]
        o_at = np.interp(grid, mono[keep], od[keep], left=np.nan, right=np.nan)
        # scale-normalise per run (remove wheel diameter error): divide by run median ratio
        r = np.diff(o_at) / W
        r = r / np.nanmedian(r)
        R.append(r)
    R = np.array(R)
    med = np.nanmedian(R, axis=0)
    cnt = np.sum(np.isfinite(R), axis=0)
    bad = np.where((np.abs(med - 1) > 0.01) & (cnt >= 5))[0]
    print('==', dirn, 'windows', len(med), ' cumulative (odo - map) over route: %.1f m' % np.nansum((med - 1) * W))
    for b in bad:
        print('  s=%5.0f-%5.0f  median odo/map %.4f (=> map %+.1f m vs odo)  n=%d  spread(IQR) %.4f' % (grid[b], grid[b] + W, med[b], (1 - med[b]) * W, cnt[b], np.subtract(*np.nanpercentile(R[:, b], [75, 25]))))
