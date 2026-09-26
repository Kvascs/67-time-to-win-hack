"""Where do the stub runs actually leave the main line, and what is the curvature of both paths
for the first 120 m after the divergence?  Built from the RTK traces of the stub runs (the stub
polyline has a kink artefact at its first 3 m: yaw jumps 22 deg in 1.5 m).

Output: stub_path.csv (combined path: main before the toe + mean stub-run trace after),
        paths_kappa.csv (p, kappa_main, kappa_stub_trace, kappa_stub_file), stub_geometry.txt
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import common as C

_lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def resample_path(x, y, step=0.25):
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.r_[0, np.cumsum(seg)]
    keep = np.r_[True, seg > 1e-3]
    s, x, y = s[keep], x[keep], y[keep]
    g = np.arange(0, s[-1], step)
    return g, np.interp(g, s, x), np.interp(g, s, y)


def curvature_of(x, y, step, sigma_m=1.5):
    xs = gaussian_filter1d(x, sigma_m / step, mode='nearest')
    ys = gaussian_filter1d(y, sigma_m / step, mode='nearest')
    dx, dy = np.gradient(xs, step), np.gradient(ys, step)
    ddx, ddy = np.gradient(dx, step), np.gradient(dy, step)
    k = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    return xs, ys, k


def main():
    origin = C.map_origin()
    pl = C.load_main()
    stub, st = C.load_stub()
    hits = pd.read_csv(C.HERE / 'stub_hits.csv')
    S = pd.read_pickle(C.HERE / 'samples.pkl')
    # --- lateral offset of stub runs and main runs vs main s around the switch ---
    grid = np.arange(5360, 5402, 2.0)
    lat_stub, lat_main = [], []
    traces = {}
    for bag in hits.bag:
        d = C.load_bag(bag)
        fx = C.master_fix_enu(d, origin)
        fx = fx[fx.status == 2].drop_duplicates('t').reset_index(drop=True)
        s_m, lat, dist = pl.project(fx.x.values, fx.y.values)
        s_s, _, d_s = stub.project(fx.x.values, fx.y.values)
        # the pass: from main s 5250 (approach) to the last fix on the stub
        on_st = np.flatnonzero(d_s < 2.0)
        t_end = fx.t.values[on_st].max()
        app = np.flatnonzero((np.abs(s_m - 5300) < 50) & (dist < 1.5) & (fx.t.values < t_end))
        t_beg = fx.t.values[app].min()
        m = (fx.t.values >= t_beg) & (fx.t.values <= t_end)
        traces[bag] = fx[m].reset_index(drop=True)
        near = m & (s_m > 5355) & (s_m < 5402) & (dist < 3)
        lat_stub.append(np.interp(grid, s_m[near], dist[near], left=np.nan, right=np.nan))
    main_bags = []
    for bag, g in S.groupby('bag'):
        if bag in set(hits.bag):
            continue
        sm = g.sm.values
        if np.sum((sm > 5390) & (sm < 5530)) > 50:
            main_bags.append(bag)
    for bag in main_bags:
        d = C.load_bag(bag)
        fx = C.master_fix_enu(d, origin)
        fx = fx[fx.status == 2].drop_duplicates('t').reset_index(drop=True)
        s_m, lat, dist = pl.project(fx.x.values, fx.y.values)
        near = (s_m > 5355) & (s_m < 5402) & (dist < 1.5)
        if near.sum() < 10:
            continue
        o = np.argsort(s_m[near])
        lat_main.append(np.interp(grid, s_m[near][o], dist[near][o], left=np.nan, right=np.nan))
    lat_stub = np.array(lat_stub)
    lat_main = np.array(lat_main)
    say(f'stub runs: {len(lat_stub)}, main runs through the switch area: {len(lat_main)}')
    say('main s | stub runs dist to main map [m] (each run) | main runs dist: median / p95')
    for i, s in enumerate(grid):
        say(f'  {s:7.1f} | ' + ' '.join(f'{v:5.2f}' for v in lat_stub[:, i]) +
            f' | {np.nanmedian(lat_main[:, i]):5.3f} / {np.nanpercentile(lat_main[:, i], 95):5.3f}')
    # divergence: first grid point where every stub run is > p95 of main runs + 0.05 m and stays so
    thr = np.nanpercentile(lat_main, 95, axis=0) + 0.05
    div = np.all(lat_stub > thr, axis=0)
    first = next(i for i in range(len(grid)) if div[i:].all())
    say(f'stub runs beyond main-run p95 + 5 cm from main s ~ {grid[first]:.1f} on (stub polyline starts at 5395.2)')
    # the offset of the stub runs is 0.1-0.2 m at 5382-5388 but falls back to 0.08 m at 5390 and only then
    # grows monotonically (0.36 m at 5394, 1.5 m at 5400): the divergence point is taken at 5390
    mono = np.all(np.diff(lat_stub, axis=1) > 0, axis=0)
    last_nonmono = max(i for i in range(len(mono)) if not mono[i])
    s_toe = float(grid[last_nonmono + 1])
    say(f'divergence point (stub-run offset grows monotonically from here): main s = {s_toe:.1f}')
    # --- mean stub-run trace after the toe, as a path ---
    step = 0.25
    paths = []
    for bag, fx in traces.items():
        x, y = fx.x.values, fx.y.values
        g, xr, yr = resample_path(x, y, step)
        s_m, _, dist = pl.project(xr, yr)
        # arc position of the toe along this trace: last point with main s < s_toe and dist < 0.15
        pre = np.flatnonzero((s_m < s_toe) & (s_m > s_toe - 60) & (dist < 0.3))
        i0 = pre.max()
        # arc coordinate p relative to the toe
        p = g - g[i0] - (s_toe - s_m[i0])
        paths.append((bag, p, xr, yr))
    pg = np.arange(-40, 125.01, step)
    X = np.array([np.interp(pg, p, xr, left=np.nan, right=np.nan) for _, p, xr, _ in paths])
    Y = np.array([np.interp(pg, p, yr, left=np.nan, right=np.nan) for _, p, _, yr in paths])
    spread = np.nanmax(np.hypot(X - np.nanmean(X, 0), Y - np.nanmean(Y, 0)), 0)
    say(f'stub-run traces: max deviation from their mean (p 0..120): median {np.nanmedian(spread[(pg >= 0) & (pg <= 120)]):.2f} m, '
        f'max {np.nanmax(spread[(pg >= 0) & (pg <= 120)]):.2f} m')
    mx, my = np.nanmean(X, 0), np.nanmean(Y, 0)
    ok = np.isfinite(mx)
    pg2, mx, my = pg[ok], mx[ok], my[ok]
    # combined stub path: main map up to the toe (p<0), mean trace after (p>=0)
    pm = np.arange(-150, 0, step)
    xm, ym = pl.xy_at(s_toe + pm)
    keep = pg2 >= 0
    cx = np.r_[xm, mx[keep]]
    cy = np.r_[ym, my[keep]]
    g, xr, yr = resample_path(cx, cy, step)
    xs, ys, ks = curvature_of(xr, yr, step, 1.5)
    p_stub = g - 150.0
    # main path curvature on the same footing (recomputed from map xy, same smoothing) and file kappa
    pmain = np.arange(-150, 200, step)
    xm2, ym2 = pl.xy_at(s_toe + pmain)
    _, _, km_geo = curvature_of(xm2, ym2, step, 1.5)
    km_file = pl.k_at(s_toe + pmain)
    pd.DataFrame({'p': p_stub, 'x': xs, 'y': ys, 'kappa': ks}).to_csv(C.HERE / 'stub_path.csv', index=False)
    pd.DataFrame({'p': pmain, 'x': xm2, 'y': ym2, 'kappa_geo': km_geo, 'kappa_file': km_file}).to_csv(
        C.HERE / 'main_path.csv', index=False)
    # stub file kappa on the p axis: stub polyline s=0 is at main 5395.2 -> p = s + (5395.2 - s_toe)
    off_file = 5395.2 - s_toe
    say(f'stub polyline s=0 corresponds to p = {off_file:.1f} m after the toe')
    rows = []
    for p in np.arange(0, 120.01, 1.0):
        kst = np.interp(p, p_stub, ks)
        ksf = stub.k_at(p - off_file) if p - off_file >= 0 else pl.k_at(s_toe + p)
        rows.append(dict(p=p, k_main_file=float(np.interp(p, pmain, km_file)), k_main_geo=float(np.interp(p, pmain, km_geo)),
                         k_stub_trace=float(kst), k_stub_file=float(ksf),
                         lat_sep=float(np.min(np.hypot(np.interp(p, p_stub, xs) - xm2, np.interp(p, p_stub, ys) - ym2)))))
    K = pd.DataFrame(rows)
    K.to_csv(C.HERE / 'paths_kappa.csv', index=False)
    say('p after toe | kappa main (file, geo) | kappa stub (RTK trace, file) | separation of the paths [m]')
    for _, r in K.iloc[::5].iterrows():
        say(f'  {r.p:5.0f} | {r.k_main_file:+.4f} {r.k_main_geo:+.4f} | {r.k_stub_trace:+.4f} {r.k_stub_file:+.4f} | {r.lat_sep:6.2f}')
    for a, b in ((0, 20), (0, 50), (0, 100), (0, 120)):
        m = (K.p >= a) & (K.p <= b)
        say(f'  p {a}-{b}: max|k| main {K.k_main_file[m].abs().max():.4f}, stub(trace) {K.k_stub_trace[m].abs().max():.4f}; '
            f'rms k difference {np.sqrt(np.mean((K.k_main_file[m] - K.k_stub_trace[m]) ** 2)):.4f}; '
            f'length with |k|>0.02: main {int((K.k_main_file[m].abs() > 0.02).sum())} m, stub {int((K.k_stub_trace[m].abs() > 0.02).sum())} m')
    (C.HERE / 'stub_geometry.txt').write_text('\n'.join(_lines), encoding='utf-8')
    np.save(C.HERE / 'stub_toe.npy', np.array([s_toe]))


if __name__ == '__main__':
    main()
