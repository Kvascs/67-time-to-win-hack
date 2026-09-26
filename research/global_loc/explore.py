"""Data exploration on TRAIN bags (GNSS truth vs the no-GNSS replay) to size the cue models.

    python explore.py odo      # wheel scale kappa per run + odometry residual random walk
    python explore.py stops    # stop statistics vs landmarks.csv (p_stop, random-stop rate, run starts)
    python explore.py cut      # traction cut-off statistics vs cutoffs.csv
    python explore.py grade    # estimator disturbance d vs map grade
    python explore.py speed    # speed profile along the map
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common as C
import data as D
from mapmodel import MapModel

pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 200)


def bags(split='train'):
    for name in C.SPLITS[split]:
        if not (C.CACHE / f'truth_{name}.npz').exists():  # never trigger a replay from here
            continue
        b = D.load_bag(name)
        if b.has_truth:
            yield b


def cyc(a, L):
    return (a + 0.5 * L) % L - 0.5 * L


def odo(split='train'):
    mm = MapModel()
    rows, incs = [], {D_: [] for D_ in (50, 100, 200, 500, 1000, 2000, 4000)}
    for b in bags(split):
        ok = b.truth_ok & np.isfinite(b.truth_s)
        i0 = np.flatnonzero(ok)[0]
        u = mm.u_of_s(b.truth_s[ok])
        x = u - u[0]
        y = b.s_rel[ok] - b.s_rel[i0]
        if x[-1] < 500:
            continue
        kap = float(np.sum(x * y) / np.sum(x * x)) - 1.0
        # kappa without the curve model (plain s) for comparison
        xs = b.truth_s[ok] - b.truth_s[i0]
        kap_s = float(np.sum(xs * y) / np.sum(xs * xs)) - 1.0
        e = y / (1 + kap) - x
        # first-half vs second-half kappa (within-run drift)
        h = x < x[-1] / 2
        k1 = float(np.sum(x[h] * y[h]) / np.sum(x[h] ** 2)) - 1
        dx, dy = x[~h] - x[~h][0], y[~h] - y[~h][0]
        k2 = float(np.sum(dx * dy) / np.sum(dx ** 2)) - 1
        rows.append(dict(bag=b.name, dist=x[-1], kappa=kap, kappa_s=kap_s, k1=k1, k2=k2,
                         e_rms=float(np.sqrt(np.mean(e ** 2))), e_max=float(np.abs(e).max())))
        for L_ in incs:
            # increments of the residual over travelled distance L_
            j = np.searchsorted(x, x + L_)
            v = j < len(x)
            if v.sum() < 10:
                continue
            de = e[j[v]] - e[v]
            incs[L_].append(de[::max(1, len(de) // 200)])
    df = pd.DataFrame(rows)
    print(df.round(5).to_string(index=False))
    print('kappa  pct:', np.percentile(df.kappa, [0, 5, 50, 95, 100]).round(5), 'std', df.kappa.std().round(5))
    print('kappa_s-kappa median', (df.kappa_s - df.kappa).median().round(5))
    print('within-run k2-k1 pct:', np.percentile(df.k2 - df.k1, [5, 50, 95]).round(5))
    for L_, lst in incs.items():
        if lst:
            a = np.abs(np.concatenate(lst))
            print(f'residual increment over {L_:5d} m: |.| p50 {np.percentile(a, 50):.2f} p90 {np.percentile(a, 90):.2f}'
                  f' p95 {np.percentile(a, 95):.2f} p99 {np.percentile(a, 99):.2f}  (p68^2/D = {np.percentile(a, 68) ** 2 / L_:.4f} m)')


def stops(split='train', dwell=1.5):
    lm = C.load_places(C.MAPS / 'landmarks.csv')
    L = C.load_map().length
    rows = []
    passes = np.zeros(len(lm))
    total_dist = 0.0
    for b in bags(split):
        ok = b.truth_ok & np.isfinite(b.truth_s)
        s0, s1 = np.nanmin(b.truth_s[ok]), np.nanmax(b.truth_s[ok])
        total_dist += s1 - s0
        for i, r in lm.iterrows():
            n0 = np.ceil((s0 + 5 - r.s) / L)
            n1 = np.floor((s1 - 5 - r.s) / L)
            passes[i] += max(0, n1 - n0 + 1)
        for st in D.stop_events(b, dwell):
            ss = b.truth_s[st['i0']:st['i1'] + 1]
            ss = ss[np.isfinite(ss)]
            if len(ss) == 0:
                continue
            sw = float(np.median(ss)) % L
            dd = cyc(lm.s.to_numpy() - sw, L)
            j = int(np.argmin(np.abs(dd)))
            rows.append(dict(bag=b.name, s=sw, dwell=st['t1'] - st['t0'], initial=st['initial'], final=st['final'],
                             lm=j, dlm=dd[j], sig=lm.sigma[j]))
    df = pd.DataFrame(rows)
    df['assoc'] = np.abs(df.dlm) <= np.maximum(3 * np.hypot(df.sig, 0.3), 1.5)
    mid = df[~df.initial & ~df.final]
    print(f'dwell>={dwell}s: {len(df)} stops, {len(mid)} mid-run, total distance {total_dist / 1000:.1f} km')
    print('mid-run assoc share', mid.assoc.mean().round(3), ' random stops per km',
          round((~mid.assoc).sum() / (total_dist / 1000), 3))
    print('|dlm| pct of assoc:', np.percentile(np.abs(mid.dlm[mid.assoc]), [50, 90, 99]).round(2))
    ini = df[df.initial]
    print(f'initial standstills: {len(ini)}, at a landmark {ini.assoc.mean():.2f}; dlm pct',
          np.percentile(np.abs(ini.dlm), [25, 50, 75]).round(1))
    print(ini[['bag', 's', 'dwell', 'lm', 'dlm']].round(2).to_string(index=False))
    cnt = mid[mid.assoc].groupby('lm').size()
    lm2 = lm.copy()
    lm2['passes'] = passes
    lm2['stops'] = cnt.reindex(range(len(lm))).fillna(0).to_numpy()
    lm2['p_emp'] = lm2.stops / lm2.passes.clip(lower=1)
    print(lm2.round(3).to_string())
    return df, lm2


def cut(split='train'):
    cu = C.load_places(C.MAPS / 'cutoffs.csv')
    L = C.load_map().length
    rows, prow = [], []
    for b in bags(split):
        ok = b.truth_ok & np.isfinite(b.truth_s)
        s0, s1 = np.nanmin(b.truth_s[ok]), np.nanmax(b.truth_s[ok])
        for e in D.cutoff_events(b):
            st = np.interp(e['t'], b.t, b.truth_s) + e['v'] * D.GNSS_LEAD_S
            if not np.isfinite(st):
                continue
            sw = st % L
            dd = cyc(cu.s.to_numpy() - sw, L)
            j = int(np.argmin(np.abs(dd)))
            rows.append(dict(bag=b.name, s=sw, v=e['v'], nb=e['notch_before'], c=j, dc=dd[j]))
        # passes: notch just before reaching each cut-off place
        for i, r in cu.iterrows():
            for n in range(int(np.ceil((s0 + 30 - r.s) / L)), int(np.floor((s1 - 5 - r.s) / L)) + 1):
                sc = r.s + n * L
                k = np.flatnonzero(ok & (b.truth_s >= sc - 15) & (b.truth_s <= sc - 3))
                if len(k) == 0:
                    continue
                tk = b.t[k]
                nt = b.notch[np.clip(np.searchsorted(b.notch_t, tk) - 1, 0, len(b.notch) - 1)]
                vk = b.v[k]
                prow.append(dict(bag=b.name, c=i, notch_max=int(nt.max()), notch_last=int(nt[-1]),
                                 v=float(vk.mean())))
    df = pd.DataFrame(rows)
    df['assoc'] = np.abs(df.dc) < 3 * cu.sigma.to_numpy()[df.c] + 2.0
    print(f'{len(df)} cut-off events, at a known place: {df.assoc.mean():.3f}')
    print(df.groupby('c').dc.describe().round(2))
    print(df[~df.assoc].round(1).to_string(index=False))
    pr = pd.DataFrame(prow)
    ev = df[df.assoc].groupby(['bag', 'c']).size()
    pr['event'] = [ev.get((r.bag, r.c), 0) > 0 for r in pr.itertuples()]
    pr['high'] = pr.notch_last >= 4
    print(pr.groupby(['c', 'high']).event.agg(['mean', 'size']).round(3))
    return df, pr


def grade(split='train', tau=0.0, kg=7.8):
    mm = MapModel()
    xs, ys, bias, lagc = [], [], [], {}
    for b in bags(split):
        ok = b.truth_ok & np.isfinite(b.truth_s) & (b.v > 3.0) & ((b.flags & C.FLAG_STANDSTILL) == 0) & (b.mu3 < 0.5)
        k = np.flatnonzero(ok)
        k = k[np.unique(np.floor(b.t[k]), return_index=True)[1]]  # ~1 Hz
        if len(k) < 50:
            continue
        g = np.interp(mm.wrap_s(b.truth_s[k]), mm.sg, mm.grade_body)
        x = -kg * g
        y = b.d[k]
        xs.append(x)
        ys.append(y)
        bias.append(np.mean(y - x))
        for lag in (-40, -20, -10, 0, 5, 10, 20, 30, 40, 60):
            gl = np.interp(mm.wrap_s(b.truth_s[k] - lag), mm.sg, mm.grade_body)
            lagc.setdefault(lag, []).append((-kg * gl, y))
    x, y = np.concatenate(xs), np.concatenate(ys)
    r = np.corrcoef(x, y)[0, 1]
    slope = np.polyfit(x, y, 1)
    res = y - x
    print(f'n={len(x)} corr={r:.3f} fit y = {slope[0]:.3f} x + {slope[1]:.3f}; resid std {res.std():.3f}, '
          f'x std {x.std():.3f}; per-bag bias pct {np.percentile(bias, [5, 50, 95]).round(3)}')
    for lag, lst in lagc.items():
        xx = np.concatenate([a for a, _ in lst])
        yy = np.concatenate([b_ for _, b_ in lst])
        print(f'  distance lag {lag:4d} m: corr {np.corrcoef(xx, yy)[0, 1]:.3f}  resid std {(yy - xx).std():.3f}')


def speed(split='train', bin_m=5.0):
    L = C.load_map().length
    nb = int(np.ceil(L / bin_m))
    vmax = np.zeros(nb)
    cnt = np.zeros(nb)
    allv = [[] for _ in range(nb)]
    for b in bags(split):
        ok = b.truth_ok & np.isfinite(b.truth_s)
        idx = (np.mod(b.truth_s[ok], L) / bin_m).astype(int) % nb
        np.maximum.at(vmax, idx, b.v[ok])
        np.add.at(cnt, idx, 1)
    print('bins covered', (cnt > 0).mean().round(3), 'vmax pct', np.percentile(vmax[cnt > 0], [5, 25, 50, 75, 95]).round(2))
    s = np.arange(nb) * bin_m
    for a in range(0, nb, 100):
        print(f's {s[a]:7.0f}: vmax ' + ' '.join(f'{v:4.1f}' for v in vmax[a:a + 100:10]))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'odo'
    split = sys.argv[2] if len(sys.argv) > 2 else 'train'
    {'odo': odo, 'stops': stops, 'cut': cut, 'grade': grade, 'speed': speed}[what](split)
