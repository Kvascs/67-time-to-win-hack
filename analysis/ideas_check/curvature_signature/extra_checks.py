"""Extra checks for the verdict.
1. Stub test confound: is the stub roughness from hard acceleration? rms y on straight track by acceleration.
2. rms y by segment of front-pivot arc u after the divergence, stub vs main runs.
3. Sharp-curve landmark: how precisely does the ratio roughness locate a curve entry?
   (first crossing of rolling rms > threshold, per passage, relative to the zone start; spread over runs)
   and false alarms per km of straight track for that threshold.
Output: extra_checks.txt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
import regress as R

_lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def main():
    # ---------- 1. acceleration confound on straight track ----------
    S, _ = R.load('train', 1.0)
    S = R.add_features(S)
    st = S.kmax < 0.003
    say('=== 1. rms of y on straight track (both pivots |k|<0.003) by acceleration and speed, train ===')
    for vlo, vhi in ((1, 3), (3, 5), (5, 8)):
        row = []
        for alo, ahi in ((-9, -0.5), (-0.5, -0.15), (-0.15, 0.15), (0.15, 0.5), (0.5, 9)):
            m = st & (S.v >= vlo) & (S.v < vhi) & (S.a >= alo) & (S.a < ahi)
            row.append(f'a[{alo},{ahi}): {100 * np.sqrt(np.mean(S.y[m] ** 2)):.2f}% (n={m.sum()})' if m.sum() > 30 else f'a[{alo},{ahi}): -')
        say(f'  v {vlo}-{vhi} m/s: ' + ' | '.join(row))
    # ---------- 2. segment rms, stub vs main ----------
    say('=== 2. rms y [%] by front-pivot arc segment after the divergence (from stub_test inputs) ===')
    import stub_test as T  # noqa: F401  (re-use its data preparation by re-running the minimal part)
    A = rebuild_stub_samples()
    for a, b in ((0, 27), (27, 40), (40, 60), (60, 100)):
        out = []
        for cls in ('main', 'stub'):
            q = A[(A.cls == cls) & (A.u >= a) & (A.u < b)]
            per = q.groupby('bag').y.apply(lambda y: np.sqrt(np.mean(y ** 2)))
            vv = q.groupby('bag').v.median()
            aa = q.groupby('bag').a.median()
            if len(per):
                out.append(f'{cls}: runs {len(per)} rms median {100 * per.median():.2f}% [p10 {100 * per.quantile(0.1):.2f}, p90 {100 * per.quantile(0.9):.2f}] '
                           f'v {vv.median():.1f} m/s a {aa.median():+.2f}')
        say(f'  u {a:3d}-{b:3d} m: ' + ' || '.join(out))
    # ---------- 3. curve landmark precision ----------
    say('=== 3. sharp-curve entry located from the ratio roughness (train, v>1.5) ===')
    S, _ = R.load('train', 1.5)
    S = R.add_features(S)
    S['rr'] = R.rolling_stat(S, 'y', 1.0, 'rms')
    pl = C.load_main()
    zones = R.zone_table(pl)
    straight = S.kmax < 0.003
    for thr_q in (0.99, 0.995, 0.999):
        thr = float(np.quantile(S.rr[straight], thr_q))
        # false alarms on straight track: number of upward crossings per km of straight travel
        fa, dist = 0, 0.0
        for b, idx in S.groupby('bag').indices.items():
            g = S.iloc[idx]
            m = (g.kmax < 0.003).values
            s = g.s.values
            above = (g.rr.values > thr) & m
            fa += int(np.sum(above[1:] & ~above[:-1]))
            ds = np.diff(s)
            dist += float(np.sum(np.clip(ds[m[1:] & m[:-1]], 0, 5)))
        rows = []
        for a, e in zones:
            devs = []
            for b, idx in S.groupby('bag').indices.items():
                g = S.iloc[idx]
                sf = g.s.values + C.D_FRONT     # front pivot arc (unwrapped)
                L = pl.L
                for lap in np.unique(np.floor((sf - (a - 40)) / L)):
                    lo, hi = a - 40 + lap * L, e + 10 + lap * L
                    m = (sf >= lo) & (sf <= hi)
                    if m.sum() < 10 or (sf[m].max() - sf[m].min()) < 0.8 * (hi - lo):
                        continue
                    r = g.rr.values[m]
                    k = np.flatnonzero(r > thr)
                    if len(k):
                        devs.append(sf[m][k[0]] - (a + lap * L))
                    else:
                        devs.append(np.nan)
            devs = np.array(devs)
            if len(devs) >= 5:
                det = np.isfinite(devs)
                d = devs[det]
                rows.append((a, len(devs), det.mean(), np.median(d) if len(d) else np.nan,
                             (np.quantile(d, 0.75) - np.quantile(d, 0.25)) if len(d) > 3 else np.nan,
                             np.median(np.abs(d - np.median(d))) * 1.4826 if len(d) > 3 else np.nan))
        say(f' threshold = straight-track q{thr_q} of rolling rms(+-1 s) = {100 * thr:.2f}%: '
            f'false alarms {fa} in {dist / 1000:.1f} km straight = {fa / (dist / 1000):.2f} per km')
        for a, n, dr, med, iqr, rsd in rows:
            say(f'   zone {a:8.1f}: passages {n:3d} detected {dr:.2f}; first crossing at front pivot {med:+6.1f} m from zone start, '
                f'IQR {iqr:5.1f} m, robust sd {rsd:5.1f} m')
    (C.HERE / 'extra_checks.txt').write_text('\n'.join(_lines), encoding='utf-8')


def rebuild_stub_samples():
    """Same run/arc preparation as stub_test.main (kept in sync by copy)."""
    import stub_test as T
    pl = C.load_main()
    origin = C.map_origin()
    sp = pd.read_csv(C.HERE / 'stub_path.csv')
    stub_pl = C.Polyline(sp.x.values, sp.y.values, sp.p.values, sp.kappa.values, cyclic=False)
    S = pd.read_pickle(C.HERE / 'samples.pkl')
    with np.errstate(divide='ignore', invalid='ignore'):
        S['lr'] = np.log(S.vf / S.vr)
    st = (S.kf.abs() < 0.003) & (S.kr.abs() < 0.003) & (S.v > 3) & np.isfinite(S.lr)
    off = S[st].groupby('bag').lr.median()
    stub_bags = list(pd.read_csv(C.HERE / 'stub_hits.csv').bag)
    runs = []
    for bag, g in S.groupby('bag'):
        g = g.reset_index(drop=True)
        if bag in stub_bags:
            d = C.load_bag(bag)
            fx = C.master_fix_enu(d, origin)
            fx = fx[fx.status == 2].drop_duplicates('t').reset_index(drop=True)
            p_fix, _, dist = stub_pl.project(fx.x.values, fx.y.values)
            k = (dist < 1.5) & (p_fix > -140) & (p_fix < stub_pl.s[-1] - 1)
            tq = g.t.values - C.FIX_LEAD_S
            tf, pf = fx.t.values[k], p_fix[k]
            j = np.clip(np.searchsorted(tf, tq), 1, len(tf) - 1)
            okj = (tq >= tf[0]) & (tq <= tf[-1]) & ((tf[j] - tf[j - 1]) < 0.5)
            p = np.where(okj, np.interp(tq, tf, pf), np.nan)
            cls = 'stub'
        else:
            sm = g.sm.values
            if np.sum((sm > T.S_DIV) & (sm < T.S_DIV + 140)) < 30:
                continue
            p = np.where((sm > T.S_DIV - 150) & (sm < T.S_DIV + 250), sm - T.S_DIV, np.nan)
            cls = 'main'
        u = p + C.D_FRONT
        y = g.lr.values - off.get(bag, np.nanmedian(g.lr.values))
        m = np.isfinite(u) & np.isfinite(y) & (g.vf.values > T.VMIN) & (g.vr.values > T.VMIN) & ~g.bad_ep.values
        m &= np.abs(y) < 0.05
        if m.any():
            runs.append(pd.DataFrame({'bag': bag, 'cls': cls, 'u': u[m], 'y': y[m], 'v': g.v.values[m], 'a': g.a.values[m]}))
    return pd.concat(runs, ignore_index=True)


if __name__ == '__main__':
    main()
