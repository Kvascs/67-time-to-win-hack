"""Creep check: does wheel speed / GNSS ground speed depend on traction / braking acceleration?

Hypothesis: xi = (w r - v)/v ~ F/C  ->  log(v_wheel/v_gnss) = k0 + beta * a  (per regime, per bogie).
Confound handled: time lag tau between wheel and GNSS speed gives y ~ tau * a / v (1/v shape), creep gives
beta * a (flat in v). Both are fitted jointly; tau grid also scanned.

Data: data/npz, TRAIN bags with RTK fraction > 0.8 (+ val for confirmation). Epochs: uniform 10 Hz grid at
GNSS master vel stamps; GNSS speed > 3 m/s; straight track (|curvature| < 1/500 m from GNSS course and
from the train-only track map when matched); notch sign held >= 1 s; front/rear agreement within 2 %.
y = log(v_wheel/v_gnss) - per-bag median of y over coasting epochs (notch 0).
Output: creep.txt next to this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools' / 'replay'))
from cpp_bridge import NPZ, ROOT  # noqa: E402

OUT = Path(__file__).resolve().parent
G = 9.80665
DT = 0.1
SG_WIN = 15          # 1.5 s Savitzky-Golay window
KAPPA_MAX = 1 / 500.0
HOLD_S = 1.0
VMIN = 3.0
SLIP = 0.02
TRIM = 0.03
LINES: list[str] = []


def say(s=''):
    print(s)
    LINES.append(s)


A_, F_ = 6378137.0, 1 / 298.257223563
E2 = F_ * (2 - F_)


def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A_ / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


def enu(lat, lon, h, lat0, lon0, h0):
    d = ecef(lat, lon, h) - ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))
    la, lo = np.radians(lat0), np.radians(lon0)
    r = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return d @ r.T


def load_map():
    p = ROOT / 'analysis' / 'validation_maps' / 'track_map.csv'
    hdr = [l for l in open(p) if l.startswith('#')][1]
    kv = dict(t.split('=') for t in hdr[1:].split())
    m = pd.read_csv(p, comment='#')
    xy = m[['x', 'y']].to_numpy()
    tan = np.gradient(xy, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    return dict(o=(float(kv['origin_lat']), float(kv['origin_lon']), float(kv['origin_h'])),
                tree=cKDTree(xy), grade=m.grade.to_numpy(), kap=m.curvature.to_numpy(), tan=tan)


def rtk_frac(d):
    f = d['sensing__gnss__master__fix']
    return float((f[:, 5] == 2).mean()) if len(f) else 0.0


def prep(bag, MAP):
    """GNSS-side epoch grid (independent of the lag) + wheel/notch raw arrays."""
    d = np.load(NPZ / f'{bag}.npz')
    mv = d['sensing__gnss__master__vel']
    mv = mv[np.argsort(mv[:, 1])]
    tg = mv[:, 1]
    tu = np.arange(tg[0], tg[-1], DT)
    j = np.clip(np.searchsorted(tg, tu), 1, len(tg) - 1)
    near = np.minimum(np.abs(tg[j] - tu), np.abs(tg[j - 1] - tu)) <= 0.07
    vx, vy, vz = (np.interp(tu, tg, mv[:, c]) for c in (2, 3, 4))
    vh = np.hypot(vx, vy)
    # invalidate the SG window around gaps
    bad = np.convolve((~near).astype(float), np.ones(SG_WIN), 'same') > 0
    a = savgol_filter(vh, SG_WIN, 2, deriv=1, delta=DT)
    psi = np.unwrap(np.arctan2(vy, vx))
    yr = savgol_filter(psi, SG_WIN * 2 + 1, 2, deriv=1, delta=DT)
    kap_g = yr / np.maximum(vh, 0.5)
    # map projection of the antenna position
    f = d['sensing__gnss__master__fix']
    f = f[np.isfinite(f[:, 2])]
    f = f[np.argsort(f[:, 1])]
    lat, lon, h = (np.interp(tu, f[:, 1], f[:, c]) for c in (2, 3, 4))
    p = enu(lat, lon, h, *MAP['o'])
    dist, idx = MAP['tree'].query(p[:, :2])
    onmap = dist < 2.5
    dirn = np.sign(MAP['tan'][idx, 0] * vx + MAP['tan'][idx, 1] * vy)
    grade = np.where(onmap, MAP['grade'][idx] * dirn, vz / np.maximum(vh, 0.5))
    grade = savgol_filter(grade, SG_WIN, 1) if len(grade) > SG_WIN else grade
    kap_m = np.where(onmap, MAP['kap'][idx], 0.0)
    vg = vh * np.sqrt(1 + grade ** 2)          # along-track (3-D) ground speed
    # notch: value and time since last sign change, at the grid times
    c = d['vehicle__driver_position_cmd']
    c = c[np.argsort(c[:, 1])]
    s = np.sign(c[:, 2])
    chg = np.r_[True, s[1:] != s[:-1]]
    tchg = np.maximum.accumulate(np.where(chg, c[:, 1], -np.inf))
    k = np.searchsorted(c[:, 1], tu, 'right') - 1
    kk = np.clip(k, 0, len(c) - 1)
    notch = np.where(k >= 0, c[kk, 2], np.nan)
    age = np.where(k >= 0, tu - tchg[kk], 0.0)
    wf, wr = d['vehicle__front_bogie_velocity'], d['vehicle__rear_bogie_velocity']
    wf, wr = wf[np.argsort(wf[:, 1])], wr[np.argsort(wr[:, 1])]
    base = pd.DataFrame(dict(t=tu, vg=vg, a=a, af=a + G * grade, grade=grade, kap_g=kap_g, kap_m=kap_m,
                             onmap=onmap, notch=notch, age=age, ok=near & ~bad))
    return base, (wf[:, 1], wf[:, 2] / 3.6), (wr[:, 1], wr[:, 2] / 3.6)


def wheel_at(t, w):
    tw, vw = w
    j = np.clip(np.searchsorted(tw, t), 1, len(tw) - 1)
    ok = (tw[j] - tw[j - 1] <= 0.25) & (t >= tw[0]) & (t <= tw[-1])
    return np.interp(t, tw, vw), ok


def epochs(bag, P, tau):
    base, wf, wr = P[bag]
    e = base.copy()
    vf, okf = wheel_at(e.t.to_numpy() + tau, wf)
    vr, okr = wheel_at(e.t.to_numpy() + tau, wr)
    e['vf'], e['vr'] = vf, vr
    m = (e.ok & okf & okr & (e.vg > VMIN) & (vf > 1) & (vr > 1) & np.isfinite(e.notch)
         & (e.kap_g.abs() < KAPPA_MAX) & (e.kap_m.abs() < KAPPA_MAX) & (e.age >= HOLD_S))
    e = e[m].copy()
    if len(e) < 200:
        return None
    e['fr'] = np.log(e.vf / e.vr)
    e = e[(e.fr - e.fr.median()).abs() < SLIP]
    e['reg'] = np.select([e.notch > 0, e.notch < 0], ['trac', 'brake'], 'coast')
    for b, col in (('F', 'vf'), ('R', 'vr')):
        y = np.log(e[col] / e.vg)
        co = y[e.reg == 'coast']
        if len(co) < 50:
            return None
        e['y' + b] = y - co.median()
    e['dfr'] = e.fr - e.fr[e.reg == 'coast'].median()
    e['inv_v'] = 1.0 / e.vg
    e['bag'] = bag
    return e


def ols_cluster(X, y, g):
    """OLS with cluster-robust (CR1) SEs by group g."""
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    r = y - X @ b
    meat = np.zeros((X.shape[1],) * 2)
    ug = np.unique(g)
    for u in ug:
        s = X[g == u].T @ r[g == u]
        meat += np.outer(s, s)
    G_, n, k = len(ug), len(y), X.shape[1]
    V = XtX @ meat @ XtX * G_ / (G_ - 1) * (n - 1) / (n - k)
    return b, np.sqrt(np.diag(V)), r


def fit(E, ycol, xcol, reg, lag_term=False, trim=True):
    e = E[E.reg == reg]
    if trim:
        e = e[e[ycol].abs() < TRIM]
    cols = [np.ones(len(e)), e[xcol].to_numpy()]
    if lag_term:
        cols.append((e.a * e.inv_v).to_numpy())       # y = tau * a / v  <- lag signature
    X = np.column_stack(cols)
    b, se, r = ols_cluster(X, e[ycol].to_numpy(), e.bag.to_numpy())
    return b, se, len(e), e.bag.nunique(), r


def boot(E, ycol, xcol, reg, lag_term, B=400, seed=1):
    """Bag (cluster) bootstrap via per-bag sufficient statistics X'X, X'y."""
    rng = np.random.default_rng(seed)
    XX, Xy = [], []
    for _, e in E[E.reg == reg].groupby('bag'):
        e = e[e[ycol].abs() < TRIM]
        cols = [np.ones(len(e)), e[xcol].to_numpy()]
        if lag_term:
            cols.append((e.a * e.inv_v).to_numpy())
        X = np.column_stack(cols)
        XX.append(X.T @ X)
        Xy.append(X.T @ e[ycol].to_numpy())
    XX, Xy = np.array(XX), np.array(Xy)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(XX), len(XX))
        out.append(np.linalg.solve(XX[pick].sum(0), Xy[pick].sum(0))[1])
    return np.percentile(out, [2.5, 97.5])


def per_bag_slopes(E, ycol, xcol, reg):
    r = []
    for b, e in E[E.reg == reg].groupby('bag'):
        e = e[e[ycol].abs() < TRIM]
        if len(e) < 100 or e[xcol].std() < 0.1:
            continue
        X = np.column_stack([np.ones(len(e)), e[xcol], e.a * e.inv_v])
        r.append(np.linalg.lstsq(X, e[ycol].to_numpy(), rcond=None)[0][1])
    return np.array(r)


def main():
    t0 = pd.Timestamp.now()
    splits = json.load(open(ROOT / 'data' / 'splits.json'))
    try:
        ck = pd.read_csv(ROOT / 'analysis' / 'timing_reference' / 'clocks_per_bag.csv').set_index('bag')
    except Exception:
        ck = None
    MAP = load_map()
    P, sel = {}, {'train': [], 'val': []}
    for split in ('train', 'val'):
        for bag in splits[split]:
            d = np.load(NPZ / f'{bag}.npz')
            rf = rtk_frac(d)
            anom = float(ck.loc[bag, 'gnss_vs_veh_anom_frac']) if ck is not None and bag in ck.index else 0.0
            if rf > 0.8 and not anom > 0.01:
                P[bag] = prep(bag, MAP)
                sel[split].append(bag)
    say(f'# creep check  ({pd.Timestamp.now():%Y-%m-%d %H:%M})')
    say(f'bags: train {len(sel["train"])}/{len(splits["train"])}, val {len(sel["val"])}/{len(splits["val"])} '
        f'(RTK frac > 0.8, no GNSS-vs-vehicle clock anomaly)')
    say(f'filters: v_gnss > {VMIN} m/s, |kappa| < 1/500 m (GNSS course and map), notch sign held >= {HOLD_S} s, '
        f'|log(front/rear) - median| < {SLIP:.0%}, |y| < {TRIM:.0%}; a = SG(1.5 s) d|v_gnss|/dt; '
        f'a_force = a + g*grade (map grade, else vz/vh)')
    say('y = log(v_wheel/v_gnss) - per-bag median over coasting epochs; beta in % of scale per m/s^2\n')

    # ---------------- lag scan (train) ----------------
    say('## 1. Lag scan (train): wheel sampled at t_gnss + tau; model y = a_r + b_r*a per regime (both bogies)')
    say('tau_s   n      rms_resid_%  b_trac_F  b_brake_F  b_trac_R  b_brake_R   (%/(m/s^2), no lag term)')
    best = None
    for tau in np.arange(-0.20, 0.41, 0.05):
        Es = [epochs(b, P, tau) for b in sel['train']]
        E = pd.concat([e for e in Es if e is not None])
        res, bs = [], []
        for bog in 'FR':
            for reg in ('trac', 'brake', 'coast'):
                b, se, n, nb, r = fit(E, 'y' + bog, 'a', reg)
                res.append(r)
                if reg != 'coast':
                    bs.append(b[1] * 100)
        rms = np.sqrt(np.mean(np.concatenate(res) ** 2)) * 100
        say(f'{tau:+.2f}  {len(E):6d}  {rms:.4f}      {bs[0]:+.3f}    {bs[1]:+.3f}     {bs[2]:+.3f}    {bs[3]:+.3f}')
        if best is None or rms < best[1]:
            best = (tau, rms)
    tau_b = round(best[0], 2)
    say(f'-> best tau = {tau_b:+.2f} s (min residual rms)\n')

    for split in ('train', 'val'):
        Es = [epochs(b, P, tau_b) for b in sel[split]]
        E = pd.concat([e for e in Es if e is not None])
        E.to_csv(OUT / f'epochs_{split}.csv.gz', index=False, float_format='%.6g') if split == 'train' else None
        say(f'## 2. {split.upper()} at tau = {tau_b:+.2f} s: {len(E)} epochs, {E.bag.nunique()} bags; '
            f'counts {E.reg.value_counts().to_dict()}')
        for reg in ('trac', 'brake'):
            q = E[E.reg == reg]
            say(f'   {reg}: a mean {q.a.mean():+.2f} (p5 {q.a.quantile(.05):+.2f}, p95 {q.a.quantile(.95):+.2f}); '
                f'a_force mean {q.af.mean():+.2f}; v mean {q.vg.mean():.1f} m/s')
        say('   bogie regime x        beta %/(m/s2)  SE_cl   95%CI(bag bootstrap)  intercept %   tau_fit s(+/-SE)   n / bags')
        for bog in 'FR':
            for reg in ('trac', 'brake'):
                for x in ('a', 'af'):
                    for lag in (False, True):
                        b, se, n, nb, _ = fit(E, 'y' + bog, x, reg, lag_term=lag)
                        ci = boot(E, 'y' + bog, x, reg, lag, B=1000) * 100
                        lt = f'{b[2]:+.3f}({se[2]:.3f})' if lag else '     -      '
                        say(f'   {bog}     {reg:5s}  {x:3s}{"+lag" if lag else "    "}  {b[1]*100:+.3f}     {se[1]*100:.3f}  '
                            f'[{ci[0]:+.3f}, {ci[1]:+.3f}]      {b[0]*100:+.3f}       {lt}      {n}/{nb}')
        # front/rear differential (GNSS-free, lag-free to first order)
        say('   front-rear differential dfr = log(vf/vr) - coast median  vs a:')
        for reg in ('trac', 'brake'):
            b, se, n, nb, _ = fit(E, 'dfr', 'a', reg)
            say(f'     {reg:5s} beta {b[1]*100:+.3f} +/- {se[1]*100:.3f} %/(m/s2), intercept {b[0]*100:+.3f} %')
        # per-bag slopes, noise scale
        say('   per-bag slopes (a, with lag term), %/(m/s2): median [IQR], share > 0')
        for bog in 'FR':
            for reg in ('trac', 'brake'):
                s = per_bag_slopes(E, 'y' + bog, 'a', reg) * 100
                if len(s):
                    say(f'     {bog} {reg:5s}: {np.median(s):+.3f} [{np.percentile(s,25):+.3f}, {np.percentile(s,75):+.3f}]'
                        f'  {np.mean(s>0):.0%} of {len(s)}')
        # noise reference: spread of coast residuals, 10-s block averages, and bag offsets
        co = E[E.reg == 'coast']
        blk = co.assign(blk=(co.t // 10).astype(int)).groupby(['bag', 'blk']).yF.mean()
        say(f'   noise: coast yF sd (10 Hz) {co.yF.std()*100:.3f} %, 10-s block-mean sd {blk.std()*100:.3f} %')
        # speed-binned means (shape check: creep flat in v, lag ~ 1/v)
        say('   binned mean yF by regime and speed (%), traction/braking a-normalised: mean(yF)/mean(a)')
        for reg in ('trac', 'brake'):
            q = E[(E.reg == reg) & (E.yF.abs() < TRIM)]
            row = []
            for lo, hi in ((3, 6), (6, 9), (9, 12), (12, 20)):
                z = q[(q.vg >= lo) & (q.vg < hi)]
                if len(z) > 200:
                    row.append(f'{lo}-{hi} m/s: {z.yF.mean()*100:+.3f}/{z.a.mean():+.2f} = {z.yF.mean()/z.a.mean()*100:+.3f}')
            say(f'     {reg}: ' + ';  '.join(row))
        say()
    say(f'runtime {(pd.Timestamp.now()-t0).total_seconds():.0f} s')
    (OUT / 'creep.txt').write_text('\n'.join(LINES) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
