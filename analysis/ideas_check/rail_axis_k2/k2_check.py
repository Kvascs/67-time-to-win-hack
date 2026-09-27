"""Rail-axis vs antenna path in curves: does v_wheel / v_antenna fall like kappa^2 (geometry) or like |kappa|?

Geometry (teammate's derivation): a body point at distance u from the midpoint of the bogie-pivot chord (L = 7.55 m)
moves on a circle of radius R + delta, delta = (u^2 - L^2/4) / (2R), so
    v_point / v_axis = 1 + ((u^2 - L^2/4) / 2) * kappa^2
master u = 6.098 m -> 11.47, rover u = 6.338 m -> 12.96.  Expected: y = log(v_wheel / v_ant) ~ -11.47 kappa^2.
Current code (estimator.cpp): v = v_wheel * c(k), c = 1 + min(0.415|k|, 0.009) - 0.054 k  ->  y_model = -log c(k).

Data: TRAIN bags with master RTK (status 2) fraction > 0.8. Epochs = GNSS vel header stamps; wheel speeds
(front/rear, km/h) linearly interpolated to the vel stamp (+ a common wheel-vs-GNSS lag estimated here), both
neighbours within 0.12 s. Filters: v > 3 m/s, |front/rear - 1| < 1 %, |accel| < 0.3 m/s^2, fix status 2 within
0.1 s, antenna within 1 m of the main-cycle map (validation map, built from the master antenna, curvature
smoothed with sigma = 2 m). GNSS horizontal speed corrected to 3D with the map grade.
y is normalised per bag by its median on straight track (|k| < 0.001) -> removes the wheel scale.

Run: python k2_check.py   ->  k2_check.txt (all numbers) next to this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
from cpp_bridge import NPZ  # noqa: E402

MAP_CSV = ROOT / 'analysis' / 'validation_maps' / 'track_map.csv'
SPLITS = ROOT / 'data' / 'splits.json'
OUT = HERE / 'k2_check.txt'

L_BOGIE = 7.55
U_MASTER, U_ROVER = 6.098, 6.338
K2_MASTER = (U_MASTER ** 2 - L_BOGIE ** 2 / 4) / 2
K2_ROVER = (U_ROVER ** 2 - L_BOGIE ** 2 / 4) / 2
ROVER_AHEAD = 12.44          # map meta: rover path = master path shifted by +12.44 m in s
PIV = (U_MASTER - L_BOGIE / 2, U_MASTER + L_BOGIE / 2)   # bogie pivots relative to master along s: 2.32 .. 9.87 m

A_WGS, F_WGS = 6378137.0, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)

LINES: list[str] = []


def out(s=''):
    print(s)
    LINES.append(s)


# ----------------------------------------------------------------------------------------------------------- geo
def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A_WGS / np.sqrt(1 - E2 * np.sin(la) ** 2)
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
    hdr = [ln for ln in open(MAP_CSV) if ln.startswith('#')]
    meta = dict(kv.split('=') for ln in hdr for kv in ln[1:].split() if '=' in kv)
    m = pd.read_csv(MAP_CSV, comment='#')
    org = (float(meta['origin_lat']), float(meta['origin_lon']), float(meta['origin_h']))
    return m, org


class MapLookup:
    def __init__(self, m):
        self.s = m.s.to_numpy()
        self.k = m.curvature.to_numpy()
        self.g = m.grade.to_numpy()
        self.Ltot = self.s[-1] + (self.s[1] - self.s[0])
        self.tree = cKDTree(m[['x', 'y']].to_numpy())
        # cumulative curvature for span means (cyclic)
        ds = np.diff(self.s).mean()
        self.ds = ds
        self.kk = np.r_[self.k, self.k, self.k]
        self.cum = np.r_[0.0, np.cumsum(self.kk)]

    def nearest(self, xy):
        d, i = self.tree.query(xy)
        return d, i

    def span_mean(self, i, a, b):
        """mean curvature over [s_i + a, s_i + b] (metres, cyclic)."""
        n = len(self.k)
        ia = i + n + int(round(a / self.ds))
        ib = i + n + int(round(b / self.ds))
        ib = np.maximum(ib, ia + 1)
        return (self.cum[ib + 1] - self.cum[ia]) / (ib + 1 - ia)


# ----------------------------------------------------------------------------------------------------------- data
def interp_series(t_src, v_src, t, max_gap=0.12):
    o = np.argsort(t_src)
    ts, vs = t_src[o], v_src[o]
    keep = np.r_[True, np.diff(ts) > 1e-6]
    ts, vs = ts[keep], vs[keep]
    i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    t0, t1 = ts[i - 1], ts[i]
    w = np.clip((t - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)
    v = vs[i - 1] * (1 - w) + vs[i] * w
    ok = (t - t0 <= max_gap) & (t1 - t <= max_gap) & (t >= t0) & (t <= t1)
    return v, ok


def nearest_series(t_src, v_src, t, tol=0.05):
    o = np.argsort(t_src)
    ts, vs = t_src[o], v_src[o]
    i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    pick = np.where(np.abs(t - ts[i - 1]) <= np.abs(ts[i] - t), i - 1, i)
    return vs[pick], np.abs(ts[pick] - t) <= tol


def bag_epochs(bag, ant, ml, org, lag=0.0, match='interp', tol_fr=0.01):
    d = np.load(NPZ / f'{bag}.npz')
    fr, rr = d['vehicle__front_bogie_velocity'], d['vehicle__rear_bogie_velocity']
    vel, fix = d[f'sensing__gnss__{ant}__vel'], d[f'sensing__gnss__{ant}__fix']
    if min(len(fr), len(rr), len(vel), len(fix)) < 100:
        return None
    t = vel[:, 1]
    keep = np.r_[True, np.diff(t) > 1e-6]
    vel, t = vel[keep], t[keep]
    v_hor = np.hypot(vel[:, 2], vel[:, 3])
    f = interp_series if match == 'interp' else nearest_series
    vf, okf = f(fr[:, 1], fr[:, 2] / 3.6, t + lag)
    vr, okr = f(rr[:, 1], rr[:, 2] / 3.6, t + lag)
    # acceleration from the wheel mean (central difference over +-0.5 s)
    vm_p, okp = interp_series(fr[:, 1], (fr[:, 2]) / 3.6, t + lag + 0.5, 0.3)
    vm_m, okm = interp_series(fr[:, 1], (fr[:, 2]) / 3.6, t + lag - 0.5, 0.3)
    acc = (vm_p - vm_m) / 1.0
    # fix: status 2, nearest header stamp within 0.1 s -> position -> map
    fx = fix[np.r_[True, np.diff(fix[:, 1]) > 1e-6]]
    j = np.clip(np.searchsorted(fx[:, 1], t), 1, len(fx) - 1)
    j = np.where(np.abs(t - fx[j - 1, 1]) <= np.abs(fx[j, 1] - t), j - 1, j)
    okfix = (np.abs(fx[j, 1] - t) <= 0.1) & (fx[j, 5] == 2)
    p = enu(fx[j, 2], fx[j, 3], fx[j, 4], *org)
    dist, im = ml.nearest(p[:, :2])
    k_ant = ml.k[im]
    grade = ml.g[im]
    if ant == 'master':
        k_mid = ml.span_mean(im, PIV[0], PIV[1])
    else:  # rover sits at master + 12.44 -> pivots at rover - 10.12 .. rover - 2.57
        k_mid = ml.span_mean(im, PIV[0] - ROVER_AHEAD, PIV[1] - ROVER_AHEAD)
    v_ant = v_hor * np.sqrt(1 + grade ** 2)
    vw = 0.5 * (vf + vr)
    ok = (okf & okr & okfix & okp & okm & (dist < 1.0) & (v_hor > 3.0) & (vw > 3.0)
          & (np.abs(vf / np.maximum(vr, 1e-3) - 1) < tol_fr) & (np.abs(acc) < 0.3))
    if ok.sum() < 50:
        return None
    df = pd.DataFrame(dict(bag=bag, t=t[ok], vw=vw[ok], vant=v_ant[ok], vhor=v_hor[ok], k_ant=k_ant[ok],
                           k_mid=k_mid[ok], grade=grade[ok], acc=acc[ok], fr_rr=(vf / vr)[ok]))
    df['y_raw'] = np.log(df.vw / df.vant)
    df['y_raw_nograde'] = np.log(df.vw / df.vhor)
    return df


def normalise(df, kcol, ycol='y_raw'):
    out_ = []
    for b, g in df.groupby('bag'):
        st = g[np.abs(g[kcol]) < 0.001]
        if len(st) < 50:
            continue
        g = g.copy()
        g['y'] = g[ycol] - np.median(st[ycol])
        out_.append(g)
    d = pd.concat(out_, ignore_index=True)
    # trim gross outliers (GNSS Doppler glitches)
    return d[np.abs(d.y) < 0.05].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------- fits
def c_model(k):
    return 1 + np.minimum(0.415 * np.abs(k), 0.009) - 0.054 * k


def ols(y, X, groups):
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    rss = float(e @ e)
    s2 = rss / (n - p)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    # cluster-robust (sandwich) SE, clusters = bag x 10 s block
    _, gi = np.unique(groups, return_inverse=True)
    S = np.zeros((gi.max() + 1, p))
    np.add.at(S, gi, X * e[:, None])
    G = S.shape[0]
    V = XtX_inv @ (S.T @ S) @ XtX_inv * G / (G - 1) * (n - 1) / (n - p)
    se_cl = np.sqrt(np.diag(V))
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss
    aic = n * np.log(rss / n) + 2 * p
    bic = n * np.log(rss / n) + p * np.log(n)
    return dict(b=b, se=se, se_cl=se_cl, rss=rss, r2=r2, aic=aic, bic=bic, n=n, p=p, G=G)


def fixed(y, pred, groups, p_free=1):
    """intercept-only fit around a fixed prediction."""
    r = y - pred
    c0 = r.mean()
    e = r - c0
    n = len(y)
    rss = float(e @ e)
    tss = float(((y - y.mean()) ** 2).sum())
    return dict(b=np.array([c0]), se=np.array([np.nan]), se_cl=np.array([np.nan]), rss=rss, r2=1 - rss / tss,
                aic=n * np.log(rss / n) + 2 * p_free, bic=n * np.log(rss / n) + p_free * np.log(n), n=n, p=p_free,
                G=len(np.unique(groups)))


def design(d, kcol):
    k = d[kcol].to_numpy()
    one = np.ones_like(k)
    xc = -np.log(c_model(k))
    return {
        '(a) k^2': (np.c_[one, k ** 2], ['c0', 'k^2']),
        '(b) |k|': (np.c_[one, np.abs(k)], ['c0', '|k|']),
        '(c) k^2 + |k|': (np.c_[one, k ** 2, np.abs(k)], ['c0', 'k^2', '|k|']),
        '(d) beta*[-log c(k)] (current form, scale fitted)': (np.c_[one, xc], ['c0', 'beta']),
        '(a+s) k^2 + k(signed)': (np.c_[one, k ** 2, k], ['c0', 'k^2', 'k']),
        '(b+s) |k| + k(signed)': (np.c_[one, np.abs(k), k], ['c0', '|k|', 'k']),
        '(e) |k| capped at 0.009 (slope fitted, cap fixed)': (np.c_[one, np.minimum(np.abs(k), 0.009 / 0.415)], ['c0', 'slope']),
    }


def report_fits(d, kcol, label, k2_theory):
    y = d.y.to_numpy()
    groups = (d.bag + '_' + (d.t // 10).astype(int).astype(str)).to_numpy()
    k = d[kcol].to_numpy()
    out(f'--- {label}: n = {len(d)} epochs, {d.bag.nunique()} bags, {len(np.unique(groups))} clusters (bag x 10 s); '
        f'curvature column = {kcol}')
    out(f'{"model":52s} {"R2":>8s} {"AIC":>11s} {"BIC":>11s}  coefficients (OLS SE / cluster SE)')
    res = {}
    for name, (X, cols) in design(d, kcol).items():
        r = ols(y, X, groups)
        res[name] = r
        coef = '  '.join(f'{c}={b:+.5g} ({s:.2g}/{sc:.2g})' for c, b, s, sc in zip(cols, r['b'], r['se'], r['se_cl']))
        out(f'{name:52s} {r["r2"]:8.4f} {r["aic"]:11.1f} {r["bic"]:11.1f}  {coef}')
    for name, pred in ((f'(f) theory fixed: -{k2_theory:.2f} k^2 (+c0)', -k2_theory * k ** 2),
                       ('(g) current model fixed: -log c(k) (+c0)', -np.log(c_model(k))),
                       ('(0) intercept only', np.zeros_like(k))):
        r = fixed(y, pred, groups)
        res[name] = r
        out(f'{name:52s} {r["r2"]:8.4f} {r["aic"]:11.1f} {r["bic"]:11.1f}  c0={r["b"][0]:+.5g}')
    return res


BINS = [(0.05, np.inf, 'R < 20'), (0.025, 0.05, '20-40'), (0.0125, 0.025, '40-80'), (0.005, 0.0125, '80-200'),
        (0.0, 0.005, '> 200')]


def binned(d, kcol, res, k2_theory):
    k = d[kcol].to_numpy()
    ak = np.abs(k)
    groups = (d.bag + '_' + (d.t // 10).astype(int).astype(str)).to_numpy()
    ba = res['(a) k^2']['b']
    bb = res['(b) |k|']['b']
    out(f'{"R bin [m]":10s} {"n":>7s} {"clust":>6s} {"mean|k|":>9s} {"y mean %":>9s} {"SE %":>7s} '
        f'{"theory %":>9s} {"fit k2 %":>9s} {"fit|k| %":>9s} {"current %":>10s}')
    for lo, hi, lab in BINS:
        m = (ak >= lo) & (ak < hi)
        if m.sum() < 5:
            out(f'{lab:10s} {m.sum():7d}  (too few)')
            continue
        y = d.y.to_numpy()[m]
        g = groups[m]
        # SE from cluster means (autocorrelation-safe)
        cm = pd.Series(y).groupby(g).mean()
        se = cm.std(ddof=1) / np.sqrt(len(cm)) if len(cm) > 1 else np.nan
        th = np.mean(-k2_theory * k[m] ** 2)
        fa = np.mean(ba[1] * k[m] ** 2)
        fb = np.mean(bb[1] * ak[m])
        cu = np.mean(-np.log(c_model(k[m])))
        out(f'{lab:10s} {m.sum():7d} {len(cm):6d} {ak[m].mean():9.5f} {100 * y.mean():+9.3f} {100 * se:7.3f} '
            f'{100 * th:+9.3f} {100 * fa:+9.3f} {100 * fb:+9.3f} {100 * cu:+10.3f}')
    out('  (predictions shown without the fitted intercepts; intercepts: '
        f'k^2 fit c0={100 * ba[0]:+.3f} %, |k| fit c0={100 * bb[0]:+.3f} %)')


def one_second(d, kcol):
    d = d.copy()
    d['sec'] = (d.t // 1).astype(int)
    a = d.groupby(['bag', 'sec']).agg(y=('y', 'mean'), k=(kcol, 'mean'), t=('t', 'mean'), n=('y', 'size')).reset_index()
    a = a[a.n >= 5].rename(columns={'k': kcol})
    return a


def estimate_lag(bags, ml, org):
    """Common wheel-vs-GNSS stamp lag: minimise the spread of y on straight track (acc filter off -> signal)."""
    lags = np.round(np.arange(-0.40, 0.41, 0.05), 3)
    rows = []
    for lag in lags:
        ys = []
        for b in bags:
            df = bag_epochs_noacc(b, ml, org, lag)
            if df is None:
                continue
            st = df[np.abs(df.k_ant) < 0.001]
            if len(st) < 50:
                continue
            yy = st.y_raw - np.median(st.y_raw)
            ys.append(yy[np.abs(yy) < 0.05])
        allv = np.concatenate(ys)
        rows.append((lag, float(np.sqrt(np.mean(allv ** 2))), len(allv)))
    return rows


def bag_epochs_noacc(bag, ml, org, lag):
    d = np.load(NPZ / f'{bag}.npz')
    fr, rr = d['vehicle__front_bogie_velocity'], d['vehicle__rear_bogie_velocity']
    vel, fix = d['sensing__gnss__master__vel'], d['sensing__gnss__master__fix']
    if min(len(fr), len(rr), len(vel), len(fix)) < 100:
        return None
    t = vel[:, 1]
    keep = np.r_[True, np.diff(t) > 1e-6]
    vel, t = vel[keep], t[keep]
    v_hor = np.hypot(vel[:, 2], vel[:, 3])
    vf, okf = interp_series(fr[:, 1], fr[:, 2] / 3.6, t + lag)
    vr, okr = interp_series(rr[:, 1], rr[:, 2] / 3.6, t + lag)
    fx = fix[np.r_[True, np.diff(fix[:, 1]) > 1e-6]]
    j = np.clip(np.searchsorted(fx[:, 1], t), 1, len(fx) - 1)
    j = np.where(np.abs(t - fx[j - 1, 1]) <= np.abs(fx[j, 1] - t), j - 1, j)
    okfix = (np.abs(fx[j, 1] - t) <= 0.1) & (fx[j, 5] == 2)
    p = enu(fx[j, 2], fx[j, 3], fx[j, 4], *org)
    dist, im = ml.nearest(p[:, :2])
    vw = 0.5 * (vf + vr)
    ok = okf & okr & okfix & (dist < 1.0) & (v_hor > 3.0) & (np.abs(vf / np.maximum(vr, 1e-3) - 1) < 0.01)
    if ok.sum() < 50:
        return None
    return pd.DataFrame(dict(k_ant=ml.k[im][ok], y_raw=np.log(vw[ok] / (v_hor[ok] * np.sqrt(1 + ml.g[im][ok] ** 2)))))


# ----------------------------------------------------------------------------------------------------------- main
def main():
    m, org = load_map()
    ml = MapLookup(m)
    splits = json.load(open(SPLITS))
    bags = []
    for b in splits['train']:
        f = np.load(NPZ / f'{b}.npz')['sensing__gnss__master__fix']
        if len(f) and np.mean(f[:, 5] == 2) > 0.8:
            bags.append(b)
    out('k2_check: wheel speed / antenna ground speed vs track curvature')
    out(f'map: {MAP_CSV.relative_to(ROOT)} ({len(m)} pts, origin {org}); master-antenna map, curvature sigma 2 m')
    out(f'theory coefficient: master (u={U_MASTER}) {K2_MASTER:.3f}, rover (u={U_ROVER}) {K2_ROVER:.3f} [m^2]; '
        'y = log(v_wheel/v_ant) predicted = -coef * k^2')
    out(f'current code: v = v_wheel * c(k), c = 1 + min(0.415|k|, 0.009) - 0.054 k  -> y = -log c(k)')
    out(f'train bags with master RTK fraction > 0.8: {len(bags)} of {len(splits["train"])}')
    out()

    # ---- lag between wheel stamps and GNSS vel stamps
    out('== common lag (wheel stamp = GNSS stamp + lag), rms of y on straight track, acc filter off, 12 bags ==')
    rows = estimate_lag(bags[::max(1, len(bags) // 12)][:12], ml, org)
    for lag, rms, n in rows:
        out(f'  lag {lag:+.2f} s: rms {100 * rms:.4f} %  (n={n})')
    lag = min(rows, key=lambda r: r[1])[0]
    out(f'  -> using lag = {lag:+.2f} s')
    out()

    # ---- main dataset (master)
    frames = [bag_epochs(b, 'master', ml, org, lag) for b in bags]
    raw = pd.concat([f for f in frames if f is not None], ignore_index=True)
    out(f'master epochs after filters: {len(raw)} from {raw.bag.nunique()} bags')
    ak = raw.k_ant.abs()
    out('  epochs per curvature bin (before straight-normalisation/trim): ' +
        ', '.join(f'{lab}: {int(((ak >= lo) & (ak < hi)).sum())}' for lo, hi, lab in BINS))
    out(f'  grade correction size: mean |log sqrt(1+g^2)| = {100 * np.mean(np.log(np.sqrt(1 + raw.grade ** 2))):.4f} %')
    out()

    results = {}
    for kcol, lab in (('k_ant', 'MASTER, kappa at antenna'), ('k_mid', 'MASTER, kappa averaged between bogie pivots')):
        d = normalise(raw, kcol)
        out(f'== {lab} ==')
        res = report_fits(d, kcol, lab, K2_MASTER)
        out()
        binned(d, kcol, res, K2_MASTER)
        out()
        d1 = one_second(d, kcol)
        out(f'  1-s averaged data (n = {len(d1)}): R2 / AIC')
        g1 = (d1.bag + '_' + (d1.t // 10).astype(int).astype(str)).to_numpy()
        for name, (X, cols) in design(d1, kcol).items():
            if name[:4] not in ('(a) ', '(b) ', '(c) ', '(d) '):
                continue
            r = ols(d1.y.to_numpy(), X, g1)
            out(f'    {name:52s} R2 {r["r2"]:.4f}  AIC {r["aic"]:10.1f}  BIC {r["bic"]:10.1f}  b1={r["b"][1]:+.5g}'
                f' (cl SE {r["se_cl"][1]:.2g})' + (f' b2={r["b"][2]:+.5g} (cl SE {r["se_cl"][2]:.2g})' if len(cols) > 2 else ''))
        out()
        results[kcol] = res

    # ---- sensitivity: nearest-stamp matching (no interpolation), no grade correction, 2 % bogie tolerance
    out('== sensitivity (kappa at antenna; coefficient of k^2 in (a), of |k| in (b); cluster SE) ==')
    variants = []
    fr_near = [bag_epochs(b, 'master', ml, org, lag, match='nearest') for b in bags]
    variants.append(('nearest wheel sample within 0.05 s (per spec)', pd.concat([f for f in fr_near if f is not None]), 'y_raw'))
    variants.append(('no grade correction (horizontal GNSS speed)', raw, 'y_raw_nograde'))
    fr_nolag = [bag_epochs(b, 'master', ml, org, 0.0) for b in bags]
    variants.append(('lag = 0', pd.concat([f for f in fr_nolag if f is not None]), 'y_raw'))
    for tol in (0.03, 1.0):
        fr_tol = [bag_epochs(b, 'master', ml, org, lag, tol_fr=tol) for b in bags]
        variants.append((f'front/rear tolerance {tol:.0%}' + (' (no filter)' if tol >= 1 else ''),
                         pd.concat([f for f in fr_tol if f is not None]), 'y_raw'))
    for vlab, vd, ycol in variants:
        d = normalise(vd, 'k_ant', ycol)
        y = d.y.to_numpy()
        g = (d.bag + '_' + (d.t // 10).astype(int).astype(str)).to_numpy()
        k = d.k_ant.to_numpy()
        ra = ols(y, np.c_[np.ones_like(k), k ** 2], g)
        rb = ols(y, np.c_[np.ones_like(k), np.abs(k)], g)
        out(f'  {vlab:48s} n={len(d):6d}  k^2: {ra["b"][1]:+7.3f} ({ra["se_cl"][1]:.2f}) R2 {ra["r2"]:.4f} AIC {ra["aic"]:.1f}'
            f' | |k|: {rb["b"][1]:+.4f} ({rb["se_cl"][1]:.3f}) R2 {rb["r2"]:.4f} AIC {rb["aic"]:.1f}')
        rd = ols(y, np.c_[np.ones_like(k), -np.log(c_model(k))], g)
        ak = np.abs(k)
        out(f'  {"":48s} current-form beta {rd["b"][1]:+.3f} ({rd["se_cl"][1]:.3f}) R2 {rd["r2"]:.4f} AIC {rd["aic"]:.1f}; bin means y %: ' +
            ', '.join(f'{lab} {100 * y[(ak >= lo) & (ak < hi)].mean():+.3f} (n={int(((ak >= lo) & (ak < hi)).sum())})'
                      for lo, hi, lab in BINS))
    out()

    # ---- rover
    fr_rov = []
    for b in bags:
        f = np.load(NPZ / f'{b}.npz')['sensing__gnss__rover__fix']
        if len(f) and np.mean(f[:, 5] == 2) > 0.8:
            fr_rov.append(bag_epochs(b, 'rover', ml, org, lag))
    rov = pd.concat([f for f in fr_rov if f is not None], ignore_index=True)
    for kcol, lab in (('k_ant', 'ROVER, kappa at rover antenna'), ('k_mid', 'ROVER, kappa averaged between bogie pivots')):
        d = normalise(rov, kcol)
        out(f'== {lab} (theory coefficient {K2_ROVER:.2f}) ==')
        res = report_fits(d, kcol, lab, K2_ROVER)
        out()
        binned(d, kcol, res, K2_ROVER)
        out()

    # ---- bogie-ratio filter bias check: fraction of epochs dropped by |front/rear-1| < 1 % per bin
    out('== note: share of candidate epochs rejected by the 1 % front/rear filter, per curvature bin (master) ==')
    tot, kept = {}, {}
    for b in bags:
        dd = bag_epochs_filtercheck(b, ml, org, lag)
        if dd is None:
            continue
        for lo, hi, lab in BINS:
            mm = (np.abs(dd.k) >= lo) & (np.abs(dd.k) < hi)
            tot[lab] = tot.get(lab, 0) + int(mm.sum())
            kept[lab] = kept.get(lab, 0) + int((mm & dd.ok).sum())
    out('  ' + ', '.join(f'{lab}: {1 - kept[lab] / max(tot[lab], 1):.1%} of {tot[lab]}' for _, _, lab in BINS))

    OUT.write_text('\n'.join(LINES) + '\n', encoding='utf-8')
    print(f'wrote {OUT}')


def bag_epochs_filtercheck(bag, ml, org, lag):
    d = np.load(NPZ / f'{bag}.npz')
    fr, rr = d['vehicle__front_bogie_velocity'], d['vehicle__rear_bogie_velocity']
    vel, fix = d['sensing__gnss__master__vel'], d['sensing__gnss__master__fix']
    if min(len(fr), len(rr), len(vel), len(fix)) < 100:
        return None
    t = vel[:, 1]
    t = t[np.r_[True, np.diff(t) > 1e-6]]
    vf, okf = interp_series(fr[:, 1], fr[:, 2] / 3.6, t + lag)
    vr, okr = interp_series(rr[:, 1], rr[:, 2] / 3.6, t + lag)
    fx = fix[np.r_[True, np.diff(fix[:, 1]) > 1e-6]]
    j = np.clip(np.searchsorted(fx[:, 1], t), 1, len(fx) - 1)
    j = np.where(np.abs(t - fx[j - 1, 1]) <= np.abs(fx[j, 1] - t), j - 1, j)
    okfix = (np.abs(fx[j, 1] - t) <= 0.1) & (fx[j, 5] == 2)
    p = enu(fx[j, 2], fx[j, 3], fx[j, 4], *org)
    dist, im = ml.nearest(p[:, :2])
    base = okf & okr & okfix & (dist < 1.0) & (0.5 * (vf + vr) > 3.0)
    return pd.DataFrame(dict(k=ml.k[im][base], ok=(np.abs(vf / np.maximum(vr, 1e-3) - 1) < 0.01)[base]))


if __name__ == '__main__':
    main()
