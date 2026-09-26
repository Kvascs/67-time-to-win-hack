"""Robust directional centerline builder (v2).

Changes vs v1:
 * reference polyline is binned by WHEEL-ODOMETRY distance (1 m bins, median position per bin) -> removes
   standstill jitter and GNSS jumps while the tram is stopped (v1 baked a +13 m detour into BA at s~4850);
 * spike filter on the reference;
 * after refinement: along-track length consistency check against wheel odometry of all runs (per 50 m window).
"""
import sys, pickle
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
from common import *
from scipy.ndimage import gaussian_filter1d
src = open('C:/MosTransHack/research/map_matching_code/buildmap.py').read()
exec(src.split('# collect runs')[0])  # resample, Line (project_seq)


def wheel_odo(d):
    fv = d['vehicle__front_bogie_velocity']
    rv = d['vehicle__rear_bogie_velocity']
    tw = fv[:, 1]
    vw = 0.5 * (fv[:, 2] + np.interp(tw, rv[:, 1], rv[:, 2])) / 3.6
    odo = np.r_[0, np.cumsum(0.5 * (vw[1:] + vw[:-1]) * np.diff(tw))]
    return tw, odo


runs = {'AB': [], 'BA': []}
seen = set()
for n in names():
    d = load(n)
    m = d['sensing__gnss__master__fix']
    if m.shape[0] < 3000:
        continue
    key = (m.shape[0], round(m[0, 0], 1))
    if key in seen:
        continue
    seen.add(key)
    t, p, z, st = fix_xy(m, True)
    if len(t) < 0.9 * m.shape[0]:
        continue
    dirn = 'AB' if np.linalg.norm(p[0] - A) < np.linalg.norm(p[0] - B) else 'BA'
    full = (min(np.linalg.norm(p[0] - A), np.linalg.norm(p[0] - B)) < 60) and (min(np.linalg.norm(p[-1] - A), np.linalg.norm(p[-1] - B)) < 150)
    tw, odo = wheel_odo(d)
    runs[dirn].append(dict(name=n, t=t, p=p, z=z, full=full, veh=n[:5], od=np.interp(t, tw, odo)))


def clean_reference(r):
    od = r['od'] - r['od'][0]
    b = np.floor(od).astype(int)
    P = []
    for bi in np.unique(b):
        m = b == bi
        P.append(np.median(r['p'][m], axis=0))
    P = np.array(P)
    # spike filter: distance of each point to chord of neighbours +-5 points
    keep = np.ones(len(P), bool)
    for i in range(5, len(P) - 5):
        a, c = P[i - 5], P[i + 5]
        v = c - a
        L = np.linalg.norm(v)
        if L < 1e-3:
            continue
        dev = abs(v[0] * (P[i] - a)[1] - v[1] * (P[i] - a)[0]) / L
        if dev > 1.5:
            keep[i] = False
    return P[keep], int((~keep).sum())


maps = {}
for dirn in ('AB', 'BA'):
    R = [r for r in runs[dirn] if r['full']]
    S0, S1 = (A, B) if dirn == 'AB' else (B, A)
    ref = min(R, key=lambda r: np.linalg.norm(r['p'][0] - S0) + np.linalg.norm(r['p'][-1] - S1) + (0 if len(r['t']) / (r['t'][-1] - r['t'][0]) > 9.5 else 50))
    Pref, nsp = clean_reference(ref)
    line = Line(Pref)
    print(dirn, 'ref', ref['name'], 'spikes removed', nsp, 'len %.1f' % line.s[-1], flush=True)
    train = [r for r in R if r['veh'] == '30618']
    for it in range(3):
        allS = []; allE = []; allZ = []; per = []
        for r in train:
            S, E_ = line.project_seq(r['p'])
            ok = np.isfinite(S) & (np.abs(E_) < 2.5)
            allS.append(S[ok]); allE.append(E_[ok]); allZ.append(r['z'][ok])
            per.append((r, S, E_))
        S = np.concatenate(allS); E_ = np.concatenate(allE); Z = np.concatenate(allZ)
        b = np.clip(np.round(S).astype(int), 0, len(line.s) - 1)
        med = np.zeros(len(line.s))
        order = np.argsort(b); bs = b[order]; es = E_[order]; zs = Z[order]
        splits = np.searchsorted(bs, np.arange(len(line.s) + 1))
        zmed = np.full(len(line.s), np.nan)
        for i in range(len(line.s)):
            a_, c_ = splits[i], splits[i + 1]
            if c_ - a_ >= 3:
                med[i] = np.median(es[a_:c_]); zmed[i] = np.median(zs[a_:c_])
        med = gaussian_filter1d(med, 3)
        resid = E_ - med[b]
        print(' it', it, 'pts', len(E_), 'resid RMS %.3f p50|.| %.3f p95 %.3f p99 %.3f' % (np.sqrt(np.mean(resid ** 2)), np.median(np.abs(resid)), np.percentile(np.abs(resid), 95), np.percentile(np.abs(resid), 99)), flush=True)
        newP = line.P + med[:, None] * line.n
        okz = np.isfinite(zmed)
        zf = gaussian_filter1d(np.interp(line.s, line.s[okz], zmed[okz]), 5)
        line = Line(newP)
        line.z = np.interp(line.s, np.linspace(0, line.s[-1], len(zf)), zf)
    # along-track length consistency vs wheel odometry (per 50 m window, scale-normalised per run)
    W = 50.0
    grid = np.arange(0, len(line.s), W)
    RR = []
    for r in train:
        S, E_ = line.project_seq(r['p'])
        ok = np.isfinite(S) & (np.abs(E_) < 3)
        mono = np.maximum.accumulate(S[ok]); keep = np.r_[True, np.diff(mono) > 0]
        o_at = np.interp(grid, mono[keep], r['od'][ok][keep], left=np.nan, right=np.nan)
        rr = np.diff(o_at) / W
        RR.append(rr / np.nanmedian(rr))
    RR = np.array(RR)
    medr = np.nanmedian(RR, axis=0)
    cnt = np.sum(np.isfinite(RR), axis=0)
    bad = np.where((np.abs(medr - 1) > 0.01) & (cnt >= 5))[0]
    print(' length check: cumulative (odo-map) %.1f m; windows with |odo/map-1|>1%%:' % np.nansum((medr - 1) * W), [(int(grid[i]), round(float(medr[i]), 4)) for i in bad], flush=True)
    maps[dirn] = dict(s=line.s, P=line.P, z=line.z, ref=ref['name'])
    E2 = []
    for r in R:
        if r['veh'] != '30639':
            continue
        S, E_ = line.project_seq(r['p']); ok = np.isfinite(S) & (np.abs(E_) < 5); E2.append(E_[ok])
    if E2:
        E2 = np.concatenate(E2)
        print(' 30639 cross-track vs map: RMS %.3f median|e| %.3f p95 %.3f' % (np.sqrt(np.mean(E2 ** 2)), np.median(np.abs(E2)), np.percentile(np.abs(E2), 95)), flush=True)
pickle.dump(dict(maps=maps), open(SP + 'map.pkl', 'wb'))
for dirn, mp in maps.items():
    P = mp['P']; s = mp['s']
    psi = np.unwrap(np.arctan2(*np.gradient(P, axis=0)[:, ::-1].T))
    kap = gaussian_filter1d(np.gradient(gaussian_filter1d(psi, 2), s), 2)
    gr = np.gradient(gaussian_filter1d(mp['z'], 10), s)
    print(dirn, 'L=%.1f m  z %.1f..%.1f  max|grade| %.3f  p99|grade| %.3f  min R %.1f m  frac R<500: %.2f  frac R<100: %.3f' % (
        s[-1], mp['z'].min(), mp['z'].max(), np.abs(gr).max(), np.percentile(np.abs(gr), 99), 1 / np.abs(kap).max(), np.mean(np.abs(kap) > 1 / 500), np.mean(np.abs(kap) > 1 / 100)))
print('saved')
