import sys, pickle
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
import numpy as np

SP = 'C:/MosTransHack/research/map_matching_code/'
M = pickle.load(open(SP + 'map.pkl', 'rb'))['maps']
C = pickle.load(open(SP + 'cache.pkl', 'rb'))
ST = pickle.load(open(SP + 'stops.pkl', 'rb'))
EV = ST['ev']


def vsmooth(v):  # causal 5-sample moving average
    c = np.cumsum(np.r_[0, v])
    idx = np.arange(len(v))
    a = np.maximum(0, idx - 4)
    return (c[idx + 1] - c[a]) / (idx + 1 - a)


def zone_events(v):
    ev = []
    up = False
    dn = False
    for i in range(len(v)):
        if v[i] < 18:
            up = True
        if v[i] > 26:
            dn = True
        if up and v[i] >= 22:
            ev.append((i, 'up'))
            up = False
        if dn and v[i] < 22:
            ev.append((i, 'dn'))
            dn = False
    return ev


# precompute events + true s for every run
ZE = {}
for n, c in C.items():
    v = vsmooth(0.5 * (c['vf'] + c['vr']))
    c['vs'] = v
    ok = np.isfinite(c['Sg']) & (np.abs(np.nan_to_num(c['Eg'], nan=99)) < 3)
    tt = c['t'][ok]
    ss = c['Sg'][ok]
    lst = []
    for i, typ in zone_events(v):
        ti = c['tw'][i]
        j = np.searchsorted(tt, ti)
        if j <= 0 or j >= len(tt):
            continue
        if tt[j] - tt[j - 1] > 1.0:
            continue
        lst.append((typ, np.interp(ti, tt, ss)))
    ZE[n] = lst


def cluster(vals, runs, nruns, gap, minfrac, maxsig):
    if len(vals) == 0:
        return np.zeros((0, 3))
    o = np.argsort(vals)
    v = np.array(vals)[o]
    r = np.array(runs)[o]
    cl = np.r_[0, np.cumsum(np.diff(v) > gap)]
    db = []
    for c_ in np.unique(cl):
        m = cl == c_
        k = len(set(r[m]))
        if k < max(3, minfrac * nruns):
            continue
        med = np.median(v[m])
        sig = 1.4826 * np.median(np.abs(v[m] - med))
        if sig > maxsig:
            continue
        db.append((med, sig, k / nruns))
    return np.array(db) if db else np.zeros((0, 3))


def dbs(dirn, exclude):
    rr = [n for n, c in C.items() if c['dir'] == dirn and n != exclude]
    nr = len(rr)
    ev = [e for e in EV if e[1] == dirn and e[0] != exclude and e[0] in rr]
    stop = cluster([e[2] for e in ev], [e[0] for e in ev], nr, 15, 0.12, 3.0)
    if len(stop):
        stop[:, 1] = np.maximum(stop[:, 1], 0.3)
    out = {'stop': stop}
    for typ in ('up', 'dn'):
        vals = []
        runs = []
        for n in rr:
            for t_, s_ in ZE[n]:
                if t_ == typ:
                    vals.append(s_)
                    runs.append(n)
        db = cluster(vals, runs, nr, 8, 0.3, 3.5)
        if len(db):
            db[:, 1] = np.maximum(db[:, 1], 1.0)
        out[typ] = db
    return out


def interp_map(dirn, s):
    P = M[dirn]['P']
    z = M[dirn]['z']
    S = np.arange(len(P)).astype(float)
    s = np.clip(s, 0, S[-1])
    return np.c_[np.interp(s, S, P[:, 0]), np.interp(s, S, P[:, 1]), np.interp(s, S, z)]


def run(n, use_stop, use_zone, use_scale, q_abs=0.002, eps=0.004):
    c = C[n]
    dirn = c['dir']
    D = dbs(dirn, n) if (use_stop or use_zone) else None
    Sg = c['Sg']
    t = c['t']
    tw = c['tw']
    vw = 0.5 * (c['vf'] + c['vr']) / 3.6
    vs = c['vs']
    f0 = np.where(np.isfinite(Sg))[0][0]
    t0 = t[f0]
    st = {'s': Sg[f0], 'P': 0.5 ** 2, 'k': 1.0, 'Pk': 0.004 ** 2, 'last': None, 'odo': 0.0, 'dfix': 0.0}
    nfix = {'stop': 0, 'up': 0, 'dn': 0}
    i0 = max(np.searchsorted(tw, t0), 1)
    S_est = np.full(len(tw), np.nan)
    SIG = np.full(len(tw), np.nan)

    def update(kind):
        db = D[kind]
        if len(db) == 0:
            return
        s = st['s']
        P = st['P']
        j = np.argmin(np.abs(db[:, 0] - s))
        inn = db[j, 0] - s
        R = db[j, 1] ** 2
        gate = max(3 * np.sqrt(P + R), 5.0)
        if abs(inn) > gate:
            return
        K = P / (P + R)
        st['s'] = s + K * inn
        st['P'] = (1 - K) * P
        nfix[kind] += 1
        st['dfix'] = 0.0
        if use_scale:
            last = st['last']
            if last is not None and db[j, 0] - last[0] > 150 and st['odo'] - last[1] > 100:
                kobs = np.clip((db[j, 0] - last[0]) / (st['odo'] - last[1]), 0.95, 1.05)
                Rk = (np.sqrt(R + last[2]) / (db[j, 0] - last[0])) ** 2
                Kk = st['Pk'] / (st['Pk'] + Rk)
                st['k'] = st['k'] + Kk * (kobs - st['k'])
                st['Pk'] = (1 - Kk) * st['Pk'] + 1e-8
            st['last'] = (db[j, 0], st['odo'], R)

    zero_t = None
    used = False
    up = False
    dn = False
    for i in range(i0, len(tw)):
        dt = tw[i] - tw[i - 1]
        if dt <= 0 or dt > 1.0:
            dt = min(max(dt, 0), 0.2)
        ds = st['k'] * vw[i] * dt
        st['s'] += ds
        st['odo'] += vw[i] * dt
        st['dfix'] += ds
        st['P'] += q_abs * ds + 2 * eps ** 2 * st['dfix'] * ds
        if use_stop:
            if vw[i] < 0.05 / 3.6:
                if zero_t is None:
                    zero_t = tw[i]
                    used = False
                if (not used) and tw[i] - zero_t >= 1.0:
                    used = True
                    update('stop')
            else:
                zero_t = None
        if use_zone:
            if vs[i] < 18:
                up = True
            if vs[i] > 26:
                dn = True
            if up and vs[i] >= 22:
                up = False
                update('up')
            if dn and vs[i] < 22:
                dn = False
                update('dn')
        S_est[i] = st['s']
        SIG[i] = np.sqrt(st['P'])
    fin = np.isfinite(S_est)
    ok = np.isfinite(Sg) & (t >= tw[i0]) & (t <= tw[-1]) & (np.abs(np.nan_to_num(c['Eg'], nan=99)) < 3)
    # reference-glitch mask: GNSS along-track position inconsistent with wheel odometry over +-5 s
    from scipy.ndimage import median_filter
    odo_raw = np.r_[0, np.cumsum(vw[1:] * np.diff(tw))]
    rr = Sg[ok] - np.interp(t[ok], tw, odo_raw)
    hp = rr - median_filter(rr, size=101, mode='nearest')
    good = np.abs(hp) < 2.0
    idx_ok = np.where(ok)[0]
    ok[idx_ok[~good]] = False
    glitch_frac = 1 - good.mean()
    se = np.interp(t[ok], tw[fin], S_est[fin])
    sig = np.interp(t[ok], tw[fin], SIG[fin])
    Pe = interp_map(dirn, se)
    ref = np.c_[c['p'][ok], c['z'][ok]]
    e3 = np.linalg.norm(Pe - ref, axis=1)
    ea = se - Sg[ok]
    ta = c['tall']
    oka = (ta >= tw[i0]) & (ta <= tw[-1])
    sea = np.interp(ta[oka], tw[fin], S_est[fin])
    Pa = interp_map(dirn, sea)
    e3a = np.linalg.norm(Pa - np.c_[c['pall'][oka], c['zall'][oka]], axis=1)
    dist = np.nansum(np.abs(np.diff(Sg[ok])))
    nees = np.mean((ea / np.maximum(sig, 1e-3)) ** 2)
    return dict(n=n, rmse3=np.sqrt(np.mean(e3 ** 2)), ea_mean=np.mean(ea), ea_abs=np.mean(np.abs(ea)),
                ea_rmse=np.sqrt(np.mean(ea ** 2)), ea_max=np.max(np.abs(ea)), end=e3[-1],
                endpct=100 * e3[-1] / max(dist, 1), dist=dist, fs=nfix['stop'], fz=nfix['up'] + nfix['dn'],
                k=st['k'], rmse3all=np.sqrt(np.mean(e3a ** 2)), p95all=np.percentile(e3a, 95), nees=nees, glitch=glitch_frac,
                ea_p95=np.percentile(np.abs(ea), 95))


if __name__ == '__main__':
    modes = [('A: map + wheel odometry (k=1)', 0, 0, 0), ('B: A + stop landmarks', 1, 0, 0),
             ('C: B + speed-zone landmarks', 1, 1, 0), ('D: C + online scale calib.', 1, 1, 1)]
    for lab, a, b, cc in modes:
        res = [run(n, a, b, cc) for n in C]
        print('=====', lab)
        for veh in ('30618', '30639', 'all'):
            R = [r for r in res if veh == 'all' or r['n'].startswith(veh)]
            f = lambda key: np.mean([r[key] for r in R])
            print(' %-5s n=%2d | along: mean %+.2f mean|.| %.2f RMSE %.2f p95 %.2f max(avg) %.2f worst %.2f | 3D RMSE(RTK,clean) %.2f 3D RMSE(all fixes) %.2f p95 %.2f | end %.2f m = %.3f%% (worst %.3f%%) | fixes stop %.1f zone %.1f | NEES %.2f | ref-glitch %.2f%%' % (
                veh, len(R), f('ea_mean'), f('ea_abs'), f('ea_rmse'), f('ea_p95'), f('ea_max'), max(r['ea_max'] for r in R),
                f('rmse3'), f('rmse3all'), f('p95all'), f('end'), f('endpct'), max(r['endpct'] for r in R),
                f('fs'), f('fz'), f('nees'), 100 * f('glitch')))
        if cc:
            print('  final k per run (30639):', [round(r['k'], 4) for r in res if r['n'].startswith('30639')])
    for d_ in ('AB', 'BA'):
        D = dbs(d_, 'none')
        print('landmark counts %s: stops %d, up %d, dn %d' % (d_, len(D['stop']), len(D['up']), len(D['dn'])))
        print('  stop sigma median %.2f m; zone sigma median %.2f m' % (np.median(D['stop'][:, 1]), np.median(np.r_[D['up'][:, 1], D['dn'][:, 1]])))
