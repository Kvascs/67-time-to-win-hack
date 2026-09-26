"""Variants of the along-track filter: which landmarks feed the scale estimator, gates, noise; coverage of sigma."""
import sys
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
import numpy as np
from scipy.ndimage import median_filter
import evaldr2 as E


def run(n, use_stop=1, use_zone=1, use_scale=1, scale_types=('stop',), min_base=300.0, q_abs=0.002, eps=0.004,
        zone_sig_floor=1.0, gate_min=5.0, k0=1.0, zone_min_sigma=0.0, pk0=0.004):
    c = E.C[n]
    dirn = c['dir']
    D = E.dbs(dirn, n)
    if len(D['up']):
        D['up'][:, 1] = np.maximum(D['up'][:, 1], zone_sig_floor)
    if len(D['dn']):
        D['dn'][:, 1] = np.maximum(D['dn'][:, 1], zone_sig_floor)
    Sg = c['Sg']; t = c['t']; tw = c['tw']
    vw = 0.5 * (c['vf'] + c['vr']) / 3.6
    vs = c['vs']
    f0 = np.where(np.isfinite(Sg))[0][0]
    st = dict(s=Sg[f0], P=0.5 ** 2, k=k0, Pk=pk0 ** 2, last=None, odo=0.0, dfix=0.0)
    nfix = {'stop': 0, 'up': 0, 'dn': 0}
    i0 = max(np.searchsorted(tw, t[f0]), 1)
    S_est = np.full(len(tw), np.nan); SIG = np.full(len(tw), np.nan)

    def update(kind):
        db = D[kind]
        if len(db) == 0:
            return
        s, P = st['s'], st['P']
        j = np.argmin(np.abs(db[:, 0] - s)); inn = db[j, 0] - s; R = db[j, 1] ** 2
        if abs(inn) > max(3 * np.sqrt(P + R), gate_min):
            return
        K = P / (P + R); st['s'] = s + K * inn; st['P'] = (1 - K) * P; nfix[kind] += 1; st['dfix'] = 0.0
        if use_scale and kind in scale_types:
            last = st['last']
            if last is not None and db[j, 0] - last[0] > min_base and st['odo'] - last[1] > 0.5 * min_base:
                kobs = np.clip((db[j, 0] - last[0]) / (st['odo'] - last[1]), 0.97, 1.03)
                Rk = (np.sqrt(R + last[2]) / (db[j, 0] - last[0])) ** 2 + 0.001 ** 2
                Kk = st['Pk'] / (st['Pk'] + Rk); st['k'] += Kk * (kobs - st['k']); st['Pk'] = (1 - Kk) * st['Pk'] + 1e-8
            st['last'] = (db[j, 0], st['odo'], R)

    zero_t = None; used = False; up = False; dn = False
    for i in range(i0, len(tw)):
        dt = tw[i] - tw[i - 1]
        if dt <= 0 or dt > 1.0:
            dt = min(max(dt, 0), 0.2)
        ds = st['k'] * vw[i] * dt
        st['s'] += ds; st['odo'] += vw[i] * dt; st['dfix'] += ds
        st['P'] += q_abs * ds + 2 * eps ** 2 * st['dfix'] * ds
        if use_stop:
            if vw[i] < 0.05 / 3.6:
                if zero_t is None:
                    zero_t = tw[i]; used = False
                if (not used) and tw[i] - zero_t >= 1.0:
                    used = True; update('stop')
            else:
                zero_t = None
        if use_zone:
            if vs[i] < 18: up = True
            if vs[i] > 26: dn = True
            if up and vs[i] >= 22:
                up = False
                if np.sqrt(st['P']) > zone_min_sigma: update('up')
            if dn and vs[i] < 22:
                dn = False
                if np.sqrt(st['P']) > zone_min_sigma: update('dn')
        S_est[i] = st['s']; SIG[i] = np.sqrt(st['P'])
    fin = np.isfinite(S_est)
    ok = np.isfinite(Sg) & (t >= tw[i0]) & (t <= tw[-1]) & (np.abs(np.nan_to_num(c['Eg'], nan=99)) < 3)
    odo_raw = np.r_[0, np.cumsum(vw[1:] * np.diff(tw))]
    rr = Sg[ok] - np.interp(t[ok], tw, odo_raw)
    good = np.abs(rr - median_filter(rr, size=101, mode='nearest')) < 2.0
    idx = np.where(ok)[0]; ok[idx[~good]] = False
    se = np.interp(t[ok], tw[fin], S_est[fin]); sig = np.interp(t[ok], tw[fin], SIG[fin])
    ea = se - Sg[ok]
    Pe = E.interp_map(dirn, se); ref = np.c_[c['p'][ok], c['z'][ok]]
    e3 = np.linalg.norm(Pe - ref, axis=1)
    dist = np.nansum(np.abs(np.diff(Sg[ok])))
    return dict(n=n, ea_mean=ea.mean(), ea_abs=np.abs(ea).mean(), ea_rmse=np.sqrt(np.mean(ea ** 2)), ea_p95=np.percentile(np.abs(ea), 95),
                ea_max=np.abs(ea).max(), rmse3=np.sqrt(np.mean(e3 ** 2)), end=e3[-1], endpct=100 * e3[-1] / max(dist, 1), k=st['k'],
                cov1=np.mean(np.abs(ea) <= sig), cov2=np.mean(np.abs(ea) <= 2 * sig), fs=nfix['stop'], fz=nfix['up'] + nfix['dn'])


def summary(lab, **kw):
    res = [run(n, **kw) for n in E.C]
    out = []
    for veh in ('30618', '30639', 'all'):
        R = [r for r in res if veh == 'all' or r['n'].startswith(veh)]
        f = lambda key: np.mean([r[key] for r in R])
        out.append('%s: along mean %+.2f |.| %.2f RMSE %.2f p95 %.2f max %.1f | 3D %.2f | end %.2f m %.3f%% | cov1 %.2f cov2 %.2f' % (
            veh, f('ea_mean'), f('ea_abs'), f('ea_rmse'), f('ea_p95'), f('ea_max'), f('rmse3'), f('end'), f('endpct'), f('cov1'), f('cov2')))
    print('==', lab)
    for o in out:
        print('   ', o)
    return res


if __name__ == '__main__':
    summary('H  stops + zones only when sigma_s>2 m, no scale', use_scale=0, zone_min_sigma=2.0)
    summary('I  stops + scale(stop pairs>500 m, prior 0.2%)', use_zone=0, scale_types=('stop',), min_base=500, pk0=0.002)
    summary('J  stops + zones(sigma_s>2) + scale(stop pairs>500 m, prior 0.3%)', zone_min_sigma=2.0, scale_types=('stop',), min_base=500, pk0=0.003)
    summary('K  stops + zones(sigma_s>1.5) + scale(stop pairs>300 m, prior 0.4%)', zone_min_sigma=1.5, scale_types=('stop',), min_base=300, pk0=0.004)
