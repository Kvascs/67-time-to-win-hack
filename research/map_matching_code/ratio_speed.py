import sys, pickle
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
from common import *
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

SP = 'C:/MosTransHack/research/map_matching_code/'
M = pickle.load(open(SP + 'map.pkl', 'rb'))['maps']
geo = {}
for k, v in M.items():
    P = v['P']
    psi = np.unwrap(np.arctan2(*np.gradient(P, axis=0)[:, ::-1].T))
    kap = gaussian_filter1d(np.gradient(gaussian_filter1d(psi, 2)), 2)
    geo[k] = (cKDTree(P), kap)
X = []
seen = set()
loopB = 0
nAB = 0
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
    if dirn == 'AB':
        nAB += 1
        if np.any(p[:, 0] < -5115):
            loopB += 1
    tree, kap = geo[dirn]
    dd, j = tree.query(p)
    fv = d['vehicle__front_bogie_velocity']
    rv = d['vehicle__rear_bogie_velocity']
    mv = d['sensing__gnss__master__vel']
    vf = np.interp(t, fv[:, 1], fv[:, 2]) / 3.6
    vr = np.interp(t, rv[:, 1], rv[:, 2]) / 3.6
    gs = np.hypot(np.interp(t, mv[:, 1], mv[:, 2]), np.interp(t, mv[:, 1], mv[:, 3]))
    # GNSS speed from position differences (antenna path) as a second reference
    ok = (dd < 1.5)
    X.append(np.c_[kap[j][ok], 0.5 * (vf + vr)[ok], gs[ok], np.full(ok.sum(), n.startswith('30639'))])
X = np.vstack(X)
k_, vw, gs, veh = X.T
print('AB runs %d, of which traverse the western loop at B (x<-5115): %d' % (nAB, loopB))
straight = np.abs(k_) < 1 / 2000
for lo, hi in ((1, 3), (3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 16)):
    m = straight & (gs >= lo) & (gs < hi) & (veh == 0)
    if m.sum() > 100:
        print('straight, GNSS speed %4.1f-%4.1f m/s: n=%6d wheel/GNSS mean %.4f median %.4f' % (lo, hi, m.sum(), np.mean(vw[m] / gs[m]), np.median(vw[m] / gs[m])))
for lo, hi in ((3, 5), (5, 7)):
    for a, b in ((0, 1 / 2000), (1 / 200, 1 / 100), (1 / 100, 1 / 25)):
        m = (np.abs(k_) >= a) & (np.abs(k_) < b) & (gs >= lo) & (gs < hi) & (veh == 0)
        if m.sum() > 100:
            print('speed %d-%d m/s, R in (%.0f, %.0f]: n=%6d wheel/GNSS median %.4f' % (lo, hi, 1 / b, (1 / a if a > 0 else np.inf), m.sum(), np.median(vw[m] / gs[m])))
alat = gs ** 2 * np.abs(k_)
m = gs > 2
print('lateral accel v^2*|kappa|: p50 %.2f p99 %.2f p99.9 %.2f max %.2f m/s^2' % (np.percentile(alat[m], 50), np.percentile(alat[m], 99), np.percentile(alat[m], 99.9), alat[m].max()))
