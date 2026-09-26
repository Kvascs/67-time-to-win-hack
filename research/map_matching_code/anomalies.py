import sys, pickle
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
import numpy as np
from common import load
import evaldr2 as E

rows = []
for n, c in E.C.items():
    d = load(n)
    mv = d['sensing__gnss__master__vel']
    tw = c['tw']
    vf = c['vf'] / 3.6
    vr = c['vr'] / 3.6
    gs = np.interp(tw, mv[:, 1], np.hypot(mv[:, 2], mv[:, 3]))
    dt = np.diff(tw)
    gaps = np.sum(dt > 0.5)
    maxgap = dt.max()
    both = 0.5 * (vf + vr)
    zero_move = np.mean((both < 0.05) & (gs > 1.0))
    fr_dis = np.mean(np.abs(vf - vr) > 1.0)
    over = np.mean(both - gs > 1.0)   # wheel faster than ground (slip)
    under = np.mean(gs - both > 1.0)  # wheel slower (slide) / dropout
    r = E.run(n, 1, 0, 0)
    ra = E.run(n, 0, 0, 0)
    rows.append((n, gaps, maxgap, 100 * zero_move, 100 * fr_dis, 100 * over, 100 * under, ra['ea_rmse'], ra['ea_max'], r['ea_rmse'], r['ea_max'], r['endpct']))
rows.sort(key=lambda x: -x[10])
print('run             gaps>0.5s maxgap | %zero&moving %front!=rear %wheel>gnss+1 %wheel<gnss-1 | A:RMSE max | B:RMSE max end%')
for x in rows[:15]:
    print('%s %4d %6.2f | %5.2f %5.2f %5.2f %5.2f | %6.2f %6.2f | %6.2f %6.2f %.3f' % x)
print('...')
for x in rows[-5:]:
    print('%s %4d %6.2f | %5.2f %5.2f %5.2f %5.2f | %6.2f %6.2f | %6.2f %6.2f %.3f' % x)
