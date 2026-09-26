"""Geometry of the west_arrival_2 stub vs the main line around main s ~ 5300..5800."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(r'C:\MosTransHack')
HERE = Path(__file__).parent
MAPS = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps'
STUB = ROOT / 'analysis' / 'map_build' / 'map' / 'edge_west_arrival_2.csv'

m = pd.read_csv(MAPS / 'track_map.csv', comment='#')
st = pd.read_csv(STUB)


def seg_dist(px, py, x, y):
    """Distance from points (px,py) to polyline (x,y); returns dist, arc index float."""
    ax, ay = x[:-1], y[:-1]
    bx, by = x[1:], y[1:]
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    out_d = np.empty(len(px))
    out_i = np.empty(len(px))
    for j, (qx, qy) in enumerate(zip(px, py)):
        t = np.clip(((qx - ax) * dx + (qy - ay) * dy) / l2, 0, 1)
        cx, cy = ax + t * dx, ay + t * dy
        dd = np.hypot(qx - cx, qy - cy)
        k = np.argmin(dd)
        out_d[j] = dd[k]
        out_i[j] = k + t[k]
    return out_d, out_i


sel = (m.s > 5300) & (m.s < 5540)
mo = m[sel].reset_index(drop=True)
d_out, i_out = seg_dist(st.x.values, st.y.values, mo.x.values, mo.y.values)
s_out = np.interp(i_out, np.arange(len(mo)), mo.s.values)
sel2 = (m.s > 5600) & (m.s < 5800)
mr = m[sel2].reset_index(drop=True)
d_ret, i_ret = seg_dist(st.x.values, st.y.values, mr.x.values, mr.y.values)
s_ret = np.interp(i_ret, np.arange(len(mr)), mr.s.values)
print('stub_s  d_outbound  s_outbound  d_return  s_return  k_stub')
for j in range(0, 80, 2):
    print(f'{st.s[j]:6.2f} {d_out[j]:8.2f} {s_out[j]:9.1f} {d_ret[j]:8.2f} {s_ret[j]:9.1f} {st.curvature[j]:+.4f}')
for j in range(80, len(st), 10):
    print(f'{st.s[j]:6.2f} {d_out[j]:8.2f} {s_out[j]:9.1f} {d_ret[j]:8.2f} {s_ret[j]:9.1f} {st.curvature[j]:+.4f}')

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
w = (m.s > 5250) & (m.s < 5850)
ax[0].plot(m.x[w], m.y[w], 'k.-', ms=2, lw=0.5, label='main')
for sv in range(5250, 5851, 50):
    j = np.argmin(np.abs(m.s - sv))
    ax[0].annotate(f'{sv}', (m.x[j], m.y[j]), fontsize=7)
ax[0].plot(st.x, st.y, 'r.-', ms=2, lw=0.5, label='stub')
for sv in range(0, 130, 20):
    j = np.argmin(np.abs(st.s - sv))
    ax[0].annotate(f'st{sv}', (st.x[j], st.y[j]), fontsize=7, color='r')
ax[0].axis('equal'); ax[0].legend(); ax[0].grid()
ax[1].plot(m.s[w], m.curvature[w], 'k-', label='main kappa')
ax[1].plot(st.s + 5395.1, st.curvature, 'r-', label='stub kappa (s shifted to switch 5395.1)')
ax[1].grid(); ax[1].legend()
plt.tight_layout()
plt.savefig(HERE / 'fig_stub_geometry.png', dpi=80)
