"""Shared helpers for the curvature-signature check (bogie speed ratio vs track curvature).

Map: shipped main-cycle map (path of the MASTER antenna), local ENU of the map header origin.
Front bogie pivot is 9.873 m ahead of the master antenna along the car, rear pivot 2.323 m ahead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(r'C:\MosTransHack')
HERE = Path(__file__).resolve().parent
NPZ = ROOT / 'data' / 'npz'
MAPS = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps'
MAP_CSV = MAPS / 'track_map.csv'
STUB_CSV = ROOT / 'analysis' / 'map_build' / 'map' / 'edge_west_arrival_2.csv'
EPISODES = ROOT / 'analysis' / 'wheel_anomalies' / 'episodes_all.csv'
SPLITS = json.load(open(ROOT / 'data' / 'splits.json'))

D_FRONT = 9.873   # front pivot ahead of master antenna [m]
D_REAR = 2.323    # rear pivot ahead of master antenna [m]
KMH = 1 / 3.6
FIX_LEAD_S = 0.0435   # master fix positions lead wheel/vel header stamps by ~43.5 ms (DATA_FINDINGS #1)
BAD_KINDS = {'slip', 'slide', 'lock', 'stuck_zero', 'overspeed/slip', 'frozen', 'negative', 'gap'}

A_WGS, F_WGS = 6378137.0, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)


def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A_WGS / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


def enu(lat, lon, h, lat0, lon0, h0):
    """Same formula as tools/replay/quick_eval.py::enu."""
    d = ecef(lat, lon, h) - ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))
    la, lo = np.radians(lat0), np.radians(lon0)
    r = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return d @ r.T


def map_origin(path=MAP_CSV):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') and 'origin_lat' in line:
                kv = dict(tok.split('=') for tok in line[1:].split() if '=' in tok)
                return float(kv['origin_lat']), float(kv['origin_lon']), float(kv['origin_h'])
    raise ValueError('no origin in map header')


class Polyline:
    """Polyline with arc length, nearest-point projection and curvature/heading lookup."""

    def __init__(self, x, y, s=None, kappa=None, cyclic=False):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        seg = np.hypot(np.diff(self.x), np.diff(self.y))
        self.s = np.r_[0, np.cumsum(seg)] if s is None else np.asarray(s, float)
        self.cyclic = cyclic
        if cyclic:
            gap = np.hypot(self.x[0] - self.x[-1], self.y[0] - self.y[-1])
            self.L = self.s[-1] + gap
            # close the polyline for projection
            self.px = np.r_[self.x, self.x[0]]
            self.py = np.r_[self.y, self.y[0]]
            self.ps = np.r_[self.s, self.L]
        else:
            self.L = self.s[-1]
            self.px, self.py, self.ps = self.x, self.y, self.s
        self.kappa = None if kappa is None else np.asarray(kappa, float)
        self.tree = cKDTree(np.c_[self.px, self.py])
        # unit tangents at vertices (central differences, cyclic aware)
        if cyclic:
            xx = np.r_[self.x[-1], self.x, self.x[0]]
            yy = np.r_[self.y[-1], self.y, self.y[0]]
            tx, ty = xx[2:] - xx[:-2], yy[2:] - yy[:-2]
        else:
            tx, ty = np.gradient(self.x), np.gradient(self.y)
        n = np.hypot(tx, ty)
        self.tx, self.ty = tx / n, ty / n

    def project(self, qx, qy, k=6):
        """Nearest point on the polyline: returns s, signed lateral offset (left +), distance."""
        qx = np.asarray(qx, float)
        qy = np.asarray(qy, float)
        _, idx = self.tree.query(np.c_[qx, qy], k=k)
        nseg = len(self.px) - 1
        best_d = np.full(len(qx), np.inf)
        best_s = np.full(len(qx), np.nan)
        best_lat = np.full(len(qx), np.nan)
        for col in range(idx.shape[1]):
            for off in (-1, 0):
                j = np.clip(idx[:, col] + off, 0, nseg - 1)
                ax, ay = self.px[j], self.py[j]
                dx, dy = self.px[j + 1] - ax, self.py[j + 1] - ay
                l2 = dx * dx + dy * dy
                t = np.clip(((qx - ax) * dx + (qy - ay) * dy) / l2, 0, 1)
                cx, cy = ax + t * dx, ay + t * dy
                d = np.hypot(qx - cx, qy - cy)
                better = d < best_d
                best_d = np.where(better, d, best_d)
                best_s = np.where(better, self.ps[j] + t * (self.ps[j + 1] - self.ps[j]), best_s)
                cross = dx * (qy - ay) - dy * (qx - ax)
                best_lat = np.where(better, np.sign(cross) * d, best_lat)
        return best_s, best_lat, best_d

    def _wrap(self, s):
        return np.mod(s, self.L) if self.cyclic else np.clip(s, self.s[0], self.s[-1])

    def interp(self, arr, s):
        s = self._wrap(np.asarray(s, float))
        if self.cyclic:
            return np.interp(s, np.r_[self.s, self.L], np.r_[arr, arr[0]])
        return np.interp(s, self.s, arr)

    def k_at(self, s):
        return self.interp(self.kappa, s)

    def xy_at(self, s):
        return self.interp(self.x, s), self.interp(self.y, s)

    def t_at(self, s):
        tx, ty = self.interp(self.tx, s), self.interp(self.ty, s)
        n = np.hypot(tx, ty)
        return tx / n, ty / n


def load_main():
    m = pd.read_csv(MAP_CSV, comment='#')
    return Polyline(m.x.values, m.y.values, m.s.values, m.curvature.values, cyclic=True)


def load_stub():
    st = pd.read_csv(STUB_CSV)
    return Polyline(st.x.values, st.y.values, st.s.values, st.curvature.values, cyclic=False), st


def chord_pred(pl: Polyline, s_ant, d_front=D_FRONT, d_rear=D_REAR):
    """Rigid-car kinematic prediction of log(v_front/v_rear).

    Both pivots move along the track tangent; their velocity components along the car axis
    (chord rear->front) must be equal: v_f cos(a_f) = v_r cos(a_r).  Pivot positions are taken on
    the antenna-path map at arc s+d (approximation)."""
    sf, sr = np.asarray(s_ant) + d_front, np.asarray(s_ant) + d_rear
    xf, yf = pl.xy_at(sf)
    xr, yr = pl.xy_at(sr)
    cx, cy = xf - xr, yf - yr
    n = np.hypot(cx, cy)
    cx, cy = cx / n, cy / n
    tfx, tfy = pl.t_at(sf)
    trx, try_ = pl.t_at(sr)
    cf = np.clip(cx * tfx + cy * tfy, 1e-6, 1)
    cr = np.clip(cx * trx + cy * try_, 1e-6, 1)
    return np.log(cr) - np.log(cf)


def load_bag(bag):
    return np.load(NPZ / f'{bag}.npz')


def wheel_pairs(d):
    """Front/rear samples paired on identical header stamps. Returns DataFrame t, recv, vf, vr [m/s]."""
    fr = d['vehicle__front_bogie_velocity']
    rr = d['vehicle__rear_bogie_velocity']
    f = pd.DataFrame({'ts': np.round(fr[:, 1] * 1e6).astype(np.int64), 'recv': fr[:, 0],
                      'kf': fr[:, 2]}).drop_duplicates('ts', keep='last')
    r = pd.DataFrame({'ts': np.round(rr[:, 1] * 1e6).astype(np.int64), 'kr': rr[:, 2]}).drop_duplicates('ts', keep='last')
    w = f.merge(r, on='ts', how='inner').sort_values('ts').reset_index(drop=True)
    w['t'] = w.ts / 1e6
    w['vf'] = w.kf * KMH
    w['vr'] = w.kr * KMH
    return w[['t', 'recv', 'vf', 'vr']]


def bad_episode_mask(bag, d, recv):
    """True where a known wheel anomaly episode (bag time +-5 s) covers the sample."""
    ep = pd.read_csv(EPISODES)
    ep = ep[(ep.bag == bag) & ep.kind.isin(BAD_KINDS)]
    out = np.zeros(len(recv), bool)
    if ep.empty:
        return out
    t0 = min(d[k][0, 0] for k in ('vehicle__front_bogie_velocity', 'vehicle__rear_bogie_velocity',
                                  'vehicle__driver_position_cmd') if len(d[k]))
    tb = recv - t0
    for _, e in ep.iterrows():
        out |= (tb >= e.t_start - 5.0) & (tb <= e.t_start + e.dur + 5.0)
    return out


def master_fix_enu(d, origin):
    g = d['sensing__gnss__master__fix']
    g = g[np.argsort(g[:, 1], kind='stable')]
    e = enu(g[:, 2], g[:, 3], g[:, 4], *origin)
    return pd.DataFrame({'t': g[:, 1], 'x': e[:, 0], 'y': e[:, 1], 'status': g[:, 5]})


def gnss_bags():
    return [(b, 'train') for b in SPLITS['train']] + [(b, 'val') for b in SPLITS['val']]
