"""Locate the front bogie (base_link) relative to antenna 1 (master) from body orientation.

A rigid car body rests on two bogies 7.55 m apart (organisers). In curves the body axis is the
chord between the bogie centres on the track. The antenna baseline (antenna 2 - antenna 1)
is parallel to the body axis, so its measured heading must equal the heading of the chord
between track points s_f = s1 + D and s_r = s1 + D - 7.55, where s1 is antenna 1's arc length.
Scan D and minimise the heading residual in curves. Also reports the lateral offset of antenna 2
from antenna 1's path and the implied antenna positions relative to the bogies.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\MosTransHack')
sys.path.insert(0, str(ROOT / 'analysis' / 'map_build'))
from track_map import TrackMap, geodetic_to_enu  # noqa: E402

ORIGIN = (55.8028, 37.424, 160.0)
BASE = 7.55
M = TrackMap(str(ROOT / 'analysis' / 'map_build' / 'map_train')).edges['main']
grid = np.arange(0.0, M.length, 0.25)
P = np.array([M.pose(s)[:2] for s in grid])      # x, y along the map path
K = np.abs(np.array([M.curvature(s) for s in grid]))


def pt(s):
    s = np.mod(s, M.length)
    return np.stack([np.interp(s, grid, P[:, 0]), np.interp(s, grid, P[:, 1])], -1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


splits = json.loads((ROOT / 'data' / 'splits.json').read_text())
S1, PSI = [], []
lat_off = []
for b in splits['train'] + splits['val']:
    d = np.load(ROOT / 'data' / 'npz' / f'{b}.npz')
    a1, a2 = d['sensing__gnss__master__fix'], d['sensing__gnss__rover__fix']
    if len(a1) < 1000 or len(a2) < 1000:
        continue
    a1 = a1[a1[:, 5] == 2]
    a2 = a2[a2[:, 5] == 2]
    if len(a1) < 1000 or len(a2) < 1000:
        continue
    j = np.searchsorted(a2[:, 1], a1[:, 1])
    j = np.clip(j, 0, len(a2) - 1)
    same = np.abs(a2[j, 1] - a1[:, 1]) < 0.005
    a1, a2 = a1[same], a2[j[same]]
    x1, y1, _ = geodetic_to_enu(a1[:, 2], a1[:, 3], a1[:, 4], ORIGIN)
    x2, y2, _ = geodetic_to_enu(a2[:, 2], a2[:, 3], a2[:, 4], ORIGIN)
    L = np.hypot(x2 - x1, y2 - y1)
    ok = np.abs(L - 12.44) < 0.3
    psi = np.arctan2(y2 - y1, x2 - x1)
    s1, dist, _, _ = M.project(x1, y1, heading=psi, max_d=1.0)
    ok &= np.isfinite(s1)
    S1.append(s1[ok])
    PSI.append(psi[ok])
    # lateral offset of antenna 2 w.r.t. antenna 1's path (map)
    s2, d2, _, _ = M.project(x2[ok], y2[ok], heading=psi[ok], max_d=3.0)
    lat_off.append(np.c_[s1[ok], d2])
S1 = np.concatenate(S1)
PSI = np.concatenate(PSI)
curvy = np.interp(np.mod(S1, M.length), grid, K) > 0.01
print(f'samples {len(S1)}, in curves (|k|>0.01): {curvy.sum()}')
res = []
for D in np.arange(-20.0, 20.01, 0.25):
    pf, pr = pt(S1 + D), pt(S1 + D - BASE)
    chord = np.arctan2(pf[:, 1] - pr[:, 1], pf[:, 0] - pr[:, 0])
    r = wrap(PSI - chord)[curvy]
    res.append((float(np.sqrt(np.mean(np.degrees(r) ** 2))), D, float(np.degrees(np.median(r)))))
res.sort()
print('best D (front bogie ahead of antenna 1): ' + ', '.join(f'D={D:+.2f} m rms={rms:.2f} deg bias={bias:+.2f}' for rms, D, bias in res[:5]))
full = sorted(res, key=lambda r: r[1])
print('rms by D: ' + ' '.join(f'{D:+.0f}:{rms:.2f}' for rms, D, _ in full[::8]))
lo = np.concatenate(lat_off)
cz = np.interp(np.mod(lo[:, 0], M.length), grid, K) > 0.02
print(f'antenna 2 distance from antenna-1 path: median {np.nanmedian(np.abs(lo[:, 1])):.3f} m, '
      f'in tight curves (|k|>0.02) median {np.nanmedian(np.abs(lo[cz, 1])):.3f} m p90 {np.nanpercentile(np.abs(lo[cz, 1]), 90):.3f} m')
