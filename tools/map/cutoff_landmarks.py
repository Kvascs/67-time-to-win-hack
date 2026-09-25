"""Traction cut-off landmarks: places where drivers abruptly cut traction (notch >= +4 -> 0 in one
50 ms step), e.g. before overhead-line section insulators. Found from TRAIN bags only and the
train-built map (honest validation); exported for the estimator as
ros2_ws/src/tram_backup_odometry/maps/cutoffs.csv  (s,sigma,n on the main cycle).
Method follows analysis/cross_check/check_cutoffs.py (RTK master fix at the cut-off stamp,
projected on the main edge with heading gate, clustered in s).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis' / 'cross_check'))
sys.path.insert(0, str(ROOT / 'analysis' / 'map_build'))
from xc_common import load, glitch_free, mono  # noqa: E402
from track_map import TrackMap, geodetic_to_enu  # noqa: E402

ORIGIN = (55.8028, 37.424, 160.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map-dir', default=str(ROOT / 'analysis' / 'map_build' / 'map_train'))
    ap.add_argument('--split', default='train', help='train | all')
    ap.add_argument('--out', default=str(ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps' / 'cutoffs.csv'))
    a = ap.parse_args()
    splits = json.loads((ROOT / 'data' / 'splits.json').read_text())
    bags = splits['train'] if a.split == 'train' else splits['train'] + splits['val']
    main_edge = TrackMap(a.map_dir).edges['main']
    ev = []
    for b in bags:
        d = load(b)
        c, fx, vm = d['cmd'], d['fixm'], d['velm']
        g = glitch_free(c)
        tc, u = mono(c[g, 1], c[g, 2])
        jj = np.flatnonzero((u[:-1] >= 4) & (u[1:] == 0) & (np.diff(tc) < 0.08))
        if not len(jj):
            continue
        gf = glitch_free(fx) & (fx[:, 5] == 2)
        if gf.sum() < 100:
            continue
        tf, la, lo, al = mono(fx[gf, 1], fx[gf, 2], fx[gf, 3], fx[gf, 4])
        gv = glitch_free(vm)
        tv, vx, vy = mono(vm[gv, 1], vm[gv, 2], vm[gv, 3])
        for j in jj:
            t = tc[j + 1]
            k = np.searchsorted(tf, t)
            if k <= 0 or k >= len(tf) or tf[k] - tf[k - 1] > 0.25:
                continue
            w = (t - tf[k - 1]) / (tf[k] - tf[k - 1])
            x, y, _ = geodetic_to_enu(la[k - 1] + w * (la[k] - la[k - 1]), lo[k - 1] + w * (lo[k] - lo[k - 1]), al[k], ORIGIN)
            ve, vn = np.interp(t, tv, vx), np.interp(t, tv, vy)
            s, _, _, _ = main_edge.project(np.atleast_1d(x), np.atleast_1d(y),
                                           heading=np.atleast_1d(np.arctan2(vn, ve)), max_d=5)
            if np.isfinite(s[0]):
                ev.append(float(s[0]))
    ss = np.sort(np.array(ev))
    br = np.flatnonzero(np.diff(ss) > 20)
    starts, ends = np.r_[0, br + 1], np.r_[br, len(ss) - 1]
    rows = []
    for a0, b0 in zip(starts, ends):
        grp = ss[a0:b0 + 1]
        if len(grp) < 3:
            continue
        med = float(np.median(grp))
        mad = float(1.4826 * np.median(np.abs(grp - med)))
        rows.append((med, max(mad, 0.5), len(grp)))
    out = Path(a.out)
    with open(out, 'w', newline='\n') as fh:
        fh.write(f'# traction cut-off landmarks (notch >= 4 -> 0 in one step), {a.split} bags, main cycle s\n')
        fh.write('s,sigma,n\n')
        for med, sig, n in rows:
            fh.write(f'{med:.3f},{sig:.3f},{n}\n')
    print(f'{len(ev)} events -> {len(rows)} clusters: ' + ', '.join(f's={m:.1f} (sd {sd:.2f}, n={n})' for m, sd, n in rows))


if __name__ == '__main__':
    main()
