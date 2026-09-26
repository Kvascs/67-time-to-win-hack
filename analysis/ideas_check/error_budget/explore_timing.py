"""Scratch exploration: timing of the judge reference position on the check bag.

(1) reference kinematic_state base_link vs the RTK base_link built from both antennas (eval_base_link TF):
    along difference regressed on speed -> lag of the reference position relative to GNSS stamps;
(2) our along error regressed on speed with a nuisance drift (piecewise linear in path, reset at fixes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge as CB  # noqa: E402
import eval_base_link as EB  # noqa: E402

BAG = '30618_88aea4d9'
d = np.load(CB.NPZ / f'{BAG}.npz')
ks = d['localization__kinematic_state']
t0 = ks[0, 1]
ref = pd.read_parquet(HERE / 'ref_check.parquet')

# (1) GNSS RTK base_link vs kinematic_state
g = EB.reference(BAG)
print('RTK epochs with both antennas:', 0 if g is None else len(g))
if g is not None:
    gt = g.t.to_numpy() - t0
    rx = np.interp(gt, ref.t, ref.x)
    ry = np.interp(gt, ref.t, ref.y)
    rz = np.interp(gt, ref.t, ref.z)
    rv = np.interp(gt, ref.t, ref.vx)
    yaw = np.interp(gt, ref.t, np.unwrap(ref.yaw))
    dx, dy = g.x.to_numpy() - rx, g.y.to_numpy() - ry
    al = dx * np.cos(yaw) + dy * np.sin(yaw)
    cr = -dx * np.sin(yaw) + dy * np.cos(yaw)
    dz = g.z.to_numpy() - rz
    m = np.abs(cr) < 1.0
    A = np.c_[np.ones(m.sum()), rv[m]]
    c, *_ = np.linalg.lstsq(A, al[m], rcond=None)
    print(f'GNSS - ref: along = {c[0]:+.3f} + {c[1]:+.4f} * v  (n={m.sum()}), cross median {np.median(cr):+.3f}, '
          f'dz median {np.median(dz):+.3f}')
    # GNSS stamp resolution: fixes are stamped to 0.1 s
    print('GNSS stamp fractional parts:', np.unique(np.round((g.t.to_numpy() * 10) % 1, 2))[:10])
    for lo, hi in ((0, 0.5), (0.5, 3), (3, 6), (6, 9), (9, 12), (12, 20)):
        k = m & (rv >= lo) & (rv < hi)
        if k.sum():
            print(f'  v {lo:4.1f}-{hi:4.1f}: n={k.sum():3d} along GNSS-ref mean {al[k].mean():+.3f} median {np.median(al[k]):+.3f}')

# (2) our along error vs speed with nuisance drift
P = pd.read_parquet(HERE / 'pairs_pos.parquet')
P = P[P.t < 1270].reset_index(drop=True)
lm = pd.read_csv(HERE / 'lm_check.csv')
acc = lm[(lm.p_known >= 0.6) & (lm.n > 0)].t.to_numpy()
seg = np.searchsorted(acc, P.t.to_numpy())
for knot in (50.0, 100.0, 200.0, 400.0, 1e9):
    cols = [P.v_ref.to_numpy()]
    for sg in np.unique(seg):
        msk = seg == sg
        p = P.path.to_numpy()
        p0 = p[msk].min()
        cols.append(msk.astype(float))
        kn = np.arange(p0, p[msk].max() + knot, knot) if knot < 1e8 else np.array([p0])
        for kk in kn:
            cols.append(np.where(msk, np.maximum(p - kk, 0.0), 0.0))
    X = np.column_stack(cols)
    X = X[:, np.abs(X).sum(0) > 0]
    y = P.along.to_numpy()
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ c
    print(f'knots {knot:6.0f} m: tau = {c[0]:+.4f} s  (cols {X.shape[1]}), resid RMSE {np.sqrt(np.mean(res**2)):.3f}')

# (3) windows around stops: along at standstill vs along while moving just before / after
st = (P.v_ref < 0.02).to_numpy()
print('pair dt ms: mean %.1f  p5 %.1f p95 %.1f' % (P.dt_ms.mean(), np.percentile(P.dt_ms, 5), np.percentile(P.dt_ms, 95)))
print('along explained by pair offset v*dt: RMSE %.4f' % np.sqrt(np.mean((P.v_ref * P.dt_ms / 1e3) ** 2)))
for delta in (-0.2, -0.15, -0.1, -0.05, 0.0, 0.05):
    y2 = P.along - P.v_ref * delta
    print(f'  shift our position by {delta:+.2f} s (along - v*delta): along RMSE {np.sqrt(np.mean(y2**2)):.4f}')
