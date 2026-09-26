"""Per-bag diagnostic figure: speed, speed error with regimes, along/cross-track error, map."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .loader import BagData, T_BAG, V_COL
from .reference import Reference
from . import metrics as M


def plot_bag(res: dict, path: str, title: str = None, tol: float = 0.05, direction: str = 'ref2out'):
    """``res`` is an evaluate_bag(..., keep_log=True) result (needs _log/_ref/_bag)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log: M.OutputLog = res['_log']
    ref: Reference = res['_ref']
    bag: BagData = res['_bag']
    t0 = ref.t_vel[0]
    fig, ax = plt.subplots(4, 1, figsize=(16, 15), gridspec_kw={'height_ratios': [1.2, 1, 1, 1.6]})

    # 1. speed
    a = ax[0]
    a.plot(ref.t_vel - t0, ref.v, color='k', lw=1.2, label='GNSS ref |v|')
    o = np.argsort(log.stamp)
    a.plot(log.stamp[o] - t0, log.v[o], color='tab:red', lw=0.9, label='estimate')
    for name, c in (('front', 'tab:blue'), ('rear', 'tab:cyan')):
        w = bag[name]
        if len(w):
            a.plot(w[:, 1] - t0, w[:, V_COL] / 3.6, color=c, lw=0.5, alpha=0.6, label=f'{name}/3.6')
    anom = M.anomaly_regimes(ref, bag)
    _shade(a, ref.t_vel - t0, anom['anomaly'], 'orange', 0.25)
    a.set_ylabel('v [m/s]')
    a.legend(loc='upper right', fontsize=8)
    a.grid(alpha=0.3)
    s = res['summary']
    a.set_title(title or f"{res['bag']}  v_rmse={s['v_rmse']:.3f} m/s  bias={s['v_bias']:+.3f}  "
                f"along_rmse={s['along_rmse']:.1f} m  drift={s['drift_pct_3d']:.2f}%  match={s['match_v']:.2f}")

    # 2. speed error with regimes
    a = ax[1]
    valid = np.isfinite(log.v) & np.isfinite(log.stamp)
    iref, iout, _ = M.pair(ref.t_vel, log.stamp, valid, tol, direction)
    e = log.v[iout] - ref.v[iref]
    reg = M.speed_regimes(ref, bag)
    tt = ref.t_vel - t0
    _shade(a, tt, reg['accel'], 'tab:green', 0.12)
    _shade(a, tt, reg['brake'], 'tab:red', 0.12)
    _shade(a, tt, reg['stopped'], 'gray', 0.12)
    a.plot(tt[iref], e, '.', ms=1.5, color='tab:purple')
    a.axhline(0, color='k', lw=0.5)
    lim = max(0.3, np.nanpercentile(np.abs(e), 99.5) * 1.2) if len(e) else 1
    a.set_ylim(-lim, lim)
    a.set_ylabel('v err [m/s]\n(green acc, red brake, grey stop)')
    a.grid(alpha=0.3)

    # 3. along / cross track
    a = ax[2]
    vp = np.all(np.isfinite(log.xyz), axis=1) & np.isfinite(log.stamp)
    ir, io, _ = M.pair(ref.t_pos, log.stamp, vp, tol, direction)
    if len(ir):
        pe = log.xyz[io]
        e2 = np.hypot(*(pe[:, :2] - ref.xyz[ir, :2]).T)
        win = np.clip(20 + 1.6 * e2, 20, 2000)
        s_est, lat, _ = ref.path.project(pe[:, :2], s_hint=ref.s_pos[ir], window=win)
        tp = ref.t_pos[ir] - t0
        a.plot(tp, s_est - ref.s_pos[ir], lw=0.8, label='along-track (arc) [m]')
        a.plot(tp, lat, lw=0.8, label='cross-track vs path [m]')
        a.plot(tp, pe[:, 2] - ref.xyz[ir, 2], lw=0.8, label='z err [m]')
        a.plot(tp, e2, lw=0.6, color='gray', alpha=0.7, label='|e2d| [m]')
        out = ref.outlier[ir]
        if np.any(out):
            a.plot(tp[out], np.zeros(out.sum()), '|', color='red', ms=6, label='ref outlier')
    a.axhline(0, color='k', lw=0.5)
    a.set_ylabel('position err [m]')
    a.set_xlabel('t [s]')
    a.legend(loc='upper left', fontsize=8)
    a.grid(alpha=0.3)

    # 4. map
    a = ax[3]
    a.plot(ref.xyz[:, 0], ref.xyz[:, 1], '.', ms=1, color='k', label='GNSS ref')
    if np.any(ref.outlier):
        a.plot(ref.xyz[ref.outlier, 0], ref.xyz[ref.outlier, 1], 'x', ms=3, color='red', label='ref outliers')
    a.plot(log.xyz[vp, 0], log.xyz[vp, 1], lw=0.8, color='tab:red', label='estimate')
    a.set_aspect('equal', adjustable='datalim')
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a.set_xlabel(f'x [m] ({ref.frame.kind})')
    a.set_ylabel('y [m]')
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=80)
    plt.close(fig)


def _shade(ax, t, mask, color, alpha):
    if not np.any(mask):
        return
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        ax.axvspan(t[s], t[e - 1], color=color, alpha=alpha, lw=0)
