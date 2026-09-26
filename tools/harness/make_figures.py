"""Diagnostic figures of timing, reference quality and frames (python -m harness.make_figures) -> harness/figures/*.png"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from harness.loader import load_bag, T_BAG, T_HDR, LAT, LON, ALT, VX, VY
from harness.reference import LocalFrame, RefConfig, build_reference, select_origin

FIG = Path(__file__).resolve().parent / 'figures'


def fig_timing(bag_name='30639_3b3d9eb8'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bag = load_bag(bag_name)
    t0 = bag.t_start
    fig, ax = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    for name, c in (('front', 'tab:blue'), ('cmd', 'tab:green'), ('fix_master', 'tab:red'), ('vel_master', 'tab:orange')):
        a = bag[name]
        off = (a[:, T_HDR] - a[:, T_BAG]) * 1e3
        ax[0].plot(a[:, T_BAG] - t0, off, '.', ms=1, color=c, label=name)
    ax[0].set_ylim(-150, 30)
    ax[0].set_ylabel('header.stamp - bag time [ms]')
    ax[0].legend(markerscale=8, fontsize=8, loc='lower right')
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f'{bag_name}: per-topic latency (zoom); wheel latency drifts/saw-tooths 10..90 ms, GNSS is stable')
    for name, c in (('fix_master', 'tab:red'), ('vel_master', 'tab:orange'), ('front', 'tab:blue')):
        a = bag[name]
        ax[1].plot(a[:, T_BAG] - t0, a[:, T_HDR] - a[:, T_BAG], '.', ms=1, color=c, label=name)
    ax[1].set_ylabel('header - bag [s]')
    ax[1].set_xlabel('bag time from start [s]')
    ax[1].set_title('full scale: GNSS header stamps jump by exactly +1 s for minutes; bag-start burst ~ -2.5..-4 s')
    ax[1].legend(markerscale=8, fontsize=8)
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / 'timing_offsets.png', dpi=80)
    plt.close(fig)


def fig_ref_quality(bag_name='30618_defd0170'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bag = load_bag(bag_name)
    ref = build_reference(bag, RefConfig(time_base='bag'))
    t = ref.tbag_pos - bag.t_start
    fig, ax = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    ax[0].plot(t, ref.xyz[:, 2], '.', ms=1, color='k', label='master fix z (ENU)')
    ax[0].plot(t[ref.outlier], ref.xyz[ref.outlier, 2], '.', ms=2, color='red', label='flagged outlier')
    ax[0].set_ylabel('z [m]')
    ax[0].legend(markerscale=6, fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f'{bag_name}: two interleaved GNSS solutions (~15 m / 11 m apart) in the first ~150 s')
    d = np.hypot(*(ref.xyz[:, :2] - ref.path.interp(ref.s_pos)[0]).T)
    ax[1].plot(t, d, '.', ms=1, color='k')
    ax[1].plot(t[ref.outlier], d[ref.outlier], '.', ms=2, color='red')
    ax[1].set_ylabel('distance of fix to cleaned path [m]')
    ax[1].set_xlabel('bag time from start [s]')
    ax[1].set_yscale('symlog', linthresh=1)
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / 'ref_quality_defd0170.png', dpi=80)
    plt.close(fig)


def fig_frames(bag_name='30618_e3d94878'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bag = load_bag(bag_name)
    f = bag['fix_master']
    oi = select_origin(f)
    enu = LocalFrame('enu', *f[oi, [LAT, LON, ALT]])
    utm = LocalFrame('utm', *f[oi, [LAT, LON, ALT]])
    a = np.stack(enu.forward(f[:, LAT], f[:, LON], f[:, ALT]), 1)
    b = np.stack(utm.forward(f[:, LAT], f[:, LON], f[:, ALT]), 1)
    d = np.hypot(*(a[:, :2] - b[:, :2]).T)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ax[0].plot(a[:, 0], a[:, 1], lw=1, label='ENU (origin = first master fix)')
    ax[0].plot(b[:, 0], b[:, 1], lw=1, label='UTM 37N minus origin')
    ax[0].set_aspect('equal', adjustable='datalim')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f'{bag_name}: same fixes, two "local metric" frames')
    r = np.hypot(a[:, 0], a[:, 1])
    ax[1].plot(r, d, '.', ms=1)
    ax[1].set_xlabel('distance from origin [m]')
    ax[1].set_ylabel('|ENU - UTM_rel| [m]')
    ax[1].set_title(f'frame mix-up error: {d.max():.0f} m at {r.max():.0f} m (grid convergence '
                    f'{np.degrees(utm.enu_rotation()):.2f} deg, scale 0.9997)')
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / 'frames_enu_vs_utm.png', dpi=80)
    plt.close(fig)


if __name__ == '__main__':
    FIG.mkdir(parents=True, exist_ok=True)
    fig_timing()
    fig_ref_quality()
    fig_frames()
    print('written to', FIG)
