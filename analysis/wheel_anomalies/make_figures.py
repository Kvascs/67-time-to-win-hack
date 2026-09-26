"""Figures for REPORT.md (scale factors, within-bag constancy, episode gallery, timing anomalies)."""
import datetime
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import anomalies as A  # noqa: E402
from build_aligned import load_aligned  # noqa: E402
from plot_episodes import plot_window  # noqa: E402


def fig_scale():
    sc = pd.read_csv(HERE / 'scale_factors.csv')
    sc = sc[sc.dur_s > 300].copy()
    sc['hour'] = [datetime.datetime.fromtimestamp(t, datetime.timezone.utc).hour +
                  datetime.datetime.fromtimestamp(t, datetime.timezone.utc).minute / 60 for t in sc.t0_utc]
    seg = pd.read_csv(HERE / 'segment_ratios.csv')
    sp = pd.read_csv(HERE / 'ratio_vs_speed.csv')
    fig, axs = plt.subplots(2, 2, figsize=(18, 11))
    ax = axs[0, 0]
    groups = sc.groupby(['vehicle', 'date'])
    cols = plt.cm.tab10(np.arange(10))
    for i, ((veh, date), g) in enumerate(groups):
        ax.plot(g.hour, g.k_speed_front, 'o-', color=cols[i], label=f'{veh} {date} (n={len(g)})')
        ax.plot(g.hour, g.k_dist_all_front, 'x', color=cols[i], alpha=0.6)
    ax.axhline(3.6, color='k', lw=0.8, ls='--')
    ax.set_xlabel('UTC hour of bag start'); ax.set_ylabel('k = wheel[km/h] / true[m/s]')
    ax.set_title('Per-bag scale factor (o: straight-track speed ratio, x: overall distance ratio), front sensor')
    ax.legend(fontsize=8); ax.grid()
    ax2 = ax.secondary_yaxis('right', functions=(lambda k: (k / 3.6 - 1) * 100, lambda e: 3.6 * (1 + e / 100)))
    ax2.set_ylabel('wheel over-read vs 3.6 [%]')
    ax = axs[0, 1]
    seg = seg.merge(sc[['bag', 'k_speed_front']], on='bag')
    seg['rel'] = 100 * (seg.k_front / seg.k_speed_front - 1)
    order = sc.sort_values('t0_utc').bag.tolist()
    data = [seg[seg.bag == b].rel.values for b in order]
    ax.boxplot(data, showfliers=True, flierprops=dict(ms=2))
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([b[:5] + '_' + b[6:9] for b in order], rotation=90, fontsize=6)
    ax.set_ylabel('segment distance ratio / bag k - 1 [%]'); ax.grid(axis='y')
    ax.set_title('Within-bag constancy: ratio per inter-stop segment (curves included)')
    ax.set_ylim(-2, 1.5)
    ax = axs[1, 0]
    for (veh, s), g in sp.groupby(['vehicle', 'sensor']):
        m = g.groupby('v_lo').k_rel.median()
        ax.plot(m.index + 0.5, 100 * (m.values - 1), 'o-', label=f'{veh} {s}')
    ax.set_xlabel('true speed [m/s]'); ax.set_ylabel('ratio / bag k - 1 [%]'); ax.grid(); ax.legend()
    ax.set_title('Speed dependence on STRAIGHT track (|kappa|<0.003), steady |a|<0.25: none beyond +-0.1 %')
    ax.set_ylim(-1, 1)
    ax = axs[1, 1]
    cv = pd.read_csv(HERE / 'curvature_ratio.csv')
    for s, c in (('front', 'b'), ('rear', 'c')):
        d = cv[cv.sensor == s]
        ax.plot(d.kappa_mean, 100 * d.e_rel_median, c + 'o-', label=s)
    kk = np.linspace(-0.06, 0.06, 200)
    ax.plot(kk, -100 * np.minimum(0.5 * np.abs(kk), 0.0095), 'k--', label='model -min(0.5|kappa|, 0.95 %)')
    ax.set_xlabel('signed path curvature kappa [1/m]'); ax.set_ylabel('(wheel/k - gnss)/gnss [%]'); ax.grid(); ax.legend()
    ax.set_title('Curve effect: wheels read ~0.9-1.1 % below GNSS for R < 60 m (symmetric)')
    plt.tight_layout(); plt.savefig(HERE / 'fig_scale_factors.png', dpi=75)


def fig_gallery():
    specs = [
        ('30618_33bec73f', 96, 112, 'SLIP (traction, rear+front, anti-slip cycles) 33bec73f'),
        ('30639_50956d6e', 72, 99, 'SLIP rear @77 s (n=11), front @95 s (n=14) 50956d6e'),
        ('30618_2050d396', 392, 446, 'SLIDE (braking) rear @397/400 s, both @441 s 2050d396'),
        ('30618_68d1748a', 986, 997, 'SLIDE both bogies, different depth, 68d1748a'),
        ('30639_50956d6e', 1020, 1044, 'slip @1026 then LOCK to 0 before stop @1037 50956d6e'),
        ('30639_3b3d9eb8', 360, 500, 'rear FROZEN->silent 30 s, STUCK_ZERO at start->silent 73 s 3b3d9eb8'),
        ('30618_87afe526', 528, 552, 'handle parked at -8 while accelerating 0->11 m/s (not a wheel fault)'),
        ('30618_4d487b0d', 92, 132, 'GNSS master outputs exact zeros while moving (reference fault)'),
    ]
    sc = pd.read_csv(HERE / 'scale_factors.csv').set_index('bag')
    fig, axs = plt.subplots(4, 2, figsize=(20, 17))
    for ax, (b, a0, a1, t) in zip(axs.flat, specs):
        plot_window(ax, b, a0, a1, sc, t)
    plt.tight_layout(); plt.savefig(HERE / 'fig_episode_gallery.png', dpi=65)


def fig_timing():
    fig, axs = plt.subplots(2, 2, figsize=(18, 9))
    # recorder clock step: header vs bag
    for ax, (b, a0, a1, title) in zip(axs[0], [('30618_2255aade', 193, 202, 'Recorder (bag) clock step -1 s: 2255aade'),
                                               ('30618_40ffd323', 80, 89, 'Recorder clock +1 s / -1 s: 40ffd323')]):
        bg = A.load_bag(b)
        for s, c, nm in ((bg.front, 'b', 'front'), (bg.cmd, 'g', 'cmd')):
            m = (s.t_bag - bg.t0 > a0) & (s.t_bag - bg.t0 < a1)
            ax.plot(s.t_bag[m] - bg.t0, s.t_hdr[m] - bg.t0, '.', color=c, ms=3, label=nm)
        ax.plot([a0, a1], [a0, a1], 'k-', lw=0.5)
        ax.set_xlabel('bag (arrival) time [s]'); ax.set_ylabel('header stamp [s]'); ax.grid(); ax.legend(); ax.set_title(title)
    for ax, b in zip(axs[1], ['30618_40ffd323', '30639_3b3d9eb8']):
        al = load_aligned(b)
        tc, lw = al.lag_windows
        ax.plot(tc - al.t0, lw, 'r.', ms=3, label='data-driven window lag (master)')
        ax.plot(al.t - al.t0, al.lags['master'], 'k-', lw=1, label='latency model lag (master)')
        ax.plot(al.t - al.t0, al.lags['rover'], 'm--', lw=0.8, label='latency model lag (rover)')
        ax.set_xlabel('t [s]'); ax.set_ylabel('GNSS header lag vs wheel header [s]'); ax.grid(); ax.legend()
        ax.set_title(f'GNSS header-clock excursions (+-1 s) {b}')
    plt.tight_layout(); plt.savefig(HERE / 'fig_timing_anomalies.png', dpi=75)


def fig_monitor():
    specs = [('30618_33bec73f', 96, 112), ('30618_2050d396', 393, 404), ('30618_68d1748a', 986, 997),
             ('30639_50956d6e', 1020, 1044), ('30639_3b3d9eb8', 410, 425), ('30618_616ec56b', 1012, 1024)]
    sc = pd.read_csv(HERE / 'scale_factors.csv').set_index('bag')
    fig, axs = plt.subplots(3, 2, figsize=(20, 13))
    for ax, (b, a0, a1) in zip(axs.flat, specs):
        bg = A.load_bag(b)
        p = A.MonitorParams(k_front=sc.loc[b, 'k_speed_front'], k_rear=sc.loc[b, 'k_speed_rear'])
        rows, _ = A.replay_monitor(bg, p)
        t0 = bg.t0
        al = load_aligned(b)
        m = (al.t - t0 >= a0) & (al.t - t0 <= a1)
        ax.plot(al.t[m] - t0, al.ref[m], 'k-', lw=2, alpha=0.5, label='GNSS reference')
        for s, c in (('front', 'b'), ('rear', 'c')):
            rr = [r for r in rows if r[0] == s and a0 <= r[2] - t0 <= a1]
            ax.plot([r[2] - t0 for r in rr], [r[3] for r in rr], '.-', color=c, ms=3, lw=0.6, label=f'{s} raw')
            bad = [r for r in rr if r[4] not in ('ok', 'stale')]
            ax.plot([r[2] - t0 for r in bad], [r[3] for r in bad], 'x', color='r', ms=6)
        ee = [r for r in rows if a0 <= r[2] - t0 <= a1]
        ax.step([r[2] - t0 for r in ee], [r[8] for r in ee], 'm-', lw=1.5, where='post', label='monitor estimate (causal)')
        ax2 = ax.twinx()
        s = bg.cmd
        mm = (s.t_hdr - t0 >= a0) & (s.t_hdr - t0 <= a1)
        ax2.step(s.t_hdr[mm] - t0, s.v[mm], 'g-', lw=0.7, where='post'); ax2.set_ylim(-16, 16); ax2.set_ylabel('notch', color='g')
        ax.set_title(f'{b}: red x = samples rejected by CausalWheelMonitor'); ax.grid(lw=0.4); ax.legend(fontsize=7, loc='upper left')
        ax.set_xlabel('t since bag start [s]'); ax.set_ylabel('m/s')
    plt.tight_layout(); plt.savefig(HERE / 'fig_monitor_examples.png', dpi=65)


if __name__ == '__main__':
    which = sys.argv[1:] or ['scale', 'gallery', 'timing', 'monitor']
    if 'monitor' in which:
        fig_monitor()
    if 'scale' in which:
        fig_scale()
    if 'gallery' in which:
        fig_gallery()
    if 'timing' in which:
        fig_timing()
