"""Plot selected episodes: wheel front/rear (per-bag scaled), GNSS master/rover raw speed (header time
+ applied lag), fused reference, notch. Usage: python plot_episodes.py out.png bag:t0:t1 [bag:t0:t1 ...]
(t relative to bag start)."""
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


def plot_window(ax, name, a0, a1, sc=None, title=None):
    b = A.load_bag(name)
    try:
        al = load_aligned(name)
    except FileNotFoundError:
        al = None
    t0 = b.t0
    kf = kr = A.KMH
    if sc is not None and name in sc.index:
        kf, kr = sc.loc[name, 'k_speed_front'], sc.loc[name, 'k_speed_rear']
    lag = 0.0
    if al is not None:
        m = (al.t - t0 >= a0) & (al.t - t0 <= a1)
        lag = np.nanmedian(al.lag[m]) if m.any() else 0.0
    for s, c, k in ((b.front, 'b', kf), (b.rear, 'c', kr)):
        mm = (s.t_hdr - t0 >= a0) & (s.t_hdr - t0 <= a1)
        ax.plot(s.t_hdr[mm] - t0, s.v[mm] / k, '.-', color=c, ms=4, lw=0.8, label=f'{"front" if c == "b" else "rear"}/k')
    for rx, c in (('master', 'r'), ('rover', 'm')):
        g = b.gnss_vel[rx]
        if len(g) == 0:
            continue
        mm = (g.t_hdr - lag - t0 >= a0) & (g.t_hdr - lag - t0 <= a1)
        ax.plot(g.t_hdr[mm] - lag - t0, np.hypot(g.v[mm, 0], g.v[mm, 1]), 'x', color=c, ms=4, label=f'gnss {rx} (hdr-lag)')
    if al is not None:
        ax.plot(al.t[m] - t0, al.ref[m], 'k-', lw=0.8, alpha=0.6, label='fused ref')
    ax2 = ax.twinx()
    s = b.cmd
    mm = (s.t_hdr - t0 >= a0) & (s.t_hdr - t0 <= a1)
    ax2.step(s.t_hdr[mm] - t0, s.v[mm], 'g-', lw=0.8, where='post')
    ax2.set_ylim(-16, 16); ax2.set_ylabel('notch', color='g')
    ax.set_title(title or f'{name}  lag={lag:+.2f}s', fontsize=9)
    ax.grid(lw=0.4); ax.legend(fontsize=6, loc='upper left')
    ax.set_xlabel('t since bag start [s]'); ax.set_ylabel('m/s')


def main(out, specs, ncol=2):
    sc = pd.read_csv(HERE / 'scale_factors.csv').set_index('bag') if (HERE / 'scale_factors.csv').exists() else None
    n = len(specs)
    nr = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nr, ncol, figsize=(9 * ncol, 3.3 * nr), squeeze=False)
    for ax, sp in zip(axs.flat, specs):
        parts = sp.split(':')
        name, a0, a1 = parts[0], float(parts[1]), float(parts[2])
        title = parts[3] if len(parts) > 3 else None
        plot_window(ax, name, a0, a1, sc, title)
    for ax in list(axs.flat)[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out, dpi=70)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
