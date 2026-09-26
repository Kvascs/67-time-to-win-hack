"""Plot helpers for map_build (overlay polylines on fix clouds in zoom windows)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WINDOWS = {
    'west_yard': (-2290, -2040, -420, -250),
    'east_loop': (2280, 2420, 690, 895),
    'mid': (0, 400, 60, 160),
    'west_curve': (-1300, -700, -450, 350),
}


def overlay(runs, polys, fname, windows=WINDOWS, good_only=True, arrows=False):
    n = len(windows)
    fig, axs = plt.subplots(1, n, figsize=(7 * n, 7))
    axs = np.atleast_1d(axs)
    for ax, (nm, (x0, x1, y0, y1)) in zip(axs, windows.items()):
        for r in runs.values():
            m = (r.x > x0) & (r.x < x1) & (r.y > y0) & (r.y < y1)
            if good_only:
                m &= r.good
            ax.plot(r.x[m], r.y[m], '.', ms=0.6, color='0.6')
        for name, (P, c) in polys.items():
            ax.plot(P.xy[:, 0], P.xy[:, 1], '-', color=c, lw=1.0, label=name)
            if arrows:
                k = np.arange(0, len(P.xy) - 1, max(len(P.xy) // 200, 1))
                ax.quiver(P.xy[k, 0], P.xy[k, 1], P.t[k, 0], P.t[k, 1], color=c, scale=80, width=0.002)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.grid(True); ax.set_title(nm)
    axs[0].legend(loc='upper left', fontsize=7)
    fig.savefig(fname, dpi=80, bbox_inches='tight')
    plt.close(fig)
