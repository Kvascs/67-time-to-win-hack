"""Report figures for the track map (reads map/ and caches).  python make_plots.py"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data_io as D
import runs as R
import track_map as TM

OUT = D.OUT
PLOTS = OUT / 'plots'
EDGE_COLORS = {'main': 'k', 'wb_detour': 'tab:red', 'fan_F2': 'tab:blue', 'fan_F3': 'tab:purple',
               'west_arrival_2': 'tab:orange'}


def load():
    tm = TM.TrackMap(OUT / 'map')
    runs = {**R.load_runs(D.split('train')), **R.load_runs(D.split('val'))}
    return tm, runs


def draw_edges(ax, tm, lw=1.2, arrows=True, arrow_scale=45):
    for eid, E in tm.edges.items():
        ax.plot(E.x, E.y, '-', color=EDGE_COLORS.get(eid, 'tab:green'), lw=lw, label=f'{eid} ({E.length:.0f} m)')
        if arrows:
            k = np.arange(10, len(E.x) - 10, max(len(E.x) // (60 if eid == 'main' else 4), 1))
            ax.quiver(E.x[k], E.y[k], np.cos(E.yaw[k]), np.sin(E.yaw[k]), color=EDGE_COLORS.get(eid, 'tab:green'),
                      scale=arrow_scale, width=0.0018, headwidth=4)


def draw_stops(ax, tm, text=True, xlim=None, ylim=None):
    for st in tm.stops:
        if xlim and not (xlim[0] < st['x'] < xlim[1] and ylim[0] < st['y'] < ylim[1]):
            continue
        if st['cls'] == 'platform':
            ax.plot(st['x'], st['y'], '^', color='tab:green', ms=8, mec='k')
        elif st['cls'] == 'terminal':
            ax.plot(st['x'], st['y'], 's', color='gold', ms=7, mec='k')
        elif st['landmark']:
            ax.plot(st['x'], st['y'], 'o', color='tab:cyan', ms=5, mec='k')
        else:
            continue
        if text:
            ax.annotate(f"{st['s']:.0f}", (st['x'], st['y']), fontsize=6, xytext=(3, 3), textcoords='offset points')


def overview(tm):
    fig, ax = plt.subplots(figsize=(22, 8))
    draw_edges(ax, tm, lw=1.0, arrows=True, arrow_scale=120)
    draw_stops(ax, tm, text=False)
    E = tm.edges['main']
    for sv in np.arange(0, E.length, 500):
        x, y, z, yaw = E.pose(sv)
        ax.plot(x, y, 'k+', ms=6)
        ax.annotate(f's={sv:.0f}', (x, y), fontsize=6, color='0.3', xytext=(-10, -12), textcoords='offset points')
    ax.plot([], [], '^', color='tab:green', mec='k', label='platform stop')
    ax.plot([], [], 's', color='gold', mec='k', label='terminal / layover')
    ax.plot([], [], 'o', color='tab:cyan', mec='k', label='other stop landmark (signal)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(loc='lower right', fontsize=8)
    ax.set_xlabel('x east [m] (MAP ENU)'); ax.set_ylabel('y north [m]')
    o = tm.origin
    ax.set_title(f'Track map (master antenna path). MAP ENU origin lat={o[0]}, lon={o[1]}, h={o[2]}. '
                 f'main cycle L={E.length:.2f} m; s increases in travel direction, s=0 at east platform')
    fig.savefig(PLOTS / 'map_overview.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def zoom(tm, runs, fname, win, title):
    x0, x1, y0, y1 = win
    fig, ax = plt.subplots(figsize=(13, 13 * (y1 - y0) / (x1 - x0) + 1))
    for r in runs.values():
        m = r.good & (r.x > x0) & (r.x < x1) & (r.y > y0) & (r.y < y1)
        ax.plot(r.x[m], r.y[m], '.', ms=0.6, color='0.7')
    draw_edges(ax, tm, lw=1.3)
    draw_stops(ax, tm, text=True, xlim=(x0, x1), ylim=(y0, y1))
    E = tm.edges['main']
    ss = E.s[::20]
    for sv in ss:
        x, y, _, _ = E.pose(sv)
        if x0 < x < x1 and y0 < y < y1 and int(sv) % 20 == 0:
            ax.annotate(f'{sv:.0f}', (x, y), fontsize=5, color='0.4')
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7); ax.set_title(title + ' (grey: RTK fixes train+val; numbers: s on main)')
    fig.savefig(PLOTS / fname, dpi=90, bbox_inches='tight')
    plt.close(fig)


def profiles(tm):
    E = tm.edges['main']
    secs = tm.meta['sections']
    fig, axs = plt.subplots(4, 1, figsize=(22, 14), sharex=True)
    axs[0].plot(E.s, E.z, 'k-', lw=0.8); axs[0].set_ylabel('z [m] (MAP ENU up)')
    axs[1].plot(E.s, 100 * E.grade, 'b-', lw=0.8); axs[1].set_ylabel('grade [%] (dh/ds)'); axs[1].axhline(0, color='0.5')
    axs[2].plot(E.s, E.curv, 'r-', lw=0.6); axs[2].set_ylabel('curvature [1/m] (left +)'); axs[2].set_ylim(-0.08, 0.08)
    for R_ in (25, 50, 100, 300):
        axs[2].axhline(1 / R_, color='0.8', lw=0.5); axs[2].axhline(-1 / R_, color='0.8', lw=0.5)
        axs[2].text(E.length + 30, 1 / R_, f'R={R_}', fontsize=7)
    axs[3].plot(E.s, np.degrees(np.unwrap(E.yaw)), 'g-', lw=0.8); axs[3].set_ylabel('yaw [deg] unwrapped')
    for ax in axs:
        for sc in secs:
            ax.axvline(sc['s0'], color='m', lw=0.6, ls='--')
        for st in tm.stops:
            if st['edge'] == 'main' and st['cls'] == 'platform':
                ax.axvline(st['s'], color='tab:green', lw=0.5, alpha=0.6)
        ax.grid(True, alpha=0.3)
    for sc in secs:
        axs[0].text(sc['s0'] + 20, axs[0].get_ylim()[1] - 2, sc['name'], fontsize=8, color='m')
    axs[3].set_xlabel('s on main [m] (green = platform stops)')
    fig.savefig(PLOTS / 'map_profiles.png', dpi=90, bbox_inches='tight')
    plt.close(fig)


def detour(tm, runs):
    br = tm.edges['wb_detour']
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    wins = [(br.x.min() - 20, br.x.max() + 20, br.y.min() - 15, br.y.max() + 15),
            (br.x[0] - 60, br.x[0] + 20, br.y[0] - 20, br.y[0] + 12), (br.x[-1] - 20, br.x[-1] + 60, br.y[-1] - 15, br.y[-1] + 15)]
    info = {i['bag']: i for i in D.SPLITS['info']}
    for ax, (x0, x1, y0, y1) in zip(axs, wins):
        for b, r in runs.items():
            day = int(round(info[b]['t0'] / 86400 - 20600))
            m = r.good & (r.x > x0) & (r.x < x1) & (r.y > y0) & (r.y < y1)
            ax.plot(r.x[m], r.y[m], '.', ms=0.8, color='tab:red' if day in (61, 62) else '0.6')
        draw_edges(ax, tm, lw=1.0, arrows=False)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(True, alpha=0.3)
        if ax is not axs[0]:
            ax.set_aspect('equal')
    axs[0].set_title('WB detour (red dots: runs of 27-28 Jul; grey: other dates)')
    axs[1].set_title('divergence end (east)'); axs[2].set_title('merge end (west)')
    axs[0].legend(fontsize=7)
    fig.savefig(PLOTS / 'map_wb_detour.png', dpi=85, bbox_inches='tight')
    plt.close(fig)


def wheel_curv():
    f = OUT / 'cache' / 'wheel_curv_windows.csv'
    if not f.exists():
        return
    df = pd.read_csv(f)
    cr = df[np.abs(df.acc) < 0.15]
    W = json.loads((OUT / 'cache' / 'wheel_curv_fit.json').read_text())['all_cruise']
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, col, nm, c1, c0 in ((axs[0], 'dwf', 'front', W['dwf']['c_abs'], W['dwf']['c0']),
                                (axs[1], 'dwr', 'rear', W['dwr']['c_abs'], W['dwr']['c0'])):
        r = cr.ds / cr[col]
        ax.scatter(cr.k, r, s=3, alpha=0.4)
        kk = np.linspace(-0.07, 0.07, 200)
        ax.plot(kk, 1 + c0 + c1 * np.abs(kk), 'r-', label=f'1 + {c0:.5f} + {c1:.3f}|k|')
        bins = np.array([-0.07, -0.02, -0.006, -0.002, -0.0005, 0.0005, 0.002, 0.006, 0.02, 0.07])
        idx = np.digitize(cr.k, bins)
        for i in range(1, len(bins)):
            m = idx == i
            if m.sum() > 5:
                ax.plot(cr.k[m].mean(), cr.ds[m].sum() / cr[col][m].sum(), 'ko', ms=6)
        ax.set_ylim(0.97, 1.05); ax.set_xlim(-0.07, 0.07); ax.grid(True, alpha=0.3); ax.legend()
        ax.set_xlabel('mean signed curvature in 10 s window [1/m] (left +)')
        ax.set_ylabel(f'map distance / wheel distance ({nm}, km/h / 3.6)')
        ax.set_title(f'{nm} bogie, cruising 10 s windows; black = distance-weighted bin ratio', fontsize=10)
    fig.savefig(PLOTS / 'wheel_vs_curvature.png', dpi=90, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    tm, runs = load()
    overview(tm)
    zoom(tm, runs, 'map_west_yard.png', (-2275, -2040, -410, -250), 'West terminal: yard, loop and fan tracks')
    zoom(tm, runs, 'map_east_loop.png', (2280, 2420, 680, 895), 'East terminal loop')
    profiles(tm)
    detour(tm, runs)
    wheel_curv()
    print('plots written')
