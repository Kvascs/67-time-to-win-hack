"""Figures for docs/REPORT.md (PNG, docs/img/). Inputs are the outputs of the runs quoted in the report:

  python tools/replay/eval_checker.py --tag _rm3                        # organisers' bag, submitted build
  python tools/replay/eval_checker.py --tag _qk0 --set quant_k_enable=0 # the same build without D30
  python tools/replay/eval_base_link.py --tag rm3_val --split val       # val, submitted build
  python tools/replay/eval_base_link.py --tag qk0_val --split val --set quant_k_enable=0
  docker run ... check_run.sh 30618_88aea4d9 0                          # live run: <out>/30618_88aea4d9/latency.csv

  python tools/replay/plot_report.py [--latency <latency.csv>]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpp_bridge  # noqa: E402
from eval_checker import approximate_sync  # noqa: E402

ROOT = cpp_bridge.ROOT
IMG = ROOT / 'docs' / 'img'
CHK = ROOT / 'build_core' / 'replay_tmp' / 'checker'
BAG = '30618_88aea4d9'

# paper and ink, one accent (validated: contrast >= 3:1 on the surface)
SURFACE, INK, INK2, MUTED, GRID, AXIS, ACCENT = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#2a78d6'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
    'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 1.6,
})


def pairs(tag):
    """Their pairing (ApproximateTimeSynchronizer replay) of our published output with the reference."""
    o = pd.read_csv(CHK / f'{BAG}_out{tag}.csv').drop_duplicates('stamp_ns').sort_values('stamp_ns').reset_index(drop=True)
    ks = np.load(cpp_bridge.NPZ / f'{BAG}.npz')['localization__kinematic_state']
    r_arr = np.round(ks[:, 0] * 1e9).astype(np.int64)
    r_st = np.round(ks[:, 1] * 1e9).astype(np.int64)
    o_arr = o.recv_ns.to_numpy(np.int64) + 500_000
    o_st = o.stamp_ns.to_numpy(np.int64)
    op = np.flatnonzero(o.pos_valid.to_numpy() == 1)
    pp = approximate_sync(r_arr, r_st, o_arr[op], o_st[op])
    oi, ri = op[pp[:, 1]], pp[:, 0]
    d = np.sqrt((o.x.to_numpy()[oi] - ks[ri, 2]) ** 2 + (o.y.to_numpy()[oi] - ks[ri, 3]) ** 2 +
                (o.z.to_numpy()[oi] - ks[ri, 4]) ** 2)
    t = (r_st[ri] - r_st[0]) * 1e-9
    return o, ks, t, d


def fig_speed(o, ks):
    t0 = ks[0, 1]
    tr, vr = ks[:, 1] - t0, ks[:, 9]
    to, vo = o.stamp_ns.to_numpy() * 1e-9 - t0, o.v.to_numpy()
    lo, hi = 505.0, 625.0
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    m = (tr >= lo) & (tr <= hi)
    ax.plot(tr[m], vr[m], color=MUTED, lw=2.4, label='эталон судьи (kinematic_state)')
    m = (to >= lo) & (to <= hi)
    ax.plot(to[m], vo[m], color=ACCENT, lw=1.4, label='наш /result/velocity')
    ax.set_xlim(lo, hi)
    ax.set_xlabel('время от начала бэга, с')
    ax.set_ylabel('скорость, м/с')
    ax.set_title('Бэг организаторов: наша скорость и эталон судьи, 2 минуты', loc='left')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(IMG / 'checker_speed.png', dpi=110)
    plt.close(fig)


def fig_error_time():
    _, _, t0, d0 = pairs('_qk0')
    _, _, t1, d1 = pairs('_rm3')
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(t0 / 60, d0, color=MUTED, lw=1.6, label=f'прежняя сборка: RMSE {np.sqrt(np.mean(d0 ** 2)):.2f} м')
    ax.plot(t1 / 60, d1, color=ACCENT, lw=1.6, label=f'сдаваемая сборка: RMSE {np.sqrt(np.mean(d1 ** 2)):.2f} м')
    ax.set_ylim(0, 5.0)
    ax.set_xlabel('время от начала бэга, мин')
    ax.set_ylabel('3-D ошибка base_link, м')
    ax.set_title('Бэг организаторов, их сопоставление: ошибка положения по времени', loc='left')
    tail0, tail1 = t0 / 60 > 21.0, t1 / 60 > 21.0
    if tail0.any() and tail1.any():
        ax.annotate(f'съезд в тупик: до {d0[tail0].max():.1f} м → до {d1[tail1].max():.1f} м', xy=(21.2, 4.5),
                    xytext=(11.5, 4.45), color=INK2, fontsize=9, va='center',
                    arrowprops=dict(arrowstyle='->', color=INK2, lw=0.8))
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(IMG / 'checker_error_time.png', dpi=110)
    plt.close(fig)


def fig_val():
    ev = ROOT / 'build_core' / 'eval'
    a = pd.read_csv(ev / 'bl_qk0_val.csv').set_index('bag').p3_rmse
    b = pd.read_csv(ev / 'bl_rm3_val.csv').set_index('bag').p3_rmse
    df = pd.DataFrame({'before': a, 'after': b}).dropna().sort_values('after')
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for yi, (lo, hi) in zip(y, zip(df.before, df.after)):
        ax.plot([lo, hi], [yi, yi], color=AXIS, lw=1.4, zorder=1)
    ax.scatter(df.before, y, s=46, color=MUTED, zorder=2, label='прежняя сборка')
    ax.scatter(df.after, y, s=46, color=ACCENT, edgecolor=SURFACE, linewidth=1.5, zorder=3, label='сдаваемая сборка')
    ax.set_yticks(y)
    ax.set_yticklabels([s.split('_')[1] for s in df.index], fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks([0.3, 0.5, 1, 2, 5, 10])
    ax.set_xticklabels(['0.3', '0.5', '1', '2', '5', '10'])
    ax.set_xlabel('3-D RMSE base_link против RTK, м (лог. шкала)')
    ax.set_title(f'val, {len(df)} бэгов: медиана {df.before.median():.2f} → {df.after.median():.2f} м, '
                 f'среднее {df.before.mean():.2f} → {df.after.mean():.2f} м', loc='left')
    ax.grid(axis='y', visible=False)
    for yi, (bag, r) in zip(y, df.iterrows()):
        if abs(r.after - r.before) > 0.1:
            ax.annotate(f'{r.before:.2f} → {r.after:.2f}', xy=(max(r.before, r.after), yi), xytext=(8, -3),
                        textcoords='offset points', color=INK2, fontsize=8)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(IMG / 'val_per_bag.png', dpi=110)
    plt.close(fig)


def fig_latency(path):
    lat = pd.read_csv(path).latency_ms.dropna().to_numpy()
    lat = np.sort(lat[lat > 0])
    cdf = np.arange(1, len(lat) + 1) / len(lat)
    p50, p99 = np.percentile(lat, 50), np.percentile(lat, 99)
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.plot(lat, cdf, color=ACCENT, lw=1.8)
    ax.set_xscale('log')
    ax.set_xlim(0.1, 150)
    ax.set_ylim(0, 1.02)
    for v, name in ((p50, 'p50'), (p99, 'p99')):
        ax.axvline(v, color=INK2, lw=0.8, ls=':')
        ax.annotate(f'{name} {v:.2f} мс', xy=(v, 0.08 if name == 'p50' else 0.22), xytext=(4, 0),
                    textcoords='offset points', color=INK2, fontsize=9)
    ax.axvline(100, color=INK, lw=1.0)
    ax.annotate('лимит жюри 100 мс', xy=(100, 0.5), xytext=(-6, 0), textcoords='offset points', ha='right',
                color=INK, fontsize=9)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100])
    ax.set_xticklabels(['0.1', '0.3', '1', '3', '10', '30', '100'])
    ax.set_xlabel('задержка «вход → выход», мс (лог. шкала)')
    ax.set_ylabel('доля выходов')
    ax.set_title(f'Живой прогон, бэг организаторов: {len(lat)} выходов, максимум {lat.max():.1f} мс', loc='left')
    fig.tight_layout()
    fig.savefig(IMG / 'live_latency.png', dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latency', default=str(ROOT / 'build_core' / 'live9' / BAG / 'latency.csv'))
    a = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    o, ks, _, _ = pairs('_rm3')
    fig_speed(o, ks)
    fig_error_time()
    fig_val()
    fig_latency(a.latency)
    for f in sorted(IMG.glob('*.png')):
        print(f.relative_to(ROOT), f.stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
