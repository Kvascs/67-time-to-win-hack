"""Clock consistency: GNSS header clock vs vehicle header clock vs bag (recorder) clock.

Detects 1-s steps / ~8 %/s slews ("chrony-like") and checks, inside anomaly windows, which pair of clocks is
physically consistent by re-estimating the wheel-vs-GNSS lag there.

Outputs: clocks_per_bag.csv, clock_anomalies.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing as T  # noqa: E402

OUT = Path(__file__).resolve().parent


def episodes(t: np.ndarray, m: np.ndarray, join: float = 2.0):
    """Contiguous True-runs of mask m over times t (runs closer than ``join`` s are merged)."""
    if not m.any():
        return []
    tt = t[m]
    br = np.flatnonzero(np.diff(tt) > join)
    starts = np.r_[tt[0], tt[br + 1]]
    ends = np.r_[tt[br], tt[-1]]
    return list(zip(starts, ends))


def vehicle_vs_bag_steps(bag: T.Bag, thresh: float = 0.3):
    c = T.cmd(bag)
    ok = ~T.stale_mask(c.t_bag, c.t_hdr)
    off = (c.t_bag - c.t_hdr)[ok]
    med = np.median(off)
    return c.t_hdr[ok], off - med, np.abs(off - med) > thresh


def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    s = T.splits()
    bags = T.list_bags('train', 'val', 'short', 'no_gnss_long')
    rows, plots = [], []
    for name in bags:
        bag = T.load_bag(name)
        r = dict(bag=name, vehicle=bag.vehicle, has_gnss=bag.has('master_fix') and len(T.gnss_fix(bag)) > 50)
        tv, dv, av = vehicle_vs_bag_steps(bag)
        r['veh_vs_bag_anom_s'] = float(sum(e - a + 0.05 for a, e in episodes(tv, av)))
        r['veh_vs_bag_episodes'] = len(episodes(tv, av))
        if r['has_gnss']:
            tg, rel, an = T.clock_offsets(bag)
            good = ~T.stale_mask(T.gnss_fix(bag, drop_invalid=False).t_bag, tg)
            r['gnss_minus_veh_med_ms'] = float(np.median(rel[good]) * 1e3)
            r['gnss_minus_veh_p01_ms'] = float(np.percentile(rel[good], 1) * 1e3)
            r['gnss_minus_veh_p99_ms'] = float(np.percentile(rel[good], 99) * 1e3)
            eps = episodes(tg, an)
            r['gnss_vs_veh_anom_s'] = float(sum(e - a + 0.1 for a, e in eps))
            r['gnss_vs_veh_anom_frac'] = float(an.mean())
            r['gnss_vs_veh_episodes'] = len(eps)
            r['gnss_vs_veh_max_abs_ms'] = float(np.max(np.abs(rel[good] - np.median(rel[good]))) * 1e3)
            if eps:
                plots.append((name, tg, rel, tv, dv, bag.t0))
                # physical consistency inside the longest episode
                a, e = max(eps, key=lambda p: p[1] - p[0])
                if e - a > 20:
                    vm = T.gnss_vel(bag)
                    w = T.wheel(bag, 'front')
                    ok = ~T.stale_mask(vm.t_bag, vm.t_hdr)
                    m = (vm.t_hdr > a + 3) & (vm.t_hdr < e - 3)
                    # hdr-hdr and bag-bag lags restricted to the anomaly window
                    rh = T.estimate_lag(vm.t_hdr[ok], vm.speed[ok], np.sort(w.t_hdr), w.v[np.argsort(w.t_hdr)],
                                        lo=-1.6, hi=1.6, coarse=0.005, mask=m[ok])
                    rb = T.estimate_lag(vm.t_bag[ok], vm.speed[ok], np.sort(w.t_bag), w.v[np.argsort(w.t_bag)],
                                        lo=-1.6, hi=1.6, coarse=0.005, mask=m[ok])
                    r['longest_ep_start_s'] = float(a - bag.t0)
                    r['longest_ep_dur_s'] = float(e - a)
                    r['lag_hh_in_episode'] = rh['lag']
                    r['lag_bb_in_episode'] = rb['lag']
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'clocks_per_bag.csv', index=False, float_format='%.5g')
    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 200)
    print(df.to_string())
    g = df[df.has_gnss]
    tot = g['gnss_vs_veh_anom_s'].sum()
    print(f"\nbags with GNSS: {len(g)}; with GNSS-vs-vehicle clock anomalies: {(g.gnss_vs_veh_episodes > 0).sum()}; "
          f"total anomalous time {tot:.0f} s")
    print(f"bags with vehicle-vs-bag clock steps: {(df.veh_vs_bag_episodes > 0).sum()} of {len(df)}")
    # plot
    if plots:
        n = len(plots)
        fig, axs = plt.subplots(n, 1, figsize=(14, 2.6 * n), squeeze=False)
        for ax, (name, tg, rel, tv, dv, t0) in zip(axs[:, 0], plots):
            ax.plot(tg - t0, rel * 1e3, 'r.', ms=1.5, label='GNSS hdr clock - vehicle hdr clock')
            ax.plot(tv - t0, dv * 1e3, 'k.', ms=1, label='(bag - hdr)_cmd minus median: recorder vs vehicle clock')
            ax.set_ylabel('ms'); ax.set_ylim(-1300, 1300); ax.grid(alpha=.3)
            ax.set_title(name, fontsize=9, loc='left')
            ax.legend(fontsize=7, loc='upper right', markerscale=6)
        axs[-1, 0].set_xlabel('bag time since start [s]')
        fig.tight_layout()
        fig.savefig(OUT / 'clock_anomalies.png', dpi=80)


if __name__ == '__main__':
    main()
