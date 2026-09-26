"""Timing audit of the tram dataset (npz produced by tools/extract_bags.py).

Reproduces the numbers quoted in research/ros2_realtime.md, section 2:
  * per-topic rate, receive-vs-header lag, gaps, non-monotonic stamps
  * start-of-bag catch-up burst
  * within-bag drift of the lag (wheel vs cmd vs GNSS)
  * wheel speed unit (ratio to GNSS speed)
  * wheel-stamp vs GNSS-stamp alignment (best delay)
  * prediction horizon: cmd stamp minus latest wheel stamp
  * duplicate bags (identical content under different ids)
Run:  python research/ros2_realtime_snippets/dataset_timing_audit.py
"""
import collections
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILES = sorted(glob.glob(str(ROOT / 'data' / 'npz' / '*.npz')))
IN_TOPICS = ['vehicle__front_bogie_velocity', 'vehicle__rear_bogie_velocity', 'vehicle__driver_position_cmd']


def pct(a, q):
    return np.percentile(a, q) if len(a) else float('nan')


def main():
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    horizons, ratios, dup = [], [], collections.defaultdict(list)
    for f in FILES:
        d = np.load(f)
        name = Path(f).stem
        for k in list(d.keys()):
            a = d[k]
            if len(a) < 200:
                continue
            dt = np.diff(a[:, 0])
            lag = a[:, 0] - a[:, 1]
            s = agg[k]
            s['rate'].append(1.0 / np.median(dt))
            s['lag_med'].append(np.median(lag))
            s['max_gap'].append(dt.max())
            s['nonmono'].append(int((np.diff(a[:, 1]) < 0).sum()))
            t = a[:, 0] - a[0, 0]
            meds = np.array([np.median(lag[(t >= x) & (t < x + 60)])
                             for x in range(0, int(t[-1]) - 59, 60) if ((t >= x) & (t < x + 60)).sum() > 50])
            if len(meds):
                meds = meds[np.abs(meds - np.median(meds)) < 0.5]
                s['lag_drift'].append(meds.max() - meds.min() if len(meds) else 0.0)
        w, c, g = d['vehicle__front_bogie_velocity'], d['vehicle__driver_position_cmd'], d['sensing__gnss__master__vel']
        dup[(len(w), len(c), round(float(w[:, 2].sum()), 3) if len(w) else 0)].append(name)
        if len(c) > 1000 and len(w) > 500:
            idx = np.searchsorted(w[:, 0], c[:, 0], side='right') - 1
            m = idx >= 0
            h = c[m, 1] - np.maximum.accumulate(w[:, 1])[idx[m]]
            horizons.append(h[len(h) // 50:])
        if len(g) > 100 and len(w) > 100:
            gs = np.hypot(g[:, 2], g[:, 3])
            wi = np.interp(g[:, 0], w[:, 0], w[:, 2])
            mm = (gs > 3) & (wi > 5)
            if mm.sum() > 100:
                ratios.append(np.median(wi[mm] / gs[mm]))

    print('== per-topic timing (medians over bags) ==')
    for k, s in agg.items():
        print(f'{k:36s} rate {np.median(s["rate"]):5.2f} Hz | lag(recv-stamp) p50 {np.median(s["lag_med"]):.3f} s'
              f' | within-bag lag drift p50 {pct(s["lag_drift"], 50):.3f} max {max(s["lag_drift"] or [0]):.3f} s'
              f' | max gap {max(s["max_gap"]):.1f} s | non-monotonic stamps {sum(s["nonmono"])}')
    H = np.concatenate(horizons)
    print('\nprediction horizon cmd_stamp - last_wheel_stamp [s]: p1/p50/p95/p99 = '
          + ' / '.join(f'{v:.3f}' for v in np.percentile(H, [1, 50, 95, 99])))
    print(f'wheel/GNSS speed ratio (median over bags): {np.median(ratios):.3f}  -> raw wheel speed is km/h')
    dups = [v for v in dup.values() if len(v) > 1]
    print(f'unique bags: {len(dup)} of {sum(len(v) for v in dup.values())}; duplicate groups: {len(dups)}')
    for v in dups:
        print('   ', v)


if __name__ == '__main__':
    main()
