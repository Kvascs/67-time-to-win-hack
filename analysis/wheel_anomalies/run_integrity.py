"""GNSS-free stream integrity of the input topics (+ GNSS reference integrity) for all unique bags.

Outputs:
  integrity_topics.csv   one row per bag x topic: rate, header-dt percentiles, gap histogram, stamp issues,
                         latency, start-up burst, value checks
  stream_events.csv      discrete events (gaps, stamp excursions, non-monotonic stamps, negative/frozen values,
                         controller jumps, arrival-order gaps)
  gnss_integrity.csv     per bag x receiver: gaps, exact-zero velocity messages, spikes, header-clock excursions
  fig_gap_hist.png, fig_hdr_dt_hist.png
"""
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

SP = A.splits()
INFO = {x['bag']: x for x in SP['info']}
GAP_EDGES = [0.15, 0.25, 0.35, 0.5, 1.0, 2.0, 5.0, 1e9]


def topic_row(b, name, s, nominal):
    r = dict(bag=b.name, vehicle=b.vehicle, topic=name, n=len(s))
    if len(s) < 5:
        return r
    ok = A.stamp_ok_mask(s)
    th = np.sort(s.t_hdr[ok])
    d = np.diff(th)
    dtb = np.diff(np.sort(s.t_bag))
    dur = th[-1] - th[0]
    nb, span = A.startup_burst(s)
    st = A.stamp_anomalies(s)
    arr_nonmono = int((np.diff(s.t_hdr[np.argsort(s.t_bag, kind='stable')]) < 0).sum())
    r.update(dur_s=dur, rate_hz=(len(th) - 1) / dur if dur > 0 else np.nan,
             hdr_dt_p1=np.percentile(d, 1), hdr_dt_p5=np.percentile(d, 5), hdr_dt_p50=np.percentile(d, 50),
             hdr_dt_p95=np.percentile(d, 95), hdr_dt_p99=np.percentile(d, 99), hdr_dt_max=d.max(),
             bag_dt_p50=np.percentile(dtb, 50), bag_dt_p99=np.percentile(dtb, 99),
             frac_missing_est=float(max(0.0, 1 - (len(th) - 1) * nominal / dur)) if dur > 0 else np.nan,
             n_gap_gt_0p5=int((d > 0.5).sum()), n_gap_gt_1=int((d > 1.0).sum()), max_gap_s=float(d.max()),
             total_gap_time_gt_0p5=float(d[d > 0.5].sum()),
             n_dup_stamp=int((~ok).sum()), n_nonmono_arrival=arr_nonmono,
             burst_n=nb, burst_span_s=span,
             lat_median=st.get('lat_median'), lat_p99=st.get('lat_p99'), lat_min=st.get('lat_min'),
             lat_max=st.get('lat_max'), n_excursion_msgs=st.get('n_excursion_msgs', 0) + st.get('n_slow_excursion_msgs', 0))
    for lo, hi in zip(GAP_EDGES[:-1], GAP_EDGES[1:]):
        r[f'gaps_{lo:g}_{hi:g}'] = int(((d > lo) & (d <= hi)).sum())
    v = s.v
    if name in ('front', 'rear'):
        r.update(v_min_kmh=float(np.nanmin(v)), v_max_kmh=float(np.nanmax(v)), n_nonfinite=int((~np.isfinite(v)).sum()),
                 n_negative=int((v < 0).sum()), frac_zero=float(np.mean(v == 0)),
                 min_positive_kmh=float(v[v > 0].min()) if (v > 0).any() else np.nan,
                 frac_repeat_moving=A.repeat_stats(v, 0.3 * A.KMH),
                 n_frozen_runs=len(A.frozen_runs(s.t_hdr, v, 1.0, 0.3 * A.KMH)))
    else:
        c = A.cmd_checks(s)
        r.update(cmd_min=c['min'], cmd_max=c['max'], n_out_of_range=c['n_out_of_range'],
                 n_non_integer=c['n_non_integer'], n_big_steps=c['n_big_steps'], frac_zero=c['frac_zero'],
                 frac_trac=c['frac_trac'], frac_brake=c['frac_brake'])
    return r


def gnss_row(b, rx):
    s = b.gnss_vel[rx]
    f = b.gnss_fix[rx]
    r = dict(bag=b.name, receiver=rx, n_vel=len(s), n_fix=len(f))
    if len(s) < 20:
        return r
    o = np.argsort(s.t_hdr, kind='stable')
    th = s.t_hdr[o]
    d = np.diff(th)
    sp = np.hypot(s.v[:, 0], s.v[:, 1])
    zero = (s.v[:, 0] == 0) & (s.v[:, 1] == 0) & (s.v[:, 2] == 0)
    clean = A.gnss_speed_stream(b, rx)
    lat = s.t_bag - s.t_hdr
    nb, _ = A.startup_burst(s)
    base = np.median(lat[nb:])
    exc = np.abs(A.rolling_median(lat, 21) - base) > 0.3
    exc[:nb] = False
    r.update(dur_s=th[-1] - th[0], n_gap_gt_0p5=int((d > 0.5).sum()), max_gap_s=float(d.max()),
             total_gap_time_s=float(d[d > 0.5].sum()), n_exact_zero=int(zero.sum()),
             n_removed_by_cleaning=int(len(s) - len(clean)), v_max=float(sp.max()),
             lat_median=float(base), hdr_clock_excursion_s=float(np.sum(np.r_[0, np.diff(s.t_bag)][exc])),
             frac_status2=float(np.mean(f.v[:, 3] == 2)) if len(f) else np.nan)
    return r


def main():
    rows, ev, grows = [], [], []
    for name in sorted(INFO, key=lambda x: INFO[x]['t0']):
        b = A.load_bag(name)
        for tn, s, nom in (('front', b.front, 0.1), ('rear', b.rear, 0.1), ('cmd', b.cmd, 0.05)):
            rows.append(topic_row(b, tn, s, nom))
        for e in A.stream_events(b):
            e['bag'] = name
            ev.append(e)
        for rx in ('master', 'rover'):
            grows.append(gnss_row(b, rx))
        print(name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'integrity_topics.csv', index=False)
    pd.DataFrame(ev)[['bag', 'topic', 'kind', 't_rel', 'dur', 'detail']].to_csv(HERE / 'stream_events.csv', index=False)
    pd.DataFrame(grows).to_csv(HERE / 'gnss_integrity.csv', index=False)

    # gap histogram figure (header dt of wheel/cmd topics pooled)
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    for tn, c in (('front', 'b'), ('rear', 'c'), ('cmd', 'g')):
        dd = []
        for name in INFO:
            b = A.load_bag(name)
            s = getattr(b, tn)
            if len(s) > 5:
                dd.append(np.diff(np.sort(s.t_hdr[A.stamp_ok_mask(s)])))
        dd = np.concatenate(dd)
        axs[0].hist(dd, bins=np.arange(0, 0.5, 0.005), histtype='step', color=c, label=tn, density=True)
        big = dd[dd > 0.15]
        axs[1].hist(big, bins=np.logspace(np.log10(0.15), np.log10(100), 60), histtype='step', color=c, label=f'{tn} (n={len(big)})')
    axs[0].set_yscale('log'); axs[0].set_xlabel('header dt [s]'); axs[0].legend(); axs[0].grid(); axs[0].set_title('Header-stamp interval distribution (all bags)')
    axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel('gap [s]'); axs[1].legend(); axs[1].grid(); axs[1].set_title('Intervals > 0.15 s (dropouts)')
    plt.tight_layout(); plt.savefig(HERE / 'fig_gap_hist.png', dpi=80)


if __name__ == '__main__':
    main()
