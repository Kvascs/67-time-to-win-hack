"""Per-bag scale factors (speed ratio & distance ratio, per sensor) and GNSS-labelled wheel anomaly
episodes. Requires aligned/ cache (build_aligned.py).

Outputs (in this directory):
  scale_factors.csv      one row per bag with GNSS
  segment_ratios.csv     distance ratio per inter-stop segment (within-bag constancy)
  ratio_vs_speed.csv     speed-binned ratio per bag/sensor
  episodes_gnss.csv      slip / slide / lock / mismatch episodes labelled vs GNSS
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import anomalies as A  # noqa: E402
from build_aligned import load_aligned  # noqa: E402

SP = A.splits()
INFO = {x['bag']: x for x in SP['info']}


def split_of(b):
    for k in ('train', 'val', 'no_gnss_long', 'short'):
        if b in SP[k]:
            return k
    return '?'


def date_of(b):
    import datetime
    return datetime.datetime.fromtimestamp(INFO[b]['t0'], datetime.timezone.utc).strftime('%Y-%m-%d')


def main():
    scale_rows, seg_rows, spd_rows, ep_rows = [], [], [], []
    for name in sorted(INFO, key=lambda b: INFO[b]['t0']):
        al = load_aligned(name)
        if np.isfinite(al.ref).sum() < 300:
            continue
        b = A.load_bag(name)
        kappa = A.gnss_curvature(b, al.t + al.lag)
        sc = A.bag_scale(al, kappa=kappa)
        eps = sc.pop('eps')
        k = sc.pop('k')
        if not all(np.isfinite(list(k.values()))):
            continue
        dur = al.t[-1] - al.t[0]
        row = dict(bag=name, vehicle=name[:5], date=date_of(name), split=split_of(name), dur_s=dur,
                   t0_utc=INFO[name]['t0'], **sc)
        row['n_episodes'] = len(eps)
        row['frac_lag_nonzero'] = float(np.mean(np.abs(al.lag) > 0.3))
        # travel direction: sign of net east displacement from GNSS fixes (route runs roughly W-E)
        f = b.gnss_fix['master'] if len(b.gnss_fix['master']) > 50 else b.gnss_fix['rover']
        row['direction'] = ('E' if f.v[-1, 1] > f.v[0, 1] else 'W') if len(f) > 50 else '?'
        row['lon_start'] = float(f.v[0, 1]) if len(f) else np.nan
        row['lon_end'] = float(f.v[-1, 1]) if len(f) else np.nan
        scale_rows.append(row)
        # within-bag: per movement segment
        em = A.episode_mask(al, eps) | A.lag_transition_mask(al)
        for r in A.segment_ratios(al, em):
            r.update(bag=name, vehicle=name[:5], date=date_of(name), t_rel=r['t_mid'] - al.t0)
            i0 = np.searchsorted(al.t, r['t_mid'] - r['dur'] / 2); i1 = np.searchsorted(al.t, r['t_mid'] + r['dur'] / 2)
            kk = kappa[i0:i1]; vv = al.ref[i0:i1]
            mm = np.isfinite(kk) & np.isfinite(vv)
            r['frac_curve'] = float(np.sum(vv[mm] * (np.abs(kk[mm]) > 0.003)) / max(np.sum(vv[mm]), 1e-9))
            seg_rows.append(r)
        # speed dependence (steady, straight samples)
        steady = A.steady_mask(al) & ~em & (np.abs(np.nan_to_num(kappa, nan=1.0)) < 0.003)
        bins = np.arange(2, 17, 1.0)
        for s in ('front', 'rear'):
            w = getattr(al, s) * A.KMH
            for lo in bins:
                m = steady & (al.ref >= lo) & (al.ref < lo + 1) & np.isfinite(w)
                if m.sum() < 30:
                    continue
                spd_rows.append(dict(bag=name, vehicle=name[:5], date=date_of(name), sensor=s, v_lo=lo,
                                     k=float(np.sum(w[m] * al.ref[m]) / np.sum(al.ref[m] ** 2)), n=int(m.sum()),
                                     k_rel=float(np.sum(w[m] * al.ref[m]) / np.sum(al.ref[m] ** 2) / k[s])))
        for r in A.episode_table(al, k, t0=al.t0):
            r.update(vehicle=name[:5], date=date_of(name), split=split_of(name))
            ep_rows.append(r)
        print(name, {kk: round(vv, 4) for kk, vv in k.items()}, 'episodes', len(eps), flush=True)
    pd.DataFrame(scale_rows).to_csv(HERE / 'scale_factors.csv', index=False)
    pd.DataFrame(seg_rows).to_csv(HERE / 'segment_ratios.csv', index=False)
    pd.DataFrame(spd_rows).to_csv(HERE / 'ratio_vs_speed.csv', index=False)
    er = pd.DataFrame(ep_rows)
    if len(er):
        cl = [A.classify_episode(r) for r in er.to_dict('records')]
        er['validity'] = [c[0] for c in cl]; er['confidence'] = [c[1] for c in cl]; er['note'] = [c[2] for c in cl]
    er.to_csv(HERE / 'episodes_gnss_raw.csv', index=False)
    er[er.validity == 'wheel'].to_csv(HERE / 'episodes_gnss.csv', index=False)


if __name__ == '__main__':
    main()
