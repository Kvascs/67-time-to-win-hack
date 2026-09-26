"""End-to-end check of the CUSUM allowances chosen on train (cusum_theory.py kappa scan) on val.

    python analysis/consistency/kappa_ab.py [--slip 0.4] [--slide 0.5]

Replays the 17 val bags with the frozen binary and --set cusum_slip_accel / cusum_slide_accel,
compares quick_eval metrics with the base replay and counts joint-monitor latches (mu3 = 1, mu0 = 0
in the published outputs), each labelled by the GNSS Doppler speed (max |wheel - v_ref| > 0.5 m/s
within [-1, +3] s of the onset = real slip/slide). Writes results/kappa_ab_val.csv / .json.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import samples as S  # noqa: E402

KEYS = ['v_rmse', 'v_p99', 'along_rmse', 'along_max', 'end_err', 'model_only']


def _one(args):
    bag, tag, sets = args
    return C.run_bag(bag, tag=tag, sets=sets)


def latches(bag: str, tag: str) -> list[dict]:
    _, o = C.load_replay(bag, tag)
    lat = (o.mu3.to_numpy() >= 0.99999) & (o.mu0.to_numpy() <= 0.00001)
    on = np.flatnonzero(lat & ~np.concatenate([[False], lat[:-1]]))
    s = pd.read_parquet(S.OUT / f'{bag}.parquet', columns=['t', 'zc_f', 'zc_r', 'k_pr', 'v_ref'])
    out = []
    for i in on:
        t0 = o.stamp_ns.iloc[i] * 1e-9
        w = s[(s.t > t0 - 1.0) & (s.t < t0 + 3.0)]
        dev = np.nanmax(np.abs(np.stack([w.zc_f, w.zc_r], 1) / (1 + w.k_pr.to_numpy()[:, None])
                               - w.v_ref.to_numpy()[:, None])) if len(w) else np.nan
        out.append({'bag': bag, 'tag': tag, 't': t0, 'max_wheel_minus_gnss': float(dev),
                    'gnss_confirms': bool(dev > 0.5)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slip', type=float, default=0.4)
    ap.add_argument('--slide', type=float, default=0.5)
    a = ap.parse_args()
    tag = f'kappa_{a.slip:g}_{a.slide:g}'
    sets = {'cusum_slip_accel': a.slip, 'cusum_slide_accel': a.slide}
    bags = C.splits()['val']
    with ProcessPoolExecutor(2) as ex:
        new = list(ex.map(_one, [(b, tag, sets) for b in bags]))
    base = [C.load_replay(b)[0] for b in bags]
    df = pd.DataFrame([{**{f'base_{k}': bm[k] for k in KEYS}, **{f'new_{k}': nm[k] for k in KEYS},
                        'bag': b, 'ref_good': bm['ref_good']} for b, bm, nm in zip(bags, base, new)])
    lat = pd.DataFrame(sum([latches(b, 'base') + latches(b, tag) for b in bags], []))
    df.to_csv(C.RESULTS / 'kappa_ab_val.csv', index=False)
    lat.to_csv(C.RESULTS / 'kappa_ab_val_latches.csv', index=False)
    summ = {'sets': sets, 'bags': len(bags)}
    for k in KEYS:
        summ[k] = {'base_mean': float(df[f'base_{k}'].mean()), 'new_mean': float(df[f'new_{k}'].mean()),
                   'base_median': float(df[f'base_{k}'].median()), 'new_median': float(df[f'new_{k}'].median()),
                   'max_abs_change': float((df[f'new_{k}'] - df[f'base_{k}']).abs().max())}
    for t in ('base', tag):
        sub = lat[lat.tag == t] if len(lat) else lat
        summ[f'latches_{t}'] = {'total': int(len(sub)), 'gnss_confirmed': int(sub.gnss_confirms.sum()) if len(sub) else 0}
    C.write_json(summ, C.RESULTS / 'kappa_ab_val.json')
    import json
    print(json.dumps(summ, indent=1))
    pd.set_option('display.width', 250)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
