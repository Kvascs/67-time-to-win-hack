"""Step 0: replay the frozen estimator (tbo_replay_fix2.exe) on every unique long bag and cache the
outputs plus the GNSS reference. At most 2 replays run in parallel (shared machine).

usage: python analysis/products/run_replays.py [bag ...] [--force]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from common import OUT, bag_meta, load_ref, load_run, long_bags, run_estimator


def one(bag: str, force: bool) -> dict:
    t0 = time.time()
    run_estimator(bag, force=force)
    t1 = time.time()
    o = load_run(bag, force=force)
    ref = load_ref(bag, force=force)
    return {'bag': bag, 'replay_s': round(t1 - t0, 1), 'rows': len(o),
            'dur_s': float(o.t.iloc[-1]), 'dist_km': float(np.trapezoid(o.v, o.t) / 1e3),
            'map_assigned': float(np.isfinite(o.s_map).mean()), 'has_ref': ref is not None}


def main():
    args = sys.argv[1:]
    force = '--force' in args
    bags = [a for a in args if not a.startswith('--')] or long_bags()
    rows = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(one, b, force): b for b in bags}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"{r['bag']}: {r['rows']} rows, {r['dur_s']:.0f} s, {r['dist_km']:.2f} km, "
                  f"map {r['map_assigned']:.3f}, replay {r['replay_s']} s", flush=True)
    df = pd.DataFrame(rows).set_index('bag').sort_index()
    meta = bag_meta()
    df = meta.loc[df.index].join(df.drop(columns=['dur_s']))
    if len(bags) == len(long_bags()):
        df.to_csv(OUT / 'runs.csv')
    print(df.groupby(['vehicle', 'date']).size())


if __name__ == '__main__':
    main()
