"""Run the no-GNSS replay (frozen tbo_replay_fix2.exe) and the GNSS ground-truth projection for bags.

    python prep.py                 # train + val + no_gnss_long
    python prep.py --split val

At most 2 replays run in parallel (shared machine). Results are cached in research/global_loc/cache/.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as C


def _one(bag: str):
    t0 = time.time()
    try:
        r = C.replay_nognss(bag)
        out = C.CACHE / f'truth_{bag}.npz'
        has_gnss = len(np.load(C.NPZ / f'{bag}.npz')['sensing__gnss__master__fix']) > 0
        if has_gnss and not out.exists():
            m = C.load_map()
            tr = C.truth_track(bag, m)
            np.savez_compressed(out, t=tr.t.to_numpy(), s_map=tr.s_map.to_numpy(), s_unw=tr.s_unw.to_numpy(),
                                dist=tr.dist.to_numpy(), status=tr.status.to_numpy(), ok=tr.ok.to_numpy())
        return bag, len(r['t']), time.time() - t0, ''
    except Exception as e:  # keep going, report
        return bag, 0, time.time() - t0, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='all')
    ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    if a.split == 'all':
        bags = C.SPLITS['train'] + C.SPLITS['val'] + C.SPLITS['no_gnss_long']
    else:
        bags = C.SPLITS[a.split]
    with ProcessPoolExecutor(min(a.jobs, 2)) as ex:
        for bag, n, dt, err in ex.map(_one, bags):
            print(f'{bag}: {n} outputs, {dt:.1f} s {err}', flush=True)


if __name__ == '__main__':
    main()
