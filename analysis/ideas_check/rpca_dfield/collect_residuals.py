"""Per-sample model residuals r = dv/dt - (g*a_drive + a_ext) for train and val bags.

Reuses tools/model/learn_dfield.residuals() unchanged (same replay, same filters: v > 1 m/s,
mu0 > 0.9, on the main cycle, no slip flags, |r| < 2 m/s^2), replay WITHOUT a field, validation
(train-only) maps. Writes residuals_<split>.parquet (bag, s_map, r, v) next to this script.

    python analysis/ideas_check/rpca_dfield/collect_residuals.py --exe build_core/tbo_replay_final2.exe --jobs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'model'))
import learn_dfield  # noqa: E402  (also puts tools/replay on sys.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=str(ROOT / 'build_core' / 'tbo_replay_final2.exe'))
    ap.add_argument('--splits', default='train,val')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--redo', nargs='*', default=[], help='only re-run these bags and merge them into the '
                    'existing residuals_<split>.parquet (e.g. after a transient out-of-memory failure)')
    a = ap.parse_args()
    maps = ROOT / 'analysis' / 'validation_maps'
    tmp = ROOT / 'build_core' / 'replay_tmp' / 'rpca_res'
    splits = json.load(open(ROOT / 'data' / 'splits.json'))
    if a.redo:
        for split in a.splits.split(','):
            redo = [b for b in a.redo if b in splits[split]]
            if not redo:
                continue
            out = HERE / f'residuals_{split}.parquet'
            old = pd.read_parquet(out)
            new = [learn_dfield._job((b, maps, a.exe, tmp, {})) for b in redo]
            df = pd.concat([old[~old.bag.isin(redo)]] + [p for p in new if p is not None], ignore_index=True)
            df.to_parquet(out, index=False)
            print(f'{split}: redo {redo} -> {df.bag.nunique()}/{len(splits[split])} runs, {len(df)} samples, '
                  f'r std {df.r.std():.4f}', flush=True)
        return
    for split in a.splits.split(','):
        bags = splits[split]
        t0 = time.time()
        with ProcessPoolExecutor(a.jobs) as ex:
            parts = list(ex.map(learn_dfield._job, [(b, maps, a.exe, tmp, {}) for b in bags]))
        ok = [p for p in parts if p is not None]
        df = pd.concat(ok, ignore_index=True)
        out = HERE / f'residuals_{split}.parquet'
        df.to_parquet(out, index=False)
        print(f'{split}: {len(ok)}/{len(bags)} runs, {len(df)} samples, r std {df.r.std():.4f}, '
              f'{time.time() - t0:.0f} s -> {out.name}', flush=True)


if __name__ == '__main__':
    main()
