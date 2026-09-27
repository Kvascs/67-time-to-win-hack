"""Continue the D33 parameter search with MOOtation's IBEA-eps+ (mootation.minimize), from our journal.

MOOtation is the team's C++ library of multi-objective optimisers with a Python API
(https://github.com/DedMoroz132/MOOtation, `pip install .` in its root; needs a C++ compiler and CMake).
The pattern follows its batch interface:
- the parameters live in the unit cube; `decode` (tools/tune/ibea_d33.py) maps them to real values;
- `batch` receives a whole generation and evaluates it with the same evaluator and per-bag normalised
  objectives as tools/tune/ibea_d33.py (median and p90 of the along-track RMSE ratio to the hand-set values,
  median speed ratio); every evaluation goes to the same journal (analysis/tuning/ibea_d33/);
- the first population is seeded from the journal (the best `pop` points by IBEA selection, the hand-set point
  included), so the continuation costs no re-evaluations;
- past the wall-clock deadline `batch` raises Deadline; the journal keeps everything, and the answer is read from
  the journal (`python tools/tune/ibea_d33.py select`), not from the final population;
- the evaluator uses ibea_d33.BASE, which pins ratio_update_k = 0, so the journal stays the D33 tuning with any
  build that has the bogie-ratio corrections (D33 or D34 binary as --exe).

    python tools/tune/mootation_d33.py --pop 10 --evals 40 --workers 7 --until 22:40 --exe build_core/tbo_replay.exe
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ibea_d33 as base  # noqa: E402


class Deadline(Exception):
    pass


def main():
    import mootation
    from mootation.persistence import LoadedPopulation

    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', type=int, default=10)
    ap.add_argument('--evals', type=int, default=40, help='new evaluations inside minimize')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--until', default='23:00')
    ap.add_argument('--split', default='train')
    ap.add_argument('--exe', default=str(base.ROOT / 'build_core' / 'tbo_replay.exe'))
    a = ap.parse_args()
    hh, mm = (int(t) for t in a.until.split(':'))
    deadline = dt.datetime.now().replace(hour=hh, minute=mm, second=0).timestamp()
    bags = base.rtk_bags(a.split)
    with ProcessPoolExecutor(a.workers, initializer=base._init, initargs=(a.exe,)) as pool:
        ev = base.Evaluator(bags, pool, a.split + '-mootation', norm=True)
        ev([base.X0])
        keys = list(ev.rows)
        U = np.array([(np.array([json.loads(k)[n] for n in base.NAMES]) - base.LO) / (base.HI - base.LO) for k in keys])
        F = np.array([ev.objectives(ev.rows[k]) for k in keys])
        keep = base.environmental_selection(F, a.pop) if len(U) > a.pop else np.arange(len(U))
        seed = LoadedPopulation(U[keep].tolist(), F[keep].tolist(), [0.0] * len(keep), {})
        print(f'seeded from {len(keys)} journal points, population {len(keep)}', flush=True)

        def batch(X):
            if time.time() > deadline:
                raise Deadline()
            out = ev([base.decode(np.asarray(x)) for x in X]).tolist()
            print(f'batch of {len(X)}: journal {ev.n} evaluations, best median ratio {min(f[0] for f in out):.4f}', flush=True)
            return out

        try:
            res = mootation.minimize(lambda x: batch([x])[0], [(0.0, 1.0)] * len(base.NAMES), 3,
                                     algorithm='ibea_eplus', pop_size=len(keep), max_evaluations=a.evals, seed=1,
                                     batch=batch, seed_population=seed, pm=1.0 / len(base.NAMES), kappa=0.05)
            print(f'finished: {res.evaluations} evaluations inside minimize; ignored knobs: {res.ignored}')
        except Deadline:
            print('stopped at the deadline; the journal keeps every evaluation')


if __name__ == '__main__':
    main()
