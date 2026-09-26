"""Black-box objective for parameter tuning experiments: the frozen submitted build (tbo_replay_final2.exe)
replayed on a set of bags with train-only maps (analysis/validation_maps), scored like the jury:

  v_rmse     : published speed vs |v_h| of /sensing/gnss/master/vel (nearest output within 0.05 s)
  along_rmse : published base_link (MGRS) vs the RTK two-antenna base_link reference of
               tools/replay/eval_base_link.py (nearest output within 0.05 s), along the body axis
  p3_rmse    : same, 3-D distance

Every (bag, parameter set) result is cached in <scratch>/qn/cache_<maps snapshot>.jsonl; all replays read a
frozen, content-hashed snapshot of analysis/validation_maps (see _snapshot).
At most JOBS=2 replay processes run at the same time (shared machine).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r'C:\MosTransHack')
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge  # noqa: E402

EXE = ROOT / 'build_core' / 'tbo_replay_final2.exe'
SCR = Path(os.environ.get('QN_SCRATCH', r'<session-scratch>'
                          r'\53977e3d-3af2-4796-9c03-8895f50bc4dd\scratchpad\qn'))
JOBS = 2


def _snapshot() -> Path:
    """Frozen copy of the train-only maps and the traction table, named by their content hash.
    Other sessions edit analysis/validation_maps while this runs (landmarks.csv was replaced at 01:34 during
    run 1, commit 8592b5f "honest validation landmarks"); every replay reads the snapshot, and the cache is
    keyed by the snapshot hash."""
    import hashlib
    import shutil
    src = sorted((ROOT / 'analysis' / 'validation_maps').glob('*.csv'))
    lut = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'config' / 'traction_lut.csv'
    h = hashlib.md5()
    for f in src + [lut, EXE]:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    d = SCR / f'maps_{h.hexdigest()[:8]}'
    if not d.exists():
        tmp = SCR / f'maps_tmp_{os.getpid()}'
        tmp.mkdir(parents=True, exist_ok=True)
        for f in src + [lut]:
            shutil.copy2(f, tmp / f.name)
        try:
            tmp.rename(d)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    return d


VAL = Path(os.environ['QN_MAPS']) if os.environ.get('QN_MAPS') else _snapshot()
os.environ['QN_MAPS'] = str(VAL)  # worker processes (spawned) use the same snapshot
LUT = VAL / 'traction_lut.csv'
CACHE = SCR / f'cache_{VAL.name}.jsonl'
KEEP = SCR / f'keep_{VAL.name}'

BASE_SETS = {'output_frame': 'mgrs', 'landmark_file': str(VAL / 'landmarks.csv'),
             'cutoff_file': str(VAL / 'cutoffs.csv'), 'dfield_file': str(VAL / 'dfield.csv'),
             'gl_stops_file': str(VAL / 'gl_stops.csv'), 'gl_cutoffs_file': str(VAL / 'gl_cutoffs.csv'),
             'speed_envelope_file': str(VAL / 'speed_envelope.csv')}
BRANCHES = ','.join(str(b) for b in sorted(VAL.glob('branch_*.csv')))


def splits():
    return json.load(open(ROOT / 'data' / 'splits.json'))


def _fmt(v: float) -> str:
    return f'{float(v):.10g}'


def key(bag: str, sets: dict) -> str:
    return bag + '|' + ';'.join(f'{k}={_fmt(v)}' for k, v in sorted(sets.items()))


def prep(bag: str):
    """Export the arrival-ordered event CSV (GNSS only in the first 5 s) and the references once."""
    ev = SCR / 'ev' / f'{bag}.csv'
    ref = SCR / 'ref' / f'{bag}.pkl'
    if not ev.exists():
        ev.parent.mkdir(parents=True, exist_ok=True)
        cpp_bridge.export_events(bag, ev)
    if not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        import eval_base_link  # pyproj
        bl = eval_base_link.reference(bag)
        d = np.load(cpp_bridge.NPZ / f'{bag}.npz')
        mv = d['sensing__gnss__master__vel']
        pd.to_pickle({'bl': bl, 'vt': mv[:, 1].copy(), 'vv': np.hypot(mv[:, 2], mv[:, 3])}, ref)
    return ev, ref


def _nearest(ref_t, out_t, tol=0.05):
    i = np.clip(np.searchsorted(out_t, ref_t), 1, len(out_t) - 1)
    i = np.where(np.abs(out_t[i - 1] - ref_t) < np.abs(out_t[i] - ref_t), i - 1, i)
    return i, np.abs(out_t[i] - ref_t) <= tol


KEEP_COLS = ['stamp_ns', 'v', 's', 's_var', 'x', 'y', 'z', 'yaw', 'flags', 'pos_valid', 'mu3', 'k', 'g', 'd']


def _run(args):
    bag, sets, keep = args if len(args) == 3 else (*args, False)
    ev, refp = SCR / 'ev' / f'{bag}.csv', SCR / 'ref' / f'{bag}.pkl'
    out = SCR / 'out' / f'{bag}_{os.getpid()}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    cpp_bridge.REPLAY_EXE = EXE
    t0 = time.time()
    o = cpp_bridge.run_replay(ev, out, map_csv=VAL / 'track_map.csv',
                              traction_csv=LUT,
                              sets={**BASE_SETS, **sets}, branches=BRANCHES)
    wall = time.time() - t0
    if keep:  # reduced output kept for error-budget analyses
        KEEP.mkdir(parents=True, exist_ok=True)
        o[[c for c in KEEP_COLS if c in o.columns]].to_pickle(KEEP / f'{bag}.pkl')
    try:
        out.unlink()
    except OSError:
        pass
    R = pd.read_pickle(refp)
    o = o.drop_duplicates('stamp_ns').sort_values('stamp_ns')
    ot = o.stamp_ns.to_numpy() * 1e-9
    res = {'bag': bag, 'wall_s': wall}
    i, ok = _nearest(R['vt'], ot)
    ev_ = o.v.to_numpy()[i[ok]] - R['vv'][ok]
    res['v_rmse'] = float(np.sqrt(np.mean(ev_ ** 2)))
    bl = R['bl']
    op = o[o.pos_valid == 1]
    if bl is not None and len(op):
        opt = op.stamp_ns.to_numpy() * 1e-9
        i, ok = _nearest(bl.t.to_numpy(), opt)
        e = op[['x', 'y', 'z']].to_numpy()[i[ok]] - bl[['x', 'y', 'z']].to_numpy()[ok]
        yaw = bl.yaw.to_numpy()[ok]
        along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
        res.update(along_rmse=float(np.sqrt(np.mean(along ** 2))),
                   p3_rmse=float(np.sqrt(np.mean(np.sum(e ** 2, 1)))), n_pos=int(ok.sum()))
    else:
        res.update(along_rmse=float('nan'), p3_rmse=float('nan'), n_pos=0)
    return res


class Evaluator:
    """evaluate(bags, sets) -> DataFrame, one row per bag; counts replays actually run."""

    def __init__(self, jobs: int = JOBS):
        SCR.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if CACHE.exists():
            for ln in CACHE.read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    self.cache[r['key']] = r
        self.pool = ProcessPoolExecutor(jobs)
        self.n_replays = 0
        self.replay_wall = 0.0

    def evaluate(self, bags, sets: dict) -> pd.DataFrame:
        return self.evaluate_many(bags, [sets])[0]

    def evaluate_many(self, bags, sets_list, keep=False) -> list:
        """All (bag, point) pairs go to the pool at once (no idle tail between points).
        keep=True: also store a reduced output of every replay of this call in <scratch>/keep (forces a run)."""
        todo, seen = [], set()
        for sets in sets_list:
            for b in bags:
                k = key(b, sets)
                if (keep or k not in self.cache) and k not in seen:
                    seen.add(k)
                    todo.append((b, sets))
        for b in {b for b, _ in todo}:
            prep(b)
        if todo:
            for (b, sets), r in zip(todo, self.pool.map(_run, [(b, s, keep) for b, s in todo])):
                r['key'] = key(b, sets)
                r['sets'] = {k: float(v) for k, v in sets.items()}
                self.cache[r['key']] = r
                self.n_replays += 1
                self.replay_wall += r['wall_s']
                with open(CACHE, 'a') as fh:
                    fh.write(json.dumps(r) + '\n')
        return [pd.DataFrame([self.cache[key(b, s)] for b in bags]) for s in sets_list]

    def close(self):
        self.pool.shutdown()
