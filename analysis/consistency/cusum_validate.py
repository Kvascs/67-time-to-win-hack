"""Checks the Python re-implementation of the joint monitor (cusum_theory.reconstruct) against what
the frozen C++ binary actually did, on train and val.

    python analysis/consistency/cusum_validate.py     (needs cusum_theory.py: cache/cusum_reconstruction.parquet)

C++ latch onset   = first posterior output with P(both bad) = 1.00000 (the latch sets mu3 = 1 - 4e-6)
C++ cmd flag      = CMD_INCONSISTENT flag onsets (the flag is held 8 s after the last alarm)
Python joint alarm matched if a C++ latch onset lies within 0.5 s; Python controller alarms are merged
into events over the same 8 s hold and matched to flag onsets within 1 s.
Writes results/cusum_validation.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def onsets(m: np.ndarray) -> np.ndarray:
    return np.flatnonzero(m & ~np.concatenate([[False], m[:-1]]))


def main():
    R = pd.read_parquet(C.CACHE / 'cusum_reconstruction.parquet',
                        columns=['bag', 'split', 't', 'joint', 'cmd_alarm', 'mu3_po', 'flags_po'])
    out = {}
    rows = []
    for bag, g in R.groupby('bag', sort=False):
        t = g.t.to_numpy()
        cpp_lat = t[onsets(g.mu3_po.to_numpy() >= 0.99999)]
        py_joint = t[np.flatnonzero(g.joint.to_numpy())]
        cpp_cmd = t[onsets((g.flags_po.to_numpy() & C.F_CMD_INCONS) > 0)]
        ia = np.flatnonzero(g.cmd_alarm.to_numpy())
        py_cmd = t[ia[np.concatenate([[True], np.diff(t[ia]) > 8.0])]] if len(ia) else np.array([])
        rows.append({'bag': bag, 'split': g.split.iloc[0],
                     'cpp_latches': len(cpp_lat), 'py_joint': len(py_joint),
                     'cpp_latches_reproduced': int(sum(np.any(np.abs(py_joint - x) <= 0.5) for x in cpp_lat)),
                     'cpp_cmd_onsets': len(cpp_cmd), 'py_cmd_events': len(py_cmd),
                     'cpp_cmd_reproduced': int(sum(np.any(np.abs(py_cmd - x) <= 1.0) for x in cpp_cmd)),
                     'py_cmd_confirmed_by_cpp': int(sum(np.any(np.abs(cpp_cmd - x) <= 1.0) for x in py_cmd))})
    df = pd.DataFrame(rows)
    for split in ('train', 'val'):
        s = df[df.split == split]
        out[split] = {k: int(s[k].sum()) for k in ('cpp_latches', 'py_joint', 'cpp_latches_reproduced', 'cpp_cmd_onsets',
                                                   'py_cmd_events', 'cpp_cmd_reproduced', 'py_cmd_confirmed_by_cpp')}
        out[split]['latch_bags'] = df[(df.split == split) & ((df.cpp_latches > 0) | (df.py_joint > 0))][
            ['bag', 'cpp_latches', 'py_joint', 'cpp_latches_reproduced']].to_dict('records')
    # false C++ latches on train: GNSS labels from cusum_theory.py, 95 % upper bound (rule of three if 0)
    tr = R[R.split == 'train']
    dt = np.diff(tr.t.to_numpy())
    same_bag = tr.bag.to_numpy()[1:] == tr.bag.to_numpy()[:-1]
    op_h = float(dt[same_bag & (dt > 0) & (dt < 1.0)].sum() / 3600)
    lat = pd.read_csv(C.RESULTS / 'cusum_cpp_latches_train.csv')
    n_false = int((~lat.gnss_confirms).sum())
    out['train_false_latches'] = {'latches': len(lat), 'unconfirmed': n_false, 'operating_hours': op_h,
                                  'rate_upper95_per_h': (3.0 if n_false == 0 else n_false + 2 * np.sqrt(n_false)) / op_h}
    C.write_json(out, C.RESULTS / 'cusum_validation.json')
    import json
    print(json.dumps(out, indent=1, default=int))


if __name__ == '__main__':
    main()
