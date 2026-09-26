"""Re-run the base point and the L-BFGS-B best point on the train subset (bypassing the cache) and compare
with the cached values: guards against a changed input file during the experiment (validation_maps/landmarks.csv
was replaced at 01:34 while run 1 ran; run 2 reads a frozen snapshot) and checks determinism.

    python a_recheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a_tune as A  # noqa: E402
import qn_harness as H  # noqa: E402


def main():
    best = json.loads((HERE / 'a_lbfgsb_best.json').read_text())['x_exact']
    exact = {'base': A.sets_of(np.zeros(len(A.PARAMS))), 'lbfgsb_best': A.sets_of(np.array(best, float))}
    ev = H.Evaluator()
    try:
        for name, sets in exact.items():
            cached = {b: ev.cache[H.key(b, sets)] for b in A.SUBSET}
            new = ev.evaluate_many(A.SUBSET, [sets], keep=True)[0].set_index('bag')
            for b in A.SUBSET:
                c = cached[b]
                print(f'{name:12s} {b}: v {c["v_rmse"]:.5f} -> {new.loc[b, "v_rmse"]:.5f}  along {c["along_rmse"]:.4f} -> '
                      f'{new.loc[b, "along_rmse"]:.4f}  {"SAME" if abs(c["along_rmse"] - new.loc[b, "along_rmse"]) < 1e-9 and abs(c["v_rmse"] - new.loc[b, "v_rmse"]) < 1e-9 else "DIFFERENT"}',
                      flush=True)
    finally:
        ev.close()


if __name__ == '__main__':
    main()
