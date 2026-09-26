"""Exploration: fine structure of the bogie speed values (quantum of the km/h steps vs speed)."""
import json
import sys

import numpy as np

ROOT = r"C:\MosTransHack"
bags = json.load(open(ROOT + r"\data\splits.json"))["train"]
np.set_printoptions(precision=6, suppress=True, linewidth=180)

rows = []
for b in bags[: int(sys.argv[1]) if len(sys.argv) > 1 else 5]:
    z = np.load(ROOT + rf"\data\npz\{b}.npz")
    for key in ("vehicle__front_bogie_velocity", "vehicle__rear_bogie_velocity"):
        a = z[key]
        v = a[:, 2]
        dv = np.diff(v)
        vm = 0.5 * (v[1:] + v[:-1])
        ok = (dv != 0) & (vm > 1)
        rows.append(np.c_[vm[ok], np.abs(dv[ok])])
R = np.vstack(rows)
print("pairs", len(R))
# smallest nonzero step per speed band
for lo in (1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60):
    m = (R[:, 0] >= lo) & (R[:, 0] < lo + 5)
    if m.sum() < 50:
        continue
    s = np.sort(R[m, 1])
    print(f"v {lo:2d}-{lo+5:2d} km/h n={m.sum():6d} smallest steps {s[:6]}  q01 {np.quantile(s,0.01):.6f} "
          f"median {np.median(s):.4f}")
