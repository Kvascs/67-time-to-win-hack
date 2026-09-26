"""Exploration: per bag, estimate the step quantum q and check whether values sit on a lattice q*(n+phi)."""
import json

import numpy as np

ROOT = r"C:\MosTransHack"
bags = json.load(open(ROOT + r"\data\splits.json"))["train"]
np.set_printoptions(precision=6, suppress=True, linewidth=180)


def quantum(v):
    dv = np.abs(np.diff(v))
    dv = dv[(dv > 0.003) & (dv < 0.006)]
    return np.median(dv) if len(dv) > 20 else np.nan, len(dv)


for b in bags[:12]:
    z = np.load(ROOT + rf"\data\npz\{b}.npz")
    out = [b]
    for key in ("vehicle__front_bogie_velocity", "vehicle__rear_bogie_velocity"):
        v = z[key][:, 2]
        q, n = quantum(v)
        vv = v[v > 1]
        # lattice phase: circular mean of v/q
        ph = np.angle(np.mean(np.exp(2j * np.pi * vv / q))) / (2 * np.pi)
        R = np.abs(np.mean(np.exp(2j * np.pi * vv / q)))
        # refine q by maximising lattice coherence
        qs = np.linspace(q * 0.995, q * 1.005, 2001)
        coh = np.array([np.abs(np.mean(np.exp(2j * np.pi * vv / qq))) for qq in qs])
        qb = qs[np.argmax(coh)]
        out.append(f"{key[9:14]} q={q:.6f} (n={n}) coh={R:.3f} best q={qb:.7f} coh={coh.max():.3f}")
    print(" | ".join(out))
