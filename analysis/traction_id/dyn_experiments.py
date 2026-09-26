"""Quick dynamics-structure experiments with the fitted LUT table held fixed (one-step val RMSE).

Variants: regime-specific lag (tau by regime of the delayed notch), two cascaded poles, jerk limit,
regime-specific dead time. Output: dyn_experiments.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numba as nb
import numpy as np
import pandas as pd

from final_lut import load_lut_sim
from tid_data import OUT, list_bags
from tm_core import DT, apply_matched, get_arrays

sim = load_lut_sim()


@nb.njit(cache=True)
def run_dyn(table, tv0, tdv, umin, u, v, kdB, kdT, kd0, aB, aT, a0, a2, jd):
    n = len(u)
    y1 = 0.0
    y2 = 0.0
    out = np.empty(n)
    for i in range(n):
        ui = u[i]
        # regime of the (undelayed) current command decides which dead time applies
        kd = kdB if ui < 0 else (kdT if ui > 0 else kd0)
        j = i - kd
        uu = u[j] if j >= 0 else u[0]
        if v[i] <= 0.05 and uu <= 0:
            c = 0.0
        else:
            x = (v[i] - tv0) / tdv
            nv = table.shape[1]
            if x <= 0:
                c = table[uu - umin, 0]
            elif x >= nv - 1:
                c = table[uu - umin, nv - 1]
            else:
                k = int(x)
                f = x - k
                c = table[uu - umin, k] * (1 - f) + table[uu - umin, k + 1] * f
        al = aB if uu < 0 else (aT if uu > 0 else a0)
        d = al * (c - y1)
        if d > jd:
            d = jd
        elif d < -jd:
            d = -jd
        y1 += d
        y2 += a2 * (y1 - y2)
        out[i] = y2
    return out


va = list_bags("val")
data = []
for b in va:
    A = get_arrays(b)
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    m = A.fit & ((vv > 0.3) | (A.u > 0))
    cls = np.where(A.u < 0, 0, np.where(A.u == 0, 1, 2))
    data.append((A, vv, m, sim.kg[cls] * np.nan_to_num(A.gr)))


def score(tauB, tauT, tau0, tau2=None, jerk=None, dB=0.0, dT=0.0, d0=0.0):
    al = lambda t: 1 - np.exp(-DT / t)
    a2 = 1.0 if tau2 is None else al(tau2)
    jd = 1e9 if jerk is None else jerk * DT
    se = n = 0
    for A, vv, m, grav in data:
        y = run_dyn(sim.table, sim.tv0, sim.tdv, sim.umin, A.u.astype(np.int64), vv, int(round(dB / DT)),
                    int(round(dT / DT)), int(round(d0 / DT)), al(tauB), al(tauT), al(tau0), a2, jd)
        e = (apply_matched(y - grav) - A.a)[m]
        se += float(e @ e)
        n += len(e)
    return (se / n) ** 0.5


rows = []


def rec(name, **kw):
    r = score(**kw)
    rows.append(dict(variant=name, rmse=r, **{k: v for k, v in kw.items()}))
    print(name, kw, round(r, 5), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "dyn_experiments.csv", index=False)


rec("base", tauB=0.3, tauT=0.3, tau0=0.3)
for tB in (0.15, 0.2, 0.4, 0.5):
    rec("tauB", tauB=tB, tauT=0.3, tau0=0.3)
for tT in (0.2, 0.4, 0.5):
    rec("tauT", tauB=0.3, tauT=tT, tau0=0.3)
for t0 in (0.15, 0.5, 0.8):
    rec("tau0", tauB=0.3, tauT=0.3, tau0=t0)
for t1, t2 in ((0.15, 0.15), (0.2, 0.1), (0.1, 0.2), (0.25, 0.1)):
    rec("two_pole", tauB=t1, tauT=t1, tau0=t1, tau2=t2)
for J in (0.8, 1.2, 1.6, 2.5):
    rec("jerk", tauB=0.2, tauT=0.2, tau0=0.2, jerk=J)
for dB in (0.1, 0.2):
    rec("delayB", tauB=0.3, tauT=0.3, tau0=0.3, dB=dB)
for dT in (0.1, 0.2):
    rec("delayT", tauB=0.3, tauT=0.3, tau0=0.3, dT=dT)
