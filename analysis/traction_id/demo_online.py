"""Online (causal, scalar) use of traction_model.py on validation bags: model + windowed acceleration +
GainAdapter, as the estimator would run it (wheel speeds as measurement, GNSS only for scoring here).
Prints the adapted gains per bag and bridging errors from a few start points.  Output: demo_online.csv"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from tid_data import OUT, list_bags
from tm_core import DT, get_arrays
from traction_model import BRAKE, COAST, TRAC, GainAdapter, TractionModel, WindowedAccel, regime

rows = []
for b in list_bags("val"):
    A = get_arrays(b)
    tm = TractionModel()
    wa = WindowedAccel(DT, 1.0)
    ga = GainAdapter()
    vw = A.vw  # calibrated wheel speed (online: wheel speed / per-vehicle factor)
    gr = np.nan_to_num(A.gr)  # online: grade from map at dead-reckoned s
    errs = {H: [] for H in (5.0, 10.0)}
    for i in range(A.n):
        a = tm.step(DT, int(A.u[i]), float(vw[i]), float(gr[i]))
        r = regime(int(A.u[i]))
        off = tm.y[COAST] - tm.kg[r] * gr[i]
        out = wa.push(float(vw[i]), (tm.y[TRAC], tm.y[BRAKE], off))
        if out is not None:
            # clean = both bogies consistent (here: offline clean flag) and not in an automation episode
            ga.update(DT, out[0], out[1], out[2], out[3], valid=bool(A.clean[i] and not A.auto[i] and vw[i] > 0.3))
        tm.g_t, tm.g_b = ga.gains
        if i % 400 == 0 and i + 600 < A.n and A.fit[i] and np.isfinite(A.v[i]):
            vp = tm.predict_speed(float(A.v[i]), A.u[i + 1:i + 201], DT, float(A.s[i]), float(A.dirn[i]))
            for H, k in ((5.0, 99), (10.0, 199)):
                if np.isfinite(A.v[i + k + 1]) and not A.auto[i:i + k + 2].any():
                    errs[H].append(vp[k] - A.v[i + k + 1])
    g = ga.gains
    rows.append(dict(bag=b, group=A.group, g_trac=round(g[0], 3), g_brake=round(g[1], 3),
                     rmse5=np.sqrt(np.mean(np.square(errs[5.0]))), rmse10=np.sqrt(np.mean(np.square(errs[10.0]))),
                     n=len(errs[10.0])))
    print(rows[-1], flush=True)
D = pd.DataFrame(rows)
D.to_csv(OUT / "demo_online.csv", index=False)
print(D.groupby("group")[["g_trac", "g_brake", "rmse5", "rmse10"]].mean().round(3))
