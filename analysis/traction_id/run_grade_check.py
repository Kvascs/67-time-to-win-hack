"""Does the residual correlate with grade?  LUT fitted WITHOUT grade vs WITH grade (same d, tau).

Outputs: fig_residual_vs_grade.png, eval_grade_check.csv
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_lut import lut_sim_from_model
from lut_model import LutModel, fit_lut, score_accel
from simulate import evaluate, summarize
from tid_data import OUT, list_bags
from tm_core import apply_matched, get_arrays

tr, va = list_bags("train"), list_bags("val")
m_g = LutModel.from_json(json.loads((OUT / "lut_model.json").read_text()))
m_0, rmse0, _ = fit_lut(tr, m_g.delay, 0.5 * (m_g.tau_up + m_g.tau_dn), decim=2, fixed_kg=0.0)
m_0.tau_up, m_0.tau_dn = m_g.tau_up, m_g.tau_dn
print("no-grade LUT: val one-step", round(score_accel(m_0, va), 4), " with grade:", round(score_accel(m_g, va), 4),
      flush=True)

rows = []
for b in va:
    A = get_arrays(b)
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    msk = A.fit & ((vv > 0.3) | (A.u > 0))
    for name, m in (("with_grade", m_g), ("no_grade", m_0)):
        am, _ = m.accel_series(A.u, vv, A.gr)
        r = (A.a - apply_matched(am))[msk]
        rows.append(pd.DataFrame(dict(model=name, gr=A.gr[msk], r=r, u=A.u[msk], group=A.group)))
R = pd.concat(rows)
R["gb"] = pd.cut(R.gr * 100, np.arange(-4.5, 4.6, 0.5))
T = R.groupby(["model", "gb"], observed=True).r.agg(["mean", "std", "size"]).reset_index()
T.to_csv(OUT / "residual_vs_grade.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 6))
for name, col in (("no_grade", "tab:red"), ("with_grade", "tab:blue")):
    q = T[T.model == name]
    x = [b.mid for b in q.gb]
    ax.errorbar(x, q["mean"], yerr=q["std"] / np.sqrt(q["size"] / 200), fmt="o-", color=col, label=name, capsize=3)
    sl = np.polyfit(R[R.model == name].gr * 100, R[R.model == name].r, 1)[0]
    print(name, "residual slope vs grade [m/s2 per %]:", round(sl, 4), flush=True)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("grade felt [%] (+ uphill)")
ax.set_ylabel("mean one-step residual a_meas - a_model [m/s2]")
ax.set_title("validation residual vs grade: LUT without grade term vs with grade term")
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_residual_vs_grade.png", dpi=80)
plt.close()

E = pd.concat([evaluate(lut_sim_from_model(m_g, "lut_with_grade"), va, stride=2.0),
               evaluate(lut_sim_from_model(m_0, "lut_fit_without_grade", use_grade=False), va, stride=2.0)])
S = summarize(E)
S.to_csv(OUT / "eval_grade_check.csv", index=False)
print(S.to_string(), flush=True)
