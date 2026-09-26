"""Export the final model (OE-LUT) to traction_model_params.json + tables for the C++ port, check that the
scalar reference implementation (traction_model.py) reproduces the vectorised numba simulator, and plot
the final static map.  usage: python export_final.py [model_json]"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_lut import lut_sim_from_model
from lut_model import LutModel
from route_map import grade_profile
from simulate import _grade_grid, run_states, sim_hammerstein
from tid_data import OUT, list_bags
from tm_core import DT, get_arrays
from traction_model import TractionModel

src = sys.argv[1] if len(sys.argv) > 1 else "lut_oe_lut_s1.json"
NAME = sys.argv[2] if len(sys.argv) > 2 else "lut_oe_s1"
m = LutModel.from_json(json.loads((OUT / src).read_text()))
prof = grade_profile()
gains = json.loads((OUT / "group_gains.json").read_text()) if (OUT / "group_gains.json").exists() else {}


def horizon_table(csv, model):
    if not (OUT / csv).exists():
        return None
    S = pd.read_csv(OUT / csv)
    S = S[S.model == model]
    return {f"{h:g}s": dict(rmse=round(r.rmse, 4), mae=round(r.mae, 4), bias=round(r.bias, 4), dist_rmse_m=round(r.d_rmse, 3))
            for h, r in zip(S.H, S.itertuples())}


params = dict(
    model="OE-LUT: 2-D notch x speed table fitted by differentiable open-loop simulation (10-s windows)",
    units=dict(a="m/s^2", v="m/s", s="m", grade="dimensionless dh/ds (+uphill in travel direction)"),
    lut=dict(u_min=-15, u_max=15, v_knots=[float(x) for x in json.loads((OUT / src).read_text())["v_knots"]],
             table=np.round(m.table, 5).tolist(),
             note="rows = notch -15..15, cols = v_knots; linear interpolation in v, clamp outside; flat-track "
                  "steady-state acceleration incl. running resistance"),
    kg=[round(float(x), 4) for x in m.kg],
    kg_order=["brake (u<0)", "coast (u=0)", "traction (u>0)"],
    tau=round(float(m.tau_up), 4),
    delay=0.0,
    v_still=0.05,
    grade=dict(s0=float(prof.s.iloc[0]), ds=float(prof.s.iloc[1] - prof.s.iloc[0]),
               values=np.round(prof.grade.to_numpy(), 6).tolist(),
               note="s = along-track distance from the western terminus (centerline.csv, UTM 37N / EPSG:32637); "
                    "grade felt = values(s) * dir, dir = +1 when s increases (eastbound)"),
    adapter=dict(T=300.0, lam=0.02, g_min=0.7, g_max=1.3, window_s=1.0,
                 note="traction/brake gains by exponentially forgotten LS on clean samples; bias adaptation "
                      "was tested and does NOT help (not recommended)"),
    group_gain_priors={k: dict(g_trac=round(v["g_trac"], 3), g_brake=round(v["g_brake"], 3)) for k, v in gains.items()},
    noise=dict(one_step_accel_rmse=None, speed_rmse_vs_horizon=horizon_table(f"eval_gain_{NAME}.csv", NAME),
               speed_rmse_vs_horizon_adapted=horizon_table(f"eval_gain_{NAME}.csv", f"{NAME}+gain300s"),
               random_walk_q="speed error grows ~ 0.11*sqrt(H) m/s -> process noise q ~ 0.012 (m/s)^2/s on v "
                             "while propagating with the model"),
    detector=dict(innovation="a_meas(1-s causal LS slope of wheel speed) - model (same weights)",
                  ema_T_s=1.0, threshold=0.3,
                  note="EMA(|innovation|) > 0.3 m/s^2 -> model not in control (automation / emergency brake): "
                       "stop trusting the model, trust bogie-consistent wheels"),
)
out = OUT / "traction_model_params.json"
out.write_text(json.dumps(params, indent=1))
print("written", out, "size %.0f kB" % (out.stat().st_size / 1024), flush=True)

# CSV copies for the C++ side
pd.DataFrame(m.table, index=np.arange(-15, 16), columns=[f"v{v:g}" for v in params["lut"]["v_knots"]]).to_csv(
    OUT / "traction_lut_table.csv", float_format="%.5f")

# ---- consistency check: scalar reference vs numba ------------------------------------------------
sim = lut_sim_from_model(m, "final")
tm = TractionModel(out)
b = list_bags("val")[0]
A = get_arrays(b)
vv = np.where(np.isfinite(A.v), A.v, A.vw)
a_nb, y_nb = sim.accel_series(A)
a_py = np.empty(A.n)
tm.reset()
gr = np.nan_to_num(A.gr)
for i in range(A.n):
    a_py[i] = tm.step(DT, int(A.u[i]), float(vv[i]), float(gr[i]))
d1 = np.max(np.abs(a_py - a_nb))
print("max |a_python - a_numba| along bag %s: %.2e" % (b, d1), flush=True)
# rollouts from a few start points
gs0, gds, gg = _grade_grid()
i0s = np.array([2000, 6000, 12000, 18000])
i0s = i0s[i0s < A.n - 700]
H = 600
rec = np.array([200, 600], dtype=np.int64)
kd, a_up, a_dn = sim.params()
vout, _ = sim_hammerstein(sim.table, sim.tv0, sim.tdv, sim.umin, sim.kg, 0, a_up, a_dn, A.u.astype(np.int64), A.dirn,
                          i0s.astype(np.int64), A.v[i0s], A.s[i0s], y_nb[i0s], np.zeros(len(i0s)), H, DT, gs0, gds, gg,
                          True, rec)
worst = 0.0
for k, i0 in enumerate(i0s):
    tm.reset()
    for i in range(i0 + 1):
        tm.step(DT, int(A.u[i]), float(vv[i]), float(gr[i]))
    vp = tm.predict_speed(float(A.v[i0]), A.u[i0 + 1:i0 + 1 + H], DT, float(A.s[i0]), float(A.dirn[i0]))
    worst = max(worst, abs(vp[199] - vout[k, 0]), abs(vp[599] - vout[k, 1]))
print("max |v_python - v_numba| over rollouts (10 s, 30 s): %.2e" % worst, flush=True)
params["consistency_check"] = dict(bag=b, max_abs_accel_diff=float(d1), max_abs_rollout_speed_diff=float(worst))
out.write_text(json.dumps(params, indent=1))

# ---- plots of the final static map ---------------------------------------------------------------
vk = np.array(params["lut"]["v_knots"])
fig, axs = plt.subplots(1, 3, figsize=(24, 7))
cm = plt.get_cmap("viridis")
for i, u in enumerate(range(-15, 0)):
    axs[0].plot(vk, m.table[u + 15], color=cm(i / 14), label=f"{u}" if u % 2 else None)
axs[0].set_title("final OE-LUT: brake notches (flat track)")
for i, u in enumerate(range(0, 16)):
    axs[1].plot(vk, m.table[u + 15], color=cm(i / 15), label=f"{u}" if u % 3 == 0 else None)
axs[1].set_title("final OE-LUT: coast (0) and traction notches (flat track)")
for ax in axs[:2]:
    ax.set_xlabel("v [m/s]")
    ax.set_ylabel("a_ss [m/s2]")
    ax.grid()
    ax.legend(fontsize=7, ncol=2)
im = axs[2].pcolormesh(np.r_[vk, vk[-1] + 1.5] - 0.25, np.arange(-15, 17) - 0.5, m.table, cmap="RdBu_r", vmin=-1.6,
                       vmax=1.6)
plt.colorbar(im, ax=axs[2])
axs[2].set_xlabel("v knot [m/s]")
axs[2].set_ylabel("notch")
axs[2].set_title(f"OE-LUT table; tau = {m.tau_up:.3f} s, kg = {np.round(m.kg, 2)}")
plt.tight_layout()
plt.savefig(OUT / "fig_final_lut.png", dpi=80)
print("done", flush=True)
