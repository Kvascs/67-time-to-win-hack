"""Experiment (b): full-batch L-BFGS on the output-error objective of the traction LUT (analysis/traction_id/oe_torch.py).

The shipped table (lut_oe_lut_s1.json / oe_lut_s1.pt) was trained with Adam on mini-batches of 10-s open-loop
windows with early stopping. Here the same differentiable simulation (oe_torch.Model, same windows, same split:
fit = train minus train[::5], early-stop set ES = train[::5]) is minimised with torch.optim.LBFGS
(full batch, strong-Wolfe line search):

    python b_oe_lbfgs.py warm  [iters]   # start from the shipped Adam solution (s1)
    python b_oe_lbfgs.py cold  [iters]   # start from the equation-error LS table (lut_model.json), like Adam did

Loss = data MSE + OE_LAM * roughness + 1e-2 * anchor-to-EE + OE_MONO * monotonicity (s1 has zero monotonicity
violations and roughness 1.4e-2, so OE_LAM=1e-2, OE_MONO=10 are used; the data term is what is reported).
Reports ES RMSE (dense 0.1..10 s) after every iteration and the val protocol of oe_torch.evaluate_torch
(stride 2 s, horizons 1/3/5/10/30 s, grade at the predicted position) for the result.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
TID = Path(r"C:\MosTransHack\analysis\traction_id")
sys.path.insert(0, str(TID))
MODE = sys.argv[1] if len(sys.argv) > 1 else "warm"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
sys.argv = [sys.argv[0], "lut"]  # oe_torch reads its MODE from argv at import

import numpy as np  # noqa: E402
import torch  # noqa: E402

import oe_torch as OT  # noqa: E402
from lut_model import LutModel, _reg_matrix  # noqa: E402
from simulate import summarize  # noqa: E402
from tid_data import OUT, list_bags  # noqa: E402

LAM = float(os.environ.get("OE_LAM", "1e-2"))
MONO = float(os.environ.get("OE_MONO", "10"))
LOG = HERE / f"b_oe_lbfgs_{MODE}.log"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(s + "\n")


def mono_pen(T):
    tr_ = torch.relu(T[16:30] - T[17:31])
    br_ = torch.relu(T[0:14] - T[1:15])
    return (tr_ ** 2).mean() + (br_ ** 2).mean()


def main():
    torch.set_num_threads(1)
    tr = list_bags("train")
    es = tr[::5]
    fit = [b for b in tr if b not in es]
    lut_ee = LutModel.from_json(json.loads((OUT / "lut_model.json").read_text()))
    t0 = time.time()
    Wtr = OT.build_windows(fit)
    Wes = OT.build_windows(es)
    log(f"[{MODE}] windows fit {Wtr[0].shape[0]} ES {Wes[0].shape[0]} ({time.time() - t0:.0f} s); "
        f"OE_LAM {LAM} OE_MONO {MONO}")
    model = OT.Model(lut_ee, Wtr[4].shape[2], False)
    table0 = model.table.detach().clone()  # anchor = EE solution, as in oe_torch
    if MODE == "warm":
        model.load_state_dict(torch.load(TID / "oe_lut_s1.pt"))
    R, _ = _reg_matrix(3.0, 3.0)
    Rt = torch.tensor(R, dtype=torch.float32)
    params = [model.table, model.kg, model.log_tau]

    def data_loss(W, chunk=None):
        U, V, GR, OK, EX = W
        vp = model(U, V, GR, EX)
        vr, m = V[:, OT.BURN + 1:], OK[:, OT.BURN + 1:]
        e = torch.where(m, vp - torch.nan_to_num(vr), torch.zeros_like(vp))
        return (e ** 2).sum() / m.sum()

    def total():
        d = data_loss(Wtr)
        pen = LAM * ((Rt @ model.table.reshape(-1)) ** 2).mean() + 1e-2 * ((model.table - table0) ** 2).mean() \
            + MONO * mono_pen(model.table)
        return d + pen, d

    def es_rmse():
        with torch.no_grad():
            return float(data_loss(Wes).sqrt())

    # state at the start: loss, gradient norm (is the Adam solution stationary for the full-batch objective?)
    t1 = time.time()
    L, d = total()
    L.backward()
    gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in params)))
    log(f"[{MODE}] start: loss {float(L):.5f} data rmse(fit) {float(d.sqrt()):.4f} ES rmse {es_rmse():.4f} "
        f"|grad| {gn:.3e}; one full-batch loss+grad {time.time() - t1:.1f} s")
    for p in params:
        p.grad = None

    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=1, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)
    ncl = [0]

    def closure():
        opt.zero_grad()
        L, _ = total()
        L.backward()
        ncl[0] += 1
        return L

    best = (es_rmse(), {k: v.detach().clone() for k, v in model.state_dict().items()}, 0)
    t2 = time.time()
    for it in range(1, ITERS + 1):
        opt.step(closure)
        with torch.no_grad():
            L, d = total()
        r = es_rmse()
        log(f"[{MODE}] it {it:3d} evals {ncl[0]:4d} loss {float(L):.5f} fit rmse {float(d.sqrt()):.4f} ES rmse {r:.4f} "
            f"tau {float(torch.exp(model.log_tau)):.4f} kg {np.round(model.kg.detach().numpy(), 3).tolist()} "
            f"({time.time() - t2:.0f} s)")
        if r < best[0] - 1e-5:
            best = (r, {k: v.detach().clone() for k, v in model.state_dict().items()}, it)
    log(f"[{MODE}] best ES rmse {best[0]:.4f} at it {best[2]}; {ncl[0]} full-batch evaluations, {time.time() - t2:.0f} s")
    last = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(last, HERE / f"b_oe_lbfgs_{MODE}_last.pt")
    torch.save(best[1], HERE / f"b_oe_lbfgs_{MODE}_best.pt")
    Ss = []
    for tag, sd in (("last", last), ("best_es", best[1])):
        if tag == "best_es" and best[2] == ITERS:
            continue
        model.load_state_dict(sd)
        E = OT.evaluate_torch(model, list_bags("val"), f"lbfgs_{MODE}_{tag}")
        Ss.append(summarize(E))
    import pandas as pd
    S = pd.concat(Ss, ignore_index=True)
    S.to_csv(HERE / f"b_eval_val_{MODE}.csv", index=False)
    log(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
