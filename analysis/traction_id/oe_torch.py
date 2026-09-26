"""Differentiable-simulation (output-error) training of the Hammerstein LUT, optionally + MLP residual.

Windows of 10 s (dt = 0.1 s) start every 3 s on TRAIN bags (early stopping on held-out train bags):
    burn-in 2 s with the true speed (teacher forcing) to initialise the lag state, then free-running:
    c = LUT(u, v) (0 at standstill with brake), y += alpha (c - y), a = y - kg[regime] gr + NN(features),
    v <- snap(v + a dt)
Loss: mean squared open-loop speed error over all steps of the window (all horizons 0.1..10 s weighted
equally) + smoothness penalty on the table (same structure as the least-squares fit).
The grade along the window is taken at the true position (exogenous) during training; evaluation uses
the standard numba simulator with the grade looked up at the predicted position.

usage: python oe_torch.py lut|lut_nn|nn_on_lut
  lut       : train LUT + kg + tau                      -> lut_oe_model.json
  lut_nn    : train LUT + kg + tau + MLP residual jointly -> lut_nn_oe.pt/json
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch

from lut_model import NU, NV, U_MIN, V_KNOTS, LutModel, _reg_matrix
from ml_model import exog_features
from simulate import V_STILL, start_indices
from tid_data import OUT, list_bags
from tm_core import DT, get_arrays

torch.set_num_threads(1)
torch.manual_seed(0)
MODE = sys.argv[1] if len(sys.argv) > 1 else "lut"
DTS = 0.1
STEP = int(round(DTS / DT))
K = int(round(10.0 / DTS))
BURN = int(round(2.0 / DTS))
VK = torch.tensor(V_KNOTS, dtype=torch.float32)


def build_windows(bags, stride=3.0):
    U, V, GR, OK, EX = [], [], [], [], []
    for b in bags:
        A = get_arrays(b)
        X_ex, names = exog_features(A.u, A.t)
        i0s = start_indices(A, stride)
        offs = STEP * np.arange(-BURN, K + 1)
        idx = i0s[:, None] + offs[None, :]
        good = (idx[:, 0] >= 0) & (idx[:, -1] < A.n)
        idx = idx[good]
        vref = A.v[idx]
        fit = A.fit[idx] & np.isfinite(vref)
        auto = A.auto[idx].any(1)
        still = (np.nan_to_num(vref).max(1) < 0.2) & ~(A.u[idx] > 0).any(1)
        keep = ~auto & ~still & fit[:, BURN] & (fit[:, BURN:].mean(1) > 0.8) & np.isfinite(vref[:, :BURN + 1]).all(1)
        idx = idx[keep]
        U.append(A.u[idx])
        V.append(A.v[idx])
        GR.append(np.nan_to_num(A.gr[idx]))
        OK.append((A.fit[idx] & np.isfinite(A.v[idx])))
        EX.append(X_ex[idx].astype(np.float32))
    f = lambda L, dt: torch.tensor(np.concatenate(L), dtype=dt)
    return f(U, torch.long), f(V, torch.float32), f(GR, torch.float32), f(OK, torch.bool), f(EX, torch.float32)


class Model(torch.nn.Module):
    def __init__(self, lut: LutModel, n_ex: int, use_nn: bool):
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor(lut.table, dtype=torch.float32))
        self.kg = torch.nn.Parameter(torch.tensor(lut.kg, dtype=torch.float32))
        self.log_tau = torch.nn.Parameter(torch.tensor(np.log(lut.tau_up), dtype=torch.float32))
        self.use_nn = use_nn
        if use_nn:
            self.nn = torch.nn.Sequential(torch.nn.Linear(n_ex + 3, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32),
                                          torch.nn.Tanh(), torch.nn.Linear(32, 1))
            torch.nn.init.zeros_(self.nn[-1].weight)
            torch.nn.init.zeros_(self.nn[-1].bias)
            self.register_buffer("mu", torch.zeros(n_ex + 3))
            self.register_buffer("sd", torch.ones(n_ex + 3))

    def lut(self, u, v):
        iu = (u - U_MIN).clamp(0, NU - 1)
        vc = v.clamp(0, float(V_KNOTS[-1]) - 1e-4)
        j = torch.searchsorted(VK, vc.contiguous(), right=True) - 1
        j = j.clamp(0, NV - 2)
        w = (vc - VK[j]) / (VK[j + 1] - VK[j])
        row = self.table[iu]
        c0 = torch.gather(row, 1, j[:, None])[:, 0]
        c1 = torch.gather(row, 1, (j + 1)[:, None])[:, 0]
        return c0 * (1 - w) + c1 * w

    def forward(self, U, V, GR, EX):
        """Returns predicted speed for steps 1..K (free-running after burn-in)."""
        alpha = 1 - torch.exp(-DTS / torch.exp(self.log_tau))
        y = torch.zeros(U.shape[0])
        for k in range(BURN + 1):  # teacher-forced burn-in
            u, v = U[:, k], V[:, k]
            c = torch.where((v <= V_STILL) & (u <= 0), torch.zeros_like(v), self.lut(u, v))
            y = y + alpha * (c - y)
        v = V[:, BURN]
        out = []
        for k in range(BURN + 1, BURN + K + 1):
            u = U[:, k]
            c = torch.where((v <= V_STILL) & (u <= 0), torch.zeros_like(v), self.lut(u, v))
            y = y + alpha * (c - y)
            reg = torch.where(u < 0, 0, torch.where(u == 0, 1, 2))
            a = y - self.kg[reg] * GR[:, k]
            if self.use_nn:
                f = torch.cat([EX[:, k], v[:, None], GR[:, k:k + 1], y[:, None]], 1)
                a = a + self.nn((f - self.mu) / self.sd)[:, 0]
            vn = v + a * DTS
            vn = torch.where((vn <= V_STILL) & (a < 0), torch.zeros_like(vn), vn)
            vn = torch.relu(vn)
            out.append(vn)
            v = vn
        return torch.stack(out, 1)


def main():
    tr = list_bags("train")
    es = tr[::5]
    fit = [b for b in tr if b not in es]
    lut = LutModel.from_json(json.loads((OUT / "lut_model.json").read_text()))
    t0 = time.time()
    Wtr = build_windows(fit)
    Wes = build_windows(es)
    print("windows", Wtr[0].shape[0], Wes[0].shape[0], "%.0f s" % (time.time() - t0), flush=True)
    use_nn = MODE in ("lut_nn", "nn_on_lut")
    model = Model(lut, Wtr[4].shape[2], use_nn)
    if use_nn:
        ex = Wtr[4][:, BURN:, :].reshape(-1, Wtr[4].shape[2])
        vv = Wtr[1][:, BURN:].reshape(-1, 1)
        gg = Wtr[2][:, BURN:].reshape(-1, 1)
        feats = torch.cat([ex, torch.nan_to_num(vv), gg, torch.zeros_like(gg)], 1)
        model.mu.copy_(feats.mean(0))
        model.sd.copy_(feats.std(0) + 1e-3)
        model.mu[-1] = 0.0  # lag state y: physical scale instead of the (all-zero) placeholder
        model.sd[-1] = 0.5
    R, _ = _reg_matrix(3.0, 3.0)
    Rt = torch.tensor(R, dtype=torch.float32)
    table0 = model.table.detach().clone()
    params = [{"params": [model.table], "lr": 2e-3}, {"params": [model.kg, model.log_tau], "lr": 5e-3}]
    if MODE == "nn_on_lut":
        model.table.requires_grad_(False)
        model.kg.requires_grad_(False)
        model.log_tau.requires_grad_(False)
        params = []
    if use_nn:
        params.append({"params": model.nn.parameters(), "lr": 2e-3})
    opt = torch.optim.Adam(params)

    def loss_fn(W, idx):
        U, V, GR, OK, EX = (w[idx] for w in W)
        vp = model(U, V, GR, EX)
        vr = V[:, BURN + 1:]
        m = OK[:, BURN + 1:]
        e = torch.where(m, vp - torch.nan_to_num(vr), torch.zeros_like(vp))
        return (e ** 2).sum() / m.sum(), e, m

    def es_rmse():
        with torch.no_grad():
            tot = cnt = 0.0
            per_h = torch.zeros(K)
            per_n = torch.zeros(K)
            for k in range(0, Wes[0].shape[0], 4096):
                idx = torch.arange(k, min(k + 4096, Wes[0].shape[0]))
                l, e, m = loss_fn(Wes, idx)
                per_h += (e ** 2).sum(0)
                per_n += m.sum(0)
            r = (per_h / per_n).sqrt()
            return float((per_h.sum() / per_n.sum()).sqrt()), r

    base, rh = es_rmse()
    print("ES rmse before: %.4f  @1s %.3f @5s %.3f @10s %.3f" % (base, rh[9], rh[49], rh[99]), flush=True)
    best = (base, {k: v.detach().clone() for k, v in model.state_dict().items()})
    n = Wtr[0].shape[0]
    bs = 1024
    lam = float(os.environ.get("OE_LAM", "1e-3"))      # smoothness weight
    lam_mono = float(os.environ.get("OE_MONO", "0"))   # monotonicity-in-notch weight
    tag = os.environ.get("OE_TAG", "")

    def mono_pen(T):
        # traction rows (u=1..15, idx 16..30): non-decreasing in notch; brake rows (u=-15..-1, idx 0..14):
        # deeper brake -> more negative, i.e. table[u] <= table[u+1]; coast row excluded
        tr_ = torch.relu(T[16:30] - T[17:31])
        br_ = torch.relu(T[0:14] - T[1:15])
        return (tr_ ** 2).mean() + (br_ ** 2).mean()
    bad = 0
    for ep in range(40):
        perm = torch.randperm(n)
        t1 = time.time()
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            opt.zero_grad()
            l, _, _ = loss_fn(Wtr, idx)
            if MODE != "nn_on_lut":
                l = l + lam * ((Rt @ model.table.reshape(-1)) ** 2).mean() + 1e-2 * ((model.table - table0) ** 2).mean()
                if lam_mono > 0:
                    l = l + lam_mono * mono_pen(model.table)
            l.backward()
            opt.step()
        r, rh = es_rmse()
        print("ep %d  ES rmse %.4f  @1s %.3f @5s %.3f @10s %.3f  tau %.3f kg %s  (%.0f s)" % (
            ep, r, rh[9], rh[49], rh[99], float(torch.exp(model.log_tau)), np.round(model.kg.detach().numpy(), 2),
            time.time() - t1), flush=True)
        if r < best[0] - 1e-4:
            best = (r, {k: v.detach().clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best[1])
    torch.save(model.state_dict(), OUT / f"oe_{MODE}{tag}.pt")
    tau = float(torch.exp(model.log_tau))
    m = LutModel(model.table.detach().numpy().astype(float), model.kg.detach().numpy().astype(float), 0.0, tau, tau)
    js = m.to_json()
    js["objective"] = "open-loop speed error over 10-s windows (dense), differentiable simulation"
    (OUT / f"lut_oe_{MODE}{tag}.json").write_text(json.dumps(js))
    print("saved; best ES rmse", best[0], flush=True)




# ------------------------------------------------------------------------------------------------
# standard-protocol evaluation of a torch model (stride 2 s, horizons up to 30 s, grade looked up at
# the PREDICTED position, dt = 0.1 s)
# ------------------------------------------------------------------------------------------------
def evaluate_torch(model, bags, name, horizons=(1.0, 3.0, 5.0, 10.0, 30.0), stride=2.0):
    from route_map import grade_profile
    from simulate import reference_windows
    prof = grade_profile()
    gs = torch.tensor(prof["s"].to_numpy(), dtype=torch.float32)
    gv = torch.tensor(prof["grade"].to_numpy(), dtype=torch.float32)

    def grade_at(s):
        x = ((s - gs[0]) / (gs[1] - gs[0])).clamp(0, len(gv) - 1 - 1e-4)
        i = x.floor().long()
        f = x - i
        return gv[i] * (1 - f) + gv[i + 1] * f

    rows = []
    Kmax = int(round(max(horizons) / DTS))
    for b in bags:
        A = get_arrays(b)
        X_ex, _ = exog_features(A.u, A.t)
        i0s = start_indices(A, stride)
        i0s = i0s[i0s - STEP * BURN >= 0]
        if len(i0s) == 0:
            continue
        ref = reference_windows(A, i0s, horizons)
        offs = STEP * np.arange(-BURN, Kmax + 1)
        idx = np.clip(i0s[:, None] + offs[None, :], 0, A.n - 1)
        U = torch.tensor(A.u[idx], dtype=torch.long)
        V = torch.tensor(np.nan_to_num(np.where(np.isfinite(A.v), A.v, A.vw)[idx]), dtype=torch.float32)
        GRt = torch.tensor(np.nan_to_num(A.gr[idx]), dtype=torch.float32)
        EX = torch.tensor(X_ex[idx].astype(np.float32))
        dirn = torch.tensor(A.dirn[i0s], dtype=torch.float32)
        s = torch.tensor(np.nan_to_num(A.s[i0s]), dtype=torch.float32)
        with torch.no_grad():
            alpha = 1 - torch.exp(-DTS / torch.exp(model.log_tau))
            y = torch.zeros(len(i0s))
            for k in range(BURN + 1):
                u, v = U[:, k], V[:, k]
                c = torch.where((v <= V_STILL) & (u <= 0), torch.zeros_like(v), model.lut(u, v))
                y = y + alpha * (c - y)
            v = V[:, BURN]
            vout = {}
            dout = {}
            d = torch.zeros(len(i0s))
            for k in range(1, Kmax + 1):
                u = U[:, BURN + k]
                c = torch.where((v <= V_STILL) & (u <= 0), torch.zeros_like(v), model.lut(u, v))
                y = y + alpha * (c - y)
                reg = torch.where(u < 0, 0, torch.where(u == 0, 1, 2))
                gr = grade_at(s) * dirn
                a = y - model.kg[reg] * gr
                if model.use_nn:
                    f = torch.cat([EX[:, BURN + k], v[:, None], gr[:, None], y[:, None]], 1)
                    a = a + model.nn((f - model.mu) / model.sd)[:, 0]
                vn = v + a * DTS
                vn = torch.relu(torch.where((vn <= V_STILL) & (a < 0), torch.zeros_like(vn), vn))
                d = d + 0.5 * (v + vn) * DTS
                s = s + dirn * 0.5 * (v + vn) * DTS
                v = vn
                for H in horizons:
                    if k == int(round(H / DTS)):
                        vout[H] = v.numpy().copy()
                        dout[H] = d.numpy().copy()
        for H in horizons:
            R = ref[H]
            ok = R["ok"] & np.isfinite(vout[H])
            rows.append(pd.DataFrame(dict(model=name, bag=b, group=A.group, H=H, i0=i0s[ok], v0=A.v[i0s[ok]],
                                          u0=A.u[i0s[ok]], ev=vout[H][ok] - R["v1"][ok], ed=dout[H][ok] - R["d1"][ok],
                                          still=R["still"][ok])))
    return pd.concat(rows, ignore_index=True)


def run_eval(mode_name):
    from simulate import summarize
    lut = LutModel.from_json(json.loads((OUT / "lut_model.json").read_text()))
    sd = torch.load(OUT / f"oe_{mode_name}.pt")
    use_nn = any(k.startswith("nn.") for k in sd)
    n_ex = sd["mu"].shape[0] - 3 if use_nn else 26
    model = Model(lut, n_ex, use_nn)
    model.load_state_dict(sd)
    E = evaluate_torch(model, list_bags("val"), f"oe_{mode_name}")
    E.to_pickle(OUT / "cache" / f"eval_oe_{mode_name}_torch.pkl")
    S = summarize(E)
    S.to_csv(OUT / f"eval_oe_{mode_name}_torch.csv", index=False)
    print(S.to_string(), flush=True)


if __name__ == "__main__":
    if MODE.startswith("eval:"):
        run_eval(MODE.split(":", 1)[1])
    else:
        main()
        run_eval(MODE + os.environ.get("OE_TAG", ""))
