"""Train LightGBM models (direct and residual-on-LUT) and evaluate open-loop rollouts on val.

usage: python run_ml.py direct|residual
Early stopping uses a held-out subset of TRAIN bags (every 5th), never the validation bags.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml_model import MLSim, bag_design, training_matrix
from simulate import evaluate, summarize
from tid_data import OUT, list_bags

mode = sys.argv[1] if len(sys.argv) > 1 else "direct"
target = "residual" if mode.endswith("residual") else "direct"
tr = list_bags("train")
va = list_bags("val")
es_bags = tr[::5]
fit_bags = [b for b in tr if b not in es_bags]

base = None
if target == "residual":
    from final_lut import load_lut_sim
    base = load_lut_sim()

t0 = time.time()
Xtr, ytr, names, _ = training_matrix(fit_bags, base, decim=4, target=target)
Xes, yes, _, _ = training_matrix(es_bags, base, decim=4, target=target)
print("design", Xtr.shape, Xes.shape, "%.0f s" % (time.time() - t0), flush=True)
if mode.startswith("mlp"):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    tgt = mode.split("_")[1] if "_" in mode else "direct"
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                              torch.nn.Tanh(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)[:, None]
    Xe = torch.tensor((Xes - mu) / sd, dtype=torch.float32)
    ye = torch.tensor(yes, dtype=torch.float32)[:, None]
    best, best_state, bad = 1e9, None, 0
    for ep in range(200):
        perm = torch.randperm(len(Xt))
        net.train()
        for k in range(0, len(Xt), 1024):
            idx = perm[k:k + 1024]
            opt.zero_grad()
            loss = torch.mean((net(Xt[idx]) - yt[idx]) ** 2)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            le = float(torch.mean((net(Xe) - ye) ** 2)) ** 0.5
        if le < best - 1e-5:
            best, bad = le, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0:
            print("epoch", ep, "es rmse", round(le, 4), flush=True)
        if bad >= 8:
            break
    net.load_state_dict(best_state)
    print("mlp es rmse", best, flush=True)
    W = {k: v.numpy().tolist() for k, v in net.state_dict().items()}
    (OUT / f"{mode}_weights.json").write_text(json.dumps(dict(mu=mu.tolist(), sd=sd.tolist(), names=names, weights=W)))

    def pred(X):
        with torch.no_grad():
            return net(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy()[:, 0]
    sim = MLSim(pred, mode, base_sim=base, target="residual" if mode == "mlp_residual" else "direct")
else:
  params = dict(objective="l2", learning_rate=0.08, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1)
  dtr = lgb.Dataset(Xtr, ytr, feature_name=names)
  des = lgb.Dataset(Xes, yes, reference=dtr)
  bst = lgb.train(params, dtr, num_boost_round=600, valid_sets=[des],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
  print("trees", bst.best_iteration, "es rmse", bst.best_score["valid_0"]["l2"] ** 0.5, flush=True)
  bst.save_model(str(OUT / f"gbm_{mode}.txt"), num_iteration=bst.best_iteration)
  imp = pd.Series(bst.feature_importance("gain"), index=names).sort_values(ascending=False)
  imp.to_csv(OUT / f"gbm_{mode}_importance.csv")
  print(imp.head(12), flush=True)

  pred = lambda X: bst.predict(X, num_iteration=bst.best_iteration, num_threads=1)
  sim = MLSim(pred, f"gbm_{mode}", base_sim=base, target=target)

# one-step accel RMSE on val (matched: GBM is fitted to the smoothed target directly)
from tm_core import apply_matched, get_arrays
se = n = 0
for b in va:
    A = get_arrays(b)
    a, _ = sim.accel_series(A)
    if target == "residual":
        # base part must be matched-filtered; ML part already predicts the smoothed residual

        _, X, _, basea = bag_design(b, base)
        a = pred(X) + apply_matched(basea)
    else:
        a = pred(bag_design(b, None)[1])
    vv = np.where(np.isfinite(A.v), A.v, A.vw)
    m = A.fit & ((vv > 0.3) | (A.u > 0))
    e = (a - A.a)[m]
    se += float(e @ e); n += len(e)
print("val one-step rmse", (se / n) ** 0.5, flush=True)

t0 = time.time()
E = evaluate(sim, va, stride=2.0)
E.to_pickle(OUT / "cache" / f"eval_{sim.name}.pkl")
S = summarize(E)
S["onestep_rmse"] = (se / n) ** 0.5
print(S.to_string(), flush=True)
print("rollout %.0f s" % (time.time() - t0))
S.to_csv(OUT / f"eval_{sim.name}.csv", index=False)
