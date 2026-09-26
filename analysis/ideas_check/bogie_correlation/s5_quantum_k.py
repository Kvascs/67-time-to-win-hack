"""Side finding: the smallest value step q of each bogie speed (km/h) moves together with the wheel scale k.

Hypothesis: speed = (configured wheel factor) x (integer-ish raw count), so q is proportional to the configured
factor, and k = configured / true - 1. With the true wheel diameter fixed over a period, q / (1 + k) = const.
Per bag and bogie (train + val):
  q      : median of consecutive |steps| within 5e-5 km/h of the modal smallest step (0.0035..0.0062 km/h)
  q1, q2 : the same on the first / second half of the bag
  k      : wheel / GNSS - 1 on straight steady samples (bc_common.true_k)
Test: c = q / (1 + k) per (vehicle, date) group; within-group spread of c vs spread of k, within-group
regression k ~ q; train -> val prediction  k_hat = q / c_group(train) - 1.
Outputs: s5_quantum_k.csv, s5_summary.txt, s5_quantum_k.png
"""
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bc_common import HERE, load_bag, splits, true_k


def smallest_step(v):
    dv = np.abs(np.diff(v))
    dv = dv[(dv > 0.0035) & (dv < 0.0062)]
    if len(dv) < 20:
        return np.nan, 0
    h, e = np.histogram(dv, bins=np.arange(0.0035, 0.00621, 2e-6))
    mode = e[np.argmax(h)] + 1e-6
    sel = dv[np.abs(dv - mode) < 5e-5]
    return float(np.median(sel)), len(sel)


def main():
    sp = splits()
    info = {r["bag"]: r for r in sp["info"]}
    rows = []
    for split in ("train", "val"):
        for b in sp[split]:
            d = load_bag(b)
            kt = true_k(d)
            t0 = info[b]["t0"]
            date = dt.datetime.fromtimestamp(t0, dt.timezone(dt.timedelta(hours=3))).strftime("%Y-%m-%d")
            r = {"bag": b, "split": split, "vehicle": info[b]["vehicle"], "date": date, "t0": t0, **kt}
            for nm, key in (("front", "vf_all"), ("rear", "vr_all")):
                v = d[key]
                n = len(v)
                r[f"q_{nm}"], r[f"nq_{nm}"] = smallest_step(v)
                r[f"q1_{nm}"] = smallest_step(v[: n // 2])[0]
                r[f"q2_{nm}"] = smallest_step(v[n // 2:])[0]
            rows.append(r)
            print(b, split, r["vehicle"], date, f"q {r['q_front']:.7f}/{r['q_rear']:.7f} "
                  f"k {r['k_front']:+.5f}/{r['k_rear']:+.5f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["vehicle", "date", "t0"])
    for nm in ("front", "rear"):
        df[f"c_{nm}"] = df[f"q_{nm}"] / (1 + df[f"k_{nm}"])
    df.to_csv(HERE / "s5_quantum_k.csv", index=False)

    L = ["Per (vehicle, date, bogie) group: spread of k vs spread of c = q/(1+k); slope of k on (q/median q - 1)"]
    rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))  # noqa: E731
    pred = []
    for (veh, date), g in df.groupby(["vehicle", "date"]):
        for nm in ("front", "rear"):
            gg = g.dropna(subset=[f"q_{nm}", f"k_{nm}"])
            if len(gg) < 2:
                continue
            x = gg[f"q_{nm}"] / gg[f"q_{nm}"].median() - 1
            y = gg[f"k_{nm}"]
            slope = np.polyfit(x, y, 1)[0] if x.std() > 0 else np.nan
            c = gg[f"c_{nm}"]
            L.append(f"  {veh} {date} {nm:5s} n={len(gg):2d} (train {int((gg.split == 'train').sum())}): "
                     f"k range {y.min() * 100:+.3f}..{y.max() * 100:+.3f} % (sd {y.std() * 100:.3f} %), "
                     f"q range {gg[f'q_{nm}'].min():.7f}..{gg[f'q_{nm}'].max():.7f}, "
                     f"c sd {c.std() / c.mean() * 100:.3f} %, corr(q,k) {np.corrcoef(x, y)[0, 1]:+.2f}, "
                     f"slope dk/(dq/q) {slope:+.2f}")
            # train -> val prediction inside the group
            tr = gg[gg.split == "train"]
            va = gg[gg.split == "val"]
            if len(tr) and len(va):
                cg = tr[f"c_{nm}"].median()
                for _, row in va.iterrows():
                    pred.append({"bag": row.bag, "bogie": nm, "k_true": row[f"k_{nm}"],
                                 "k_hat": row[f"q_{nm}"] / cg - 1,
                                 "k_hat_groupmean": tr[f"k_{nm}"].mean()})
    P = pd.DataFrame(pred)
    if len(P):
        e1 = P.k_hat - P.k_true
        e0 = P.k_hat_groupmean - P.k_true
        L.append(f"VAL prediction ({len(P)} bag-bogies in groups that also have train bags): "
                 f"k = q / c_group(train) - 1: error median {e1.median() * 100:+.3f} %, rms {np.sqrt(np.mean(e1 ** 2)) * 100:.3f} %, "
                 f"max |.| {e1.abs().max() * 100:.3f} %  |  baseline k = mean k of the group's train bags: "
                 f"rms {np.sqrt(np.mean(e0 ** 2)) * 100:.3f} %, max |.| {e0.abs().max() * 100:.3f} %")
        worst = P.assign(err=(P.k_hat - P.k_true) * 100).sort_values("err", key=abs, ascending=False).head(6)
        L.append("  largest val errors (%): " + ", ".join(f"{r.bag}/{r.bogie} {r.err:+.3f} (k {r.k_true * 100:+.2f})"
                                                        for r in worst.itertuples()))
    # c per (vehicle, date): is it carried across dates?
    L.append("c = q/(1+k) per vehicle, date, bogie (median; sd in % of c) -- c ~ true wheel diameter if q ~ configured one")
    for (veh, nm), g in df.melt(id_vars=["vehicle", "date", "split"], value_vars=["c_front", "c_rear"],
                                var_name="bogie", value_name="c").groupby(["vehicle", "bogie"]):
        parts = []
        for date, gg in g.groupby("date"):
            parts.append(f"{date}: {gg.c.median() * 1e3:.5f}e-3 (sd {gg.c.std() / gg.c.median() * 100:.3f} %, n={len(gg)})")
        L.append(f"  {veh} {nm[2:]:5s} " + " | ".join(parts))
    # cross-date: c from the previous date of the same vehicle (train bags) -> all bags of the next date
    xd = []
    for veh, g in df.groupby("vehicle"):
        dates = sorted(g.date.unique())
        for d0, d1 in zip(dates[:-1], dates[1:]):
            for nm in ("front", "rear"):
                c0 = g[(g.date == d0) & (g.split == "train")][f"c_{nm}"].median()
                tgt = g[g.date == d1]
                e = tgt[f"q_{nm}"] / c0 - 1 - tgt[f"k_{nm}"]
                xd.append((veh, d0, d1, nm, e.median() * 100, np.sqrt(np.mean(e ** 2)) * 100, len(e)))
    for r in xd:
        L.append(f"  cross-date {r[0]} c({r[1]}) -> {r[2]} {r[3]:5s}: k error median {r[4]:+.3f} %, rms {r[5]:.3f} % (n={r[6]})")
    # within-bag drift
    for nm in ("front", "rear"):
        dq = (df[f"q2_{nm}"] / df[f"q1_{nm}"] - 1).dropna()
        L.append(f"within-bag change of q ({nm}), 2nd half vs 1st half: median {dq.median() * 100:+.3f} %, "
                 f"max |.| {dq.abs().max() * 100:.3f} % ({df.loc[dq.abs().idxmax(), 'bag']})")
    txt = "\n".join(L)
    print(txt)
    (HERE / "s5_summary.txt").write_text(txt, encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, nm in zip(ax, ("front", "rear")):
        for (veh, date), g in df.groupby(["vehicle", "date"]):
            a.scatter(g[f"q_{nm}"] * 1e3, g[f"k_{nm}"] * 100, s=16, label=f"{veh} {date}")
        a.set_xlabel(f"smallest step q, {nm} bogie (x1e-3 km/h)")
        a.set_ylabel("k (GNSS, straight steady), %")
        a.grid(alpha=0.4)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "s5_quantum_k.png", dpi=90)


if __name__ == "__main__":
    main()
