"""Feasibility of the wheel scale from the speed quantum inside the filter (before any C++).

Per bag of the submitted build's replays (train / val), in the FILTER's convention of k
(z = wheel_kmh * 1.00037 / 3.6 = (1 + k) v; the research k uses km/h / (3.6 * 1.00037), so
k_filter = (1 + k_research) * 1.00037^2 - 1):
  k_true   : wheel / GNSS on straight steady samples (s5_quantum_k.csv), converted
  k_first  : the filter's k right after its first landmark (first change of the Schmidt state)
  k_end    : the filter's k at the end of the run
  k_q      : q / c - 1 with c from TRAIN bags only; candidates = per vehicle the latest wheel epoch not later
             than the bag's date; the candidate nearest to k_first is taken if it is within the gate
Output: s6_quant_k_feasibility.csv and a printed summary.
"""
import datetime as dt
import sys

import numpy as np
import pandas as pd

from bc_common import HERE, ROOT, load_bag, splits
from s5_quantum_k import smallest_step

CONV = 1.00037 ** 2
REPLAY = ROOT / "build_core" / "replay_tmp"
GATE = 0.005          # |k_q - k_first| must be below this
AMBIG = 2.0           # second candidate must be this many times farther


def to_filter(k_research):
    return (1.0 + k_research) * CONV - 1.0


def first_change(k):
    k0 = k[0]
    idx = np.nonzero(np.abs(k - k0) > 1e-7)[0]
    return int(idx[0]) if len(idx) else -1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s5 = pd.read_csv(HERE / "s5_quantum_k.csv")
    tr = s5[s5.split == "train"]
    # epoch table from train only: (vehicle, date) -> median c per bogie
    table = tr.groupby(["vehicle", "date"])[["c_front", "c_rear"]].median().reset_index()
    rows = []
    for split, folder in (("train", "bl_rough_train"), ("val", "bl_rough_val")):
        for f in sorted((REPLAY / folder).glob("*_out.csv")):
            bag = f.name[:-8]
            r5 = s5[s5.bag == bag]
            if not len(r5):
                continue
            r5 = r5.iloc[0]
            o = pd.read_csv(f, usecols=["stamp_ns", "k", "s"])
            k = o.k.to_numpy()
            i1 = first_change(k)
            row = {"bag": bag, "split": split, "vehicle": int(r5.vehicle), "date": r5.date,
                   "k_true": to_filter(r5.k_mean), "k_first": k[i1] if i1 >= 0 else np.nan,
                   "s_first": o.s.iloc[i1] if i1 >= 0 else np.nan, "k_end": k[-1],
                   "q_front": r5.q_front, "q_rear": r5.q_rear}
            # candidates: per vehicle the latest epoch not later than the bag date (else the earliest)
            cands = []
            for veh, g in table.groupby("vehicle"):
                g = g.sort_values("date")
                past = g[g.date <= r5.date]
                e = past.iloc[-1] if len(past) else g.iloc[0]
                kq = to_filter(0.5 * (r5.q_front / e.c_front + r5.q_rear / e.c_rear) - 1.0)
                cands.append((veh, e.date, kq))
            row["cands"] = "; ".join(f"{v}@{d}:{kq * 100:+.3f}%" for v, d, kq in cands)
            if np.isfinite(row["k_first"]) and cands:
                dist = sorted((abs(kq - row["k_first"]), v, d, kq) for v, d, kq in cands)
                d1 = dist[0]
                d2 = dist[1][0] if len(dist) > 1 else np.inf
                ok = d1[0] < GATE and d2 > AMBIG * d1[0]
                row.update({"sel_vehicle": d1[1], "sel_date": d1[2], "k_q": d1[3], "d1": d1[0], "d2": d2,
                            "accepted": ok, "right_vehicle": d1[1] == row["vehicle"]})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "s6_quant_k_feasibility.csv", index=False)
    pd.set_option("display.width", 250)
    for split in ("train", "val"):
        g = df[df.split == split].dropna(subset=["k_true", "k_first"])
        acc = g[g.accepted == True]  # noqa: E712
        e_first = (g.k_first - g.k_true) * 100
        e_end = (g.k_end - g.k_true) * 100
        e_q = (acc.k_q - acc.k_true) * 100
        e_first_acc = (acc.k_first - acc.k_true) * 100
        print(f"== {split}: {len(g)} bags with true k; accepted {len(acc)}, wrong vehicle among accepted "
              f"{int((acc.right_vehicle == False).sum())}")  # noqa: E712
        print(f"  |k_first - k_true| %: median {e_first.abs().median():.3f}, rms {np.sqrt((e_first ** 2).mean()):.3f}, max {e_first.abs().max():.3f}")
        print(f"  |k_end   - k_true| %: median {e_end.abs().median():.3f}, rms {np.sqrt((e_end ** 2).mean()):.3f}, max {e_end.abs().max():.3f}")
        if len(acc):
            print(f"  accepted: |k_q - k_true| %: median {e_q.abs().median():.3f}, rms {np.sqrt((e_q ** 2).mean()):.3f}, max {e_q.abs().max():.3f}"
                  f"  vs k_first on the same bags: median {e_first_acc.abs().median():.3f}, rms {np.sqrt((e_first_acc ** 2).mean()):.3f}")
        rej = g[g.accepted != True]  # noqa: E712
        for r in rej.itertuples():
            print(f"  rejected {r.bag}: k_true {r.k_true * 100:+.3f} k_first {r.k_first * 100:+.3f} | {r.cands}")
        worst = acc.assign(e=(acc.k_q - acc.k_true) * 100).sort_values("e", key=abs, ascending=False).head(4)
        for r in worst.itertuples():
            print(f"  worst accepted {r.bag} ({r.vehicle} {r.date}): k_true {r.k_true * 100:+.3f} k_q {r.k_q * 100:+.3f} "
                  f"k_first {r.k_first * 100:+.3f} k_end {r.k_end * 100:+.3f} sel {r.sel_vehicle}@{r.sel_date}")

    # the organisers' bag: vehicle 30618, 2026-09-09; epoch table from ALL bags (as shipped)
    allt = s5.groupby(["vehicle", "date"])[["c_front", "c_rear"]].median().reset_index()
    d = load_bag("30618_88aea4d9")
    qf, qr = smallest_step(d["vf_all"])[0], smallest_step(d["vr_all"])[0]
    print(f"== organisers' bag: q {qf:.7f} / {qr:.7f}")
    for veh, g in allt.groupby("vehicle"):
        g = g.sort_values("date")
        e = g[g.date <= "2026-09-09"].iloc[-1]
        kq = to_filter(0.5 * (qf / e.c_front + qr / e.c_rear) - 1.0)
        print(f"  candidate {veh}@{e.date}: k_q {kq * 100:+.3f} %")
    ck = pd.read_csv(REPLAY / "checker" / "30618_88aea4d9_out.csv", usecols=["k", "s"])
    i1 = first_change(ck.k.to_numpy())
    print(f"  filter: k_first {ck.k.iloc[i1] * 100:+.3f} % at s {ck.s.iloc[i1]:.0f} m, k_end {ck.k.iloc[-1] * 100:+.3f} %")


if __name__ == "__main__":
    main()
