"""Side finding, follow-up: value quantum q of the bags without GNSS (no_gnss_long, short) -> which wheel epoch
(c = q/(1+k) of the same vehicle, from s5_quantum_k.csv) they belong to, and the k it implies.
Output: s5b_q_other_bags.csv, printed table."""
import datetime as dt

import numpy as np
import pandas as pd

from bc_common import HERE, load_bag, splits
from s5_quantum_k import smallest_step


def main():
    sp = splits()
    info = {r["bag"]: r for r in sp["info"]}
    ref = pd.read_csv(HERE / "s5_quantum_k.csv")
    cg = ref.groupby(["vehicle", "date"])[["c_front", "c_rear"]].median().reset_index()
    rows = []
    for split in ("no_gnss_long", "short"):
        for b in sp[split]:
            d = load_bag(b)
            t0 = info[b]["t0"]
            date = dt.datetime.fromtimestamp(t0, dt.timezone(dt.timedelta(hours=3))).strftime("%Y-%m-%d")
            veh = int(info[b]["vehicle"])
            qf, nf = smallest_step(d["vf_all"])
            qr, nr = smallest_step(d["vr_all"])
            g = cg[cg.vehicle == veh].copy()
            # nearest calibrated date of the same vehicle
            g["gap_days"] = [abs((dt.date.fromisoformat(x) - dt.date.fromisoformat(date)).days) for x in g.date]
            near = g.sort_values("gap_days").iloc[0] if len(g) else None
            rows.append({"bag": b, "split": split, "vehicle": veh, "date": date, "q_front": qf, "n_qf": nf,
                         "q_rear": qr, "n_qr": nr, "c_date": near.date if near is not None else "",
                         "gap_days": near.gap_days if near is not None else np.nan,
                         "k_front_implied": qf / near.c_front - 1 if near is not None else np.nan,
                         "k_rear_implied": qr / near.c_rear - 1 if near is not None else np.nan})
    df = pd.DataFrame(rows).sort_values(["vehicle", "date"])
    df.to_csv(HERE / "s5b_q_other_bags.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
