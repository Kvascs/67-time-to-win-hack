"""Speed quantum per vehicle and wheel epoch -> wheel_epochs.csv for the estimator.

A bogie reports speed in steps of q km/h and q = c (1 + k): c is fixed for a vehicle and a wheel epoch
(the dates of our recordings), the configured wheel factor sets q per run. Source: per-bag q and GNSS
wheel scale in analysis/ideas_check/bogie_correlation/s5_quantum_k.csv (s5_quantum_k.py).

The research k is wheel / (3.6 * 1.00037 * v_gnss) - 1, the filter's k is wheel * 1.00037 / 3.6 / v - 1,
so the table stores c in the filter's convention: c = q / (1 + k_research) / 1.00037^2.

  python tools/map/export_wheel_epochs.py            # both tables
    ros2_ws/src/tram_backup_odometry/maps/wheel_epochs.csv   all bags (shipped)
    analysis/validation_maps/wheel_epochs.csv                train bags only (honest validation)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "analysis" / "ideas_check" / "bogie_correlation" / "s5_quantum_k.csv"
CONV = 1.00037 ** 2


def table(df):
    rows = []
    for (veh, date), g in df.groupby(["vehicle", "date"]):
        cf = (g.q_front / (1 + g.k_front)).median() / CONV
        cr = (g.q_rear / (1 + g.k_rear)).median() / CONV
        rows.append({"vehicle": int(veh), "date": int(str(date).replace("-", "")), "c_front": cf, "c_rear": cr, "n": len(g)})
    return pd.DataFrame(rows).sort_values(["vehicle", "date"])


def write(t, path, what):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# speed quantum per vehicle and wheel epoch ({what}); k = q / c - 1, filter convention\n")
        fh.write("vehicle,date,c_front,c_rear,n\n")
        for r in t.itertuples():
            fh.write(f"{r.vehicle},{r.date},{r.c_front:.9e},{r.c_rear:.9e},{r.n}\n")
    print(path.relative_to(ROOT), len(t), "epochs")


def main():
    df = pd.read_csv(SRC).dropna(subset=["q_front", "q_rear", "k_front", "k_rear"])
    write(table(df), ROOT / "ros2_ws" / "src" / "tram_backup_odometry" / "maps" / "wheel_epochs.csv", "all bags")
    write(table(df[df.split == "train"]), ROOT / "analysis" / "validation_maps" / "wheel_epochs.csv", "train bags only")


if __name__ == "__main__":
    main()
