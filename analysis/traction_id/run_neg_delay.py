"""Timing-offset check: LUT fit with negative dead time (notch look-ahead), analysis only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import pandas as pd
from lut_model import fit_lut, score_accel
from tid_data import OUT, list_bags
tr, va = list_bags("train"), list_bags("val")
rows = []
for d in (-0.1, -0.2, -0.3, -0.45):
    m, rt, _ = fit_lut(tr[::4], d, 0.4, decim=3)
    rv = score_accel(m, va)
    rows.append(dict(delay=d, tau=0.4, rmse_train=rt, rmse_val=rv, kg_b=m.kg[0], kg_c=m.kg[1], kg_t=m.kg[2]))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "lut_negdelay.csv", index=False)
