"""Follow-up on epochs_train.csv.gz (tau = 0): one common lag term tau*a/v for traction AND braking
(a real clock/filter lag must be the same in both regimes), regime slopes beta_T, beta_B, regime intercepts."""
import numpy as np, pandas as pd
from pathlib import Path
from creep import ols_cluster, TRIM
OUT = Path(__file__).resolve().parent
E = pd.read_csv(OUT / 'epochs_train.csv.gz')
L = []
for bog in 'FR':
    e = E[E['y' + bog].abs() < TRIM]
    T, B = (e.reg == 'trac').to_numpy(float), (e.reg == 'brake').to_numpy(float)
    a, z = e.a.to_numpy(), (e.a * e.inv_v).to_numpy()
    X = np.column_stack([np.ones(len(e)), T, B, a * T, a * B, z])
    b, se, _ = ols_cluster(X, e['y' + bog].to_numpy(), e.bag.to_numpy())
    L.append(f'{bog}: beta_trac {b[3]*100:+.3f}+/-{se[3]*100:.3f}  beta_brake {b[4]*100:+.3f}+/-{se[4]*100:.3f} %/(m/s2); '
             f'common tau {b[5]:+.4f}+/-{se[5]:.4f} s; step trac {b[1]*100:+.3f} brake {b[2]*100:+.3f} % (vs coast)')
    X2 = np.column_stack([np.ones(len(e)), T, B, a * (T + B), z])   # symmetric creep, one beta
    b2, se2, _ = ols_cluster(X2, e['y' + bog].to_numpy(), e.bag.to_numpy())
    L.append(f'{bog}: one beta (trac+brake) {b2[3]*100:+.3f}+/-{se2[3]*100:.3f} %/(m/s2), common tau {b2[4]:+.4f}+/-{se2[4]:.4f} s')
print('\n'.join(L)); (OUT / 'creep_joint.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
