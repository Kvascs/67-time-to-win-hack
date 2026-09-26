"""Noise level and curvature of J along each coordinate (from a_scan.csv): quadratic fit over the scan points
without the cliffs (J > 1.5); eps_f = residual RMS, L = 2c; forward-difference step of Berahas et al. (2019):
h* = 2 sqrt(eps_f / |L|); signal = |J(+-0.35) - 1| vs eps_f."""
import numpy as np
import pandas as pd

d = pd.read_csv('a_scan.csv')
base = d[d.param == 'base']
rows = []
for p in ['sigma_accel', 'sigma_wheel', 'q_disturbance', 'landmark_assoc_q']:
    q = pd.concat([base, d[d.param == p]])
    cliff = q[q.J > 1.5]
    q = q[q.J <= 1.5]
    c2, c1, c0 = np.polyfit(q.x, q.J, 2)
    res = q.J - np.polyval([c2, c1, c0], q.x)
    eps = float(np.sqrt(np.sum(res ** 2) / max(len(q) - 3, 1)))
    L = 2 * c2
    rows.append({'param': p, 'n_pts': len(q), 'cliff_at_x': list(cliff.x), 'slope_b': c1, 'curv_L': L, 'eps_f': eps,
                 'h_star': 2 * np.sqrt(eps / abs(L)) if L != 0 else np.inf,
                 'J_range_pct': 100 * (q.J.max() - q.J.min())})
r = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(r.round(4).to_string(index=False))
r.to_csv('a_noise_fit.csv', index=False)
