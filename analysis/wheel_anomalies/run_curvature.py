"""Does the wheel/GNSS ratio depend on path curvature (sensor on one wheel side, antenna offset)?
Bins the relative ratio residual (w/k - ref)/ref against signed GNSS curvature kappa (1/m).
Output: curvature_ratio.csv, fig_ratio_vs_curvature.png"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import anomalies as A  # noqa: E402
from build_aligned import load_aligned  # noqa: E402


def main():
    sc = pd.read_csv(HERE / 'scale_factors.csv')
    sc = sc[sc.dur_s > 300]
    allk, alle, allv, alld, allb = [], [], [], [], []
    for _, r in sc.iterrows():
        al = load_aligned(r.bag)
        b = A.load_bag(r.bag)
        kap = A.gnss_curvature(b, al.t + al.lag)
        excl = A.lag_transition_mask(al)
        steady = A.steady_mask(al, vmin=1.5, amax=0.3) & ~excl
        for s in ('front', 'rear'):
            w = getattr(al, s) * A.KMH / r[f'k_speed_{s}']
            e = (w - al.ref) / al.ref
            m = steady & np.isfinite(e) & np.isfinite(kap) & (np.abs(e) < 0.1)
            allk.append(kap[m]); alle.append(e[m]); allv.append(al.ref[m]); allb.append(np.full(m.sum(), s))
            # direction label: sign of net eastward displacement over bag (from GNSS fix)
            alld.append(np.full(m.sum(), r.bag))
    K = np.concatenate(allk); E = np.concatenate(alle); V = np.concatenate(allv); S = np.concatenate(allb)
    edges = np.array([-0.06, -0.04, -0.03, -0.02, -0.015, -0.01, -0.006, -0.003, -0.001, 0.001, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06])
    rows = []
    for s in ('front', 'rear'):
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (S == s) & (K >= lo) & (K < hi)
            if m.sum() < 50:
                continue
            rows.append(dict(sensor=s, k_lo=lo, k_hi=hi, kappa_mean=float(np.mean(K[m])), n=int(m.sum()),
                             e_rel_median=float(np.median(E[m])), e_rel_mean=float(np.mean(E[m])),
                             v_mean=float(np.mean(V[m]))))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'curvature_ratio.csv', index=False)
    print(df.round(5).to_string())
    # linear fit e = a + b*kappa (+ c*|kappa|) on |kappa|<0.05
    for s in ('front', 'rear'):
        m = (S == s) & (np.abs(K) < 0.05)
        X = np.c_[np.ones(m.sum()), K[m], np.abs(K[m])]
        coef, *_ = np.linalg.lstsq(X, E[m], rcond=None)
        print(s, 'fit e_rel = a + b*kappa + c*|kappa|:', coef.round(5), '-> b [m] (lateral lever arm), c [m]')
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for s, c in (('front', 'b'), ('rear', 'c')):
        d = df[df.sensor == s]
        ax[0].plot(d.kappa_mean, 100 * d.e_rel_median, c + 'o-', label=f'{s} median')
    ax[0].set_xlabel('GNSS path curvature kappa [1/m] (>0 left turn)'); ax[0].set_ylabel('(wheel/k - gnss)/gnss [%]')
    ax[0].grid(); ax[0].legend(); ax[0].set_title('Wheel/GNSS ratio residual vs curvature (steady, v>1.5 m/s)')
    m = (S == 'front')
    h = ax[1].hexbin(K[m], 100 * E[m], gridsize=80, extent=(-0.06, 0.06, -3, 3), bins='log', cmap='viridis')
    ax[1].set_xlabel('kappa [1/m]'); ax[1].set_ylabel('front residual [%]'); ax[1].set_title('front: density')
    plt.colorbar(h, ax=ax[1])
    plt.tight_layout()
    plt.savefig(HERE / 'fig_ratio_vs_curvature.png', dpi=80)


if __name__ == '__main__':
    main()
