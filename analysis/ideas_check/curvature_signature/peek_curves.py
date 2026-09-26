"""Plot log(front/rear) vs antenna arc s through the sharp-curve zones for several runs."""
import matplotlib
import numpy as np
import pandas as pd

import common as C

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

S = pd.read_pickle(C.HERE / 'samples.pkl')
S = S[S.split == 'train']
with np.errstate(divide='ignore', invalid='ignore'):
    S['lr'] = np.log(S.vf / S.vr)
ok = np.isfinite(S.kf) & (S.v > 1.0) & (S.vf > 0.5) & (S.vr > 0.5) & np.isfinite(S.lr) & ~S.bad_ep
ok &= (np.abs(S.dsdt / S.v - 1) < 0.1)
T = S[ok].copy()
T['y'] = T.lr - T.groupby('bag').lr.transform('median')
pl = C.load_main()
zones = [(5330, 5470, 'west approach (switch 5395)'), (5480, 5830, 'west loop'),
         (3570, 3700, 's 3600'), (7450, 7580, 's 7500'), (10900, 11053.6, 'east loop a'), (0, 240, 'east loop b')]
fig, axs = plt.subplots(len(zones), 1, figsize=(15, 4 * len(zones)))
for ax, (a, b, name) in zip(axs, zones):
    sg = np.arange(a, b, 0.5)
    ax2 = ax.twinx()
    ax2.plot(sg, pl.k_at(sg), 'k-', lw=1, label='kappa(s) antenna')
    ax2.plot(sg, pl.k_at(sg + C.D_FRONT), 'g-', lw=0.7, label='kappa front pivot')
    ax2.plot(sg, pl.k_at(sg + C.D_REAR), 'm-', lw=0.7, label='kappa rear pivot')
    ax2.set_ylabel('kappa')
    n = 0
    for bag, g in T.groupby('bag'):
        m = (g.sm >= a) & (g.sm < b)
        if m.sum() < 30:
            continue
        ax.plot(g.sm[m], 100 * g.y[m], '.', ms=1.5, alpha=0.5)
        n += 1
    ax.plot(sg, 100 * C.chord_pred(pl, sg), 'r-', lw=2, label='chord model pred')
    ax.set_ylim(-4, 4)
    ax.set_title(f'{name}: {n} runs; y = log(vf/vr) - bag median [%] (dots), v>1 m/s')
    ax.set_xlabel('antenna s [m]')
    ax.set_ylabel('y [%]')
    ax.grid()
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(C.HERE / 'fig_peek_curves.png', dpi=70)
