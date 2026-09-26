"""Is the ratio pattern y(s) repeatable run to run (a place signature beyond curvature)?
Split the train bags into two halves (alternating), build the 1 m-bin mean of y and of |y| per half,
correlate the halves, separately for straight bins and sharp-curve bins. Also: how much of the
bin-mean variance does the kappa model (M6) explain.
Output: repeatability.txt
"""
import json

import numpy as np
import pandas as pd

import common as C
import regress as R

_lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def main():
    pl = C.load_main()
    L = pl.L
    nb = int(np.ceil(L))
    for vmin in (3.0, 1.5):
        S, _ = R.load('train', vmin)
        S = R.add_features(S)
        S['bin'] = np.floor(np.mod(S.s, L)).astype(int) % nb
        bags = sorted(S.bag.unique())
        half = {b: i % 2 for i, b in enumerate(bags)}
        S['h'] = S.bag.map(half)
        S['ay'] = S.y.abs()
        # speed-normalised z (removes the speed dependence of the noise level)
        st = S[S.kmax < 0.003]
        vg = np.arange(1.5, 16.01, 0.5)
        sv = np.array([st.y[(st.v >= a) & (st.v < a + 0.5)].std() if ((st.v >= a) & (st.v < a + 0.5)).sum() > 200 else np.nan for a in vg])
        ok = np.isfinite(sv)
        S['az'] = S.ay / np.interp(S.v, vg[ok] + 0.25, sv[ok])
        mj = json.load(open(C.HERE / 'model_v1.5.json'))
        b6 = np.array(mj['models']['M6 M3+kf^2,kr^2']['beta'])
        sg = np.arange(nb) + 0.5
        kf, kr = pl.k_at(sg + C.D_FRONT), pl.k_at(sg + C.D_REAR)
        mu_k = np.c_[kf, kr, np.abs(kf), np.abs(kr), kf ** 2, kr ** 2] @ b6
        kmx = np.maximum(np.abs(kf), np.abs(kr))
        say(f'=== v > {vmin} m/s, train bags split in halves ({len(bags)} bags) ===')
        for col, nm in (('y', 'mean y'), ('ay', 'mean |y|'), ('az', 'mean |y|/sigma_v(v)')):
            h = [S[S.h == k].groupby('bin')[col].agg(['mean', 'size']).reindex(range(nb)) for k in (0, 1)]
            both = (h[0]['size'] >= 8) & (h[1]['size'] >= 8)
            for cls, msk in (('straight max|k|<0.003', kmx < 0.003), ('mild 0.003-0.02', (kmx >= 0.003) & (kmx < 0.02)),
                             ('sharp max|k|>0.02', kmx >= 0.02)):
                m = both.values & msk
                a, b = h[0]['mean'].values[m], h[1]['mean'].values[m]
                r = np.corrcoef(a, b)[0, 1]
                line = f'  {nm:22s} {cls:22s} bins {m.sum():5d}: split-half r = {r:.3f}'
                if col == 'y':
                    full = S.groupby('bin').y.mean().reindex(range(nb)).values[m]
                    r2k = 1 - np.var(full - mu_k[m]) / np.var(full)
                    line += f'; std of bin means {100 * np.std(full):.3f}%; kappa-model explains {r2k:.3f} of bin-mean variance'
                say(line)
    (C.HERE / 'repeatability.txt').write_text('\n'.join(_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
