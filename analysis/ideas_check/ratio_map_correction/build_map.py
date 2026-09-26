"""Step 1: ratio map from TRAIN runs (and a train+val map for the organisers' check bag).

Samples: previous study's samples.pkl (paired front/rear wheel stamps, master-antenna arc s from RTK at
t - 43.5 ms, known anomaly episodes flagged). Selection as regress.load(vmin=1.5): both bogies > 1.5 m/s,
no anomaly episode, arc rate consistent with wheel speed, per-bag offset (straight-track median) removed,
|y| < 5 %. z = y / sigma_v(v) (speed-only straight-track noise), then per 1 m bin: n, sum z, sum z^2,
kept per bag so a leave-one-bag-out map can be formed.
The shipped and the train-only (validation_maps) map arcs agree within 1 cm, so s is used as is.

Output: ratio_map_parts.npz, build_map.txt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rm_common as M

_lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def main():
    pl = M.load_valmap()
    L = pl.L
    nb = int(np.ceil(L))
    parts = {}
    for split in ('train', 'val'):
        S, info = M.R.load(split, M.VMIN)
        S = M.R.add_features(S)
        say(f'{split}: {info}')
        if split == 'train':
            sv_tab = M.sigma_v_table(S)
            say('sigma_v(v) straight [%]: ' + ' '.join(f'{v:.2f}:{100 * s:.3f}' for v, s in zip(*sv_tab)))
            offs = S.groupby('bag').off.first()
            say(f'per-bag offset log(vf/vr): median {100 * offs.median():.3f} %, IQR {100 * offs.quantile(0.25):.3f}..'
                f'{100 * offs.quantile(0.75):.3f} %, range {100 * offs.min():.3f}..{100 * offs.max():.3f} %')
        S['z'] = S.y / M.sigma_v(S.v.values, sv_tab)
        for bag, idx in S.groupby('bag').indices.items():
            q = S.iloc[idx]
            parts[bag] = (split, *M.bin_sums(q.s.values, q.z.values, nb, L))
    bags = sorted(parts)
    np.savez_compressed(M.HERE / 'ratio_map_parts.npz', bags=np.array(bags), split=np.array([parts[b][0] for b in bags]),
                        n=np.stack([parts[b][1] for b in bags]), s1=np.stack([parts[b][2] for b in bags]),
                        s2=np.stack([parts[b][3] for b in bags]), L=L, sv_v=sv_tab[0], sv_s=sv_tab[1])
    tr = [b for b in bags if parts[b][0] == 'train']
    n = sum(parts[b][1] for b in tr)
    rm = M.RatioMap(n, sum(parts[b][2] for b in tr), sum(parts[b][3] for b in tr), L, sv_tab)
    say(f'train map: {nb} bins, {int((~rm.few).sum())} with >= 10 samples; samples per bin median {np.median(n):.0f}; '
        f'std of bin mean z {np.std(rm.mu[~rm.few]):.3f}; median bin sd_z {np.median(rm.sd[~rm.few]):.3f}')
    (M.HERE / 'build_map.txt').write_text('\n'.join(_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
