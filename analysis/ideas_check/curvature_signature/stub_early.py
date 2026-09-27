"""Early stub decision: how the switch-window roughness builds up along the window, main-line passes vs stub runs.

For every replayed run (submitted build, train-only maps; the organisers' bag with the package maps) take the filter's
antenna arc on the main cycle (s_map of the outputs), pair front and rear bogie readings with a common stamp (both
> 1 m/s, as in the estimator) and accumulate y = log(front/rear) from the window start d = -15 m (antenna arc
relative to the stub start 5395.19). Print the running rms at d = -10, -5, 0, +5, +12 m for every pass.
Output: stub_early.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r'C:\MosTransHack')
RT = ROOT / 'build_core' / 'replay_tmp'
JOIN = 5395.19
KMH = 1.00037 / 3.6
POINTS = (-10.0, -5.0, 0.0, 5.0, 12.0)


def passes(out_csv, bag):
    o = pd.read_csv(out_csv, usecols=['stamp_ns', 's_map', 'v']).drop_duplicates('stamp_ns').sort_values('stamp_ns')
    z = np.load(ROOT / 'data' / 'npz' / f'{bag}.npz')
    f = pd.DataFrame(z['vehicle__front_bogie_velocity'][:, 1:], columns=['t', 'vf'])
    r = pd.DataFrame(z['vehicle__rear_bogie_velocity'][:, 1:], columns=['t', 'vr'])
    f['k'] = np.round(f.t * 1e3).astype(np.int64)
    r['k'] = np.round(r.t * 1e3).astype(np.int64)
    w = f.merge(r[['k', 'vr']], on='k')
    w = w[(w.vf * KMH > 1.0) & (w.vr * KMH > 1.0)]
    ts = o.stamp_ns.to_numpy() * 1e-9
    sm = o.s_map.to_numpy()
    idx = np.clip(np.searchsorted(ts, w.t.to_numpy()), 0, len(ts) - 1)
    w = w.assign(s_map=sm[idx])
    w = w[w.s_map >= 0]
    w['d'] = w.s_map - JOIN
    w['y'] = np.log(w.vf / w.vr)
    rows = []
    # a pass = consecutive samples inside [-15, +12]
    inside = (w.d >= -15.0) & (w.d <= 12.0)
    grp = (inside != inside.shift()).cumsum()
    for _, g in w[inside].groupby(grp[inside]):
        if len(g) < 10:
            continue
        row = {'bag': bag, 'n': len(g), 'd_max': g.d.max()}
        cs = np.cumsum(g.y.to_numpy() ** 2)
        n = np.arange(1, len(g) + 1)
        for p in POINTS:
            m = np.flatnonzero(g.d.to_numpy() <= p)
            row[f'rms@{p:+.0f}'] = float(np.sqrt(cs[m[-1]] / n[m[-1]])) * 100 if len(m) >= 10 else np.nan
            row[f'n@{p:+.0f}'] = int(len(m))
        rows.append(row)
    return rows


def main():
    stub_runs = {'30618_49fe4c54', '30618_8158f0b0', '30639_0be558e2', '30618_a869780d'}
    rows = []
    for folder in ('bl_qk_train', 'bl_qk_val'):
        for f in sorted((RT / folder).glob('*_out.csv')):
            bag = f.name[:-8]
            for r in passes(f, bag):
                r['cls'] = 'stub' if bag in stub_runs and r['d_max'] > 11 else 'main'
                r['set'] = folder[6:]
                rows.append(r)
    for r in passes(RT / 'checker' / '30618_88aea4d9_out_qk.csv', '30618_88aea4d9'):
        r['cls'] = 'stub (organisers)'
        r['set'] = 'check'
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).with_name('stub_early.csv'), index=False)
    pd.set_option('display.width', 220)
    cols = ['set', 'bag', 'cls', 'n'] + [f'rms@{p:+.0f}' for p in POINTS]
    print(df.sort_values(['cls', 'rms@+0'], ascending=[True, False])[cols].round(2).to_string(index=False))
    main_ = df[df.cls == 'main']
    print('\nmain passes:', len(main_), ' max running rms %:', {c: round(main_[c].max(), 2) for c in cols[4:]})


if __name__ == '__main__':
    main()
