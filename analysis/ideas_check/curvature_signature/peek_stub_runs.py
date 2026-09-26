"""What do the stub runs look like (speed, position, RTK) around the west_arrival_2 switch?"""
import numpy as np
import pandas as pd

import common as C

S = pd.read_pickle(C.HERE / 'samples.pkl')
hits = pd.read_csv(C.HERE / 'stub_hits.csv')
origin = C.map_origin()
pl = C.load_main()
stub, _ = C.load_stub()
for bag in hits.bag:
    g = S[S.bag == bag].reset_index(drop=True)
    d = C.load_bag(bag)
    fx = C.master_fix_enu(d, origin)
    rtk = fx[fx.status == 2]
    s_m, _, d_m = pl.project(rtk.x.values, rtk.y.values)
    s_s, _, d_s = stub.project(rtk.x.values, rtk.y.values)
    near = (np.abs(s_m - 5395) < 150) | (d_s < 2)
    t_near = rtk.t.values[near]
    print(f'=== {bag}: RTK fixes near switch/stub: {near.sum()}, t span {t_near.min() - g.t.min():.1f}..{t_near.max() - g.t.min():.1f} s '
          f'(bag wheel span {g.t.max() - g.t.min():.1f} s)')
    # time series every ~2 s in the relevant window
    t0, t1 = t_near.min() - 5, t_near.max() + 5
    gi = g[(g.t >= t0) & (g.t <= t1)]
    for _, r in gi.iloc[::20].iterrows():
        print(f'  t={r.t - g.t.min():8.1f} v={r.v:5.2f} vf={r.vf:5.2f} vr={r.vr:5.2f} s={r.sm:8.1f} '
              f'dmain={r.dmain:5.2f} s_stub={r.s_stub:7.2f} dmain@stub={r.dmain_at_stub:6.2f}')
    # fixes after leaving the stub
    last = rtk.t.values[d_s < 2].max()
    print(f'  last on-stub RTK fix at {last - g.t.min():.1f} s; wheel data end {g.t.max() - g.t.min():.1f} s; '
          f'RTK after that: {(rtk.t.values > last).sum()}; all fixes after: {(fx.t.values > last).sum()}')
