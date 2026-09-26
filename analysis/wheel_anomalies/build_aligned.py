"""Build and cache wheel/controller/GNSS-reference series on a common 0.1 s grid (wheel header time)
for every unique bag. Output: aligned/<bag>.npz (t, front, rear, notch, ref, ref_q, ref3d, lag, lag_tc, lag_w).

GNSS reference = fused master/rover Doppler horizontal speed, shifted by a locally estimated lag
(handles the +-1 s GNSS header-stamp excursions found in ~10 bags)."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anomalies as A  # noqa: E402

OUT = Path(__file__).parent / 'aligned'


def build(name):
    out = OUT / f'{name}.npz'
    if out.exists():
        return name, 'cached'
    import json
    b = A.load_bag(name)
    al = A.align_bag(b)
    tc, lw = getattr(al, 'lag_windows', (np.zeros(0), np.zeros(0)))
    lags = getattr(al, 'lags', {})
    nan = np.full(len(al.t), np.nan)
    np.savez_compressed(out, t=al.t, front=al.front, rear=al.rear, notch=al.notch, ref=al.ref, ref_q=al.ref_q,
                        ref3d=al.ref3d, lag=al.lag, lag_tc=tc, lag_w=lw, t0=b.t0,
                        lag_master=lags.get('master', nan), lag_rover=lags.get('rover', nan),
                        lag_info=json.dumps(getattr(al, 'lag_info', {})), kappa=getattr(al, 'kappa', nan))
    return name, len(al.t)


def load_aligned(name) -> A.Aligned:
    import json
    z = np.load(OUT / f'{name}.npz')
    al = A.Aligned(name, z['t'], z['front'], z['rear'], z['notch'], z['ref'], z['ref_q'], z['lag'], z['ref3d'])
    al.lag_windows = (z['lag_tc'], z['lag_w'])
    al.t0 = float(z['t0'])
    al.lags = {'master': z['lag_master'], 'rover': z['lag_rover']}
    al.lag_info = json.loads(str(z['lag_info']))
    al.kappa = z['kappa'] if 'kappa' in z.files else np.full(len(al.t), np.nan)
    return al


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    names = [x['bag'] for x in A.splits()['info']]
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for n, r in ex.map(build, names):
            print(n, r, flush=True)
