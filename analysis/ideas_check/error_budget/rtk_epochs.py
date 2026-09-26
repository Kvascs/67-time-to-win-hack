"""Error budget, step 2: per-epoch errors on the RTK bags (val + train), judge frame.

Same setup as tools/replay/eval_base_link.py (train-only validation maps, branches, GNSS only in the first
5 s, reference base_link from both RTK antennas with the organisers' TF, nearest output within 50 ms), but
every matched epoch is kept together with our state (flags, k, s_map, v, s) and the reference quality
(GNSS header-vs-arrival offset, gap to the previous RTK epoch), and every place-fix attempt is logged
(TBO_DEBUG_LM=1). Output: rtk_epochs/<bag>.parquet, rtk_epochs/<bag>_lm.csv, rtk_summary.csv.

    python analysis/ideas_check/error_budget/rtk_epochs.py --split val --jobs 2
    python analysis/ideas_check/error_budget/rtk_epochs.py --split all --tag _max_dk_1e-3 --set landmark_max_dk=0.001
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge as CB  # noqa: E402
import eval_base_link as EB  # noqa: E402

EXE = ROOT / 'build_core' / 'tbo_replay_final2.exe'
OUT = HERE / 'rtk_epochs'
SETS: dict = {}
TAG = ''
LM_RE = re.compile(r'LM t=([\d.]+) s_map=([-\d.]+) sd=([\d.]+) n=(\d+) d0=([-\d.]+) p_known=([\d.]+) k=([-\d.]+)')


def ref_quality(bag):
    """GNSS header-minus-arrival offset of master RTK fixes (clock anomalies show as +-1 s steps)."""
    d = np.load(CB.NPZ / f'{bag}.npz')
    m = d['sensing__gnss__master__fix']
    return m[:, 1], m[:, 0] - m[:, 1], m[:, 5]


def one(args):
    bag, sets, tag, kmap = args
    if kmap and bag in kmap:  # oracle wheel scale: calibration divided by (1 + k), k frozen
        sets = {**sets, 'wheel_kmh_to_ms': f'{0.2778805556 / (1.0 + kmap[bag]):.10f}', 'init_sigma_scale': '0.00001'}
    out_dir = HERE / f'rtk_epochs{tag}'
    ref = EB.reference(bag)
    if ref is None:
        return {'bag': bag, 'error': 'no RTK'}
    CB.REPLAY_EXE = EXE
    tmp = HERE / 'tmp'
    tmp.mkdir(exist_ok=True)
    ev, out = tmp / f'{bag}_ev{tag}.csv', tmp / f'{bag}_out{tag}.csv'
    CB.export_events(bag, ev)
    VAL = EB.VAL
    br = ','.join(str(b) for b in sorted(VAL.glob('branch_*.csv')))
    s = {'output_frame': 'mgrs', 'landmark_file': str(VAL / 'landmarks.csv'), 'cutoff_file': str(VAL / 'cutoffs.csv'),
         'dfield_file': str(VAL / 'dfield.csv'), 'gl_stops_file': str(VAL / 'gl_stops.csv'),
         'gl_cutoffs_file': str(VAL / 'gl_cutoffs.csv'), 'speed_envelope_file': str(VAL / 'speed_envelope.csv'), **sets}
    os.environ['TBO_DEBUG_LM'] = '1'
    o = CB.run_replay(ev, out, map_csv=VAL / 'track_map.csv',
                      traction_csv=CB.PKG / 'config' / 'traction_lut.csv', sets=s, branches=br)
    err = o.attrs.get('stderr', '')
    out.unlink()
    ev.unlink()
    t_first = o.stamp_ns.min() * 1e-9
    lm = sorted({m.groups() for m in LM_RE.finditer(err)}, key=lambda g: float(g[0]))
    lm = pd.DataFrame([[float(v) for v in g] for g in lm], columns=['t_abs', 's_map', 'sd', 'n', 'd0', 'p_known', 'k'])
    lm['t'] = lm.t_abs - t_first
    o = o[o.pos_valid == 1].drop_duplicates('stamp_ns').sort_values('stamp_ns')
    if o.empty:
        return {'bag': bag, 'error': 'no valid position'}
    ot = o.stamp_ns.to_numpy() * 1e-9
    rt = ref.t.to_numpy()
    i = np.clip(np.searchsorted(ot, rt), 1, len(ot) - 1)
    i = np.where(np.abs(ot[i - 1] - rt) < np.abs(ot[i] - rt), i - 1, i)
    ok = np.abs(ot[i] - rt) <= 0.05
    e = o[['x', 'y', 'z']].to_numpy()[i[ok]] - ref[['x', 'y', 'z']].to_numpy()[ok]
    yaw = ref.yaw.to_numpy()[ok]
    E = pd.DataFrame({'t': rt[ok] - t_first, 'dt_ms': (ot[i[ok]] - rt[ok]) * 1e3,
                      'along': e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw),
                      'cross': -e[:, 0] * np.sin(yaw) + e[:, 1] * np.cos(yaw), 'dz': e[:, 2]})
    for c in ('v', 's', 's_var', 's_map', 'k', 'flags', 'mu3'):
        E[c] = o[c].to_numpy()[i[ok]]
    # reference quality at each epoch: header-minus-arrival offset relative to the bag median
    qt, lat, st = ref_quality(bag)
    j = np.clip(np.searchsorted(qt, rt[ok]), 0, len(qt) - 1)
    E['ref_lat'] = lat[j] - np.median(lat)
    E['ref_gap'] = np.r_[0.0, np.diff(rt[ok])]
    out_dir.mkdir(exist_ok=True)
    E.to_parquet(out_dir / f'{bag}.parquet')
    lm.to_csv(out_dir / f'{bag}_lm.csv', index=False)
    e3 = np.sqrt(E.along ** 2 + E.cross ** 2 + E.dz ** 2)
    r = lambda x: float(np.sqrt(np.mean(np.square(x))))
    return {'bag': bag, 'n': len(E), 'p3_rmse': r(e3), 'along_rmse': r(E.along), 'cross_rmse': r(E.cross),
            'z_rmse': r(E.dz), 'n_lm': len(lm), 'n_lm_acc': int(((lm.p_known >= 0.6) & (lm.n > 0)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--tag', default='')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--oracle-k', default='', help='csv bag,k: per-bag frozen wheel scale (upper bound)')
    a = ap.parse_args()
    splits = json.load(open(ROOT / 'data' / 'splits.json'))
    bags = splits['train'] + splits['val'] if a.split == 'all' else splits[a.split]
    with ProcessPoolExecutor(a.jobs) as ex:
        sets = dict(x.split('=', 1) for x in a.set)
        kmap = dict(pd.read_csv(a.oracle_k)[['bag', 'k']].itertuples(index=False)) if a.oracle_k else {}
        rows = list(ex.map(one, [(b, sets, a.tag, kmap) for b in bags]))
    df = pd.DataFrame(rows)
    df['split'] = [('val' if b in splits['val'] else 'train') for b in df.bag]
    f = HERE / f'rtk_summary{a.tag}.csv'
    if f.exists():
        old = pd.read_csv(f)
        df = pd.concat([old[~old.bag.isin(df.bag)], df], ignore_index=True)
    df.to_csv(f, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
