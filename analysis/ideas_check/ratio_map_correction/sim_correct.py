"""Step 3: simulate the ratio-map correction ("balises all along the track") on VAL with the estimator's own position.

Train-only ratio map (ratio_map_parts.npz). Every STEP m of the filter's antenna arc the last W m of wheel-pair
samples are matched against the map over corrections GRID around the current estimate (free per-window offset of
log(front/rear), rm_common.match). A correction is ACCEPTED when the likelihood peak is sharp (curvature sigma <
SIGMAX), not at the grid edge, and every other local maximum farther than 2 m is at least MARGIN below it.
Evaluation against the RTK truth at the trigger (prep: master fixes projected on the train map at t - 43.5 ms):
  e_filt = s_est - s_true (the estimator's along error), e_corr = s_est + d_hat - s_true (after the correction).
Proxy of the gain: RMS of e_filt over all triggers vs RMS of (e_corr if accepted else e_filt).

  python sim_correct.py --prep-from bl_so_val     # re-prepare VAL from the submitted build's replays first
  python sim_correct.py                           # use prep_so/ as prepared
Output: sim_correct.csv, sim_correct.txt
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

_ap = argparse.ArgumentParser()
_ap.add_argument('--prep-from', default='')
ARGS = _ap.parse_args()
sys.argv = sys.argv[:1]  # regress.py (imported by rm_common) reads sys.argv[1] as a speed

import rm_common as M  # noqa: E402

STEP, W = 50.0, 200.0
GRID = np.arange(-8.0, 8.0001, 0.1)
MARGIN = 3.0
SIGMAX = 0.8
PREP = M.HERE / 'prep_so'


def load_map():
    z = np.load(M.HERE / 'ratio_map_parts.npz')
    tr = z['split'] == 'train'
    return M.RatioMap(z['n'][tr].sum(0), z['s1'][tr].sum(0), z['s2'][tr].sum(0), float(z['L']), (z['sv_v'], z['sv_s']))


def run_bag(rm, df):
    ok = (df.est_ok & np.isfinite(df.s_est) & np.isfinite(df.y) & (df.v > M.VMIN) & ((df['flags'].astype(np.int64) & M.BAD_FLAGS) == 0)
          & (df.y.abs() < 0.05)).to_numpy()
    s, y, v, st = df.s_est.to_numpy(), df.y.to_numpy(), df.v.to_numpy(), df.s_true.to_numpy()
    idx = np.flatnonzero(ok)
    out = []
    last = -np.inf
    for i in idx:
        if s[i] - last < STEP:
            continue
        last = s[i]
        w = idx[(s[idx] >= s[i] - W) & (s[idx] <= s[i]) & (idx <= i)]
        if len(w) < 100 or not np.isfinite(st[i]):
            continue
        rel = s[w] - s[i]
        sv = M.sigma_v(v[w], rm.sv_tab)
        ll, _ = M.match(rm, y[w], sv, rel, s[i], GRID)
        d_hat, sig_c, sig_p, margin, margin_lm, edge = M.peak_stats(GRID, ll)
        acc = bool(margin_lm > MARGIN and sig_c < SIGMAX and not edge)
        out.append({'s': s[i], 'n': len(w), 'e_filt': s[i] - st[i], 'e_corr': s[i] + d_hat - st[i], 'd_hat': d_hat,
                    'sig_c': sig_c, 'margin_lm': margin_lm, 'accepted': acc})
    return out


def main():
    if ARGS.prep_from:
        import prep_val as P
        P.M.REPLAY_VAL = M.ROOT / 'build_core' / 'replay_tmp' / ARGS.prep_from
        P.OUT = PREP
        PREP.mkdir(exist_ok=True)
        P.main()
    rm = load_map()
    rows = []
    for p in sorted(PREP.glob('*.pkl')):
        if p.stem.endswith('_fix'):
            continue
        for r in run_bag(rm, pd.read_pickle(p)):
            r['bag'] = p.stem
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(M.HERE / 'sim_correct.csv', index=False)
    acc = df[df.accepted]
    after = np.where(df.accepted, df.e_corr, df.e_filt)
    lines = [
        f'VAL, train-only map, step {STEP:.0f} m, window {W:.0f} m, grid +-8 m: {len(df)} triggers in {df.bag.nunique()} bags',
        f'accepted {len(acc)} ({len(acc) / len(df) * 100:.0f} %); accepted |e_corr|: median {acc.e_corr.abs().median():.2f} m, '
        f'p90 {acc.e_corr.abs().quantile(0.9):.2f}, > 1.5 m: {int((acc.e_corr.abs() > 1.5).sum())}',
        f'at the same accepted triggers the filter had |e_filt| median {acc.e_filt.abs().median():.2f} m, p90 {acc.e_filt.abs().quantile(0.9):.2f}',
        f'RMS over all triggers: filter {np.sqrt(np.mean(df.e_filt ** 2)):.3f} m -> with accepted corrections {np.sqrt(np.mean(after ** 2)):.3f} m',
        f'triggers with |e_filt| > 1 m: {int((df.e_filt.abs() > 1).sum())}; of them accepted with |e_corr| < 0.5 m: '
        f'{int(((df.e_filt.abs() > 1) & df.accepted & (df.e_corr.abs() < 0.5)).sum())}',
    ]
    for bag, g in df.groupby('bag'):
        ga = np.where(g.accepted, g.e_corr, g.e_filt)
        lines.append(f'  {bag}: triggers {len(g)}, accepted {int(g.accepted.sum())}, RMS e_filt {np.sqrt(np.mean(g.e_filt ** 2)):.2f} '
                     f'-> {np.sqrt(np.mean(ga ** 2)):.2f} m')
    txt = '\n'.join(lines)
    print(txt)
    (M.HERE / 'sim_correct.txt').write_text(txt, encoding='utf-8')


if __name__ == '__main__':
    main()
