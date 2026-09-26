"""Place-fix attempts of the val replays (replay_ab_fixes.csv) against GNSS truth.

Truth s: RTK master fixes on map_train/main; where a run has no RTK nearby, usable non-RTK fixes are used
and flagged (truth_rtk = False, error ~0.5-2 m). Reports, for the current landmark file, where the
estimator's real fix attempts happen relative to the landmarks (bands 0-1.5 / 1.5-4 / >4 m), how often a
stop in the 1.5-4 m "unknown place" band is accepted as a fix, and how many of those a train mixture
component (or the leaky oracle queue row) would explain.
Run: python analysis/ideas_check/stop_mixture/analyze_fixes.py   (after replay_ab.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'analysis' / 'map_build'))
sys.path.insert(0, str(HERE))
from fit_mixture import Tee, cyc  # noqa: E402


def truth_tables(bags):
    import runs as R
    from validate import MapProjector
    E = MapProjector(ROOT / 'analysis' / 'map_build' / 'map_train').polys['main']
    out = {}
    for b, r in R.load_runs(bags).items():
        s, dd, dist, seg, dpsi = E.project(r.x, r.y, r.psi, max_d=3.0, max_dpsi=np.radians(60))
        ok = np.isfinite(s)
        out[b] = {'rtk': (r.th[ok & r.good], s[ok & r.good]), 'any': (r.th[ok & r.usable], s[ok & r.usable])}
    return out, float(E.length)


def interp(tt, ss, t, L):
    i = np.searchsorted(tt, t)
    if i == 0 or i >= len(tt) or tt[i] - tt[i - 1] > 1.0:
        return np.nan
    return float(np.interp(t, tt[i - 1:i + 1], np.unwrap(ss[i - 1:i + 1], period=L)) % L)


def main():
    sys.stdout = Tee(HERE / 'analyze_fixes_log.txt')
    pd.set_option('display.width', 250)
    fr = pd.read_csv(HERE / 'replay_ab_fixes.csv')
    tru, L = truth_tables(sorted(fr.bag.unique()))
    st, rtk = [], []
    for r in fr.itertuples(index=False):
        a = interp(*tru[r.bag]['rtk'], r.t, L)
        if np.isfinite(a):
            st.append(a)
            rtk.append(True)
        else:
            st.append(interp(*tru[r.bag]['any'], r.t, L))
            rtk.append(False)
    fr['s_true'] = st
    fr['truth_rtk'] = rtk
    fr['err_pred'] = cyc(fr.s_map - fr.s_true, L)
    fr['err_after'] = cyc(fr.s_map_after - fr.s_true, L)
    lm = pd.read_csv(ROOT / 'analysis' / 'validation_maps' / 'landmarks.csv', comment='#')
    mix = pd.read_csv(HERE / 'landmarks_mixture_train.csv', comment='#')
    orc = pd.read_csv(HERE / 'landmarks_oracle_queue_LEAKY.csv', comment='#')
    d = cyc(fr.s_true.values[:, None] - lm.s.values[None, :], L)
    j = np.nanargmin(np.where(np.isfinite(d), np.abs(d), np.inf), 1)
    fr['lm_s'] = lm.s.values[j]
    fr['d_lm'] = d[np.arange(len(fr)), j]
    for nm, rows in (('mix', mix), ('oracle', orc)):
        extra = rows[rows.cls == 'secondary']
        dd = np.abs(cyc(fr.s_true.values[:, None] - extra.s.values[None, :], L))
        fr[f'expl_{nm}'] = (dd <= np.maximum(1.0, 2 * extra.sigma.values)[None, :]).any(1) if len(extra) else False
    fr.to_csv(HERE / 'replay_ab_fixes_truth.csv', index=False)
    c = fr[(fr.variant == 'cur') & fr.s_true.notna()].copy()
    ad = c.d_lm.abs()
    c['band'] = np.select([ad < 1.5, ad <= 4.0], ['<1.5 m', '1.5-4 m'], '>4 m')
    print(f'val place-fix attempts (current file): {int((fr.variant == "cur").sum())}, with truth {len(c)} '
          f'(RTK {int(c.truth_rtk.sum())}); association sd: median {c.sd.median():.2f} m, p90 {c.sd.quantile(.9):.2f} m')
    g = c.groupby('band').agg(n=('t', 'size'), accepted=('accepted', 'sum'), rtk=('truth_rtk', 'sum'),
                              med_abs_err_pred=('err_pred', lambda x: x.abs().median()),
                              med_abs_err_after=('err_after', lambda x: x.abs().median()),
                              expl_mix=('expl_mix', 'sum'), expl_oracle=('expl_oracle', 'sum'))
    print(g.round(3).to_string())
    b = c[c.band == '1.5-4 m'].sort_values(['bag', 't'])
    print('\nattempts in the 1.5-4 m band (d_lm = true stop - nearest landmark):')
    print(b[['bag', 's_true', 'truth_rtk', 'lm_s', 'd_lm', 'sd', 'n', 'p_known', 'accepted', 'err_pred', 'err_after',
             'expl_mix', 'expl_oracle']].round(2).to_string(index=False))
    acc = b[b.accepted]
    worse = acc[acc.err_after.abs() > acc.err_pred.abs() + 0.2]
    print(f'\n1.5-4 m band: {len(b)} attempts, {len(acc)} accepted as a fix, {len(worse)} of them made the error '
          f'> 0.2 m worse; explained by a train mixture row: {int(b.expl_mix.sum())}, by the leaky oracle queue row: '
          f'{int(b.expl_oracle.sum())}')


if __name__ == '__main__':
    main()
