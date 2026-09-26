"""Post-hoc analysis of saved traces (cache/trace_<tag>_<bag>.csv) and eval tables.

    python analyze.py thresholds full        # fix time / false fixes vs the confidence threshold
    python analyze.py table full_train full_val ...   # one summary line per eval csv
    python analyze.py nognss full full_start # no-GNSS bags: fix rate / time, post-fix stop-landmark match
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd

import common as C

FALSE_M = 20.0


def thresholds(tag='full'):
    files = sorted(glob.glob(str(C.CACHE / f'trace_{tag}_*.csv')))
    rows = []
    for th in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
        for mc in (1, 2, 3):
            tf, df, false, n, nfix, wrong_any = [], [], 0, 0, 0, 0
            for f in files:
                t = pd.read_csv(f)
                if t.err.notna().mean() < 0.5:
                    continue
                n += 1
                k = np.flatnonzero((t.conf >= th) & (t.n_pos >= mc))
                if not len(k):
                    continue
                nfix += 1
                i = k[0]
                tf.append(t.t[i])
                df.append(t.r[i])
                e = t.err[i]
                false += int(np.isfinite(e) and abs(e) > FALSE_M)
                # confident-and-wrong at any later update (reference-consistent samples only)
                sel = (t.index >= i) & (t.conf >= th) & t.ref_ok & t.err.notna()
                wrong_any += int((np.abs(t.err[sel]) > FALSE_M).any())
            rows.append(dict(p_fix=th, min_cues=mc, bags=n, fixed=nfix, t_fix_med=np.median(tf) if tf else np.nan,
                             t_fix_p90=np.percentile(tf, 90) if tf else np.nan,
                             d_fix_med=np.median(df) if df else np.nan, false_at_fix=false,
                             confident_wrong_later=wrong_any))
    print(pd.DataFrame(rows).round({"t_fix_med": 1, "t_fix_p90": 1, "d_fix_med": 1}).to_string(index=False))


def table(*tags):
    from run_eval import summarize
    out = []
    for tag in tags:
        for f in sorted(glob.glob(str(C.CACHE / f'eval_{tag}.csv'))):
            df = pd.read_csv(f)
            if 'dist' in df and 't_start' in df:
                df = df[df.dist >= 2000]                  # trials with enough track left to localise
            out.append(summarize(df, f.split('eval_')[1][:-4]))
    print(pd.DataFrame(out).round(3).to_string(index=False))


def nognss(*tags):
    for tag in tags:
        df = pd.read_csv(C.CACHE / f'eval_{tag}_no_gnss_long.csv')
        fx = df[df.fixed == True]  # noqa: E712
        print(json.dumps(dict(tag=tag, n=len(df), fixed=len(fx), t_fix_med=round(fx.t_fix.median(), 1),
                              t_fix_p90=round(fx.t_fix.quantile(0.9), 1), d_fix_med=round(fx.d_fix.median(), 1),
                              d_fix_p90=round(fx.d_fix.quantile(0.9), 1),
                              stop_match_med=round(fx.stop_match.median(), 3),
                              stop_match_min=round(fx.stop_match.min(), 3),
                              kappa_fix=[round(x, 4) for x in fx.kappa_fix])))
        print(df[['bag', 'dist', 'fixed', 't_fix', 'd_fix', 'kappa_fix', 'n_pos_fix', 'stops_after',
                  'stop_match']].round(3).to_string(index=False))


if __name__ == '__main__':
    {'thresholds': thresholds, 'table': table, 'nognss': nognss}[sys.argv[1]](*sys.argv[2:])
