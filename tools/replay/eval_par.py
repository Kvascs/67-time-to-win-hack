"""Parallel quick_eval over a split with a chosen tbo_replay binary (A/B comparisons of core changes).

    python tools/replay/eval_par.py --tag head --exe build_head/tbo_replay.exe --split val
    python tools/replay/eval_par.py --tag new --split val --set init_sigma_scale=0.015
    python tools/replay/eval_par.py --tag model --split val --dropwin 60:10   # model-only windows

--dropwin P:D removes both bogies for D seconds every P seconds (first window at P/2 after the
first message): inside those windows the estimate is pure model prediction, which measures the
traction/grade model (speed RMSE and along-track drift per window vs the GNSS reference).

Writes build_core/eval/<tag>.csv (one row per bag) and prints ALL / GOOD-REF means and medians.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpp_bridge  # noqa: E402
import quick_eval  # noqa: E402

KEYS = ['v_rmse', 'v_mae', 'v_bias', 'v_p99', 'along_rmse', 'along_max', 'cross_rmse', 'end_err', 'drift_pct',
        'model_only', 'win_v_rmse', 'win_ds_abs']


def _dropwin_export(period: float, dur: float):
    """Wraps cpp_bridge.export_events: drop W0/W1 lines inside the model-only windows."""
    orig = cpp_bridge.export_events

    def export(bag, out_csv, gnss_seconds=5.0):
        info = orig(bag, out_csv, gnss_seconds)
        lines = Path(out_csv).read_text().splitlines()
        stamps = [int(ln.split(',')[2]) for ln in lines if ln[:2] in ('W0', 'W1', 'C,')]
        t0 = min(stamps) * 1e-9
        keep, wins = [], []
        for ln in lines:
            if ln[:2] in ('W0', 'W1'):
                t = int(ln.split(',')[2]) * 1e-9 - t0
                ph = t - 0.5 * period
                if ph >= 0 and (ph % period) < dur:
                    continue
            keep.append(ln)
        Path(out_csv).write_text('\n'.join(keep) + '\n', newline='\n')
        info['t0'] = t0
        return info

    return export


def _window_metrics(bag: str, o: pd.DataFrame, period: float, dur: float) -> dict:
    d = np.load(cpp_bridge.NPZ / f'{bag}.npz')
    ins = [d[k][:, 1] for k in ('vehicle__front_bogie_velocity', 'vehicle__rear_bogie_velocity',
                                'vehicle__driver_position_cmd') if len(d[k])]
    t0 = min(float(a.min()) for a in ins)
    mv = d['sensing__gnss__master__vel']
    ref_t, ref_v = mv[:, 1], np.hypot(mv[:, 2], mv[:, 3])
    out_t = o.stamp_ns.to_numpy() * 1e-9
    v = o.v.to_numpy()
    s = o.s.to_numpy()
    errs, ds = [], []
    t_end = out_t[-1]
    k = 0
    while True:
        a = t0 + 0.5 * period + k * period
        b = a + dur
        if b > t_end:
            break
        k += 1
        m = (ref_t >= a) & (ref_t < b)
        if m.sum() < 5 or ref_v[m].mean() < 1.0:  # only windows with motion
            continue
        vi = np.interp(ref_t[m], out_t, v)
        errs.append(vi - ref_v[m])
        # distance travelled in the window: estimate vs GNSS speed integral
        tt = ref_t[m]
        s_ref = float(np.sum(0.5 * (ref_v[m][1:] + ref_v[m][:-1]) * np.diff(tt)))
        s_est = np.interp(tt[-1], out_t, s) - np.interp(tt[0], out_t, s)
        ds.append(abs(s_est - s_ref))
    if not errs:
        return {'win_v_rmse': np.nan, 'win_ds_abs': np.nan, 'n_win': 0}
    e = np.concatenate(errs)
    return {'win_v_rmse': float(np.sqrt(np.mean(e ** 2))), 'win_ds_abs': float(np.mean(ds)), 'n_win': len(ds)}


def _one(args):
    bag, exe, tag, sets, dropwin = args
    cpp_bridge.REPLAY_EXE = Path(exe)
    quick_eval.TMP = cpp_bridge.ROOT / 'build_core' / 'replay_tmp' / tag
    try:
        if dropwin:
            quick_eval.export_events = _dropwin_export(*dropwin)
        res, o = quick_eval.eval_bag(bag, sets=sets)
        if dropwin:
            res.update(_window_metrics(bag, o, *dropwin))
        return res
    except Exception as e:  # keep the sweep going, report the failure
        return {'bag': bag, 'error': str(e)[:300]}


def summarize(df: pd.DataFrame) -> dict:
    num = df[[k for k in KEYS if k in df.columns]].astype(float)
    good = df['ref_good'].astype(bool) if 'ref_good' in df.columns else pd.Series(True, index=df.index)
    return {'all_mean': num.mean().round(4).to_dict(), 'all_median': num.median().round(4).to_dict(),
            'good_mean': num[good].mean().round(4).to_dict(), 'good_median': num[good].median().round(4).to_dict(),
            'n': len(df), 'n_good': int(good.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--exe', default=str(cpp_bridge.REPLAY_EXE))
    ap.add_argument('--split', default='val')
    ap.add_argument('--bags', nargs='*')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--dropwin', default='', help='P:D  drop both bogies D s every P s')
    a = ap.parse_args()
    sets = dict(s.split('=', 1) for s in a.set)
    dropwin = tuple(float(x) for x in a.dropwin.split(':')) if a.dropwin else None
    if a.bags:
        bags = a.bags
    else:
        splits = json.load(open(cpp_bridge.ROOT / 'data' / 'splits.json'))
        bags = splits['train'] + splits['val'] if a.split == 'all' else splits[a.split]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(_one, [(b, a.exe, a.tag, sets, dropwin) for b in bags]))
    df = pd.DataFrame(rows)
    out = cpp_bridge.ROOT / 'build_core' / 'eval'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f'{a.tag}.csv', index=False)
    if 'error' in df.columns and df['error'].notna().any():
        print('ERRORS:', df[df['error'].notna()][['bag', 'error']].to_string(index=False))
        df = df[df['error'].isna()]
    s = summarize(df)
    print(f"[{a.tag}] n={s['n']} good={s['n_good']} sets={sets} dropwin={dropwin}")
    show = ('v_rmse', 'v_p99', 'along_rmse', 'along_max', 'end_err', 'drift_pct', 'win_v_rmse', 'win_ds_abs')
    for k in ('all_mean', 'all_median', 'good_mean', 'good_median'):
        print(f'  {k:11s}', {kk: s[k][kk] for kk in show if kk in s[k]})


if __name__ == '__main__':
    main()
