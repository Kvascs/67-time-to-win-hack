"""Evaluate the GNSS-free global localiser on bags (GNSS removed from the replay; GNSS only for scoring).

    python run_eval.py --split val --tag full
    python run_eval.py --split train --tag stops_only --set use_cut=0 use_speed=0 use_cutpass=0

Per bag: time / distance to the first confident fix, along-track error at the fix and afterwards,
false fix (error > FALSE_M at the fix), lost (error > FALSE_M at any later update).
Writes cache/eval_<tag>_<split>.csv (per bag) and cache/trace_<tag>_<bag>.npz (per-update trace).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import common as C
import data as D
from glocal import Calib, GLParams, GlobalLocalizer
from mapmodel import MapModel

FALSE_M = 20.0
STOP_AFTER_FIX_M = 0.0  # >0: end a trial this far past the fix (saves CPU in start-offset sweeps)


def build_events(b: D.Bag, p: GLParams):
    """Causal cue stream: (t, kind, r, payload) sorted by time."""
    ev = []
    for st in D.stop_events(b, p.stop_dwell):
        ev.append((st['t_emit'], 'start' if st['initial'] else 'stop', st['s_rel'], None))
    for c in D.cutoff_events(b, p.cut_notch, p.cut_vmin):
        ev.append((c['t'], 'cut', c['s_rel'], None))
    r = b.s_rel
    marks = np.arange(p.checkpoint, r.max(), p.checkpoint)
    idx = np.searchsorted(r, marks)                     # first sample at/after each mark (r is monotone)
    idx = idx[idx < len(r)]
    good = (b.v > p.g_vmin) & ((b.flags & C.FLAG_STANDSTILL) == 0) & (b.mu3 < 0.5)
    prev = 0
    for i in idx:
        sl = slice(prev, i + 1)
        vmax = float(np.max(b.v[sl])) if i >= prev else np.nan
        g = good[sl]
        d_obs = float(np.mean(b.d[sl][g])) if g.mean() > 0.5 else np.nan
        ev.append((float(b.t[i]), 'cp', float(r[i]), (vmax, d_obs)))
        prev = i + 1
    ev.sort(key=lambda e: e[0])
    return ev


def cut_bag(b: D.Bag, t_start: float) -> D.Bag:
    """Pretend the recording begins t_start s into the bag (start anywhere, possibly moving)."""
    t0 = b.t[0] + t_start
    k = b.t >= t0
    nk = b.notch_t >= t0
    s0 = b.s_rel[k][0]
    c = D.Bag(b.name, b.t[k], b.s_rel[k] - s0, b.v[k], b.d[k], b.flags[k], b.mu3[k], b.a_model[k],
              b.notch_t[nk], b.notch[nk], None if b.truth_s is None else b.truth_s[k],
              None if b.truth_ok is None else b.truth_ok[k], b.has_truth, dict(b.meta))
    return c


def run_bag(name: str, p: GLParams, tag: str = '', save_trace: bool = False, t_start: float = 0.0):
    mm = MapModel()
    cal = Calib.load(C.CACHE / 'calib_train.npz')
    b = D.load_bag(name)
    if t_start > 0:
        b = cut_bag(b, t_start)
    gl = GlobalLocalizer(mm, cal, p)
    # notch as a function of the odometer (only past values are ever queried)
    r_n = np.interp(b.notch_t, b.t, b.s_rel)

    def notch_fn(rq):
        k = np.searchsorted(r_n, rq, side='right') - 1
        return b.notch[np.clip(k, 0, len(b.notch) - 1)]

    L = mm.L
    lm_s = cal.lm_s
    rows = []
    t0 = b.t[0]
    r_fix = None
    for t, kind, r, pay in build_events(b, p):
        if r_fix is not None and STOP_AFTER_FIX_M > 0 and r > r_fix + STOP_AFTER_FIX_M:
            break
        if kind == 'start':
            if p.use_start:
                gl.apply_start_prior()
            continue
        if kind == 'stop':
            gl.on_stop(r)
        elif kind == 'cut':
            gl.on_cutoff(r)
        else:
            gl.on_checkpoint(r, pay[0], pay[1], notch_fn)
        e = gl.estimate(r)
        ts = np.interp(t, b.t, b.truth_s) if b.truth_s is not None else np.nan
        err = ((e['s'] - ts + 0.5 * L) % L - 0.5 * L) if np.isfinite(ts) else np.nan
        dlm = np.min(np.abs((lm_s - e['s'] + 0.5 * L) % L - 0.5 * L))   # estimated stop vs nearest landmark
        rows.append(dict(t=t - t0, kind=kind, r=r, conf=e['conf'], s=e['s'], kappa=e['kappa'], n_pos=e['n_pos'],
                         s_true=ts, err=err, dlm=dlm))
        if r_fix is None and e['conf'] >= p.p_fix and e['n_pos'] >= p.min_cues:
            r_fix = r
    tr = pd.DataFrame(rows)
    # reference sanity: the true arc length must advance like the odometer between updates (GNSS header-stamp
    # jumps of +-1 s and float fixes break this); errors are only scored where the reference is consistent
    if len(tr):
        dtrue = np.diff(np.r_[np.nan, tr.s_true.to_numpy()])
        dodo = np.diff(np.r_[np.nan, tr.r.to_numpy()])
        tr['ref_ok'] = np.abs(dtrue - dodo) < 3.0 + 0.03 * np.abs(dodo)
        tr['s_true'] = tr.s_true % L
    res = dict(bag=name, t_start=t_start, dur=float(b.t[-1] - t0), dist=float(b.s_rel[-1]), n_upd=len(tr),
               has_truth=bool(b.has_truth), rtk=b.meta.get('rtk_share', np.nan))
    fx = tr[(tr.conf >= p.p_fix) & (tr.n_pos >= p.min_cues)] if len(tr) else tr
    if len(fx):
        i = fx.index[0]
        res.update(fixed=True, t_fix=float(tr.t[i]), d_fix=float(tr.r[i]), err_fix=float(tr.err[i]),
                   kappa_fix=float(tr.kappa[i]), n_pos_fix=int(tr.n_pos[i]))
        after = tr.loc[i:]
        a_all = np.abs(after.err.dropna().to_numpy())
        a = np.abs(after.err[after.ref_ok].dropna().to_numpy())
        st = after[after.kind == 'stop']
        res.update(stops_after=len(st), stop_match=float((st.dlm < 3.0).mean()) if len(st) else np.nan,
                   lost_raw=float((a_all > FALSE_M).mean()) if len(a_all) else np.nan)
        if len(a):
            res.update(err_after_med=float(np.median(a)), err_after_p95=float(np.percentile(a, 95)),
                       err_after_max=float(a.max()), frac_lost=float((a > FALSE_M).mean()),
                       conf_after_min=float(after.conf.min()))
        res['false_fix'] = bool(np.isfinite(res['err_fix']) and abs(res['err_fix']) > FALSE_M)
    else:
        res.update(fixed=False)
    if save_trace:
        tr.to_csv(C.CACHE / f'trace_{tag}_{name}.csv', index=False)
    return res


def _job(args):
    global STOP_AFTER_FIX_M
    name, pd_, tag, save = args[:4]
    t_start = args[4] if len(args) > 4 else 0.0
    STOP_AFTER_FIX_M = args[5] if len(args) > 5 else 0.0
    t = time.time()
    try:
        r = run_bag(name, GLParams(**pd_), tag, save, t_start)
    except Exception as e:  # keep the sweep going
        r = dict(bag=name, error=repr(e)[:300])
    r['cpu_s'] = time.time() - t
    return r


def parse_sets(sets):
    out = {}
    for kv in sets:
        k, v = kv.split('=', 1)
        f = GLParams.__dataclass_fields__[k]
        out[k] = (v not in ('0', 'false', 'False')) if f.type in ('bool', bool) else type(f.default)(float(v))
    return out


def summarize(df: pd.DataFrame, label=''):
    ok = df[df.has_truth] if 'has_truth' in df else df
    fx = ok[ok.fixed == True]  # noqa: E712
    n = len(ok)
    s = dict(label=label, n=n, fixed=len(fx), fix_rate=len(fx) / max(n, 1))
    if len(fx):
        s.update(t_fix_med=fx.t_fix.median(), t_fix_p90=fx.t_fix.quantile(0.9), d_fix_med=fx.d_fix.median(),
                 d_fix_p90=fx.d_fix.quantile(0.9), false_fix=int(fx.false_fix.sum()),
                 err_fix_med=fx.err_fix.abs().median(), err_fix_p90=fx.err_fix.abs().quantile(0.9),
                 err_after_med=fx.err_after_med.median(), err_after_p95=fx.err_after_p95.median(),
                 lost_bags=int((fx.frac_lost > 0).sum()), lost_raw_bags=int((fx.get('lost_raw', pd.Series(dtype=float)) > 0).sum()),
                 stop_match_med=fx.get('stop_match', pd.Series(dtype=float)).median(),
                 stop_match_min=fx.get('stop_match', pd.Series(dtype=float)).min())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--tag', default='default')
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--bags', nargs='*')
    ap.add_argument('--stop-after-fix', type=float, default=0.0)
    ap.add_argument('--starts', nargs='*', type=float, default=[0.0],
                    help='start the localiser this many seconds into each bag (several = more trials)')
    a = ap.parse_args()
    global STOP_AFTER_FIX_M
    STOP_AFTER_FIX_M = a.stop_after_fix
    pd_ = parse_sets(a.set)
    bags = a.bags or (C.SPLITS['train'] + C.SPLITS['val'] if a.split == 'all' else C.SPLITS[a.split])
    jobs = [(b, pd_, a.tag, a.trace, ts, a.stop_after_fix) for ts in a.starts for b in bags]
    with ProcessPoolExecutor(min(a.jobs, 2)) as ex:
        rows = list(ex.map(_job, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(C.CACHE / f'eval_{a.tag}_{a.split}.csv', index=False)
    pd.set_option('display.width', 250)
    cols = [c for c in ['bag', 'rtk', 'dist', 'fixed', 't_fix', 'd_fix', 'err_fix', 'kappa_fix', 'n_pos_fix',
                        'err_after_med', 'err_after_p95', 'err_after_max', 'frac_lost', 'lost_raw', 'stop_match',
                        'cpu_s', 'error']
            if c in df.columns]
    print(df[cols].round(3).to_string(index=False))
    s = summarize(df, f'{a.tag}/{a.split}')
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}))


if __name__ == '__main__':
    main()
