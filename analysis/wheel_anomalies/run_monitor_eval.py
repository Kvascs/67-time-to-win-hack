"""Evaluate the GNSS-free CausalWheelMonitor.

1. GNSS bags: recall of GNSS-labelled wheel episodes (per sensor), detection delay, false-alarm runs per hour
   of motion, and speed error of the monitor's trusted wheel speed vs naive fusions (mean/front/min/max)
   overall and inside episodes.
2. No-GNSS bags: list of monitor flag runs = candidate anomaly episodes.
3. Synthetic injection on clean bags: detection and error for dropout/spike/freeze/zero/slip/slide/noise/
   NaN/stamp-jump/duplicate faults.
Outputs: monitor_flags.csv, monitor_eval_bags.csv, monitor_injection.csv, episodes_nognss_candidates.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import anomalies as A  # noqa: E402
from build_aligned import load_aligned  # noqa: E402

SP = A.splits()
INFO = {x['bag']: x for x in SP['info']}


def flag_runs(rows, t0, merge=0.6, ignore=('ok', 'stale')):
    """Group non-ok monitor classifications into runs per sensor (header time)."""
    out = []
    for s in ('front', 'rear'):
        fl = [r for r in rows if r[0] == s and r[4] not in ignore]
        cur = None
        for r in fl:
            if cur and r[2] - cur['t1'] <= merge:
                cur['t1'] = r[2]; cur['n'] += 1; cur['classes'].add(r[4]); cur['kinds'].add(r[5])
                cur['v_max'] = max(cur['v_max'], r[3]); cur['v_min'] = min(cur['v_min'], r[3])
            else:
                if cur:
                    out.append(cur)
                cur = dict(sensor=s, t0=r[2], t1=r[2], n=1, classes={r[4]}, kinds={r[5]}, v_max=r[3], v_min=r[3],
                           notch=r[7], v_hat=r[6])
        if cur:
            out.append(cur)
    for o in out:
        o['t_start'] = o['t0'] - t0
        o['dur'] = o['t1'] - o['t0'] + 0.1
        o['classes'] = '/'.join(sorted(o['classes']))
        o['kinds'] = '/'.join(sorted(o['kinds']))
    return sorted(out, key=lambda o: o['t0'])


def causal_on_grid(t_msg, val, tg):
    """Value of the latest message with t_msg <= tg (causal hold), NaN before first / if older than 0.5 s."""
    o = np.argsort(t_msg, kind='stable')
    t_msg = t_msg[o]; val = val[o]
    i = np.searchsorted(t_msg, tg, side='right') - 1
    out = np.full(len(tg), np.nan)
    ok = i >= 0
    out[ok] = val[i[ok]]
    age = np.full(len(tg), np.inf); age[ok] = tg[ok] - t_msg[i[ok]]
    out[age > 0.5] = np.nan
    return out


def eval_bag(name, eps_all, sc):
    b = A.load_bag(name)
    kf = kr = 3.597
    if sc is not None and name in sc.index:
        kf, kr = sc.loc[name, 'k_speed_front'], sc.loc[name, 'k_speed_rear']
    p = A.MonitorParams(k_front=kf, k_rear=kr)
    rows, mon = A.replay_monitor(b, p)
    runs = flag_runs(rows, b.t0)
    res = dict(bag=name, vehicle=name[:5], n_msgs=len(rows), **{f'n_{k}': v for k, v in mon.counts.items()})
    has_gnss = sc is not None and name in sc.index
    for r in runs:
        r['bag'] = name
        r['gnss_episode'] = ''
    if not has_gnss:
        res['n_flag_runs'] = len(runs)
        return res, runs, []
    al = load_aligned(name)
    eps = eps_all[eps_all.bag == name]
    # recall per GNSS episode
    ep_rows = []
    for _, e in eps.iterrows():
        t_a = e.t_start_hdr - 0.5; t_b = e.t_start_hdr + e.dur + 0.5
        hit = [r for r in runs if r['sensor'] == e.sensor and r['t1'] >= t_a and r['t0'] <= t_b]
        hit_any = [r for r in runs if r['t1'] >= t_a and r['t0'] <= t_b]
        ep_rows.append(dict(bag=name, sensor=e.sensor, kind=e.kind, t_start=e.t_start, dur=e.dur, e_peak=e.e_peak,
                            detected_same_sensor=bool(hit), detected_any=bool(hit_any),
                            delay=(min(r['t0'] for r in hit) - e.t_start_hdr) if hit else np.nan))
    for r in runs:
        m = eps[(eps.t_start_hdr - 1.0 <= r['t1']) & (eps.t_start_hdr + eps.dur + 1.0 >= r['t0'])]
        r['gnss_episode'] = ';'.join(f'{x.sensor}:{x.kind}' for x in m.itertuples()) if len(m) else ''
    # speed error on grid
    tg = al.t
    arr = np.array([(r[2], r[8]) for r in rows], dtype=float)
    tv = causal_on_grid(arr[:, 0], arr[:, 1], tg)
    # naive fusions, evaluated causally as well (latest raw sample with header <= t)
    raw = {s: np.array([(r[2], r[3]) for r in rows if r[0] == s and np.isfinite(r[3])], dtype=float) for s in ('front', 'rear')}
    f = causal_on_grid(raw['front'][:, 0], raw['front'][:, 1], tg)
    rr = causal_on_grid(raw['rear'][:, 0], raw['rear'][:, 1], tg)
    nt = np.nan_to_num(al.notch)
    with np.errstate(all='ignore'):
        mean = np.nanmean(np.c_[f, rr], axis=1)
        cands = dict(monitor=tv, mean=mean, front=f,
                     minmax=np.where((nt > 0), np.fmin(f, rr), np.where((nt < 0) & (nt >= -7), np.fmax(f, rr), mean)))
    valid = np.isfinite(al.ref) & ~A.lag_transition_mask(al)
    in_ep = A.episode_mask(al, [dict(i0=int(np.searchsorted(tg, e.t_start_hdr)),
                                     i1=int(np.searchsorted(tg, e.t_start_hdr + e.dur))) for _, e in eps.iterrows()], 1.0)
    moving_h = np.sum(np.nan_to_num(al.ref) > 0.3) * 0.1 / 3600
    res.update(moving_h=moving_h, n_flag_runs=len(runs),
               n_fa_runs=sum(1 for r in runs if not r['gnss_episode']),
               n_gnss_eps=len(eps), n_detected=sum(x['detected_same_sensor'] for x in ep_rows))
    for nm, v in cands.items():
        e = v - al.ref
        m = valid & np.isfinite(e)
        res[f'rmse_{nm}'] = float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else np.nan
        mi = m & in_ep
        res[f'rmse_ep_{nm}'] = float(np.sqrt(np.mean(e[mi] ** 2))) if mi.any() else np.nan
        res[f'maxabs_ep_{nm}'] = float(np.max(np.abs(e[mi]))) if mi.any() else np.nan
        res[f'cov_{nm}'] = float(np.mean(np.isfinite(v[valid])))
    return res, runs, ep_rows


def injection_tests(bags=('30618_073f08d1', '30618_b95ca60a', '30639_253671cc'), seed=1):
    rng = np.random.default_rng(seed)
    kinds = [('dropout', 'front', 3.0, None), ('dropout', 'rear', 20.0, None), ('spike', 'front', 5.0, 80.0),
             ('freeze', 'rear', 5.0, None), ('zero', 'front', 3.0, None), ('slip', 'rear', 3.0, 2.0),
             ('slide', 'front', 3.0, 2.0), ('noise', 'rear', 5.0, 1.0), ('nan', 'front', 2.0, None),
             ('stamp_jump', 'front', 2.0, 1.0), ('dup', 'rear', 5.0, None), ('scale', 'front', 10.0, 0.05)]
    out = []
    for name in bags:
        b = A.load_bag(name)
        sc = pd.read_csv(HERE / 'scale_factors.csv').set_index('bag')
        p = A.MonitorParams(k_front=sc.loc[name, 'k_speed_front'], k_rear=sc.loc[name, 'k_speed_rear'])
        rows0, _ = A.replay_monitor(b, p)
        arr0 = np.array([(r[2], r[8]) for r in rows0], float)
        # candidate injection times: moving at 4..12 m/s
        f = b.front
        cand = f.t_bag[(f.v / A.KMH > 4) & (f.v / A.KMH < 12)] - b.t0
        for kind, sensor, dur, mag in kinds:
            for rep in range(4):
                t_rel = float(rng.choice(cand))
                nb = A.inject(b, kind, t_rel, dur, sensor, mag, seed=rep)
                rows, mon = A.replay_monitor(nb, p)
                arr = np.array([(r[2], r[8]) for r in rows if np.isfinite(r[2])], float)
                t_a = b.t0 + t_rel; t_b = t_a + dur
                # detection: any non-ok flag on the injected sensor within the window (+0.5 s)
                det = any(r[0] == sensor and r[4] not in ('ok',) and t_a - 0.2 <= r[1] <= t_b + 0.5 for r in rows)
                # error of trusted speed vs clean-run trusted speed, on a 0.1 s grid over the window (+2 s)
                tg = np.arange(t_a - (b.front.t_bag[0] - b.front.t_hdr[0]) * 0, t_b + 2.0, 0.1)
                # compare in header time: map bag window to header via median latency
                lat = np.median(b.front.t_bag - b.front.t_hdr)
                tg_h = tg - lat
                v0 = causal_on_grid(arr0[:, 0], arr0[:, 1], tg_h)
                v1 = causal_on_grid(arr[:, 0], arr[:, 1], tg_h) if len(arr) else np.full(len(tg_h), np.nan)
                err = np.abs(v1 - v0)
                out.append(dict(bag=name, kind=kind, sensor=sensor, dur=dur, mag=mag, t_rel=t_rel, detected=det,
                                max_err=float(np.nanmax(err)) if np.isfinite(err).any() else np.nan,
                                mean_err=float(np.nanmean(err)) if np.isfinite(err).any() else np.nan,
                                coverage=float(np.mean(np.isfinite(v1))),
                                crashed=False))
        print('injection done', name, flush=True)
    return pd.DataFrame(out)


def _eval_one(args):
    name, has = args
    sc = pd.read_csv(HERE / 'scale_factors.csv').set_index('bag')
    eps = pd.read_csv(HERE / 'episodes_gnss.csv')
    return eval_bag(name, eps, sc if has else None)


def _inject_one(name):
    return injection_tests(bags=(name,), seed=int(name[-4:], 16) % 1000)


def main():
    from concurrent.futures import ProcessPoolExecutor
    sc = pd.read_csv(HERE / 'scale_factors.csv').set_index('bag')
    names = sorted(INFO, key=lambda x: INFO[x]['t0'])
    res, runs_all, eprows = [], [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for (r, runs, ep) in ex.map(_eval_one, [(n, n in sc.index) for n in names]):
            res.append(r); runs_all += runs; eprows += ep
            print(r['bag'], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                             if k.startswith(('n_flag', 'n_fa', 'n_det', 'n_gnss', 'rmse_ep'))}, flush=True)
    pd.DataFrame(res).to_csv(HERE / 'monitor_eval_bags.csv', index=False)
    fr = pd.DataFrame(runs_all)
    fr = fr[['bag', 'sensor', 't_start', 'dur', 'n', 'classes', 'kinds', 'v_min', 'v_max', 'notch', 'v_hat', 'gnss_episode']]
    fr.to_csv(HERE / 'monitor_flags.csv', index=False)
    pd.DataFrame(eprows).to_csv(HERE / 'monitor_recall.csv', index=False)
    nog = fr[~fr.bag.isin(sc.index)]
    nog.to_csv(HERE / 'episodes_nognss_candidates.csv', index=False)
    inj_bags = ('30618_073f08d1', '30618_b95ca60a', '30639_253671cc', '30618_e3d94878', '30618_a869780d', '30639_2b4a6347')
    with ProcessPoolExecutor(max_workers=6) as ex:
        inj = pd.concat(list(ex.map(_inject_one, inj_bags)), ignore_index=True)
    inj.to_csv(HERE / 'monitor_injection.csv', index=False)
    print(inj.groupby('kind')[['detected', 'max_err', 'mean_err', 'coverage']].mean().round(3))


if __name__ == '__main__':
    main()
