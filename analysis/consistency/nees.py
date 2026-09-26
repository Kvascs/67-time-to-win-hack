"""(a) NEES of the published speed and position covariances on train and val.

    python analysis/consistency/nees.py

Speed:     NEES_v  = (v - v_ref)^2 / v_var                   (1 dof, 95 % bound 3.841)
Along:     NEES_s  = along^2 / var_along                       (1 dof), var_along = u^T C u, u = (cos yaw, sin yaw)
2-D:       NEES_xy = e^T C^-1 e, C = [[cov_xx, cov_xy], [cov_xy, cov_yy]]   (2 dof, 95 % bound 5.991)
A consistent filter has mean NEES = dof, median 0.455 (1 dof) / 1.386 (2 dof) and 95 % of epochs inside.

Reference quality: position NEES only on RTK epochs (master fix status 2) of bags with quick_eval's
ref_good (RTK share > 0.8 and no GNSS header-clock episodes); speed NEES on bags without clock
episodes (Doppler speed does not need RTK). The excluded parts are reported as flagged rows.
Also estimates the reference noise itself (GNSS speed at standstill, cross-track scatter by status).

Writes results/nees_speed.csv, results/nees_position.csv, results/reference_noise.json and caches
the matched pairs in cache/pairs_speed.parquet / cache/pairs_pos.parquet for calibrate.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

Q1, Q2 = chi2.ppf(0.95, 1), chi2.ppf(0.95, 2)


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(C.RESULTS / 'replay_metrics.csv').set_index('bag')
    sp = C.splits()
    vs, ps, noise = [], [], []
    for split in ('train', 'val'):
        for bag in sp[split]:
            _, o = C.load_replay(bag)
            d = C.load_npz(bag)
            m = metrics.loc[bag]
            v = C.speed_pairs(o, d)
            v['bag'], v['split'] = bag, split
            v['clock_ok'] = not (m.ref_clock_anom > 0.01)
            v['ref_good'] = bool(m.ref_good)
            v['standstill'] = (v['flags'] & C.F_STANDSTILL) > 0
            vs.append(v)
            p = C.position_pairs(o, d)
            p['bag'], p['split'] = bag, split
            p['ref_good'] = bool(m.ref_good)
            p['mapped'] = (p['flags'] & (C.F_NO_MAP | C.F_NOT_INIT)) == 0
            ps.append(p)
            noise.append(reference_noise_bag(bag, split, d, o))
    return pd.concat(vs, ignore_index=True), pd.concat(ps, ignore_index=True), pd.DataFrame(noise)


def reference_noise_bag(bag, split, d, o) -> dict:
    """GNSS Doppler noise at standstill (both bogies at 0 for >= 2 s) and while cruising."""
    mv = d['sensing__gnss__master__vel']
    t = mv[:, 1]
    order = np.argsort(t)
    t, vx, vy = t[order], mv[order, 2], mv[order, 3]
    fr = d['vehicle__front_bogie_velocity']
    rr = d['vehicle__rear_bogie_velocity']
    wt = fr[np.argsort(fr[:, 1]), 1]
    wz = np.maximum(fr[np.argsort(fr[:, 1]), 2], np.interp(fr[np.argsort(fr[:, 1]), 1], rr[np.argsort(rr[:, 1]), 1],
                                                           rr[np.argsort(rr[:, 1]), 2]))
    still = wz < 0.15
    # standstill epochs: wheels at zero for >= 2 s around the epoch (both sides)
    run_ok = np.zeros(len(wt), bool)
    for a, b in C.runs(still):
        if wt[b - 1] - wt[a] >= 4.0:
            run_ok[a:b] = (wt[a:b] >= wt[a] + 2.0) & (wt[a:b] <= wt[b - 1] - 2.0)
    st_t = wt[run_ok]
    near = np.zeros(len(t), bool)
    if len(st_t):
        idx = np.clip(np.searchsorted(st_t, t), 1, len(st_t) - 1)
        near = np.minimum(np.abs(st_t[idx] - t), np.abs(st_t[idx - 1] - t)) < 0.06
    spd = np.hypot(vx, vy)
    res = {'bag': bag, 'split': split, 'n_still': int(near.sum())}
    if near.sum() > 20:
        res['still_rms_speed'] = float(np.sqrt(np.mean(spd[near] ** 2)))
        res['still_sigma_axis'] = float(np.sqrt(np.mean(vx[near] ** 2 + vy[near] ** 2) / 2))
        res['still_p95_speed'] = float(np.percentile(spd[near], 95))
    # cruising: second difference of the reference speed (white-noise estimator std(d2)/sqrt(6))
    ok = (spd > 3.0) & (np.abs(np.gradient(C.smooth(spd, 11), t)) < 0.1)
    d2 = spd[2:] - 2 * spd[1:-1] + spd[:-2]
    reg = ok[1:-1] & ok[2:] & ok[:-2] & (np.diff(t)[1:] < 0.15) & (np.diff(t)[:-1] < 0.15)
    if reg.sum() > 100:
        res['cruise_sigma_d2'] = float(np.std(d2[reg]) / np.sqrt(6))
    return res


def stats(x: np.ndarray, bound: float) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {'n': 0}
    return {'n': len(x), 'mean': float(np.mean(x)), 'median': float(np.median(x)),
            'inside95': float(np.mean(x <= bound)), 'p95': float(np.percentile(x, 95))}


def table(df: pd.DataFrame, col: str, bound: float, groups: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    rows = []
    for name, mask in groups:
        sub = df[mask]
        for split in ('train', 'val'):
            s = sub[sub.split == split]
            r = {'set': name, 'split': split, 'bags': s.bag.nunique()}
            r.update(stats(s[col].to_numpy(), bound))
            rows.append(r)
            for reg in C.REGIMES:
                rr = s[s.regime == reg]
                r2 = {'set': name, 'split': split, 'regime': reg, 'bags': rr.bag.nunique()}
                r2.update(stats(rr[col].to_numpy(), bound))
                rows.append(r2)
    out = pd.DataFrame(rows)
    out['regime'] = out['regime'].fillna('all')
    return out[['set', 'split', 'regime', 'bags', 'n', 'mean', 'median', 'inside95', 'p95']]


def main():
    C.ensure_dirs()
    v, p, noise = collect()
    v.to_parquet(C.CACHE / 'pairs_speed.parquet', index=False)
    p.to_parquet(C.CACHE / 'pairs_pos.parquet', index=False)

    # ---------------- speed ----------------
    v['nees'] = v.err ** 2 / v.v_var
    tv = table(v, 'nees', Q1, [
        ('speed, clock-ok bags', v.clock_ok),
        ('speed, clock-ok, estimator not at standstill', v.clock_ok & ~v.standstill),
        ('speed, clock-ok, estimator standstill (P_vv pinned 1e-6)', v.clock_ok & v.standstill),
        ('FLAGGED speed, bags with GNSS clock episodes', ~v.clock_ok)])
    tv.to_csv(C.RESULTS / 'nees_speed.csv', index=False)

    # ---------------- position ----------------
    p['nees_along'] = p.along ** 2 / p.var_along
    base = p.mapped & p.ref_good
    rtk = base & (p.status == 2)
    tp1 = table(p, 'nees_along', Q1, [
        ('along, ref_good bags, RTK epochs', rtk),
        ('FLAGGED along, ref_good bags, non-RTK epochs', base & (p.status != 2)),
        ('FLAGGED along, bad-reference bags, all epochs', p.mapped & ~p.ref_good)])
    tp2 = table(p, 'nees2', Q2, [
        ('2-D, ref_good bags, RTK epochs', rtk),
        ('FLAGGED 2-D, ref_good bags, non-RTK epochs', base & (p.status != 2)),
        ('FLAGGED 2-D, bad-reference bags, all epochs', p.mapped & ~p.ref_good)])
    tp = pd.concat([tp1, tp2], ignore_index=True)
    tp.to_csv(C.RESULTS / 'nees_position.csv', index=False)

    # ---------------- reference noise ----------------
    mov = p.mapped & (p.v_ref_s > 1.0)
    cross_by_status = {}
    for st in (0, 1, 2):
        c = p.cross[mov & (p.status == st)].to_numpy()
        if len(c):
            cross_by_status[f'status{st}'] = {
                'n': len(c), 'rms': float(np.sqrt(np.mean(c ** 2))), 'median_abs': float(np.median(np.abs(c))),
                'p95_abs': float(np.percentile(np.abs(c), 95))}
    along_by_status = {}
    for st in (0, 2):
        a = p.along[mov & p.ref_good & (p.status == st)].to_numpy()
        if len(a):
            along_by_status[f'status{st}_refgood'] = {'n': len(a), 'rms': float(np.sqrt(np.mean(a ** 2)))}
    nz = noise.dropna(subset=['still_rms_speed'])
    ref_noise = {
        'speed_standstill': {'bags': len(nz), 'median_rms_speed': float(nz.still_rms_speed.median()),
                             'pooled_sigma_axis_median': float(nz.still_sigma_axis.median()),
                             'median_p95_speed': float(nz.still_p95_speed.median())},
        'speed_cruise_d2_sigma_median': float(noise.cruise_sigma_d2.median()),
        'cross_track_scatter_by_status_moving': cross_by_status,
        'along_error_rms_by_status_moving': along_by_status,
        'status_share': {str(int(k)): float(vv) for k, vv in p.status.value_counts(normalize=True).items()},
        'ref_good_bags': {s: int(p[p.split == s].groupby('bag').ref_good.first().sum()) for s in ('train', 'val')},
        'bags': {s: int(p[p.split == s].bag.nunique()) for s in ('train', 'val')},
        'clock_ok_bags': {s: int(v[v.split == s].groupby('bag').clock_ok.first().sum()) for s in ('train', 'val')},
    }
    noise.to_csv(C.RESULTS / 'reference_noise_per_bag.csv', index=False)
    C.write_json(ref_noise, C.RESULTS / 'reference_noise.json')

    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)
    print('=== speed NEES (1 dof: mean 1, median 0.455, inside95 0.95) ===')
    print(tv.round(4).to_string(index=False))
    print('\n=== position NEES (along 1 dof; 2-D: mean 2, median 1.386, inside95 0.95) ===')
    print(tp.round(4).to_string(index=False))
    print('\n=== reference noise ===')
    import json
    print(json.dumps(ref_noise, indent=1))


if __name__ == '__main__':
    main()
