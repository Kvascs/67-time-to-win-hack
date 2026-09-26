"""(b) Conformal-style covariance calibration on train, checked on val.

    python analysis/consistency/calibrate.py        (needs nees.py first: cache/pairs_*.parquet)

A published variance P is replaced by alpha * P. The NEES scales by 1/alpha, so the smallest alpha
that puts a share q of train epochs inside the chi2 bound is alpha = Q_q(NEES_train) / chi2_dof(q)
(split-conformal quantile: order statistic ceil((n+1) q) of the train scores). Val only checks.
Epochs inside a bag are correlated, so val coverage gets a bag-bootstrap 95 % interval.

Calibrated quantities:
  speed    beta  for v_var while moving; a variance floor while the estimator is at standstill
           (P_vv pinned to 1e-6 there): floor = Q_0.95(err^2 | standstill, train) / 3.841
  position alpha_s for the along-track variance (the only part the filter estimates), sigma_cross
           (map_sigma_cross) from the cross-track scatter, and a single alpha for the whole 2x2
           covariance as requested; epochs where the tram is on another track than the published
           one (detour / terminal fan, |cross| >= 2 m) are reported separately
  PL       99 % along-track protection level PL = K * sqrt(var_along), K = Q_0.99(|along| / sigma)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROUTE_MISMATCH_M = 2.0


def conformal_quantile(scores: np.ndarray, q: float) -> float:
    s = np.sort(scores[np.isfinite(scores)])
    n = len(s)
    k = min(int(np.ceil((n + 1) * q)), n)
    return float(s[k - 1])


def coverage(df: pd.DataFrame, inside: np.ndarray) -> dict:
    by_bag = [inside[df.bag.to_numpy() == b].astype(float) for b in df.bag.unique()]
    lo, hi = C.bag_bootstrap_ci(by_bag, np.mean) if len(by_bag) > 1 else (np.nan, np.nan)
    return {'n': int(len(inside)), 'bags': len(by_bag), 'coverage': float(np.mean(inside)),
            'ci95_bags': [lo, hi]}


def levels(nees: np.ndarray, dof: int) -> dict:
    return {f'{lv:g}': float(np.mean(nees <= chi2.ppf(lv, dof))) for lv in (0.5, 0.6827, 0.9, 0.95, 0.99)}


def main():
    v = pd.read_parquet(C.CACHE / 'pairs_speed.parquet')
    p = pd.read_parquet(C.CACHE / 'pairs_pos.parquet')
    q1, q2 = chi2.ppf(0.95, 1), chi2.ppf(0.95, 2)
    out = {}

    # ------------------------------------------------ speed
    v = v[v.clock_ok].copy()
    v['nees'] = v.err ** 2 / v.v_var
    mov = ~v.standstill
    tr, va = v.split == 'train', v.split == 'val'
    beta = conformal_quantile(v.nees[tr & mov].to_numpy(), 0.95) / q1
    floor = conformal_quantile((v.err[tr & ~mov] ** 2).to_numpy(), 0.95) / q1
    var_cal = np.where(v.standstill, np.maximum(beta * v.v_var, floor), beta * v.v_var)
    v['nees_cal'] = v.err ** 2 / var_cal
    sp = {'beta_moving': beta, 'beta_sigma_factor': float(np.sqrt(beta)),
          'floor_var_standstill': floor, 'floor_sigma_standstill': float(np.sqrt(floor)),
          'standstill_abs_err_median_train': float(np.median(np.abs(v.err[tr & ~mov]))),
          'standstill_abs_err_median_val': float(np.median(np.abs(v.err[va & ~mov]))),
          'dead_band_bound_ms': C.STANDSTILL_KMH * C.KMH}
    for name, m in (('moving', mov), ('standstill', ~mov), ('all', np.ones(len(v), bool))):
        for split, sm in (('train', tr), ('val', va)):
            sub = v[m & sm]
            sp[f'{split}_{name}_before'] = coverage(sub, (sub.nees <= q1).to_numpy())
            sp[f'{split}_{name}_after'] = coverage(sub, (sub.nees_cal <= q1).to_numpy())
            sp[f'{split}_{name}_levels_after'] = levels(sub.nees_cal.to_numpy(), 1)
    for reg in C.REGIMES:
        sub = v[va & (v.regime == reg)]
        sp[f'val_{reg}_after'] = coverage(sub, (sub.nees_cal <= q1).to_numpy())
    # simplest implementation: v_var_pub = max(beta * P_vv, floor) on every epoch (not only at standstill)
    var_any = np.maximum(beta * v.v_var, floor)
    for split, sm in (('train', tr), ('val', va)):
        sub = v[sm]
        sp[f'{split}_all_floor_everywhere'] = coverage(sub, (sub.err ** 2 / var_any[sm] <= q1).to_numpy())
    # sensitivity: a floor from the physical dead band instead (uniform on [0, 0.042] -> var = b^2/3)
    fl_db = (C.STANDSTILL_KMH * C.KMH) ** 2 / 3
    sub = v[va & ~mov]
    sp['val_standstill_deadband_floor'] = {'floor_var': fl_db,
                                           'coverage': float(np.mean(sub.err ** 2 / fl_db <= q1))}
    out['speed'] = sp

    # ------------------------------------------------ position (ref_good bags, RTK epochs, map-matched)
    p = p[p.ref_good & (p.status == 2) & (p['flags'] & (C.F_NO_MAP | C.F_NOT_INIT) == 0)].copy()
    p['nees_along'] = p.along ** 2 / p.var_along
    p['route_ok'] = p.cross.abs() < ROUTE_MISMATCH_M
    tr, va = p.split == 'train', p.split == 'val'
    pos = {'route_mismatch_share': {s: float(1 - p.route_ok[p.split == s].mean()) for s in ('train', 'val')},
           'route_mismatch_bags': {s: int(p[(p.split == s) & ~p.route_ok].bag.nunique()) for s in ('train', 'val')}}
    alpha_s = conformal_quantile(p.nees_along[tr].to_numpy(), 0.95) / q1
    alpha_s_ok = conformal_quantile(p.nees_along[tr & p.route_ok].to_numpy(), 0.95) / q1
    sig_cross = conformal_quantile(p.cross.abs()[tr & p.route_ok].to_numpy(), 0.95) / norm.ppf(0.975)
    pos.update(alpha_along=alpha_s, alpha_along_route_ok=alpha_s_ok, sigma_cross_route_ok=sig_cross,
               sigma_cross_published=float(np.sqrt(p.var_cross.median())))
    for split, sm in (('train', tr), ('val', va)):
        sub = p[sm]
        pos[f'{split}_along_before'] = coverage(sub, (sub.nees_along <= q1).to_numpy())
        pos[f'{split}_along_after'] = coverage(sub, (sub.nees_along / alpha_s <= q1).to_numpy())
        pos[f'{split}_along_levels_after'] = levels((sub.nees_along / alpha_s).to_numpy(), 1)
        pos[f'{split}_2d_before'] = coverage(sub, (sub.nees2 <= q2).to_numpy())
        pos[f'{split}_2d_before_route_ok'] = coverage(sub[sub.route_ok], (sub.nees2[sub.route_ok] <= q2).to_numpy())

    # single scale for the whole published 2x2 covariance (as published: sigma_cross 0.3 m)
    a2_all = conformal_quantile(p.nees2[tr].to_numpy(), 0.95) / q2
    a2_ok = conformal_quantile(p.nees2[tr & p.route_ok].to_numpy(), 0.95) / q2
    pos.update(alpha_2d_all=a2_all, alpha_2d_route_ok=a2_ok)
    for split, sm in (('train', tr), ('val', va)):
        sub = p[sm]
        pos[f'{split}_2d_after_alpha_all'] = coverage(sub, (sub.nees2 / a2_all <= q2).to_numpy())
        pos[f'{split}_2d_after_alpha_ok'] = coverage(sub, (sub.nees2 / a2_ok <= q2).to_numpy())
        pos[f'{split}_2d_after_alpha_ok_route_ok'] = coverage(sub[sub.route_ok],
                                                             (sub.nees2[sub.route_ok] / a2_ok <= q2).to_numpy())

    # decomposed: along * alpha_s, cross sigma from data -> 2-D NEES in the (along, cross) axes
    def nees2_decomp(df, a, sc):
        return df.along ** 2 / (a * df.var_along) + df.cross ** 2 / sc ** 2
    for split, sm in (('train', tr), ('val', va)):
        sub = p[sm]
        n2 = nees2_decomp(sub, alpha_s, sig_cross)
        pos[f'{split}_2d_decomposed'] = coverage(sub, (n2 <= q2).to_numpy())
        pos[f'{split}_2d_decomposed_route_ok'] = coverage(sub[sub.route_ok], (n2[sub.route_ok] <= q2).to_numpy())

    # ------------------------------------------------ 99 % along-track protection level
    ratio = (p.along.abs() / np.sqrt(p.var_along)).to_numpy()
    K = conformal_quantile(ratio[tr.to_numpy()], 0.99)
    pl = K * np.sqrt(p.var_along.to_numpy())
    inside = np.abs(p.along.to_numpy()) <= pl
    prot = {'K99': K, 'K99_gaussian_equiv': float(norm.ppf(0.995) * np.sqrt(alpha_s)),
            'K95_conformal': conformal_quantile(ratio[tr.to_numpy()], 0.95)}
    for split, sm in (('train', tr.to_numpy()), ('val', va.to_numpy())):
        sub = p[sm]
        prot[f'{split}_coverage'] = coverage(sub, inside[sm])
        prot[f'{split}_PL_m'] = {'median': float(np.median(pl[sm])), 'p90': float(np.percentile(pl[sm], 90)),
                                 'p99': float(np.percentile(pl[sm], 99))}
        prot[f'{split}_abs_along_m'] = {'p95': float(np.percentile(np.abs(p.along[sm]), 95)),
                                        'p99': float(np.percentile(np.abs(p.along[sm]), 99))}
        for reg in C.REGIMES:
            rm = sm & (p.regime.to_numpy() == reg)
            prot[f'{split}_{reg}_coverage'] = float(np.mean(inside[rm])) if rm.any() else None
        # worst bags (misleading-information share)
        mi = pd.Series(~inside[sm]).groupby(sub.bag.to_numpy()).mean().sort_values(ascending=False)
        prot[f'{split}_worst_bags_MI_share'] = {k: float(x) for k, x in mi.head(4).items()}
    out['position'] = pos
    out['protection_level'] = prot
    C.write_json(out, C.RESULTS / 'calibration.json')

    import json
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
