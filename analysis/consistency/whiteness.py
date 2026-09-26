"""(c) Whiteness of the wheel residuals, empirical R and reconstructed NIS on clean cruising segments.

    python analysis/consistency/whiteness.py        (needs samples.py first)

Residuals per bogie (see samples.py for the reconstruction from published outputs):
  innovation   nu  = z*c(kappa) - (1+k) v_prior     (prediction before the sample, contains P- + R)
  posterior    e   = z*c(kappa) - (1+k) v_post      ("wheel - estimate" after the update)
  difference   dfr = z_front*c_f - z_rear*c_r       (filter-free: the true speed cancels; ACF of the
                                                     measurement noise if front/rear noise independent)
Clean cruising: both bogies, prior valid, no health flags / standstill / maneuver / controller fault,
P(nominal) > 0.95, outside known anomaly episodes (+-5 s), reference regime 'cruise' (|a_ref| <= 0.2
m/s^2) at > 2 m/s, map position known, GNSS clock-ok bags; contiguous runs of >= 30 samples (3 s).

NIS: S = P_uu^- 11^T + diag(R_f, R_r) from a shadow Kalman filter on u = (1+k) v with the estimator's
own noise model (sigma_accel 0.12, q_d 0.004, R = 0.05^2 + (0.004 v)^2, pinned at standstill), run
over the full bag. It omits the GPB1 mode-mixing spread, so it is a slight under-estimate of S.
Empirical split of the innovation covariance: Cov(nu_f, nu_r) ~ P^- (common prediction error),
Var(nu_i) - Cov(nu_f, nu_r) ~ R_i (independent bogie noise).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import samples as S  # noqa: E402

MAX_LAG = 10
MIN_SEG = 30
BAD_FLAGS = C.F_ANY_WHEEL_PROBLEM | C.F_STANDSTILL | C.F_UNMODELED | C.F_CMD_INCONS | C.F_LATE


def shadow_S(s: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prior variance of u and the model R per bogie at every sample (shadow KF, see docstring)."""
    n = len(s)
    Pu = np.full(n, np.nan)
    t = s.t.to_numpy()
    v = np.maximum(s.v_pr.to_numpy(), 0)
    Rm = C.SIGMA_WHEEL ** 2 + (C.SIGMA_WHEEL_REL * v) ** 2
    still = (s.flags_pr.to_numpy() & C.F_STANDSTILL) > 0
    hf, hr = s.has_f.to_numpy(), s.has_r.to_numpy()
    P = np.array([[0.09, 0.0], [0.0, 0.01]])
    was_still = False
    for i in range(n):
        h = 0.0 if i == 0 else min(max(t[i] - t[i - 1], 0.0), 1.0)
        if still[i]:
            P = np.array([[1e-6, 0.0], [0.0, P[1, 1] + C.Q_DIST * h]])
            Pu[i] = P[0, 0]
            was_still = True
            continue
        if was_still:
            P = np.array([[0.09, 0.0], [0.0, P[1, 1]]])
            was_still = False
        F = np.array([[1.0, h], [0.0, 1.0]])
        P = F @ P @ F.T + np.diag([C.SIGMA_ACCEL ** 2 * h, C.Q_DIST * h])
        Pu[i] = P[0, 0]
        m = int(hf[i]) + int(hr[i])
        if m == 0:
            continue
        H = np.zeros((m, 2))
        H[:, 0] = 1.0
        Sm = H @ P @ H.T + np.eye(m) * Rm[i]
        K = P @ H.T @ np.linalg.inv(Sm)
        IKH = np.eye(2) - K @ H
        P = IKH @ P @ IKH.T + K @ (np.eye(m) * Rm[i]) @ K.T
    return Pu, Rm, still


def clean_mask(s: pd.DataFrame, clock_ok: bool) -> np.ndarray:
    fl = s.flags_pr.to_numpy() | s.flags_po.to_numpy()
    m = (s.has_f & s.has_r & s.prior_ok & s.post_ok).to_numpy().copy()
    m &= (fl & BAD_FLAGS) == 0
    m &= s.mu0_pr.to_numpy() > 0.95
    m &= ~s.episode.to_numpy()
    m &= (s.regime.to_numpy() == 'cruise') & (s.v_ref_s.to_numpy() > 2.0)
    m &= np.isfinite(s.s_map.to_numpy())
    return m & clock_ok


def segments(s: pd.DataFrame, mask: np.ndarray) -> list[np.ndarray]:
    t = s.t.to_numpy()
    out = []
    for a, b in C.runs(mask):
        # split further at sampling gaps > 0.15 s
        cut = np.flatnonzero(np.diff(t[a:b]) > 0.15) + 1
        for seg in np.split(np.arange(a, b), cut):
            if len(seg) >= MIN_SEG:
                out.append(seg)
    return out


def main():
    metrics = pd.read_csv(C.RESULTS / 'replay_metrics.csv').set_index('bag')
    sp = C.splits()
    rows, acf_rows, seg_store = [], [], {}
    for split in ('train', 'val'):
        series = {k: [] for k in ('nu_f', 'nu_r', 'e_f', 'e_r', 'dfr', 'z_f_ref', 'z_r_ref')}
        pooled = []
        n_all = n_interp = n_prior_ok = 0
        lags = []
        for bag in sp[split]:
            s = pd.read_parquet(S.OUT / f'{bag}.parquet')
            n_all += len(s)
            n_interp += int(s.prior_interp.sum())
            n_prior_ok += int(s.prior_ok.sum())
            lags.append(((s.recv_f - s.ts) * 1e-6).dropna().to_numpy())
            Pu, Rm, _ = shadow_S(s)
            s['Pu'], s['Rm'] = Pu, Rm
            onek = 1.0 + s.k_pr
            s['nu_f'] = s.zc_f - onek * s.v_pr
            s['nu_r'] = s.zc_r - onek * s.v_pr
            s['e_f'] = s.zc_f - onek * s.v_po
            s['e_r'] = s.zc_r - onek * s.v_po
            s['dfr'] = s.zc_f - s.zc_r
            s['z_f_ref'] = s.zc_f / onek - s.v_ref          # wheel vs GNSS Doppler (reference noise inside)
            s['z_r_ref'] = s.zc_r / onek - s.v_ref
            m = clean_mask(s, not (metrics.loc[bag, 'ref_clock_anom'] > 0.01))
            segs = segments(s, m)
            for seg in segs:
                for k in series:
                    series[k].append(s[k].to_numpy()[seg])
                pooled.append(s.iloc[seg].assign(seg_id=f'{bag}_{seg[0]}'))
        P = pd.concat(pooled, ignore_index=True)
        seg_store[split] = P
        n_tot = sum(len(x) for x in series['nu_f'])
        band = 1.96 / np.sqrt(n_tot)
        for k, segs_k in series.items():
            r, n = C.pooled_acf(segs_k, MAX_LAG)
            tau = 1 + 2 * r[1:].sum()
            acf_rows.append({'split': split, 'series': k, 'n': n, 'segments': len(segs_k), 'white_band': band,
                             **{f'lag{j}': r[j] for j in range(1, MAX_LAG + 1)}, 'tau_int_1_10': tau})
        # ---- innovation covariance split, NIS ----
        nu = P[['nu_f', 'nu_r']].to_numpy()
        nu_c = nu - nu.mean(0)
        cov = nu_c.T @ nu_c / len(nu_c)
        Pu_emp = cov[0, 1]
        R_emp = np.diag(cov) - cov[0, 1]
        Sdet = (P.Pu + P.Rm) ** 2 - P.Pu ** 2
        nis2 = ((P.Pu + P.Rm) * (P.nu_f ** 2 + P.nu_r ** 2) - 2 * P.Pu * P.nu_f * P.nu_r) / Sdet
        nis_f = P.nu_f ** 2 / (P.Pu + P.Rm)
        nis_r = P.nu_r ** 2 / (P.Pu + P.Rm)
        lag_all = np.concatenate(lags)
        ip = P.prior_interp.to_numpy()
        row = {'split': split, 'clean_bracketed_share': float(ip.mean()),
               'nu_std_bracketed': float(np.std(np.r_[P.nu_f[ip], P.nu_r[ip]])),
               'nu_std_extrapolated': float(np.std(np.r_[P.nu_f[~ip], P.nu_r[~ip]])),
               'wheel_stamps_all': n_all, 'prior_bracketed_share': n_interp / n_all,
               'prior_ok_share': n_prior_ok / n_all, 'front_arrival_lag_median_ms': float(np.median(lag_all)),
               'front_arrival_lag_p95_ms': float(np.percentile(lag_all, 95)),
               'bags': P.seg_id.str[:14].nunique(), 'samples': len(P),
               'hours': len(P) * 0.1 / 3600, 'segments': P.seg_id.nunique(),
               'v_mean': P.v_pr.mean(), 'nu_mean_f': P.nu_f.mean(), 'nu_mean_r': P.nu_r.mean(),
               'nu_std_f': P.nu_f.std(), 'nu_std_r': P.nu_r.std(), 'e_std_f': P.e_f.std(),
               'dfr_std': P.dfr.std(), 'Pu_prior_emp': Pu_emp, 'Pu_prior_model_mean': P.Pu.mean(),
               'R_emp_f': R_emp[0], 'R_emp_r': R_emp[1], 'R_model_mean': P.Rm.mean(),
               'sigma_R_emp_f': np.sqrt(max(R_emp[0], 0)), 'sigma_R_emp_r': np.sqrt(max(R_emp[1], 0)),
               'sigma_dfr_over_sqrt2': P.dfr.std() / np.sqrt(2),
               'NIS2_mean': nis2.mean(), 'NIS2_median': nis2.median(),
               'NIS2_inside95': float(np.mean(nis2 <= chi2.ppf(0.95, 2))),
               'NIS1_mean_f': nis_f.mean(), 'NIS1_mean_r': nis_r.mean(),
               'NIS1_inside95_f': float(np.mean(nis_f <= chi2.ppf(0.95, 1)))}
        # speed dependence of the independent bogie noise: Var(dfr)/2 per speed bin
        for lo, hi in ((2, 6), (6, 10), (10, 20)):
            b = P[(P.v_pr >= lo) & (P.v_pr < hi)]
            if len(b) > 200:
                row[f'sigma_indep_{lo}_{hi}ms'] = b.dfr.std() / np.sqrt(2)
                bn = b[['nu_f', 'nu_r']].to_numpy()
                bn = bn - bn.mean(0)
                cb = bn.T @ bn / len(bn)
                row[f'sigma_R_emp_{lo}_{hi}ms'] = float(np.sqrt(max(np.mean(np.diag(cb)) - cb[0, 1], 0)))
                row[f'n_{lo}_{hi}ms'] = len(b)
        rows.append(row)
    acf_df = pd.DataFrame(acf_rows)
    summ = pd.DataFrame(rows)
    acf_df.to_csv(C.RESULTS / 'whiteness_acf.csv', index=False)
    summ.to_csv(C.RESULTS / 'whiteness_summary.csv', index=False)
    # the difference-series tau is the inflation factor of R for colored measurement noise
    rec = {}
    for split in ('train', 'val'):
        a = acf_df[(acf_df.split == split)].set_index('series')
        su = summ[summ.split == split].iloc[0]
        tau_meas = float(a.loc['dfr', 'tau_int_1_10'])
        rec[split] = {'tau_int_measurement_noise': tau_meas,
                      'tau_int_innov_f': float(a.loc['nu_f', 'tau_int_1_10']),
                      'tau_int_innov_r': float(a.loc['nu_r', 'tau_int_1_10']),
                      'R_white_equiv': float(np.mean([su.R_emp_f, su.R_emp_r])),
                      'R_effective': float(np.mean([su.R_emp_f, su.R_emp_r]) * max(tau_meas, 1.0)),
                      'sigma_R_effective': float(np.sqrt(np.mean([su.R_emp_f, su.R_emp_r]) * max(tau_meas, 1.0))),
                      'R_model_at_mean_speed': float(su.R_model_mean),
                      'ratio_R_model_to_R_white': float(su.R_model_mean / np.mean([su.R_emp_f, su.R_emp_r])),
                      'ratio_R_model_to_R_effective': float(su.R_model_mean / (np.mean([su.R_emp_f, su.R_emp_r]) *
                                                                              max(tau_meas, 1.0))),
                      'ratio_Pprior_model_to_emp': float(su.Pu_prior_model_mean / su.Pu_prior_emp),
                      'NIS2_conservatism_2_over_mean': float(2.0 / su.NIS2_mean)}
    C.write_json(rec, C.RESULTS / 'whiteness_R.json')
    pd.set_option('display.width', 250)
    print(acf_df.round(3).to_string(index=False))
    print(summ.T.to_string())
    import json
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
