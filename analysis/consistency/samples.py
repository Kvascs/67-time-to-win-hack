"""Per-bogie-sample table reconstructed from the raw bag + the frozen replay outputs.

For every wheel header stamp t_i (front and rear share stamps in >99 % of messages):
  * measured speed z (km/h * wheel_kmh_to_ms) and the estimator's curve correction c(kappa) at each
    bogie (map curvature at s + 9.9 m / s + 2.35 m, s from projecting the published point on the map);
  * PRIOR estimate at t_i: interpolated between the two published outputs that bracket t_i and were
    computed before sample i arrived (recv_ns < arrival of sample i) but after sample i-1 arrived.
    Those outputs contain the update at t_{i-1} and the model prediction, not sample i -> the
    innovation nu = z*c - (1+k) v_prior. If no bracketing output exists (about a third of the samples,
    see results/whiteness_summary.csv), v is extrapolated with the published acceleration over <= 60 ms;
  * POSTERIOR estimate at t_i: first output computed after both bogie messages of t_i arrived,
    moved back to t_i with its published acceleration (<= 100 ms);
  * the joint monitor's model acceleration at the prior (g*a_drive + clamp(d, +-0.6) = accel - d +
    clamp(d)), the notch in force (raw controller topic), a_target, flags and mode probabilities;
  * map grade averaged over the car body (the dynamics' a_ext = -kg*grade, NOT in the monitor);
  * GNSS reference speed interpolated at t_i and its regime label;
  * known anomaly episodes (analysis/wheel_anomalies/episodes_all.csv) as an exclusion mask.

    python analysis/consistency/samples.py      # builds cache/samples/<bag>.parquet for train+val
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.CACHE / 'samples'
EPISODES = C.ROOT / 'analysis' / 'wheel_anomalies' / 'episodes_all.csv'
BAD_KINDS = {'slip', 'slide', 'lock', 'stuck_zero', 'overspeed/slip', 'frozen', 'negative', 'gap'}
EPISODE_MARGIN_S = 5.0


def _topic(d, key):
    a = d[key]
    a = a[np.argsort(a[:, 1], kind='stable')]
    return a


def wheel_table(d) -> pd.DataFrame:
    fr = _topic(d, 'vehicle__front_bogie_velocity')
    rr = _topic(d, 'vehicle__rear_bogie_velocity')
    f = pd.DataFrame({'ts': np.round(fr[:, 1] * 1e9).astype(np.int64), 'recv_f': np.round(fr[:, 0] * 1e9).astype(np.int64),
                      'kmh_f': fr[:, 2]}).drop_duplicates('ts', keep='last')
    r = pd.DataFrame({'ts': np.round(rr[:, 1] * 1e9).astype(np.int64), 'recv_r': np.round(rr[:, 0] * 1e9).astype(np.int64),
                      'kmh_r': rr[:, 2]}).drop_duplicates('ts', keep='last')
    w = f.merge(r, on='ts', how='outer').sort_values('ts').reset_index(drop=True)
    w['has_f'] = w.kmh_f.notna() & np.isfinite(w.kmh_f) & (w.kmh_f >= -0.5) & (w.kmh_f <= 110)
    w['has_r'] = w.kmh_r.notna() & np.isfinite(w.kmh_r) & (w.kmh_r >= -0.5) & (w.kmh_r <= 110)
    w['z_f'] = np.where(w.has_f, np.maximum(w.kmh_f.fillna(0), 0) * C.KMH, np.nan)
    w['z_r'] = np.where(w.has_r, np.maximum(w.kmh_r.fillna(0), 0) * C.KMH, np.nan)
    big = np.iinfo(np.int64).max
    w['recv_min'] = np.minimum(w.recv_f.fillna(big).astype(np.int64), w.recv_r.fillna(big).astype(np.int64))
    w['recv_max'] = np.maximum(w.recv_f.fillna(-1).astype(np.int64), w.recv_r.fillna(-1).astype(np.int64))
    return w


def episode_mask(bag: str, d, t_bag: np.ndarray) -> np.ndarray:
    """True where a known anomaly episode (bag time, +- margin) covers the sample arrival time."""
    ep = pd.read_csv(EPISODES)
    ep = ep[(ep.bag == bag) & ep.kind.isin(BAD_KINDS)]
    if ep.empty:
        return np.zeros(len(t_bag), bool)
    t0 = min(d[k][0, 0] for k in ('vehicle__front_bogie_velocity', 'vehicle__rear_bogie_velocity',
                                  'vehicle__driver_position_cmd') if len(d[k]))
    m = np.zeros(len(t_bag), bool)
    for r in ep.itertuples():
        a = t0 + r.t_start - EPISODE_MARGIN_S
        b = t0 + r.t_start + (0 if not np.isfinite(r.dur) else r.dur) + EPISODE_MARGIN_S
        m |= (t_bag >= a) & (t_bag <= b)
    return m


def build(bag: str, split: str) -> pd.DataFrame:
    _, o = C.load_replay(bag)
    d = C.load_npz(bag)
    w = wheel_table(d)
    ost = o.stamp_ns.to_numpy()
    orc = o.recv_ns.to_numpy()
    ov, oacc, ok_ = o.v.to_numpy(), o.accel.to_numpy(), o.k.to_numpy()
    od = o.d.to_numpy()
    amon = oacc - od + np.clip(od, -C.D_MAX, C.D_MAX)   # monitor reference g*a_drive + clamp(d)
    t = w.ts.to_numpy()

    # ---- prior: outputs computed before sample i arrived, bracketing t_i ----
    j_end = np.searchsorted(orc, w.recv_min.to_numpy(), side='left') - 1      # last output with recv < arrival
    k1 = np.minimum(np.searchsorted(ost, t, side='right') - 1, j_end)       # last stamp <= t within prefix
    valid1 = (k1 >= 0) & (j_end >= 0)
    k1c = np.maximum(k1, 0)
    k2 = k1c + 1
    has2 = valid1 & (k2 <= j_end) & (k2 < len(ost))
    k2c = np.minimum(k2, len(ost) - 1)
    s1, s2 = ost[k1c], ost[k2c]
    wgt = np.where(has2, (t - s1) / np.maximum(s2 - s1, 1), 0.0)
    v_pr = np.where(has2, ov[k1c] + wgt * (ov[k2c] - ov[k1c]), ov[k1c] + oacc[k1c] * (t - s1) * 1e-9)
    am_pr = np.where(has2, amon[k1c] + wgt * (amon[k2c] - amon[k1c]), amon[k1c])
    prev_recv = np.concatenate([[-1], w.recv_max.to_numpy()[:-1]])
    prev_close = np.concatenate([[False], np.diff(t) < 0.25e9])
    prior_ok = valid1 & ((t - s1) <= 0.06e9) & (~prev_close | (orc[k1c] >= prev_recv))

    # ---- posterior: first output computed after both messages of t_i arrived ----
    jp = np.searchsorted(orc, w.recv_max.to_numpy(), side='left')           # first output with recv >= arrival
    jp = np.minimum(jp, len(ost) - 1)
    # the first such output may carry an older stamp? stamps increase with recv, so take the first >= t
    jp2 = np.maximum(jp, np.searchsorted(ost, t, side='left'))
    jp2 = np.minimum(jp2, len(ost) - 1)
    dtp = (ost[jp2] - t) * 1e-9
    v_po = ov[jp2] - oacc[jp2] * dtp
    post_ok = (dtp >= 0) & (dtp <= 0.1)

    s = pd.DataFrame({'ts': t, 't': t * 1e-9, 'recv_f': w.recv_f, 'recv_r': w.recv_r, 'has_f': w.has_f,
                      'has_r': w.has_r, 'z_f': w.z_f, 'z_r': w.z_r, 'v_pr': v_pr, 'k_pr': ok_[k1c],
                      'amon_pr': am_pr, 'acc_pr': oacc[k1c], 'd_pr': od[k1c], 'g_pr': o.g.to_numpy()[k1c],
                      'atgt_pr': o.a_model.to_numpy()[k1c], 'flags_pr': o['flags'].to_numpy()[k1c].astype(np.int64),
                      'mu0_pr': o.mu0.to_numpy()[k1c], 'mu3_pr': o.mu3.to_numpy()[k1c], 'vvar_pr': o.v_var.to_numpy()[k1c],
                      'prior_ok': prior_ok, 'prior_interp': has2, 'v_po': v_po, 'post_ok': post_ok,
                      'flags_po': o['flags'].to_numpy()[jp2].astype(np.int64), 'mu3_po': o.mu3.to_numpy()[jp2]})

    # ---- map position of the prior output -> curvature at each bogie, grade over the body ----
    tm = C.TrackMap(C.fix_origin(d))
    valid_pos = (o.pos_valid.to_numpy() == 1) & ((o['flags'].to_numpy() & (C.F_NO_MAP | C.F_NOT_INIT)) == 0)
    s_map = np.full(len(o), np.nan)
    if valid_pos.any():
        s_map[valid_pos] = tm.project(o.x.to_numpy()[valid_pos], o.y.to_numpy()[valid_pos], o.yaw.to_numpy()[valid_pos])
    s_ant = s_map[k1c] - ov[k1c] * C.POSITION_LEAD + (v_pr * ((t - s1) * 1e-9))   # antenna arc length at t_i
    s['s_map'] = s_ant
    s['curv_f'] = tm.curvature(s_ant + C.FRONT_ALONG)
    s['curv_r'] = tm.curvature(s_ant + C.REAR_ALONG)
    offs = C.BODY_REAR + (C.BODY_FRONT - C.BODY_REAR) * np.arange(5) / 4
    s['grade_body'] = np.mean([tm.grade_at(s_ant + q) for q in offs], axis=0)
    nan_s = ~np.isfinite(s_ant)
    for c in ('curv_f', 'curv_r', 'grade_body'):
        s.loc[nan_s, c] = np.nan
    s['c_f'] = np.where(np.isfinite(s.curv_f), C.curve_factor(s.curv_f.fillna(0)), 1.0)
    s['c_r'] = np.where(np.isfinite(s.curv_r), C.curve_factor(s.curv_r.fillna(0)), 1.0)
    s['zc_f'] = s.z_f * s.c_f
    s['zc_r'] = s.z_r * s.c_r

    # ---- notch in force (by stamp) ----
    cm = _topic(d, 'vehicle__driver_position_cmd')
    ci = np.searchsorted(np.round(cm[:, 1] * 1e9).astype(np.int64), t, side='right') - 1
    s['notch'] = np.where(ci >= 0, cm[np.maximum(ci, 0), 2], 0).astype(int)
    s['cmd_age'] = np.where(ci >= 0, (t - np.round(cm[np.maximum(ci, 0), 1] * 1e9)) * 1e-9, np.inf)

    # ---- reference speed at t_i ----
    ref = C.reference_speed(d)
    s['v_ref'] = np.interp(s.t, ref.t, ref.v_ref, left=np.nan, right=np.nan)
    s['v_ref_s'] = np.interp(s.t, ref.t, ref.v_ref_s, left=np.nan, right=np.nan)
    s['a_ref'] = np.interp(s.t, ref.t, ref.a_ref, left=np.nan, right=np.nan)
    s['regime'] = C.label_at(s.t.to_numpy(), ref, 'regime', tol=0.15)
    t_bag = np.where(w.recv_f.notna(), w.recv_f, w.recv_r) * 1e-9
    s['episode'] = episode_mask(bag, d, t_bag)
    s['bag'] = bag
    s['split'] = split
    return s


def _one(args):
    bag, split, force = args
    out = OUT / f'{bag}.parquet'
    if out.exists() and not force:
        return bag, 'cached'
    try:
        build(bag, split).to_parquet(out, index=False)
        return bag, 'ok'
    except Exception as e:  # report and continue
        return bag, f'error {e!r}'[:300]


def load(splits=('train', 'val')) -> pd.DataFrame:
    sp = C.splits()
    return pd.concat([pd.read_parquet(OUT / f'{b}.parquet') for s in splits for b in sp[s]], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    force = '--force' in sys.argv
    sp = C.splits()
    jobs = [(b, s, force) for s in ('train', 'val') for b in sp[s]]
    with ProcessPoolExecutor(2) as ex:
        for bag, st in ex.map(_one, jobs):
            if st != 'cached':
                print(bag, st, flush=True)


if __name__ == '__main__':
    main()
