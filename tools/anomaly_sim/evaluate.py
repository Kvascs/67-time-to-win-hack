"""Robustness evaluation harness: replay a (corrupted) run causally through an estimator and score it.

* Replay: vehicle messages are fed in *arrival* (bag-time) order, exactly as ``ros2 bag play`` would;
  GNSS is not fed (the jury keeps only a few seconds; the reference estimators do not use it).
* Estimator interface: any object with ``reset()`` and
  ``on_message(key, t_bag, t_hdr, value) -> (stamp, v) | None`` (publish a speed estimate).
  Plug the team's Python prototype in via :func:`evaluate_run` to regression-test it on the suite.
* Reference: robust GNSS speed of the *clean* run (master & rover agree, GNSS time = bag time minus
  median latency, immune to the +-1 s GNSS header-stamp segments present in ~13 bags).
* Metrics:
    - ``rmse_rt``     speed error at the real time of publication (t_bag - median wheel latency);
    - ``rmse_stamp``  judge-like: error at the published header.stamp, ``match_rate`` = share of
                      outputs whose stamp is within 0.5 s of the real measurement time;
    - ``max_err``, ``p99_err``, ``bias``; ``rmse_anom`` inside anomaly windows (+2 s), ``rmse_clean``
      outside; ``recovery_s`` median time after an event until |err| < 0.3 m/s for 1 s;
    - ``drift_pct``   (distance from integrating the output - reference distance) / reference distance;
    - ``bad_out``     number of NaN / inf / negative / >40 m/s outputs.
  The first 5 s (the jury's GNSS initialisation window) are excluded from all metrics.
"""
from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .characterize import reference_speed
from .constants import CMD, FRONT, KMH_PER_MS, REAR, VEHICLE
from .physics import BRAKE_ACC, TRACTION_ACC
from .run import Run, load_pair

VALUE_EVENT_TYPES = ('slip', 'slide', 'dropout', 'outliers', 'frozen', 'notch_fault', 'zero_stamps',
                     'stamp_glitch', 'clock_offset')


# ------------------------------------------------------------------------------------ estimators

class NaiveEstimator:
    """Mean of the latest front/rear readings / 3.6, stamps copied from the input. No validation."""
    name = 'naive'

    def reset(self):
        self.last = {FRONT: 0.0, REAR: 0.0}
        self.v = 0.0

    def on_message(self, key, t_bag, t_hdr, value):
        if key in self.last:
            self.last[key] = value
            self.v = 0.5 * (self.last[FRONT] + self.last[REAR]) / KMH_PER_MS
        return t_hdr, self.v


def model_acc(notch: float, v: float) -> float:
    n = int(np.clip(round(notch), -15, 15))
    if n > 0:
        return float(TRACTION_ACC[n]) - 0.03
    if n < 0:
        return -float(BRAKE_ACC[-n]) if v > 0.05 else 0.0
    return -0.03 if v > 0.05 else 0.0


class KFBaseline:
    """Robust reference estimator - a *yardstick* for the scenarios, not the final solution.

    2-state Kalman filter x = [v, b] (speed, acceleration bias of the notch model):
      * prediction  v' = a_model(notch(t - 0.6 s)) + b, b' = 0 (random walk); model from the measured
        notch->acceleration table (TRACTION_ACC / BRAKE_ACC); Q_v = (0.3 m/s^2)^2 dt, Q_b = (0.05)^2 dt;
      * input sanitation: finite, -2..90 km/h; notch integer in [-15, 15]; header stamp kept if not in
        the future and <= 3 s late, else arrival time minus running latency; duplicates / out-of-order
        samples ignored; samples > 0.3 s older than the state are not fused;
      * per-sensor plausibility: a wheel whose own acceleration leaves [-3, +2] m/s^2 while it deviates
        from the prediction is *suspect* (slip / slide / lock / spike) until it is smooth and back
        inside the gate, or both sensors agree and are smooth for 1 s after >= 3 s without updates
        (re-sync);
      * fusion of fresh non-suspect sensors: traction -> min, braking -> max, else mean; chi^2 gate
        |r| <= max(3 sigma, 0.4 m/s) with sigma^2 = P_vv + R; the gate opens by itself while coasting.
    """
    name = 'kf_baseline'
    A_UP, A_DOWN = 2.0, 3.0
    Q_A, Q_B, R0, LAG = 0.3 ** 2, 0.05 ** 2, 0.05 ** 2, 0.6

    def reset(self):
        self.x = np.zeros(2)
        self.P = np.diag([1.0, 0.1])
        self.t = None
        self.lat = None
        self.notch_hist: list[tuple[float, float]] = [(-1e18, 0.0)]
        self.prev = {FRONT: None, REAR: None}
        self.suspect = {FRONT: False, REAR: False}
        self.smooth_since = {FRONT: None, REAR: None}
        self.fresh = {FRONT: (None, -1e18), REAR: (None, -1e18)}
        self.t_accept = None
        self.last_stamp = {k: -1e18 for k in VEHICLE}

    # --------------------------------------------------------------------- helpers
    def _meas_time(self, t_bag, t_hdr):
        """Header stamp if plausible (not in the future, <= 3 s late), else arrival time - latency."""
        ok = math.isfinite(t_hdr) and t_hdr > 1e6
        lat = t_bag - t_hdr if ok else None
        if self.lat is None and ok and 0 <= lat < 0.5:
            self.lat = lat
        L = self.lat if self.lat is not None else 0.05
        if ok and -0.05 <= lat - L <= 3.0:
            if abs(lat - L) < 0.1:
                self.lat = L + 0.01 * (lat - L)   # follow slow clock drift using normal samples only
            return t_hdr
        return t_bag - L

    def _notch_at(self, t):
        h = self.notch_hist
        while len(h) > 1 and h[1][0] <= t:
            h.pop(0)
        return h[0][1]

    def _predict(self, tm):
        dt = min(max(tm - self.t, 0.0), 1.0)
        if dt <= 0:
            return
        v, b = self.x
        moving = v > 0.05
        a = model_acc(self._notch_at(tm - self.LAG), v) + (b if moving else 0.0)
        self.x[0] = max(0.0, v + a * dt)
        F = np.array([[1.0, dt if moving else 0.0], [0.0, 1.0]])
        self.P = F @ self.P @ F.T + np.diag([self.Q_A * dt, self.Q_B * dt])
        self.P = 0.5 * (self.P + self.P.T)
        self.t = tm

    def _gate(self, R):
        return max(3.0 * math.sqrt(max(self.P[0, 0], 0.0) + R), 0.4)

    def _update(self, z, R):
        S = self.P[0, 0] + R
        K = self.P[:, 0] / S
        r = z - self.x[0]
        self.x = self.x + K * r
        IKH = np.eye(2) - np.outer(K, [1.0, 0.0])
        self.P = IKH @ self.P @ IKH.T + np.outer(K, K) * R   # Joseph form keeps P positive definite
        self.P = 0.5 * (self.P + self.P.T)
        self.x[0] = max(self.x[0], 0.0)
        self.x[1] = float(np.clip(self.x[1], -1.5, 1.5))

    # --------------------------------------------------------------------- main entry
    def on_message(self, key, t_bag, t_hdr, value):
        tm = self._meas_time(t_bag, t_hdr)
        if tm <= self.last_stamp[key] + 1e-4:          # duplicate / out-of-order: ignore
            return None
        self.last_stamp[key] = tm
        if self.t is None:
            self.t = tm
        if tm > self.t:
            self._predict(tm)
        if key == CMD:
            if math.isfinite(value) and -15 <= value <= 15 and value == round(value):
                self.notch_hist.append((tm, value))
            return self.t, float(self.x[0])
        if not (math.isfinite(value) and -2.0 <= value <= 90.0) or tm < self.t - 0.3:
            return self.t, float(self.x[0])            # invalid or too stale to fuse: coast
        z_i = max(value, 0.0) / KMH_PER_MS
        v = float(self.x[0])
        R = self.R0 + (0.01 * v) ** 2
        gate = self._gate(R)
        prev = self.prev[key]
        smooth = True
        if prev is not None and 1e-3 < tm - prev[1] < 1.0:
            acc = (z_i - prev[0]) / (tm - prev[1])
            # Known limitation (kept on purpose, it is what the suite measures): real emergency stops reach
            # -5 m/s^2 at notch 0..3 (30618_616ec56b t=1018 s, 30618_0e41eac3 t=571 s) and are rejected here.
            # A notch-dependent bound (-5.5 when notch >= -2) fixed only part of that and doubled the S05
            # lock error: lock vs emergency stop is only resolved by what follows (spin-up vs staying at 0).
            smooth = -self.A_DOWN <= acc <= self.A_UP
            if not smooth and abs(z_i - v) > 0.3:
                self.suspect[key] = True
        self.prev[key] = (z_i, tm)
        self.smooth_since[key] = (self.smooth_since[key] or tm) if smooth else None
        self.fresh[key] = (z_i, tm)
        if self.suspect[key] and smooth and abs(z_i - v) <= gate:
            self.suspect[key] = False
        since = 0.0 if self.t_accept is None else tm - self.t_accept
        vals = [z for k, (z, tz) in self.fresh.items() if z is not None and tm - tz < 0.3 and not self.suspect[k]]
        notch = self._notch_at(tm)
        if vals:
            z = min(vals) if notch > 0 else (max(vals) if notch < 0 else float(np.mean(vals)))
            if abs(z - v) <= gate:
                self._update(z, R)
                self.t_accept = tm
                return self.t, float(self.x[0])
        # re-sync: nothing accepted for 3 s, both sensors fresh, agreeing and smooth for 1 s
        (zf, tf), (zr, tr) = self.fresh[FRONT], self.fresh[REAR]
        if (since > 3.0 and zf is not None and zr is not None and tm - tf < 0.3 and tm - tr < 0.3
                and abs(zf - zr) < 0.3 + 0.05 * max(zf, zr)
                and all(self.smooth_since[k] is not None and tm - self.smooth_since[k] >= 1.0 for k in (FRONT, REAR))):
            self.x[0] = 0.5 * (zf + zr)
            self.P = np.diag([4 * R, self.P[1, 1]])
            self.suspect = {FRONT: False, REAR: False}
            self.t_accept = tm
        return self.t, float(self.x[0])


ESTIMATORS = {'naive': NaiveEstimator, 'kf_baseline': KFBaseline}


# ------------------------------------------------------------------------------------ replay & scoring

def replay(run: Run, est) -> dict:
    keys, tb, th, val = [], [], [], []
    for k in VEHICLE:
        s = run.streams[k]
        keys += [k] * len(s)
        tb.append(s.t_bag)
        th.append(s.t_hdr)
        val.append(s.val[:, 0])
    tb, th, val = np.concatenate(tb), np.concatenate(th), np.concatenate(val)
    order = np.argsort(tb, kind='stable')
    est.reset()
    out_t, out_s, out_v = [], [], []
    for i in order:
        r = est.on_message(keys[i], float(tb[i]), float(th[i]), float(val[i]))
        if r is not None:
            out_t.append(tb[i])
            out_s.append(r[0])
            out_v.append(r[1])
    return {'t_pub': np.array(out_t), 'stamp': np.array(out_s, float), 'v': np.array(out_v, float)}


def _windows(events: list[dict], pad: float = 2.0) -> list[tuple[float, float]]:
    return [(e['t0'], e['t1'] + pad) for e in events if e['type'] in VALUE_EVENT_TYPES]


def _in_windows(t: np.ndarray, wins) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for a, b in wins:
        m |= (t >= a) & (t <= b)
    return m


def score(out: dict, clean: Run, events: list[dict], lat_w: float, skip_s: float = 5.0) -> dict:
    """Score published estimates; the first ``skip_s`` seconds (GNSS initialisation window) are ignored."""
    keep = out['t_pub'] >= clean.t_start + skip_s
    out = {k: v[keep] for k, v in out.items()}
    t_rt = out['t_pub'] - lat_w
    grid = np.arange(t_rt.min() - 1, t_rt.max() + 1, 0.05)
    _, vref = reference_speed(clean, grid)
    if vref is None:
        return {}
    v = out['v']
    bad = ~np.isfinite(v) | (v < -0.5) | (v > 40)
    vv = np.where(np.isfinite(v), v, np.nan)
    ref_rt = np.interp(t_rt, grid, vref)
    ok = np.isfinite(ref_rt)
    e_rt = vv - ref_rt
    stamp = out['stamp']
    matched = np.isfinite(stamp) & (np.abs(stamp - t_rt) < 0.5)
    ref_st = np.interp(np.where(matched, stamp, t_rt), grid, vref)
    e_st = np.where(matched, vv - ref_st, np.nan)
    wins = _windows(events)
    in_w = _in_windows(t_rt, wins)
    e_ok = np.where(ok, e_rt, np.nan)
    ae = np.abs(np.nan_to_num(e_ok, nan=0.0, posinf=1e3, neginf=1e3))
    # distance: integrate the published speed piecewise-constant over publication time
    dt = np.diff(out['t_pub'], append=out['t_pub'][-1])
    d_out = float(np.nansum(np.clip(np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0), 0, 40) * dt))
    vref_f = np.interp(grid, grid[np.isfinite(vref)], vref[np.isfinite(vref)])
    m = (grid >= t_rt[0]) & (grid <= t_rt[-1])
    d_ref = float(np.sum(vref_f[m]) * 0.05)
    rec = []
    for e in events:
        if e['type'] not in VALUE_EVENT_TYPES:
            continue
        after = (t_rt > e['t1']) & ok
        idx = np.flatnonzero(after)
        if len(idx) == 0:
            continue
        good = np.abs(np.nan_to_num(e_rt[idx], nan=9.0)) < 0.3
        # first time after which the error stays below 0.3 m/s for 1 s
        tt = t_rt[idx]
        run_start = None
        for j in range(len(idx)):
            if good[j]:
                if run_start is None:
                    run_start = tt[j]
                if tt[j] - run_start >= 1.0:
                    rec.append(run_start - e['t1'])
                    break
            else:
                run_start = None
    def rms(x):
        x = np.clip(x[~np.isnan(x)], -1e6, 1e6)
        return float(np.sqrt(np.mean(x ** 2))) if len(x) else float('nan')
    return {
        'n_out': int(len(v)), 'rate_hz': float(len(v) / max(out['t_pub'][-1] - out['t_pub'][0], 1e-9)),
        'bad_out': int(bad.sum()),
        'rmse_rt': rms(e_ok), 'mae_rt': float(np.nanmean(np.abs(e_ok))), 'bias_rt': float(np.nanmean(e_ok)),
        'max_err': float(ae.max()), 'p99_err': float(np.percentile(ae[ok], 99)) if ok.any() else float('nan'),
        'frac_err_gt_1': float(np.mean(ae[ok] > 1.0)) if ok.any() else float('nan'),
        'rmse_stamp': rms(e_st), 'match_rate': float(matched.mean()),
        'rmse_anom': rms(e_ok[in_w]) if in_w.any() else float('nan'), 'rmse_clean': rms(e_ok[~in_w]),
        'recovery_s_median': float(np.median(rec)) if rec else float('nan'),
        'recovery_s_max': float(np.max(rec)) if rec else float('nan'),
        'd_ref_m': d_ref, 'drift_pct': float(100.0 * (d_out - d_ref) / max(d_ref, 1.0)),
    }


def evaluate_run(npz_path: str | Path, estimators=('naive', 'kf_baseline')) -> list[dict]:
    bad, clean = load_pair(npz_path)
    lat_w = float(np.median(clean.streams[FRONT].t_bag[60:] - clean.streams[FRONT].t_hdr[60:]))
    rows = []
    for name in estimators:
        est = ESTIMATORS[name]() if isinstance(name, str) else name
        out = replay(bad, est)
        sc = score(out, clean, bad.events, lat_w)
        if sc:
            rows.append({'scenario': bad.meta.get('scenario', ''), 'bag': bad.name,
                         'estimator': getattr(est, 'name', str(name)), **sc})
    return rows


def _eval_one(path):
    try:
        return evaluate_run(path)
    except Exception as e:  # pragma: no cover
        return [{'scenario': Path(path).parent.name, 'bag': Path(path).stem, 'error': repr(e)}]


def evaluate_suite(npz_root: Path, out_dir: Path, scenarios=None, workers: int = 4) -> list[dict]:
    npz_root, out_dir = Path(npz_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(npz_root.glob('*/*.npz'))
    if scenarios:
        files = [f for f in files if any(f.parent.name == s or f.parent.name.split('_')[0] == s for s in scenarios)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for rr in ex.map(_eval_one, files) for r in rr]
    if scenarios and (out_dir / 'eval_runs.csv').exists():  # partial re-evaluation: keep the other scenarios
        done = {r['scenario'] for r in rows}
        with open(out_dir / 'eval_runs.csv', newline='', encoding='utf-8') as fh:
            old = [r for r in csv.DictReader(fh) if r['scenario'] not in done]
        for r in old:
            for k, v in r.items():
                if k not in ('scenario', 'bag', 'estimator', 'error'):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
        rows = old + rows
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('scenario', 'bag', 'estimator'), k))
    with open(out_dir / 'eval_runs.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    agg = aggregate(rows)
    (out_dir / 'eval_summary.json').write_text(json.dumps(agg, indent=1), encoding='utf-8')
    (out_dir / 'eval_summary.md').write_text(to_markdown(agg), encoding='utf-8')
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if 'error' in r:
            continue
        groups.setdefault((r['scenario'], r['estimator']), []).append(r)
    for (sc, est), rr in sorted(groups.items()):
        def med(k):
            x = np.array([r[k] for r in rr], float)
            x = x[np.isfinite(x)]
            return float(np.median(x)) if len(x) else float('nan')
        def mx(k):
            x = np.array([r[k] for r in rr], float)
            x = x[np.isfinite(x)]
            return float(np.max(x)) if len(x) else float('nan')
        out.append({'scenario': sc, 'estimator': est, 'n_bags': len(rr), 'rmse_rt': med('rmse_rt'),
                    'rmse_anom': med('rmse_anom'), 'p99_err': med('p99_err'), 'max_err': mx('max_err'),
                    'frac_err_gt_1': med('frac_err_gt_1'), 'match_rate': med('match_rate'),
                    'abs_drift_pct': float(np.median([abs(r['drift_pct']) for r in rr])),
                    'max_abs_drift_pct': float(np.max([abs(r['drift_pct']) for r in rr])),
                    'recovery_s': med('recovery_s_median'), 'bad_out': int(sum(r['bad_out'] for r in rr))})
    return out


def to_markdown(agg: list[dict]) -> str:
    cols = ['scenario', 'estimator', 'n_bags', 'rmse_rt', 'rmse_anom', 'p99_err', 'max_err', 'frac_err_gt_1',
            'abs_drift_pct', 'max_abs_drift_pct', 'recovery_s', 'match_rate', 'bad_out']
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in agg:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f'{v:.3g}' if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'
