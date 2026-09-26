"""(d) CUSUM theory vs data: ARL0 / false alarms per hour and detection delays.

    python analysis/consistency/cusum_theory.py [--force]   (needs samples.py first; reuses
        cache/cusum_reconstruction.parquet unless --force)

1. Re-implements jointMonitor()'s recursions (estimator.cpp) on every bag of train+val:
   excess x = (z_i - z_ref)/age - 0.5 (a_model_i + a_model_ref), z_ref = own sample 0.2-0.5 s back
   (closest to 0.3 s) from an 8-sample history; per bogie
     S+ = max(0, S+ + (x - 0.6) dt)   under traction (notch > 0, a_target > 0.1, controller trusted)
     S- = max(0, S- + (-x - 0.8) dt)  under braking  (notch < 0, controller trusted)
   alarm S > h = 0.3 m/s; joint alarm = every available bogie in alarm (single bogie: 2h), and the
   controller CUSUM  C = max(0, C + (imp - 0.5) dt), imp = mean x (notch <= 0) or -x - 1 (traction),
   alarm C > 0.4 m/s. a_model is the monitor's g*a_drive + clamp(d, +-0.6) taken from the published
   outputs at the prior; z is the curve-corrected bogie speed; standstill / latch states from outputs.
2. Clean data = GNSS clock-ok bags, outside known anomaly episodes, both bogies within 0.5 m/s of
   the GNSS Doppler speed for +-1 s. Statistics of x in the active regimes (mean, std, ACF,
   long-run variance), including its dependence on the map grade (the monitor omits a_ext).
3. Siegmund's approximation (Gaussian increments N(mu, s^2) per step, s = sigma or sigma_LR):
     ARL = (exp(-2 D b) + 2 D b - 1) / (2 D^2),  D = mu / s,  b = h / s + 1.166   [steps]
4. Monte Carlo: stationary block bootstrap (mean block 3 s) of the clean bivariate (front, rear)
   x sequences of the active regime -> per-bogie and joint false alarms per active hour; injected
   joint slips / slides (ramp of delta m/s^2 excess, 0.3 s derivative window) -> detection delay.
5. Observed alarms on clean train data: this reconstruction and the C++ outputs (latches, controller
   flags), labelled with the GNSS reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import samples as S  # noqa: E402

H = C.CUSUM_H
RNG = np.random.default_rng(20260926)
MEAN_BLOCK = 30          # samples (3 s) mean block length of the stationary bootstrap
SIM_HOURS = 1000.0       # simulated active hours per false-alarm MC
SCAN_HOURS = 300.0       # simulated active hours per point of the allowance scan
DELTAS = (0.5, 1.0, 2.0)


# ----------------------------------------------------------------------------- reconstruction

def reconstruct(s: pd.DataFrame) -> pd.DataFrame:
    """Replays jointMonitor's CUSUM logic over one bag's sample table (time order)."""
    n = len(s)
    t = s.t.to_numpy()
    z = np.stack([s.zc_f.to_numpy(), s.zc_r.to_numpy()], 1)
    has = np.stack([s.has_f.to_numpy(), s.has_r.to_numpy()], 1)
    fl_pr, fl_po = s.flags_pr.to_numpy(), s.flags_po.to_numpy()
    stuck = np.stack([(fl_pr & C.F_FRONT_STUCK) > 0, (fl_pr & C.F_REAR_STUCK) > 0], 1)
    vc = np.maximum(s.v_pr.to_numpy(), 0.0)
    am = s.amon_pr.to_numpy()
    notch = s.notch.to_numpy()
    atgt = s.atgt_pr.to_numpy()
    still = (fl_po & C.F_STANDSTILL) > 0
    latched_before = s.mu3_pr.to_numpy() > 0.99999
    cmd_fault = (fl_pr & C.F_CMD_INCONS) > 0
    out = {k: np.full(n, np.nan) for k in ('x_f', 'x_r', 'Sp_f', 'Sp_r', 'Sn_f', 'Sn_r', 'cmd')}
    out.update({k: np.zeros(n, bool) for k in ('act_pos', 'act_neg', 'alarm_f', 'alarm_r', 'joint', 'cmd_alarm',
                                                'mon')})
    out['n_av'] = np.zeros(n, int)
    hist = [[], []]                    # (t, z, a_model) per bogie, last 8
    Sp, Sn, tcs = [0.0, 0.0], [0.0, 0.0], [-1.0, -1.0]
    cmdc, tcmd = 0.0, -1.0
    for i in range(n):
        avail = has[i] & ~stuck[i] & np.isfinite(z[i])
        for b in range(2):             # bogie at 0 while the other shows motion is dead
            o = 1 - b
            if avail[b] and z[i, b] <= 0.05 and avail[o] and z[i, o] > 0.5:
                avail[b] = False
        if not avail.any():
            continue
        if still[i]:
            Sp, Sn = [0.0, 0.0], [0.0, 0.0]
            continue
        if latched_before[i]:          # latched: monitor not evaluated, reset on release
            Sp, Sn = [0.0, 0.0], [0.0, 0.0]
            continue
        out['mon'][i] = True
        traction = (not cmd_fault[i]) and notch[i] > 0 and atgt[i] > 0.1
        braking = (not cmd_fault[i]) and notch[i] < 0
        out['act_pos'][i], out['act_neg'][i] = traction, braking
        n_av = alarms = 0
        ex_sum, n_ex = 0.0, 0
        for b in range(2):
            if not avail[b]:
                continue
            n_av += 1
            hb = hist[b]
            hb.append((t[i], z[i, b], am[i]))
            if len(hb) > 8:
                hb.pop(0)
            ref, best = None, 1e9
            for (tq, zq, aq) in hb:
                age = t[i] - tq
                if 0.2 <= age <= 0.5 and abs(age - 0.3) < best:
                    best, ref = abs(age - 0.3), (tq, zq, aq)
            dtc = min(max(t[i] - tcs[b], 0.0), 0.5) if tcs[b] >= 0 else 0.0
            tcs[b] = t[i]
            if ref is None or (vc[i] < 0.5 and z[i, b] < 0.5):
                Sp[b] = Sn[b] = 0.0
            else:
                age = t[i] - ref[0]
                x = (z[i, b] - ref[1]) / age - 0.5 * (am[i] + ref[2])
                out['x_f' if b == 0 else 'x_r'][i] = x
                ex_sum += x
                n_ex += 1
                Sp[b] = max(0.0, Sp[b] + (x - C.CUSUM_SLIP) * dtc) if traction else 0.0
                Sn[b] = max(0.0, Sn[b] + (-x - C.CUSUM_SLIDE) * dtc) if braking else 0.0
            al = Sp[b] > H or Sn[b] > H
            out['alarm_f' if b == 0 else 'alarm_r'][i] = al
            alarms += int(al)
        out['Sp_f'][i], out['Sp_r'][i], out['Sn_f'][i], out['Sn_r'][i] = Sp[0], Sp[1], Sn[0], Sn[1]
        out['n_av'][i] = n_av
        if n_ex > 0 and vc[i] > 0.3:
            ex = ex_sum / n_ex
            imp = ex if notch[i] <= 0 else (-ex - 1.0 if atgt[i] > 0.1 else 0.0)
            dtc = min(max(t[i] - tcmd, 0.0), 0.5) if tcmd >= 0 else 0.0
            cmdc = max(0.0, cmdc + (imp - C.CMD_FAULT_ACCEL) * dtc)
            if cmdc > C.CMD_FAULT_H:
                out['cmd_alarm'][i] = True
                cmdc = C.CMD_FAULT_H
        tcmd = t[i]
        out['cmd'][i] = cmdc
        if n_av == 1 and alarms == 1:
            b = 0 if avail[0] else 1
            if max(Sp[b], Sn[b]) < H * 2.0:
                alarms = 0
        if alarms == n_av:
            out['joint'][i] = True
            Sp, Sn = [0.0, 0.0], [0.0, 0.0]
            hist = [[], []]
    return pd.DataFrame(out)


def clean_by_reference(s: pd.DataFrame, clock_ok: bool) -> np.ndarray:
    onek = 1.0 + s.k_pr.to_numpy()
    dev = np.fmax(np.abs(s.zc_f.to_numpy() / onek - s.v_ref.to_numpy()),
                  np.abs(s.zc_r.to_numpy() / onek - s.v_ref.to_numpy()))
    dev = np.where(np.isfinite(s.v_ref.to_numpy()), dev, np.inf)
    dev = np.nan_to_num(dev, nan=0.0)          # a missing bogie does not make the sample dirty
    bad = dev > 0.5
    # +-1 s dilation (10 samples)
    k = np.ones(21)
    bad = np.convolve(bad.astype(float), k, mode='same') > 0
    return (~bad) & (~s.episode.to_numpy()) & clock_ok


def onsets(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask & ~np.concatenate([[False], mask[:-1]]))


# ----------------------------------------------------------------------------- theory

def siegmund_arl(mu: float, sigma: float, h: float) -> float:
    if sigma <= 0:
        return np.inf
    D = mu / sigma
    b = h / sigma + 1.166
    if abs(D) < 1e-9:
        return b * b
    val = (np.exp(-2 * D * b) + 2 * D * b - 1) / (2 * D * D)
    return float(val)


def long_run_sigma(segs: list[np.ndarray], max_lag: int) -> tuple[float, np.ndarray]:
    r, _ = C.pooled_acf(segs, max_lag)
    allx = np.concatenate(segs)
    return float(np.std(allx) * np.sqrt(max(1 + 2 * r[1:].sum(), 1e-9))), r


# ----------------------------------------------------------------------------- Monte Carlo

class BlockBootstrap:
    """Stationary bootstrap over contiguous runs of a bivariate series (blocks never cross runs)."""

    def __init__(self, runs: list[np.ndarray]):
        self.runs = [r for r in runs if len(r) >= 5]
        self.data = np.concatenate(self.runs)
        starts, pos = [], 0
        self.run_start, self.run_len = [], []
        for r in self.runs:
            self.run_start.append(pos)
            self.run_len.append(len(r))
            pos += len(r)
        self.run_start = np.array(self.run_start)
        self.run_len = np.array(self.run_len)
        self.w = self.run_len / self.run_len.sum()

    def sample(self, n_chains: int, n_steps: int) -> np.ndarray:
        """(n_chains, n_steps, 2) array built from random blocks (geometric length, mean MEAN_BLOCK)."""
        out = np.empty((n_chains, n_steps, 2))
        for c in range(n_chains):
            filled = 0
            while filled < n_steps:
                ri = RNG.choice(len(self.run_len), p=self.w)
                L = int(min(RNG.geometric(1.0 / MEAN_BLOCK), self.run_len[ri], n_steps - filled))
                off = RNG.integers(0, self.run_len[ri] - L + 1)
                a = self.run_start[ri] + off
                out[c, filled:filled + L] = self.data[a:a + L]
                filled += L
        return out


def run_cusum(xs: np.ndarray, kappa: float, sign: int, dt: float, h: float = H, inject=None):
    """Per-bogie + joint CUSUM over (chains, steps, 2); returns alarm step arrays (renewal after alarm).
    inject: (onset_step, delta) adds the slip excess ramp from onset (both bogies)."""
    nc, ns, _ = xs.shape
    Sx = np.zeros((nc, 2))
    first_single = np.full(nc, -1)
    first_joint = np.full(nc, -1)
    n_single = np.zeros(nc, int)
    n_joint = np.zeros(nc, int)
    for k in range(ns):
        x = sign * xs[:, k, :]
        if inject is not None:
            on, delta = inject
            n_after = k - on + 1
            if n_after >= 1:
                x = x + abs(delta) * min(n_after * dt, 0.3) / 0.3
        Sx = np.maximum(0.0, Sx + (x - kappa) * dt)
        al = Sx > h
        anyal = al.any(1)
        both = al.all(1)
        n_single += al.sum(1)
        n_joint += both
        new_s = (first_single < 0) & anyal
        first_single[new_s] = k
        new_j = (first_joint < 0) & both
        first_joint[new_j] = k
        Sx[both] = 0.0                  # joint alarm -> latch -> monitor reset
    return first_single, first_joint, n_single, n_joint


def false_alarm_mc(bb: BlockBootstrap, kappa: float, sign: int, dt: float, hours: float) -> dict:
    steps_per_chunk = int(3600 / dt / 4)             # 15 min chunks
    n_chunks_total = int(hours * 4)
    chains = 200
    single_onsets = joint_onsets = 0
    done = 0
    while done < n_chunks_total:
        xs = bb.sample(chains, steps_per_chunk)
        # count alarm onsets (renewal): per-bogie onsets = transitions into S>h, joint = both
        Sx = np.zeros((chains, 2))
        prev = np.zeros((chains, 2), bool)
        for k in range(steps_per_chunk):
            Sx = np.maximum(0.0, Sx + (sign * xs[:, k, :] - kappa) * dt)
            al = Sx > H
            single_onsets += int((al & ~prev).sum())
            both = al.all(1)
            joint_onsets += int(both.sum())
            Sx[both] = 0.0
            prev = al & ~both[:, None]
        done += chains
    sim_h = done * steps_per_chunk * dt / 3600
    return {'sim_active_hours': sim_h, 'single_bogie_alarms': single_onsets, 'joint_alarms': joint_onsets,
            'single_per_active_h': single_onsets / sim_h / 2, 'joint_per_active_h': joint_onsets / sim_h,
            'joint_per_active_h_upper95': (3.0 if joint_onsets == 0 else joint_onsets + 2 * np.sqrt(joint_onsets)) / sim_h}


def delay_mc(bb: BlockBootstrap, kappa: float, sign: int, dt: float, delta: float, n: int = 2000,
             warm: int = 100, horizon_s: float = 20.0, h: float = H) -> dict:
    ns = warm + int(horizon_s / dt)
    xs = bb.sample(n, ns)
    fs, fj, _, _ = run_cusum(xs, kappa, sign, dt, h=h, inject=(warm, delta))
    ok = (fj < 0) | (fj >= warm)                     # no false joint alarm before the onset
    fjv = fj[ok]
    det = fjv >= warm
    delay = (fjv[det] - warm + 1) * dt
    res = {'delta': delta, 'runs': int(ok.sum()), 'detected_share_20s': float(det.mean())}
    if det.any():
        res.update(delay_mean_s=float(delay.mean()), delay_median_s=float(np.median(delay)),
                   delay_p90_s=float(np.percentile(delay, 90)),
                   detected_within_1s=float(np.mean(det & ((fjv - warm + 1) * dt <= 1.0))),
                   detected_within_3s=float(np.mean(det & ((fjv - warm + 1) * dt <= 3.0))),
                   wheel_overspeed_at_alarm_median_ms=float(np.median(abs(delta) * delay)))
    return res


# ----------------------------------------------------------------------------- main

def main():
    metrics = pd.read_csv(C.RESULTS / 'replay_metrics.csv').set_index('bag')
    sp = C.splits()
    rec_file = C.CACHE / 'cusum_reconstruction.parquet'
    if rec_file.exists() and '--force' not in sys.argv:
        R = pd.read_parquet(rec_file)
    else:
        recs = []
        for split in ('train', 'val'):
            for bag in sp[split]:
                s = pd.read_parquet(S.OUT / f'{bag}.parquet')
                r = reconstruct(s)
                clock_ok = not (metrics.loc[bag, 'ref_clock_anom'] > 0.01)
                r['clean'] = clean_by_reference(s, clock_ok)
                for c in ('t', 'v_pr', 'notch', 'atgt_pr', 'grade_body', 'v_ref', 'a_ref', 'mu3_po', 'mu0_pr',
                          'flags_po', 'k_pr', 'zc_f', 'zc_r'):
                    r[c] = s[c].to_numpy()
                r['bag'], r['split'] = bag, split
                r['dt'] = np.concatenate([[np.nan], np.diff(s.t.to_numpy())])
                recs.append(r)
        R = pd.concat(recs, ignore_index=True)
        R.to_parquet(rec_file, index=False)
    tr = R[R.split == 'train']
    res = {}

    # ---------------- exposure ----------------
    mon = tr.mon & tr.clean
    dt_mean = float(tr.dt[mon & (tr.dt < 0.5)].mean())
    op_hours = float(tr.dt[(tr.dt < 1.0)].sum() / 3600)
    exp_ = {'train_operating_hours': op_hours, 'mean_monitor_dt_s': dt_mean,
            'clean_monitor_hours': float(tr.dt[mon & (tr.dt < 0.5)].sum() / 3600),
            'clean_traction_hours': float(tr.dt[mon & tr.act_pos & (tr.dt < 0.5)].sum() / 3600),
            'clean_braking_hours': float(tr.dt[mon & tr.act_neg & (tr.dt < 0.5)].sum() / 3600),
            'clean_cmd_monitor_hours': float(tr.dt[mon & (tr.v_pr > 0.3) & (tr.dt < 0.5)].sum() / 3600)}
    exp_['traction_share_of_operation'] = exp_['clean_traction_hours'] / exp_['clean_monitor_hours']
    exp_['braking_share_of_operation'] = exp_['clean_braking_hours'] / exp_['clean_monitor_hours']
    res['exposure'] = exp_

    # ---------------- statistics of the excess in the active regimes (clean train) ----------------
    stats, boot = {}, {}
    for name, act, sign, kappa in (('slip_traction', 'act_pos', 1, C.CUSUM_SLIP),
                                   ('slide_braking', 'act_neg', -1, C.CUSUM_SLIDE)):
        m = (tr.mon & tr.clean & tr[act] & np.isfinite(tr.x_f) & np.isfinite(tr.x_r)).to_numpy()
        segs_f, segs_r, runs2 = [], [], []
        for bag, g in tr[m].groupby('bag', sort=False):
            idx = g.index.to_numpy()
            cut = np.flatnonzero((np.diff(idx) != 1) | (np.diff(g.t.to_numpy()) > 0.25)) + 1
            for part in np.split(np.arange(len(idx)), cut):
                if len(part) < 5:
                    continue
                xf, xr = g.x_f.to_numpy()[part], g.x_r.to_numpy()[part]
                segs_f.append(sign * xf)
                segs_r.append(sign * xr)
                runs2.append(np.stack([xf, xr], 1))
        allf = np.concatenate(segs_f)
        allr = np.concatenate(segs_r)
        mu, sd = float(np.mean(np.concatenate([allf, allr]))), float(np.std(np.concatenate([allf, allr])))
        slr10, r10 = long_run_sigma(segs_f + segs_r, 10)
        slr50, _ = long_run_sigma([x for x in segs_f + segs_r if len(x) > 51], 50)
        # grade dependence: sign*x vs a_ext = -kg*grade (kg by regime)
        kg = C.KG_TRACTION if sign > 0 else C.KG_BRAKE
        gm = tr[m]
        a_ext = -kg * gm.grade_body.to_numpy()
        xm = sign * 0.5 * (gm.x_f.to_numpy() + gm.x_r.to_numpy())
        okg = np.isfinite(a_ext)
        slope, icpt = np.polyfit(sign * a_ext[okg], xm[okg], 1)
        steep = okg & (sign * a_ext > 0.2)
        stats[name] = {'samples': int(len(allf)), 'runs': len(runs2), 'mean_signed_excess': mu, 'std': sd,
                       'corr_front_rear': float(np.corrcoef(allf, allr)[0, 1]),
                       'acf_1_10': r10[1:].tolist(), 'sigma_LR_lags10': slr10, 'sigma_LR_lags50': slr50,
                       'p99': float(np.percentile(np.concatenate([allf, allr]), 99)),
                       'p999': float(np.percentile(np.concatenate([allf, allr]), 99.9)),
                       'max': float(np.max(np.concatenate([allf, allr]))),
                       'grade_slope_on_signed_aext': float(slope), 'grade_intercept': float(icpt),
                       'mean_excess_where_signed_aext_gt_0.2': float(np.mean(xm[steep])) if steep.any() else None,
                       'share_signed_aext_gt_0.2': float(np.mean(steep))}
        # Siegmund per bogie (steps of the mean monitor dt)
        sieg = {}
        for lab, s_ in (('iid_sigma', sd), ('sigma_LR_10', slr10), ('sigma_LR_50', slr50)):
            arl_steps = siegmund_arl((mu - kappa) * dt_mean, s_ * dt_mean, H)
            arl_h = arl_steps * dt_mean / 3600
            share = exp_['traction_share_of_operation'] if sign > 0 else exp_['braking_share_of_operation']
            sieg[lab] = {'ARL0_steps': arl_steps, 'ARL0_active_hours': arl_h,
                         'false_alarms_per_active_hour': 1 / arl_h if np.isfinite(arl_h) and arl_h > 0 else 0.0,
                         'false_alarms_per_operating_hour': share / arl_h if arl_h > 0 else 0.0}
            for dl in DELTAS:
                a1 = siegmund_arl((mu + dl - kappa) * dt_mean, s_ * dt_mean, H)
                sieg[lab][f'ARL1_delta{dl}_s'] = a1 * dt_mean   # after the 0.3 s derivative window has filled
        stats[name]['siegmund'] = sieg
        bb = BlockBootstrap(runs2)
        mc = false_alarm_mc(bb, kappa, sign, dt_mean, SIM_HOURS)
        share = exp_['traction_share_of_operation'] if sign > 0 else exp_['braking_share_of_operation']
        mc['joint_per_operating_h'] = mc['joint_per_active_h'] * share
        mc['single_per_operating_h'] = mc['single_per_active_h'] * share
        stats[name]['mc_false_alarms'] = mc
        stats[name]['mc_delay'] = [delay_mc(bb, kappa, sign, dt_mean, dl) for dl in DELTAS]
        boot[name] = bb
        # allowance scan (h fixed): alarms on the real clean runs, bootstrap false alarms, delays
        scan = []
        for kp in ((0.3, 0.4, 0.5, 0.6) if sign > 0 else (0.4, 0.5, 0.6, 0.8)):
            obs_s = obs_j = 0
            for run in runs2:
                Sx = np.zeros(2)
                prev = np.zeros(2, bool)
                for x in sign * run:
                    Sx = np.maximum(0.0, Sx + (x - kp) * dt_mean)
                    al = Sx > H
                    obs_s += int((al & ~prev).sum())
                    if al.all():
                        obs_j += 1
                        Sx[:] = 0.0
                        al[:] = False
                    prev = al
            mck = false_alarm_mc(bb, kp, sign, dt_mean, SCAN_HOURS)
            dls = [delay_mc(bb, kp, sign, dt_mean, d_, n=1000) for d_ in (0.5, 1.0)]
            hrs = exp_['clean_traction_hours' if sign > 0 else 'clean_braking_hours']
            scan.append({'kappa': kp, 'observed_single_per_active_h': obs_s / hrs / 2,
                         'observed_joint': obs_j, 'observed_joint_per_active_h': obs_j / hrs,
                         'mc_joint_per_active_h': mck['joint_per_active_h'],
                         'mc_joint_upper95': mck['joint_per_active_h_upper95'],
                         'mc_single_per_active_h': mck['single_per_active_h'],
                         'delay05_detected_20s': dls[0]['detected_share_20s'],
                         'delay05_median_s': dls[0].get('delay_median_s'),
                         'delay10_median_s': dls[1].get('delay_median_s'),
                         'delay10_p90_s': dls[1].get('delay_p90_s')})
        stats[name]['kappa_scan'] = scan
        print(name, 'done', flush=True)
    res['excess'] = stats

    # ---------------- controller-consistency CUSUM ----------------
    # increments depend on the regime: notch <= 0 -> imp = mean excess (can alarm), traction ->
    # imp = -excess - 1 (mean ~ -1, practically never). Real controller faults (GNSS: acceleration not
    # explained by the grade > +0.3 m/s^2 with notch <= 0, or < -0.5 under traction; +-3 s) are
    # removed from the clean pool, otherwise the "false alarm" MC would count true detections.
    base = (tr.mon & (tr.v_pr > 0.3) & (np.isfinite(tr.x_f) | np.isfinite(tr.x_r))).to_numpy()
    notch = tr.notch.to_numpy()
    atgt = tr.atgt_pr.to_numpy()
    kgv = np.where(notch < 0, C.KG_BRAKE, np.where(notch > 0, C.KG_TRACTION, C.KG_COAST))
    a_comp = tr.a_ref.to_numpy() + kgv * tr.grade_body.to_numpy()
    fault = ((notch <= 0) & (a_comp > 0.3)) | ((notch > 0) & (atgt > 0.1) & (a_comp < -0.5))
    fault_d = np.convolve(fault.astype(float), np.ones(61), mode='same') > 0
    clean_cmd = tr.clean.to_numpy() & ~fault_d
    ex = np.nanmean(np.stack([tr.x_f.to_numpy(), tr.x_r.to_numpy()], 1), 1)
    dtv = tr.dt.to_numpy()
    regA = base & clean_cmd & (notch <= 0)
    regB = base & clean_cmd & (notch > 0) & (atgt > 0.1)
    mon_h = float(np.nansum(dtv[base & tr.clean.to_numpy() & (dtv < 0.5)]) / 3600)
    hA = float(np.nansum(dtv[regA & (dtv < 0.5)]) / 3600)
    cmd = {'monitor_clean_hours': mon_h, 'gnss_fault_share_of_monitor_time': float(np.mean(fault_d[base])),
           'regimeA_notch_le0_clean_hours': hA, 'regimeA_share_of_operation': hA / op_hours,
           'regimeB_traction_clean_hours': float(np.nansum(dtv[regB & (dtv < 0.5)]) / 3600)}
    bags = tr.bag.to_numpy()
    segsA = []
    for a, b in C.runs(regA):
        cut = np.flatnonzero((np.diff(dtv[a:b]) > 0.25) | (bags[a + 1:b] != bags[a:b - 1])) + 1
        for part in np.split(np.arange(a, b), cut):
            if len(part) >= 5:
                segsA.append(ex[part])
    allA = np.concatenate(segsA)
    muA, sdA = float(np.mean(allA)), float(np.std(allA))
    slr10, r10 = long_run_sigma(segsA, 10)
    slr50, _ = long_run_sigma([x for x in segsA if len(x) > 51], 50)
    impB = -ex[regB] - 1.0
    aextA = -C.KG_COAST * tr.grade_body.to_numpy()[regA]
    exA = ex[regA]
    okc = np.isfinite(aextA)
    steep = okc & (aextA > 0.2)
    cmd.update({'A_samples': int(len(allA)), 'A_runs': len(segsA), 'A_mean_imp': muA, 'A_std_imp': sdA,
                'A_acf_1_10': r10[1:].tolist(), 'A_sigma_LR_10': slr10, 'A_sigma_LR_50': slr50,
                'A_p99': float(np.percentile(allA, 99)), 'A_p999': float(np.percentile(allA, 99.9)),
                'A_grade_slope_on_aext': float(np.polyfit(aextA[okc], exA[okc], 1)[0]),
                'A_mean_imp_downhill_aext_gt_0.2': float(np.mean(exA[steep])) if steep.any() else None,
                'A_share_downhill_aext_gt_0.2': float(np.mean(steep)),
                'B_mean_imp': float(np.mean(impB)), 'B_std_imp': float(np.std(impB))})
    for lab, s_ in (('iid_sigma', sdA), ('sigma_LR_10', slr10), ('sigma_LR_50', slr50)):
        arl = siegmund_arl((muA - C.CMD_FAULT_ACCEL) * dt_mean, s_ * dt_mean, C.CMD_FAULT_H)
        arl_h = arl * dt_mean / 3600
        cmd[f'A_siegmund_{lab}'] = {'ARL0_regimeA_hours': arl_h,
                                    'false_alarms_per_regimeA_hour': 1 / arl_h if arl_h > 0 else 0.0,
                                    'false_alarms_per_operating_hour': (hA / op_hours) / arl_h if arl_h > 0 else 0.0}
        for dl in DELTAS:
            cmd[f'A_siegmund_{lab}'][f'delay_delta{dl}_s'] = siegmund_arl(
                (muA + dl - C.CMD_FAULT_ACCEL) * dt_mean, s_ * dt_mean, C.CMD_FAULT_H) * dt_mean
    # MC on regime A (alarm events merged over the 8 s hold, statistic clamped at h like the C++)
    bbc = BlockBootstrap([np.stack([x, x], 1) for x in segsA])
    steps = int(3600 / dt_mean / 4)
    hold = int(8.0 / dt_mean)
    events, done = 0, 0
    while done < int(SIM_HOURS * 4):
        xs = bbc.sample(200, steps)[:, :, 0]
        Cc = np.zeros(200)
        last = np.full(200, -10 ** 9)
        for k in range(steps):
            Cc = np.maximum(0.0, Cc + (xs[:, k] - C.CMD_FAULT_ACCEL) * dt_mean)
            al = Cc > C.CMD_FAULT_H
            events += int((al & (k - last > hold)).sum())
            last[al] = k
            Cc = np.minimum(Cc, C.CMD_FAULT_H)
        done += 200
    simh = done * steps * dt_mean / 3600
    cmd['A_mc'] = {'sim_regimeA_hours': simh, 'events': events, 'per_regimeA_hour': events / simh,
                   'per_operating_hour': events / simh * hA / op_hours}
    cmd['A_mc_delay'] = [delay_mc(bbc, C.CMD_FAULT_ACCEL, 1, dt_mean, dl, n=1000, h=C.CMD_FAULT_H) for dl in DELTAS]
    res['cmd'] = cmd
    print('cmd done', flush=True)

    # ---------------- observed on clean train data (reconstruction) ----------------
    obs = {}
    for name, act, colS in (('slip_traction', 'act_pos', ('Sp_f', 'Sp_r')), ('slide_braking', 'act_neg', ('Sn_f', 'Sn_r'))):
        m = tr.mon & tr[act]
        al_f = (tr[colS[0]] > H) & m
        al_r = (tr[colS[1]] > H) & m
        hrs = exp_['clean_traction_hours' if act == 'act_pos' else 'clean_braking_hours']
        on_f = onsets(al_f.to_numpy())
        on_r = onsets(al_r.to_numpy())
        clean = tr.clean.to_numpy()
        joint_idx = np.flatnonzero((tr.joint & m).to_numpy())
        obs[name] = {'active_clean_hours': hrs,
                     'single_onsets_clean': int(clean[on_f].sum() + clean[on_r].sum()),
                     'single_onsets_dirty': int((~clean[on_f]).sum() + (~clean[on_r]).sum()),
                     'joint_clean': int(clean[joint_idx].sum()), 'joint_dirty': int((~clean[joint_idx]).sum()),
                     'joint_clean_bags': sorted(set(tr.bag.to_numpy()[joint_idx[clean[joint_idx]]]))}
        obs[name]['single_per_active_h'] = obs[name]['single_onsets_clean'] / hrs / 2
        obs[name]['joint_per_active_h'] = obs[name]['joint_clean'] / hrs
    # controller alarms -> events: alarms closer than the 8 s hold belong to one flag episode (as in the C++)
    lab = []
    for bag, g in tr.groupby('bag', sort=False):
        ia = np.flatnonzero(g.cmd_alarm.to_numpy())
        if len(ia) == 0:
            continue
        tt = g.t.to_numpy()
        starts = ia[np.concatenate([[True], np.diff(tt[ia]) > 8.0])]
        for i in starts:
            # label with the grade-compensated GNSS acceleration over the 1 s before the event
            w = g.iloc[max(i - 10, 0):i + 1]
            nt = int(g.notch.iloc[i])
            a_comp = float(np.nanmean(w.a_ref.to_numpy() + (C.KG_COAST if nt <= 0 else C.KG_TRACTION) *
                                      w.grade_body.to_numpy()))
            true_fault = (nt <= 0 and a_comp > 0.3) or (nt > 0 and a_comp < -0.5)
            lab.append({'bag': bag, 't': float(tt[i]), 'notch': nt, 'a_ref_grade_comp': a_comp,
                        'grade_body': float(g.grade_body.iloc[i]), 'clean': bool(g.clean.iloc[i]),
                        'gnss_confirms_fault': bool(true_fault)})
    lab = pd.DataFrame(lab)
    obs['cmd'] = {'events': len(lab), 'events_clean': int(lab.clean.sum()) if len(lab) else 0,
                  'monitor_clean_hours': exp_['clean_cmd_monitor_hours'], 'operating_hours': op_hours}
    lab.to_csv(C.RESULTS / 'cusum_cmd_alarms_train.csv', index=False)
    if len(lab):
        cl = lab[lab.clean]
        obs['cmd']['gnss_confirmed'] = int(lab.gnss_confirms_fault.sum())
        obs['cmd']['unconfirmed'] = int((~lab.gnss_confirms_fault).sum())
        obs['cmd']['unconfirmed_clean'] = int((~cl.gnss_confirms_fault).sum())
        obs['cmd']['unconfirmed_per_operating_h'] = float((~lab.gnss_confirms_fault).sum() / op_hours)
        obs['cmd']['unconfirmed_downhill_share'] = float(np.mean(lab.grade_body[~lab.gnss_confirms_fault] < -0.01))
    res['observed_reconstruction_train'] = obs

    # ---------------- observed in the C++ outputs (all train bags) ----------------
    cpp = {'latch_onsets': 0, 'latch_confirmed_by_gnss': 0, 'cmd_flag_onsets': 0}
    rows = []
    for bag in sp['train']:
        _, o = C.load_replay(bag)
        lat = ((o.mu3.to_numpy() >= 0.99999) & (o.mu0.to_numpy() <= 0.00001))
        on = onsets(lat)
        d = C.load_npz(bag)
        ref = C.reference_speed(d)
        for i in on:
            t0 = o.stamp_ns.iloc[i] * 1e-9
            s = pd.read_parquet(S.OUT / f'{bag}.parquet', columns=['t', 'zc_f', 'zc_r', 'k_pr', 'v_ref'])
            w = s[(s.t > t0 - 1.0) & (s.t < t0 + 3.0)]
            dev = np.nanmax(np.abs(np.fmin(w.zc_f, w.zc_r) / (1 + w.k_pr) - w.v_ref)) if len(w) else np.nan
            dev2 = np.nanmax(np.abs(np.fmax(w.zc_f, w.zc_r) / (1 + w.k_pr) - w.v_ref)) if len(w) else np.nan
            conf = bool(np.nanmax([dev, dev2]) > 0.5)
            rows.append({'bag': bag, 't': t0, 'max_wheel_minus_gnss': float(np.nanmax([dev, dev2])), 'gnss_confirms': conf})
        cf = (o['flags'].to_numpy() & C.F_CMD_INCONS) > 0
        cpp['cmd_flag_onsets'] += len(onsets(cf))
    lat_df = pd.DataFrame(rows)
    lat_df.to_csv(C.RESULTS / 'cusum_cpp_latches_train.csv', index=False)
    cpp['latch_onsets'] = len(lat_df)
    cpp['latch_confirmed_by_gnss'] = int(lat_df.gnss_confirms.sum()) if len(lat_df) else 0
    cpp['latch_unconfirmed_per_operating_h'] = float((len(lat_df) - cpp['latch_confirmed_by_gnss']) / op_hours)
    cpp['cmd_flag_onsets_per_operating_h'] = cpp['cmd_flag_onsets'] / op_hours
    res['observed_cpp_train'] = cpp
    C.write_json(res, C.RESULTS / 'cusum_theory.json')
    import json
    print(json.dumps(res, indent=1, default=float)[:12000])


if __name__ == '__main__':
    main()
