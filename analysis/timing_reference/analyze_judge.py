"""Part 2b: judge emulation - which output stamp, and how much lead/lag, minimises the judge's errors.

(A) timing sensitivity of the reference itself: RMS[v_ref(t+d) - v_ref(t)], RMS[s_ref(t+d) - s_ref(t)]
(B) nearest-stamp pairing: for output-stamp strategies (wheel hdr / cmd hdr / union), statistics of the
    stamp mismatch dt = t_out - t_ref and the error it causes for a *perfect* estimator
(C) end-to-end with the real wheel signal:
    - speed:    RMSE / bias(accel) / bias(brake) of k*wheel(t_out - d) vs GNSS vel at the paired stamps, vs d
    - position: along-track error of wheel odometry re-anchored every 20 s, vs lead d
    under both judge time bases: GNSS header stamps (J-hdr, most likely) and GNSS bag times (J-bag).

Outputs: judge_per_bag.csv, judge_timing.png
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing as T  # noqa: E402

OUT = Path(__file__).resolve().parent
SHIFTS = np.round(np.arange(-0.20, 0.2001, 0.01), 3)      # candidate value-time shifts d [s]
WIN = 20.0                                                 # re-anchoring window for odometry [s]


def clean_mask(bag, t_hdr):
    tg, rel, anom = T.clock_offsets(bag)
    ok = np.ones(len(t_hdr), bool)
    if anom.any():
        ta = np.sort(tg[anom])
        k = np.clip(np.searchsorted(ta, t_hdr), 0, len(ta) - 1)
        d = np.abs(t_hdr - ta[k])
        k2 = np.clip(k - 1, 0, len(ta) - 1)
        d = np.minimum(d, np.abs(t_hdr - ta[k2]))
        ok &= d > 5.0
    return ok


def wheel_speed(bag):
    f, r = T.wheel(bag, 'front'), T.wheel(bag, 'rear')
    ok = ~T.stale_mask(f.t_bag, f.t_hdr)
    t = f.t_hdr[ok]
    o = np.argsort(t)
    t = t[o]
    v = 0.5 * (f.v[ok][o] + np.interp(t, np.sort(r.t_hdr), r.v[np.argsort(r.t_hdr)]))
    tb = f.t_bag[ok][o]
    keep = np.r_[True, np.diff(t) > 1e-6]
    return t[keep], tb[keep], v[keep]


def analyze_bag(name: str) -> dict:
    warnings.simplefilter('ignore', RuntimeWarning)
    bag = T.load_bag(name)
    res = dict(bag=name, vehicle=bag.vehicle)
    ref = T.reference_trajectory(bag)
    vm = T.gnss_vel(bag)
    fm = T.gnss_fix(bag)
    # reference time bases
    okv = ~T.stale_mask(vm.t_bag, vm.t_hdr)
    tvh, tvb, vsp = T._dedup_sorted(vm.t_hdr[okv], vm.t_bag[okv], vm.speed[okv])
    cm_v = clean_mask(bag, tvh)
    # ---------------- (A) sensitivity of the reference to a time shift ----------------
    moving = vsp > 0.3
    for d in (0.010, 0.025, 0.047, 0.080, 0.127):
        dv = np.interp(tvh + d, tvh, vsp) - vsp
        res[f'A_speed_rms_shift_{int(d*1e3)}ms'] = float(np.sqrt(np.mean(dv[cm_v] ** 2)))
        ds = np.interp(ref.t + d, ref.t, ref.s) - ref.s
        res[f'A_pos_rms_shift_{int(d*1e3)}ms'] = float(np.sqrt(np.mean(ds ** 2)))
    acc = np.gradient(vsp, tvh)
    res['A_acc_rms_moving'] = float(np.sqrt(np.mean(acc[moving & cm_v] ** 2)))
    res['A_speed_rms_moving'] = float(np.sqrt(np.mean(vsp[moving] ** 2)))
    # ---------------- (B) pairing statistics for output-stamp strategies ----------------
    th, tb_w, vw = wheel_speed(bag)
    c = T.cmd(bag)
    okc = ~T.stale_mask(c.t_bag, c.t_hdr)
    tc = np.sort(c.t_hdr[okc])
    strategies = {'wheel': th, 'cmd': tc, 'union': np.unique(np.r_[th, tc])}
    for sname, ts in strategies.items():
        ir, ie, dt = T.match_nearest(tvh, ts, 0.05)
        res[f'B_{sname}_match_frac'] = float(len(ir) / len(tvh))
        res[f'B_{sname}_dt_rms_ms'] = float(np.sqrt(np.mean(dt ** 2)) * 1e3)
        res[f'B_{sname}_dt_max_ms'] = float(np.max(np.abs(dt)) * 1e3)
        # error seen by the judge for a perfect estimator evaluated at its own stamp
        v_at_out = np.interp(ts[ie], tvh, vsp)
        e = v_at_out - vsp[ir]
        res[f'B_{sname}_speed_err_rms'] = float(np.sqrt(np.mean(e[cm_v[ir]] ** 2)))
        s_at_out = np.interp(ts[ie], ref.t, ref.s)
        s_at_ref = np.interp(tvh[ir], ref.t, ref.s)
        res[f'B_{sname}_pos_err_rms'] = float(np.sqrt(np.mean((s_at_out - s_at_ref)[cm_v[ir]] ** 2)))
        # judge-side: pair every OUTPUT with its nearest reference (alternative judge implementation)
        io, jr, dt2 = T.match_nearest(ts, tvh, 0.05)
        res[f'B_{sname}_outside_dt_rms_ms'] = float(np.sqrt(np.mean(dt2 ** 2)) * 1e3)
    # ---------------- (C) end-to-end with the wheel signal ----------------
    # per-bag scale (wheel -> GNSS vel), lag 0 in header time
    lagres = T.estimate_lag(tvh, vsp, th, vw, lo=-0.3, hi=0.3, coarse=0.005, mask=cm_v)
    k = lagres['k']
    res['k'] = k
    vwk = vw * k
    acc_ref = np.convolve(np.gradient(vsp, tvh), np.ones(5) / 5, 'same')
    for jname, t_ref in (('Jhdr', tvh), ('Jbag', tvb)):
        # outputs stamped with cmd header stamps (recommended); value = k*wheel interpolated at (stamp - d)
        ts = strategies['cmd']
        ir, ie, dt = T.match_nearest(t_ref, ts, 0.05)
        ok = cm_v[ir] & (np.maximum(vsp[ir], np.interp(ts[ie], th, vwk)) > 0.05)
        best = None
        for d in SHIFTS:
            est = np.interp(ts[ie] - d, th, vwk)
            e = (est - vsp[ir])[ok]
            rm = float(np.sqrt(np.mean(e ** 2)))
            res[f'C_{jname}_speed_rmse_d{int(round(d*1e3))}'] = rm
            if best is None or rm < best[1]:
                best = (d, rm)
            if abs(d) < 1e-9 or abs(d + 0.08) < 1e-9 or abs(d - 0.047) < 1e-9:
                a = acc_ref[ir][ok]
                res[f'C_{jname}_bias_acc_d{int(round(d*1e3))}'] = float(np.mean(e[a > 0.3]))
                res[f'C_{jname}_bias_brk_d{int(round(d*1e3))}'] = float(np.mean(e[a < -0.3]))
        res[f'C_{jname}_speed_best_d'] = best[0]
        res[f'C_{jname}_speed_best_rmse'] = best[1]
    # position: along-track odometry error, re-anchored every WIN seconds, slip-free windows only
    tf_h = ref.t
    o = np.argsort(fm.t_hdr)
    tf_b = np.interp(tf_h, fm.t_hdr[o], fm.t_bag[o])          # bag time of each fix row
    goodfix = (ref.status == 2) if np.mean(ref.status == 2) > 0.5 else np.ones(len(tf_h), bool)
    res['C_pos_status2_used'] = bool(np.mean(ref.status == 2) > 0.5)
    cm_p = clean_mask(bag, tf_h) & goodfix & ~T.stale_mask(tf_b, tf_h)
    # reference self-consistency: position increment vs vel (flags fix jumps)
    v_at_fix = np.interp(tf_h, tvh, vsp)
    jump = np.r_[False, np.abs(np.diff(ref.s) - 0.5 * (v_at_fix[1:] + v_at_fix[:-1]) * np.diff(tf_h)) > 0.15]
    slip = np.abs(np.interp(tf_h, th, vwk) - v_at_fix) > 0.25
    Sw = np.r_[0.0, np.cumsum(0.5 * (vwk[1:] + vwk[:-1]) * np.diff(th))]   # wheel distance, header time
    win_len = 10.0
    for jname, t_ref in (('Jhdr', tf_h), ('Jbag', tf_b), ('Dhdr', tf_h), ('Dbag', tf_b)):
        if jname.startswith('D'):
            # direct: outputs stamped exactly at the reference stamps (no pairing offset) -> pure lead
            ir = np.arange(len(t_ref)); t_all = t_ref
            ok = cm_p.copy()
            t_out, s_ref, bad = t_all[ok], ref.s[ok], (jump | slip)[ok]
        else:
            ts = strategies['cmd']
            ir, ie, dt = T.match_nearest(t_ref, ts, 0.05)
            ok = cm_p[ir]
            t_out, s_ref, bad = ts[ie][ok], ref.s[ir][ok], (jump | slip)[ir][ok]
        win = np.floor((t_out - t_out[0]) / win_len).astype(int)
        # drop whole windows containing slip / fix jumps / gaps
        badwin = np.unique(win[bad])
        gapwin = np.unique(win[1:][np.diff(t_out) > 0.3])
        keep = ~np.isin(win, np.r_[badwin, gapwin])
        t_out, s_ref, win = t_out[keep], s_ref[keep], win[keep]
        if len(t_out) < 200:
            continue
        first = np.r_[0, np.flatnonzero(np.diff(win)) + 1]
        anchor_idx = first[np.searchsorted(first, np.arange(len(win)), side='right') - 1]
        res[f'C_{jname}_pos_n'] = int(len(t_out))
        best = None
        for d in SHIFTS:
            sw = np.interp(t_out + d, th, Sw)          # odometry advanced by lead d
            e = s_ref[anchor_idx] + (sw - sw[anchor_idx]) - s_ref
            rm = float(np.sqrt(np.mean(e ** 2)))
            res[f'C_{jname}_pos_rmse_d{int(round(d*1e3))}'] = rm
            if best is None or rm < best[1]:
                best = (d, rm)
        # sub-grid refinement (parabola through the 3 best shifts)
        i = int(np.argmin([res[f'C_{jname}_pos_rmse_d{int(round(d*1e3))}'] for d in SHIFTS]))
        if 0 < i < len(SHIFTS) - 1:
            y = [res[f'C_{jname}_pos_rmse_d{int(round(d*1e3))}'] ** 2 for d in SHIFTS[i - 1:i + 2]]
            den = y[0] - 2 * y[1] + y[2]
            best = (SHIFTS[i] + (0.5 * (y[0] - y[2]) / den * 0.01 if den > 0 else 0.0), best[1])
        res[f'C_{jname}_pos_best_d'] = best[0]
        res[f'C_{jname}_pos_best_rmse'] = best[1]
    # ---------------- (B2) perfect estimator, 'value at stamp' vs 'value at nearest 0.1-s epoch' ----------------
    V = lambda t: np.interp(t, tvh, vsp)            # wheel-equivalent speed at header time t  # noqa: E731
    P = lambda t: np.interp(t, ref.t, ref.s)        # true along-track position at time t      # noqa: E731
    ts = strategies['cmd']
    E_v, E_p = tvh, ref.t                           # GNSS epochs (header) of vel / fix rows
    Tb_v, Tb_p = tvb, tf_b                          # their bag times
    for est_name, sv, sp in (('hdrOpt', 0.0, 0.0), ('bagOpt', -0.080, -0.044)):
        for snap in (False, True):
            ev_t = np.round(ts * 10) / 10 if snap else ts
            spd_out = V(ev_t + sv)
            pos_out = P(ev_t + sp)
            for jname, (tv_ref, tp_ref) in (('Jhdr', (E_v, E_p)), ('Jbag', (Tb_v, Tb_p))):
                ir, ie, _ = T.match_nearest(tv_ref, ts, 0.05)
                m = cm_v[ir]
                ev = spd_out[ie][m] - vsp[ir][m]
                ir2, ie2, _ = T.match_nearest(tp_ref, ts, 0.05)
                m2 = cm_p[ir2]
                ep = pos_out[ie2][m2] - ref.s[ir2][m2]
                tag = f'B2_{est_name}_{"snap" if snap else "stamp"}_{jname}'
                res[tag + '_speed_rms'] = float(np.sqrt(np.mean(ev ** 2)))
                res[tag + '_pos_rms'] = float(np.sqrt(np.mean(ep ** 2)))
    return res


def safe(name):
    try:
        return analyze_bag(name)
    except Exception as e:
        return dict(bag=name, skip=f'error {e!r}')


def main():
    g = pd.read_csv(OUT / 'gnss_per_bag.csv')
    d = pd.read_csv(OUT / 'delays_per_bag.csv')
    good = d[(d['front_pos_hh_rms'] < 0.06)]['bag'].tolist()          # bags with trustworthy fix positions
    speed_ok = g[(g.get('skip').isna()) & (g.duration > 600)]['bag'].tolist()
    bags = sorted(set(speed_ok))
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(safe, bags))
    df = pd.DataFrame(rows)
    df['good_pos'] = df.bag.isin(good)
    df.to_csv(OUT / 'judge_per_bag.csv', index=False, float_format='%.5g')
    print(f'bags: {len(df)}  good-position bags: {df.good_pos.sum()}')
    pd.set_option('display.width', 250, 'display.max_columns', 80, 'display.max_rows', 400)
    cols = [c for c in df.columns if c.startswith('A_') or c.startswith('B_') or c.endswith('_best_d')
            or c.endswith('_best_rmse') or '_bias_' in c]
    print(df[cols].median().to_string())
    gp = df[df.good_pos]
    print('\nposition lead (good bags only):')
    print(gp[[c for c in df.columns if '_pos_' in c and ('best' in c or c.endswith('_n'))]].describe().T.to_string())
    print('\nB2 (good bags):')
    print(gp[[c for c in df.columns if c.startswith('B2_')]].median().to_string())
    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gp = df[df.good_pos]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    ms = SHIFTS * 1e3
    for jname, col in (('Jhdr', '#1f77b4'), ('Jbag', '#d62728')):
        sp = np.array([[r.get(f'C_{jname}_speed_rmse_d{int(round(d*1e3))}', np.nan) for d in SHIFTS] for _, r in df.iterrows()])
        rel = sp - np.nanmin(sp, axis=1, keepdims=True)
        axs[0].plot(ms, np.nanmedian(rel, 0) * 1e3, '-o', ms=3, color=col, label=f'{jname}: median over {len(df)} bags')
        axs[0].fill_between(ms, np.nanpercentile(rel, 25, 0) * 1e3, np.nanpercentile(rel, 75, 0) * 1e3, color=col, alpha=.15)
    for jname, col, lab in (('Dhdr', '#1f77b4', 'header-time judge'), ('Dbag', '#d62728', 'bag-time judge')):
        ps = np.array([[r.get(f'C_{jname}_pos_rmse_d{int(round(d*1e3))}', np.nan) for d in SHIFTS] for _, r in gp.iterrows()])
        relp = ps - np.nanmin(ps, axis=1, keepdims=True)
        axs[1].plot(ms, np.nanmedian(relp, 0) * 1e2, '-o', ms=3, color=col, label=f'{lab}: median over {len(gp)} bags')
        axs[1].fill_between(ms, np.nanpercentile(relp, 25, 0) * 1e2, np.nanpercentile(relp, 75, 0) * 1e2, color=col, alpha=.15)
    axs[0].set_xlabel('speed value shift d [ms]  (output k*wheel(t_stamp - d))')
    axs[0].set_ylabel('speed RMSE above per-bag minimum [mm/s]')
    axs[0].set_title('Speed: best d = 0 (header-time judge) / +80 ms (bag-time judge)', fontsize=10)
    axs[1].set_xlabel('position lead d [ms]  (wheel odometry evaluated at t_ref + d)')
    axs[1].set_ylabel('along-track RMSE above per-bag minimum [cm]')
    axs[1].set_title('Position lead: +48 ms (header-time judge) / ~+5 ms (bag-time judge)\n'
                     '(10-s re-anchored odometry, slip-free windows, evaluated at reference stamps)', fontsize=9)
    for a in axs:
        a.grid(alpha=.3); a.legend(fontsize=8); a.axvline(0, color='k', lw=.6)
    fig.tight_layout()
    fig.savefig(OUT / 'judge_timing.png', dpi=90)


if __name__ == '__main__':
    main()
