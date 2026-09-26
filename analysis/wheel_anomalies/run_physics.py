"""Physical limits of the true longitudinal motion (from GNSS Doppler reference and from clean wheel
data): acceleration and jerk distributions by regime / notch / speed, to be used as gating thresholds.

Outputs: physics_accel_by_regime.csv, physics_accel_by_notch.csv, physics_jerk.csv, plots."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import anomalies as A  # noqa: E402
from build_aligned import load_aligned  # noqa: E402

PCT = [0.01, 0.1, 1, 5, 50, 95, 99, 99.9, 99.99]


def collect():
    sc = pd.read_csv(HERE / 'scale_factors.csv')
    ep = pd.read_csv(HERE / 'episodes_gnss.csv') if (HERE / 'episodes_gnss.csv').exists() else None
    recs = []
    for _, r in sc.iterrows():
        al = load_aligned(r.bag)
        dt = al.t[1] - al.t[0]
        k = {'front': r.k_speed_front, 'rear': r.k_speed_rear}
        wf = al.front * A.KMH / k['front']; wr = al.rear * A.KMH / k['rear']
        wm = np.nanmean(np.c_[wf, wr], axis=1)
        bad = A.lag_transition_mask(al) | (al.ref_q == 3) | ~np.isfinite(al.ref)
        # GNSS glitch: wheels agree with each other but not with GNSS
        glitch = (np.abs(wf - wr) < 0.15) & (np.abs(wm - al.ref) > 0.3)
        bad |= glitch
        # wheel anomaly episodes (for wheel-based stats)
        wbad = np.zeros(len(al.t), bool)
        if ep is not None:
            for _, e in ep[ep.bag == r.bag].iterrows():
                i0 = np.searchsorted(al.t, e.t_start_hdr) - 20
                i1 = np.searchsorted(al.t, e.t_start_hdr + e.dur) + 20
                wbad[max(i0, 0):i1] = True
        wbad |= ~(np.isfinite(wf) & np.isfinite(wr)) | (np.abs(wf - wr) > 0.15)
        out = dict(bag=r.bag, t=al.t, v=al.ref, notch=al.notch, bad=bad, wbad=wbad, vw=wm)
        for win in (0.5, 1.0, 2.0):
            out[f'a_ref_{win}'] = A.savgol_deriv(np.where(bad, np.nan, al.ref), dt, win)
            out[f'a_w_{win}'] = A.savgol_deriv(np.where(wbad, np.nan, wm), dt, win)
        out['j_ref'] = A.savgol_deriv(out['a_ref_1.0'], dt, 1.0)
        out['j_w'] = A.savgol_deriv(out['a_w_1.0'], dt, 1.0)
        # raw causal wheel accel over 0.2 s and 0.5 s on clean data (detector noise floor)
        out['a_w_raw02'] = A.causal_rate(np.where(wbad, np.nan, wm), 2, dt)
        out['a_w_raw05'] = A.causal_rate(np.where(wbad, np.nan, wm), 5, dt)
        out['a_f_raw02'] = A.causal_rate(np.where(wbad, np.nan, wf), 2, dt)
        recs.append(out)
        print(r.bag, flush=True)
    return recs


def pct_row(x, name, **kw):
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return None
    d = dict(quantity=name, n=len(x), **kw)
    for p in PCT:
        d[f'p{p:g}'] = float(np.percentile(x, p))
    d['min'] = float(x.min()); d['max'] = float(x.max())
    return d


def main():
    recs = collect()
    cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0] if k != 'bag'}
    notch = cat['notch']; v = cat['v']
    moving = v > 0.3
    regimes = {'traction(n>0)': notch > 0, 'coast(n=0)': notch == 0, 'brake(n<0)': notch < 0,
               'brake_hard(n<=-12)': notch <= -12, 'all': np.isfinite(notch)}
    rows = []
    for rn, rm in regimes.items():
        for q in ('a_ref_0.5', 'a_ref_1.0', 'a_ref_2.0', 'a_w_0.5', 'a_w_1.0', 'a_w_2.0', 'j_ref', 'j_w',
                  'a_w_raw02', 'a_w_raw05', 'a_f_raw02'):
            rr = pct_row(cat[q][rm & moving], q, regime=rn)
            if rr:
                rows.append(rr)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'physics_accel_by_regime.csv', index=False)
    # by notch value
    rows = []
    for n in range(-15, 16):
        m = (notch == n) & moving
        for q in ('a_ref_1.0', 'a_w_1.0'):
            rr = pct_row(cat[q][m], q, notch=n)
            if rr:
                rows.append(rr)
    pd.DataFrame(rows).to_csv(HERE / 'physics_accel_by_notch.csv', index=False)
    # by speed bin (all regimes), wheel-based
    rows = []
    for lo in range(0, 16, 2):
        m = (v >= lo) & (v < lo + 2)
        for q in ('a_w_1.0', 'j_w'):
            rr = pct_row(cat[q][m & moving], q, v_lo=lo)
            if rr:
                rows.append(rr)
    pd.DataFrame(rows).to_csv(HERE / 'physics_accel_by_speed.csv', index=False)

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    ax = axs[0]
    for rn, c in (('traction(n>0)', 'tab:green'), ('coast(n=0)', 'tab:gray'), ('brake(n<0)', 'tab:red')):
        x = cat['a_w_1.0'][regimes[rn] & moving]
        x = x[np.isfinite(x)]
        ax.hist(x, bins=np.arange(-3, 2.5, 0.02), histtype='step', density=True, color=c, label=f'{rn} (wheel, 1s SG)')
        x = cat['a_ref_1.0'][regimes[rn] & moving]
        x = x[np.isfinite(x)]
        ax.hist(x, bins=np.arange(-3, 2.5, 0.02), histtype='step', density=True, color=c, ls='--', label=f'{rn} (GNSS)')
    ax.set_yscale('log'); ax.set_xlabel('accel m/s^2'); ax.legend(fontsize=7); ax.grid(); ax.set_title('True acceleration by regime')
    ax = axs[1]
    nn = np.arange(-15, 16)
    for q, c in (('a_w_1.0', 'b'),):
        lo = []; hi = []; med = []; lo1 = []; hi1 = []
        for n in nn:
            x = cat[q][(notch == n) & moving]; x = x[np.isfinite(x)]
            if len(x) < 50:
                lo.append(np.nan); hi.append(np.nan); med.append(np.nan); lo1.append(np.nan); hi1.append(np.nan)
                continue
            lo.append(np.percentile(x, 0.1)); hi.append(np.percentile(x, 99.9)); med.append(np.median(x))
            lo1.append(np.percentile(x, 5)); hi1.append(np.percentile(x, 95))
        ax.fill_between(nn, lo, hi, color=c, alpha=0.2, label='p0.1-p99.9')
        ax.fill_between(nn, lo1, hi1, color=c, alpha=0.4, label='p5-p95')
        ax.plot(nn, med, c + 'o-', label='median')
    ax.set_xlabel('notch'); ax.set_ylabel('accel m/s^2 (wheel, clean)'); ax.grid(); ax.legend(); ax.set_title('Acceleration vs controller notch (moving, v>0.3)')
    ax = axs[2]
    for q, c, lab in (('j_w', 'b', 'jerk wheel (SG 1s of SG 1s accel)'), ('j_ref', 'r', 'jerk GNSS')):
        x = cat[q][moving]; x = x[np.isfinite(x)]
        ax.hist(x, bins=np.arange(-6, 6, 0.05), histtype='step', density=True, color=c, label=lab)
    for q, c, lab in (('a_w_raw02', 'g', 'wheel mean raw accel 0.2 s diff'), ('a_f_raw02', 'm', 'front raw accel 0.2 s diff')):
        x = cat[q][moving]; x = x[np.isfinite(x)]
        ax.hist(x, bins=np.arange(-6, 6, 0.05), histtype='step', density=True, color=c, label=lab)
    ax.set_yscale('log'); ax.legend(fontsize=7); ax.grid(); ax.set_title('Jerk (m/s^3) and raw wheel-difference accel')
    plt.tight_layout()
    plt.savefig(HERE / 'fig_physics_limits.png', dpi=80)
    print(df[df.quantity.isin(['a_ref_1.0', 'a_w_1.0', 'j_w', 'a_w_raw02', 'a_f_raw02'])].round(3).to_string())


if __name__ == '__main__':
    main()
