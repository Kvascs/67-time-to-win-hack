"""Step 1: per wheel sample (paired front/rear header stamps) -> master-antenna arc s on the main
cycle from RTK fixes, curvature at the front/rear pivot, rigid-car chord prediction, flags.
Also projects every RTK fix on the west_arrival_2 stub (for step 3).

Output: samples.pkl (all GNSS bags, train+val), bag_summary.csv, stub_hits.csv
Usage: python build_samples.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

GATE_MAIN = 1.5      # keep RTK fixes within this lateral distance of the main map [m]
MAX_FIX_GAP = 0.5    # interpolate s only between fixes closer than this in time [s]


def segment_ids(t, s_unw):
    """Break the fix sequence where time gaps or implausible arc jumps occur."""
    dt = np.diff(t)
    ds = np.diff(s_unw)
    brk = (dt > MAX_FIX_GAP) | (np.abs(ds) > 25.0 * np.maximum(dt, 0.02) + 1.0)
    return np.r_[0, np.cumsum(brk)]


def interp_segmented(t_fix, val, seg, t_q):
    """Linear interpolation of val(t_fix) at t_q only where both neighbours share a segment."""
    out = np.full(len(t_q), np.nan)
    if len(t_fix) < 2:
        return out
    j = np.searchsorted(t_fix, t_q)
    ok = (j > 0) & (j < len(t_fix))
    j = np.clip(j, 1, len(t_fix) - 1)
    ok &= seg[j - 1] == seg[j]
    ok &= (t_fix[j] - t_fix[j - 1]) <= MAX_FIX_GAP
    w = (t_q - t_fix[j - 1]) / np.maximum(t_fix[j] - t_fix[j - 1], 1e-9)
    v = val[j - 1] + w * (val[j] - val[j - 1])
    out[ok] = v[ok]
    return out


def smooth_deriv(t, v, half=0.3):
    """Derivative by local linear fit over +-half seconds (irregular stamps), vectorised with
    cumulative sums; NaN if any value in the window is NaN or fewer than 3 points."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    tc = t - t[0]
    fin = np.isfinite(v)
    v0 = np.where(fin, v, 0.0)

    def cs(a):
        return np.r_[0.0, np.cumsum(a)]
    c1, ct, cv, ctt, ctv, cf = cs(np.ones_like(tc)), cs(tc), cs(v0), cs(tc * tc), cs(tc * v0), cs(fin.astype(float))
    lo = np.searchsorted(t, t - half)
    hi = np.searchsorted(t, t + half, side='right')
    n = c1[hi] - c1[lo]
    st, sv, stt, stv = ct[hi] - ct[lo], cv[hi] - cv[lo], ctt[hi] - ctt[lo], ctv[hi] - ctv[lo]
    nf = cf[hi] - cf[lo]
    den = n * stt - st * st
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = (n * stv - st * sv) / den
    slope[(n < 3) | (nf < n) | (np.abs(den) < 1e-12)] = np.nan
    return slope


def main():
    origin = C.map_origin()
    pl = C.load_main()
    stub, _ = C.load_stub()
    L = pl.L
    print(f'main cycle L = {L:.2f} m')
    rows, summ, hits = [], [], []
    for bag, split in C.gnss_bags():
        d = C.load_bag(bag)
        w = C.wheel_pairs(d)
        fx = C.master_fix_enu(d, origin)
        fx = fx[fx.status == 2].drop_duplicates('t').reset_index(drop=True)
        n_rtk = len(fx)
        s_m, lat_m, d_m = pl.project(fx.x.values, fx.y.values)
        s_st, lat_st, d_st = stub.project(fx.x.values, fx.y.values)
        fx['s'] = s_m
        fx['dmain'] = d_m
        fx['s_stub'] = s_st
        fx['dstub'] = d_st
        # stub hits (for step 3)
        on_stub = (d_st < 2.0) & (d_m > 4.0)
        if on_stub.any():
            hits.append(dict(bag=bag, split=split, n_fix=int(on_stub.sum()),
                             s_stub_min=float(s_st[on_stub].min()), s_stub_max=float(s_st[on_stub].max()),
                             t_first=float(fx.t[on_stub].min()), t_last=float(fx.t[on_stub].max())))
        keep = d_m <= GATE_MAIN
        f = fx[keep].reset_index(drop=True)
        s_unw = np.unwrap(f.s.values, period=L)
        seg = segment_ids(f.t.values, s_unw)
        # re-unwrap inside segments is not needed (breaks only drop interpolation across them)
        tq = w.t.values - C.FIX_LEAD_S
        s_w = interp_segmented(f.t.values, s_unw, seg, tq)
        lat_w = interp_segmented(f.t.values, f.dmain.values, seg, tq)
        # stub coordinates at wheel stamps (only fixes near the stub)
        ks = fx.dstub.values <= 2.0
        fs = fx[ks].reset_index(drop=True)
        if len(fs) > 2:
            seg_s = segment_ids(fs.t.values, fs.s_stub.values)
            s_stub_w = interp_segmented(fs.t.values, fs.s_stub.values, seg_s, tq)
            dmain_stub_w = interp_segmented(fs.t.values, fs.dmain.values, seg_s, tq)
        else:
            s_stub_w = np.full(len(w), np.nan)
            dmain_stub_w = np.full(len(w), np.nan)
        df = pd.DataFrame({'bag': bag, 'split': split, 't': w.t.values, 'recv': w.recv.values,
                           'vf': w.vf.values, 'vr': w.vr.values, 's': s_w, 'dmain': lat_w,
                           's_stub': s_stub_w, 'dmain_at_stub': dmain_stub_w})
        df['sm'] = np.mod(df.s, L)
        ok = np.isfinite(df.s.values)
        s_ok = df.s.values[ok]
        df['kf'] = np.nan
        df['kr'] = np.nan
        df['pred'] = np.nan
        df.loc[ok, 'kf'] = pl.k_at(s_ok + C.D_FRONT)
        df.loc[ok, 'kr'] = pl.k_at(s_ok + C.D_REAR)
        df.loc[ok, 'kant'] = pl.k_at(s_ok)
        df.loc[ok, 'pred'] = C.chord_pred(pl, s_ok)
        # kinematics
        vm = 0.5 * (df.vf.values + df.vr.values)
        df['v'] = vm
        df['a'] = smooth_deriv(df.t.values, vm, 0.3)
        df['dsdt'] = smooth_deriv(df.t.values, df.s.values, 0.5)
        df['bad_ep'] = C.bad_episode_mask(bag, d, df.recv.values)
        rows.append(df)
        summ.append(dict(bag=bag, split=split, n_pairs=len(w), n_rtk=n_rtk, n_rtk_on_main=int(keep.sum()),
                         frac_s=float(ok.mean()), s_min=float(np.nanmin(df.s)) if ok.any() else np.nan,
                         s_max=float(np.nanmax(df.s)) if ok.any() else np.nan,
                         frac_rev=float(np.mean(df.dsdt[ok & (vm > 1)] < -0.5)) if (ok & (vm > 1)).any() else np.nan))
        print(f'{bag} {split}: pairs={len(w)} rtk={n_rtk} on_main={keep.sum()} s_ok={ok.mean():.3f} '
              f'span={summ[-1]["s_max"] - summ[-1]["s_min"]:.0f} m  rev={summ[-1]["frac_rev"]}', flush=True)
    S = pd.concat(rows, ignore_index=True)
    S.to_pickle(C.HERE / 'samples.pkl')
    pd.DataFrame(summ).to_csv(C.HERE / 'bag_summary.csv', index=False)
    pd.DataFrame(hits).to_csv(C.HERE / 'stub_hits.csv', index=False)
    print('stub hits:')
    print(pd.DataFrame(hits).to_string())


if __name__ == '__main__':
    main()
