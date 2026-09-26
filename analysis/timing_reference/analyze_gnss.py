"""Part 1: master vs rover antennas, GNSS status / outliers / jumps, stand-still noise, altitude,
GNSS speed vs derivative of positions.

Outputs (in this directory):
  gnss_per_bag.csv, antenna_baseline.png, gnss_quality.png, vel_vs_posderiv.png
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing as T  # noqa: E402

OUT = Path(__file__).resolve().parent


def stationary_mask(bag: T.Bag, t_query: np.ndarray, hold: float = 1.0) -> np.ndarray:
    """True where both bogie speeds are exactly zero for at least ``hold`` s around t (header clock)."""
    f, r = T.wheel(bag, 'front'), T.wheel(bag, 'rear')
    z = (f.v < 1e-3) & (np.interp(f.t_hdr, r.t_hdr, r.v) < 1e-3)
    # run-length of zeros; mark samples that lie inside a zero run lasting >= hold
    t = f.t_hdr
    edges = np.flatnonzero(np.diff(np.r_[0, z.astype(int), 0]))
    st = np.zeros_like(z)
    for a, b in zip(edges[::2], edges[1::2]):
        if t[b - 1] - t[a] >= hold:
            st[a:b] = True
    k = np.clip(np.searchsorted(t, t_query), 1, len(t) - 1)
    near = np.where(np.abs(t[k - 1] - t_query) < np.abs(t[k] - t_query), k - 1, k)
    ok = np.abs(t[near] - t_query) < 0.3
    return st[near] & ok


def analyze_bag(name: str) -> dict:
    bag = T.load_bag(name)
    res = dict(bag=name, vehicle=bag.vehicle)
    if not (bag.has('master_fix') and bag.has('rover_fix')) or len(T.gnss_fix(bag, 'master')) < 50:
        res['skip'] = 'no gnss'
        return res
    fm, fr = T.gnss_fix(bag, 'master'), T.gnss_fix(bag, 'rover')
    vm, vr = T.gnss_vel(bag, 'master'), T.gnss_vel(bag, 'rover')
    ref = T.reference_trajectory(bag)
    lat0, lon0, h0 = ref.origin
    res['duration'] = float(ref.t[-1] - ref.t[0])
    res['n_master'] = len(fm)
    res['n_rover'] = len(fr)
    res['master_status2_frac'] = float(np.mean(fm.status == 2))
    res['rover_status2_frac'] = float(np.mean(fr.status == 2))
    res['stamps_on_0.1grid_frac'] = float(np.mean(np.abs(fm.t_hdr * 10 - np.round(fm.t_hdr * 10)) < 1e-4))
    # ---------------- master / rover baseline ----------------
    km = np.round(fm.t_hdr * 10).astype(np.int64)
    kr = np.round(fr.t_hdr * 10).astype(np.int64)
    _, um = np.unique(km, return_index=True)
    _, ur = np.unique(kr, return_index=True)
    common, im, ir = np.intersect1d(km[um], kr[ur], return_indices=True)
    im, ir = um[im], ur[ir]
    xm, ym, zm = T.llh_to_enu(fm.lat[im], fm.lon[im], fm.alt[im], lat0, lon0, h0)
    xr, yr, zr = T.llh_to_enu(fr.lat[ir], fr.lon[ir], fr.alt[ir], lat0, lon0, h0)
    bx, by, bz = xr - xm, yr - ym, zr - zm
    L = np.hypot(bx, by)
    tt = fm.t_hdr[im]
    tvm, ve, vn = T._dedup_sorted(vm.t_hdr, vm.ve, vm.vn)
    ve_i, vn_i = np.interp(tt, tvm, ve), np.interp(tt, tvm, vn)
    sp = np.hypot(ve_i, vn_i)
    # yaw rate from velocity course (to select straight track)
    crs = np.unwrap(np.arctan2(vn_i, ve_i))
    yawrate = np.abs(np.gradient(crs, tt))
    both2 = (fm.status[im] == 2) & (fr.status[ir] == 2)
    mv = (sp > 2.0)
    straight = mv & (yawrate < np.radians(0.3))
    uxt, uyt = ve_i / np.maximum(sp, 1e-9), vn_i / np.maximum(sp, 1e-9)
    lon_ = bx * uxt + by * uyt           # + => rover ahead of master in the direction of travel
    lat_ = -bx * uyt + by * uxt          # + => rover to the left
    sel = straight & both2
    res['n_pairs'] = int(len(common))
    res['base_len_med_2'] = float(np.median(L[both2])) if both2.any() else np.nan
    mad = lambda a: float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan  # noqa: E731
    res['base_len_mad_2'] = mad(L[both2])
    res['base_len_outlier_frac_2'] = float(np.mean(np.abs(L[both2] - np.median(L[both2])) > 0.3)) if both2.any() else np.nan
    res['base_len_med_any0'] = float(np.median(L[~both2])) if (~both2).any() else np.nan
    res['base_len_mad_any0'] = mad(L[~both2])
    res['base_len_outlier_frac_any0'] = float(np.mean(np.abs(L[~both2] - res['base_len_med_2']) > 0.3)) if (~both2).any() else np.nan
    res['base_dz_med_2'] = float(np.median(bz[both2])) if both2.any() else np.nan
    res['base_lon_med_straight'] = float(np.median(lon_[sel])) if sel.any() else np.nan
    res['base_lat_med_straight'] = float(np.median(lat_[sel])) if sel.any() else np.nan
    res['frac_rover_ahead_moving'] = float(np.mean(lon_[mv & both2] > 0)) if (mv & both2).any() else np.nan
    # heading from baseline vs course when moving on straight track
    hb = np.arctan2(by, bx)
    dh = np.degrees((hb - np.arctan2(vn_i, ve_i) + np.pi) % (2 * np.pi) - np.pi)
    res['base_minus_course_med_deg'] = float(np.median(dh[sel])) if sel.any() else np.nan
    res['base_minus_course_mad_deg'] = mad(dh[sel])
    # baseline heading noise at stand-still (initial alignment quality)
    stat = stationary_mask(bag, tt)
    res['n_stationary_pairs'] = int(np.sum(stat & both2))
    if (stat & both2).sum() > 20:
        # heading noise inside stationary windows: deviation from the per-window median
        hb_s = hb[stat & both2]
        grp = np.cumsum(np.r_[True, np.diff(tt[stat & both2]) > 1.0])
        dev = np.concatenate([np.degrees((hb_s[grp == g] - np.median(hb_s[grp == g]) + np.pi) % (2 * np.pi) - np.pi)
                              for g in np.unique(grp)])
        res['base_heading_std_stationary_deg'] = float(np.std(dev))
        res['base_heading_p99_stationary_deg'] = float(np.percentile(np.abs(dev), 99))
    # ---------------- master fix quality: jumps vs velocity ----------------
    t, x, y, z, st = ref.t, ref.x, ref.y, ref.z, ref.status
    dt = np.diff(t)
    ve_f, vn_f = np.interp(t, tvm, ve), np.interp(t, tvm, vn)
    pred_dx = 0.5 * (ve_f[1:] + ve_f[:-1]) * dt
    pred_dy = 0.5 * (vn_f[1:] + vn_f[:-1]) * dt
    innov = np.hypot(np.diff(x) - pred_dx, np.diff(y) - pred_dy)
    okdt = (dt > 0.05) & (dt < 0.15)
    res['jump_gt0.5m'] = int(np.sum((innov > 0.5) & okdt))
    res['jump_gt2m'] = int(np.sum((innov > 2.0) & okdt))
    res['jump_max_m'] = float(innov[okdt].max()) if okdt.any() else np.nan
    trans = np.diff(st) != 0
    res['n_status_transitions'] = int(trans.sum())
    res['jump_at_transition_med_m'] = float(np.median(innov[trans & okdt])) if (trans & okdt).any() else np.nan
    res['jump_at_transition_max_m'] = float(np.max(innov[trans & okdt])) if (trans & okdt).any() else np.nan
    res['innov_rms_status2_m'] = float(np.sqrt(np.mean(innov[okdt & (st[1:] == 2) & (st[:-1] == 2)] ** 2)))
    s0 = okdt & (st[1:] == 0) & (st[:-1] == 0)
    res['innov_rms_status0_m'] = float(np.sqrt(np.mean(innov[s0] ** 2))) if s0.any() else np.nan
    gaps = dt[dt > 0.15]
    res['fix_gaps_n'] = int(len(gaps))
    res['fix_gap_max_s'] = float(gaps.max()) if len(gaps) else 0.0
    res['fix_gap_total_s'] = float(gaps.sum()) if len(gaps) else 0.0
    # ---------------- stand-still noise ----------------
    stat_f = stationary_mask(bag, t)
    res['stationary_frac'] = float(np.mean(stat_f))
    spv = np.interp(t, tvm, np.hypot(ve, vn))
    vu_i = np.interp(t, *T._dedup_sorted(vm.t_hdr, vm.vu))
    for lab, m in (('s2', stat_f & (st == 2)), ('s0', stat_f & (st == 0))):
        if m.sum() > 20:
            res[f'still_speed_mean_{lab}'] = float(np.mean(spv[m]))
            res[f'still_speed_p95_{lab}'] = float(np.percentile(spv[m], 95))
            res[f'still_speed3d_mean_{lab}'] = float(np.mean(np.sqrt(spv[m] ** 2 + vu_i[m] ** 2)))
            # position jitter: std about the per-stop median
            grp = np.cumsum(np.r_[True, np.diff(t[m]) > 1.0])
            dxy, dzz = [], []
            for g in np.unique(grp):
                gg = grp == g
                if gg.sum() >= 10:
                    dxy.append(np.hypot(x[m][gg] - np.median(x[m][gg]), y[m][gg] - np.median(y[m][gg])))
                    dzz.append(z[m][gg] - np.median(z[m][gg]))
            if dxy:
                res[f'still_pos_rms_{lab}'] = float(np.sqrt(np.mean(np.concatenate(dxy) ** 2)))
                res[f'still_alt_std_{lab}'] = float(np.std(np.concatenate(dzz)))
    # spurious path length accumulated during stand-still (naive cumulative |dp|)
    step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    res['path_naive_m'] = float(ref.s_naive[-1])
    res['path_gated_m'] = float(ref.s[-1])
    res['path_still_spurious_m'] = float(step[stat_f].sum())
    res['path_int_speed_m'] = float(np.trapezoid(np.nan_to_num(ref.speed), t))
    # ---------------- altitude ----------------
    dz = np.diff(z)
    res['alt_range_m'] = float(z.max() - z.min())
    res['alt_step_gt0.5m'] = int(np.sum((np.abs(dz) > 0.5) & okdt))
    res['alt_step_max_m'] = float(np.max(np.abs(dz[okdt]))) if okdt.any() else np.nan
    res['alt_step_at_transition_max_m'] = float(np.max(np.abs(dz[trans & okdt]))) if (trans & okdt).any() else np.nan
    # ---------------- GNSS vel vs derivative of positions ----------------
    # central-difference speed of positions, only status-2 stretches, no gaps
    cd_ok = np.r_[False, (dt[1:] < 0.15) & (dt[:-1] < 0.15), False]
    if np.sum(cd_ok & (st == 2)) > 500:
        cd_ok &= (st == 2)
    sp_pos = ref.speed_pos
    lagres = T.estimate_lag(ref.vel_t, ref.vel_speed, t[cd_ok], sp_pos[cd_ok], lo=-0.4, hi=0.4, coarse=0.005)
    res['vel_vs_posderiv_lag_s'] = lagres['lag']      # >0: position-derivative is late w.r.t. vel
    res['vel_vs_posderiv_k'] = lagres['k']
    res['vel_vs_posderiv_rms'] = lagres['rms']
    both = cd_ok & np.isfinite(ref.speed) & (ref.speed > 0.5)
    res['vel_minus_posderiv_mean'] = float(np.mean(ref.speed[both] - sp_pos[both])) if both.any() else np.nan
    res['vel_over_posderiv_ratio_med'] = float(np.median(ref.speed[both] / np.maximum(sp_pos[both], 1e-6))) if both.any() else np.nan
    # rover vs master speed
    tvr, spr = T._dedup_sorted(vr.t_hdr, vr.speed)
    mm = np.isfinite(ref.speed) & (ref.speed > 1)
    res['rover_minus_master_speed_mean'] = float(np.mean(np.interp(t[mm], tvr, spr) - ref.speed[mm]))
    res['rover_minus_master_speed_rms'] = float(np.sqrt(np.mean((np.interp(t[mm], tvr, spr) - ref.speed[mm]) ** 2)))
    return res


def safe_analyze(name: str) -> dict:
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    try:
        return analyze_bag(name)
    except Exception as e:  # keep going, report
        return dict(bag=name, vehicle=name.split('_')[0], skip=f'error: {e!r}')


def main():
    bags = T.list_bags('train', 'val', 'short')
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(safe_analyze, bags))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'gnss_per_bag.csv', index=False, float_format='%.5g')
    ok = df[df.get('skip').isna()] if 'skip' in df else df
    pd.set_option('display.width', 250, 'display.max_columns', 80, 'display.max_rows', 200)
    print(ok.describe().T.to_string())
    print(ok.groupby('vehicle')[['base_len_med_2', 'base_lon_med_straight', 'base_lat_med_straight', 'base_dz_med_2',
                                 'base_minus_course_med_deg', 'frac_rover_ahead_moving']].agg(['median', 'min', 'max']).T)


if __name__ == '__main__':
    main()
