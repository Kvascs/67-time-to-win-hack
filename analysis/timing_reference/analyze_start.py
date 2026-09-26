"""Part 4: what does the solution see during the first seconds of a run (initial alignment window)?

For every bag: start-up burst, GNSS available if the organisers keep GNSS only for the first N s of *bag time*,
stationarity at start, time to first motion, dual-antenna heading at stand-still vs. the later course over
ground, start location on the line.

Outputs: start_per_bag.csv, start_summary.txt (printed), start_alignment.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing as T  # noqa: E402

OUT = Path(__file__).resolve().parent
WEST_TERM = (399000.0, 6184950.0)
EAST_TERM = (403630.0, 6186048.0)
DEPOT = (398790.0, 6184472.0)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def first_motion(t, v, thr=0.5, hold=1.0):
    """First time the speed exceeds thr and stays above it for `hold` seconds."""
    above = v > thr
    idx = np.flatnonzero(above)
    for i in idx:
        j = np.searchsorted(t, t[i] + hold)
        if j <= len(t) and above[i:j].all():
            return t[i]
    return np.nan


def analyze_bag(name: str) -> dict:
    bag = T.load_bag(name)
    t0 = bag.t0
    res = dict(bag=name, vehicle=bag.vehicle)
    f = T.wheel(bag, 'front'); r = T.wheel(bag, 'rear')
    res['first_wheel_hdr_rel'] = float(f.t_hdr.min() - t0)
    res['burst_wheel_span_s'] = float(np.max(f.t_bag[T.stale_mask(f.t_bag, f.t_hdr)] - f.t_hdr[T.stale_mask(f.t_bag, f.t_hdr)])) \
        if T.stale_mask(f.t_bag, f.t_hdr).any() else 0.0
    tw = np.sort(f.t_hdr); vw = f.v[np.argsort(f.t_hdr)]
    res['wheel_speed_first'] = float(vw[0])
    res['moving_at_first_sample'] = bool(vw[0] > 0.5)
    tm = first_motion(tw, vw)
    res['first_motion_rel_t0'] = float(tm - t0) if np.isfinite(tm) else np.nan
    res['first_motion_rel_first_wheel'] = float(tm - tw[0]) if np.isfinite(tm) else np.nan
    res['wheel_max_first5s_hdr'] = float(vw[tw < tw[0] + 5].max())
    res['wheel_max_bag_first5s'] = float(f.v[f.t_bag < t0 + 5].max())
    if not bag.has('master_fix') or len(T.gnss_fix(bag)) < 5:
        res['has_gnss'] = False
        return res
    res['has_gnss'] = True
    fm = T.gnss_fix(bag, 'master'); fr = T.gnss_fix(bag, 'rover')
    i0 = T.first_valid_index(fm)
    res['first_fix_hdr_rel_t0'] = float(fm.t_hdr[i0] - t0)
    res['first_fix_bag_rel_t0'] = float(fm.t_bag[i0] - t0)
    res['first_fix_status'] = int(fm.status[i0])
    res['first_motion_rel_first_fix'] = float(tm - fm.t_hdr[i0]) if np.isfinite(tm) else np.nan
    for N in (1, 2, 3, 5):
        m = fm.t_bag < t0 + N
        res[f'nfix_bag<{N}s'] = int(m.sum())
        res[f'hdr_span_bag<{N}s'] = float(fm.t_hdr[m].max() - fm.t_hdr[m].min()) if m.sum() > 1 else 0.0
        res[f'nfix_status2_bag<{N}s'] = int((m & (fm.status == 2)).sum())
        res[f'nrover_bag<{N}s'] = int((fr.t_bag < t0 + N).sum())
    # GNSS speed in the first 5 s (header time from first fix)
    vm = T.gnss_vel(bag)
    mv = vm.t_hdr < fm.t_hdr[i0] + 5
    res['gnss_speed_max_first5s'] = float(vm.speed[mv].max()) if mv.any() else np.nan
    # start location
    X, Y = T.llh_to_utm(fm.lat[i0], fm.lon[i0])
    dW, dE, dD = (np.hypot(X - p[0], Y - p[1]) for p in (WEST_TERM, EAST_TERM, DEPOT))
    res['start_x_utm'], res['start_y_utm'] = float(X), float(Y)
    res['start_place'] = 'west_terminal' if dW < 350 else 'east_terminal' if dE < 350 else 'depot' if dD < 150 else 'en_route'
    # ---- dual-antenna heading in the init window (bag time < t0 + 3 s) ----
    lat0, lon0, h0 = fm.lat[i0], fm.lon[i0], fm.alt[i0]
    km = np.round(fm.t_hdr * 10).astype(np.int64); kr = np.round(fr.t_hdr * 10).astype(np.int64)
    _, um = np.unique(km, return_index=True); _, ur = np.unique(kr, return_index=True)
    _, im, ir = np.intersect1d(km[um], kr[ur], return_indices=True)
    im, ir = um[im], ur[ir]
    em, nm, _ = T.llh_to_enu(fm.lat[im], fm.lon[im], fm.alt[im], lat0, lon0, h0)
    er, nr, _ = T.llh_to_enu(fr.lat[ir], fr.lon[ir], fr.alt[ir], lat0, lon0, h0)
    base_len = np.hypot(er - em, nr - nm)
    hb = np.arctan2(nr - nm, er - em)
    tp = fm.t_hdr[im]
    tb_pair = np.maximum(fm.t_bag[im], fr.t_bag[ir])
    win = (tb_pair < t0 + 3.0) & (np.abs(base_len - 12.44) < 0.3)
    res['init_pairs_ok'] = int(win.sum())
    if win.sum() >= 3:
        h_init = float(np.angle(np.mean(np.exp(1j * hb[win]))))
        res['init_heading_deg'] = float(np.degrees(h_init))
        res['init_heading_std_deg'] = float(np.degrees(np.std(wrap(hb[win] - h_init))))
        # compare with course over ground once moving (first 3 s with |v| > 2 m/s after start)
        ok = vm.speed > 2.0
        if ok.any():
            tmv = vm.t_hdr[ok]
            sel = ok & (vm.t_hdr < tmv[0] + 3.0)
            crs = float(np.angle(np.mean(np.exp(1j * vm.course[sel]))))
            res['course_when_moving_deg'] = float(np.degrees(crs))
            res['init_heading_minus_course_deg'] = float(np.degrees(wrap(h_init - crs)))
            res['dist_to_first_course_m'] = float(np.nan)
            # baseline heading at the same moment (removes curvature of the track between start and motion)
            k = np.clip(np.searchsorted(tp, tmv[0]), 0, len(tp) - 1)
            res['baseline_minus_course_at_motion_deg'] = float(np.degrees(wrap(hb[k] - crs)))
    # direction of travel along the line: east- or west-bound at start
    if 'init_heading_deg' in res:
        res['facing'] = 'east' if np.cos(np.radians(res['init_heading_deg'])) > 0 else 'west'
    return res


def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    bags = T.list_bags('train', 'val', 'short', 'no_gnss_long')
    rows = []
    for b in bags:
        try:
            rows.append(analyze_bag(b))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(bag=b, error=repr(e)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'start_per_bag.csv', index=False, float_format='%.4g')
    pd.set_option('display.width', 250, 'display.max_columns', 60, 'display.max_rows', 200)
    g = df[df.has_gnss == True]  # noqa: E712
    lines = []
    lines.append(f'bags: {len(df)}, with GNSS: {len(g)}, without: {(df.has_gnss == False).sum()}')  # noqa: E712
    lines.append(f"first master fix header stamp rel. to bag start: median {g.first_fix_hdr_rel_t0.median():.2f} s "
                 f"[min {g.first_fix_hdr_rel_t0.min():.2f}, max {g.first_fix_hdr_rel_t0.max():.2f}]")
    lines.append(f"first wheel header stamp rel. to bag start: median {df.first_wheel_hdr_rel.median():.2f} s")
    for N in (1, 2, 3, 5):
        lines.append(f"GNSS kept for bag time < t0+{N}s -> master fixes: median {g[f'nfix_bag<{N}s'].median():.0f} "
                     f"(min {g[f'nfix_bag<{N}s'].min()}), header span median {g[f'hdr_span_bag<{N}s'].median():.2f} s, "
                     f"rover fixes median {g[f'nrover_bag<{N}s'].median():.0f}")
    lines.append(f"first fix status: {g.first_fix_status.value_counts().to_dict()}")
    lines.append(f"moving at first wheel sample: {int(df.moving_at_first_sample.sum())} of {len(df)}")
    fm = df.first_motion_rel_first_wheel
    for s in (2, 5, 10, 30, 60):
        lines.append(f"start moving (>0.5 m/s for 1 s) within {s:>3d} s of first wheel stamp: {int((fm < s).sum())} of {len(df)}")
    lines.append(f"first motion rel. to first wheel stamp: median {fm.median():.1f} s, p10 {fm.quantile(.1):.1f}, p90 {fm.quantile(.9):.1f}")
    lines.append(f"GNSS speed max in first 5 s (hdr): median {g.gnss_speed_max_first5s.median():.3f} m/s; "
                 f"bags with >0.5 m/s: {int((g.gnss_speed_max_first5s > 0.5).sum())}")
    lines.append(f"start place: {g.start_place.value_counts().to_dict()}")
    lines.append(f"facing at start: {g.facing.value_counts().to_dict() if 'facing' in g else {}}")
    lines.append(f"init heading (baseline, bag<t0+3s) std within window: median {g.init_heading_std_deg.median():.3f} deg, "
                 f"p90 {g.init_heading_std_deg.quantile(.9):.3f}")
    d = g.init_heading_minus_course_deg.dropna()
    lines.append(f"init baseline heading - course over ground at first motion: median {d.median():.2f} deg, "
                 f"|.| p50 {d.abs().median():.2f}, p90 {d.abs().quantile(.9):.2f}, max {d.abs().max():.2f} (n={len(d)})")
    d2 = g.baseline_minus_course_at_motion_deg.dropna()
    lines.append(f"baseline - course at the same instant (mounting + track curvature): median {d2.median():.2f} deg, "
                 f"|.| p90 {d2.abs().quantile(.9):.2f}")
    txt = '\n'.join(lines)
    print(txt)
    (OUT / 'start_summary.txt').write_text(txt, encoding='utf-8')
    print(df[['bag', 'first_fix_hdr_rel_t0', 'nfix_bag<2s', 'hdr_span_bag<2s', 'first_fix_status', 'wheel_speed_first',
              'first_motion_rel_first_wheel', 'start_place', 'facing', 'init_heading_std_deg',
              'init_heading_minus_course_deg']].to_string())
    # figure
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
    axs[0].hist(fm.dropna().clip(upper=120), bins=np.arange(0, 122, 2), color='#1f77b4')
    axs[0].axvline(5, color='r', lw=1); axs[0].set_xlabel('time from first wheel stamp to first motion [s] (clipped at 120)')
    axs[0].set_ylabel('bags'); axs[0].set_title('Most runs start stationary', fontsize=10)
    axs[1].hist(g.first_fix_hdr_rel_t0, bins=30, color='#2ca02c')
    axs[1].set_xlabel('first master fix header stamp - bag start [s]'); axs[1].set_title('Start-up burst: GNSS history before t0', fontsize=10)
    axs[2].hist(g.init_heading_minus_course_deg.dropna().clip(-5, 5), bins=40, color='#9467bd')
    axs[2].set_xlabel('dual-antenna heading at start - course at first motion [deg]')
    axs[2].set_title('Stand-still heading from master->rover baseline', fontsize=10)
    for a in axs:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / 'start_alignment.png', dpi=85)


if __name__ == '__main__':
    main()
