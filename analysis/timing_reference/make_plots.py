"""Figures for REPORT.md that are not produced by the analyze_* scripts:
antenna_baseline.png, gnss_quality.png, delays.png, transient_zoom.png
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


def baseline_samples(name):
    bag = T.load_bag(name)
    fm, fr = T.gnss_fix(bag, 'master'), T.gnss_fix(bag, 'rover')
    lat0, lon0, h0 = T.reference_trajectory(bag).origin
    km = np.round(fm.t_hdr * 10).astype(np.int64); kr = np.round(fr.t_hdr * 10).astype(np.int64)
    _, um = np.unique(km, return_index=True); _, ur = np.unique(kr, return_index=True)
    _, im, ir = np.intersect1d(km[um], kr[ur], return_indices=True)
    im, ir = um[im], ur[ir]
    em, nm, um_ = T.llh_to_enu(fm.lat[im], fm.lon[im], fm.alt[im], lat0, lon0, h0)
    er, nr, ur_ = T.llh_to_enu(fr.lat[ir], fr.lon[ir], fr.alt[ir], lat0, lon0, h0)
    both2 = (fm.status[im] == 2) & (fr.status[ir] == 2)
    return fm.t_hdr[im] - bag.t0, np.hypot(er - em, nr - nm), ur_ - um_, both2, fm.status[im], fr.status[ir]


def fig_baseline():
    g = pd.read_csv(OUT / 'gnss_per_bag.csv')
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
    L2, L0 = [], []
    for name in ['30618_0652866c', '30618_27e994fc', '30639_92226df0', '30618_e2dcf65f', '30639_2b4a6347', '30618_88548b02']:
        _, L, _, b2, _, _ = baseline_samples(name)
        L2.append(L[b2]); L0.append(L[~b2])
    L2, L0 = np.concatenate(L2), np.concatenate(L0)
    bins = np.linspace(12.2, 12.7, 101)
    axs[0].hist(np.clip(L2, 12.2, 12.7), bins, histtype='stepfilled', alpha=.6, color='#1f77b4', density=True,
                label=f'both status 2 (n={len(L2)})')
    axs[0].hist(np.clip(L0, 12.2, 12.7), bins, histtype='step', color='#d62728', density=True, label=f'any status 0 (n={len(L0)})')
    axs[0].set_xlabel('horizontal master->rover distance [m] (clipped)'); axs[0].set_title('Baseline 12.44 m, MAD 1 cm when both RTK', fontsize=10)
    axs[0].legend(fontsize=8)
    gg = g[g.base_lon_med_straight.notna() & (g.base_len_mad_2 < 0.05)]
    for veh, c in (('30618', '#1f77b4'), ('30639', '#ff7f0e')):
        s = gg[gg.vehicle.astype(str) == veh]
        axs[1].scatter(s.base_lat_med_straight * 100, s.base_lon_med_straight, s=18, color=c, label=f'{veh} (n={len(s)})')
    axs[1].set_xlabel('lateral offset of rover (left +) [cm]'); axs[1].set_ylabel('longitudinal offset (ahead +) [m]')
    axs[1].set_title('Rover is 12.43-12.44 m AHEAD of master (per-bag medians, straight track)', fontsize=10)
    axs[1].legend(fontsize=8)
    t, L, dz, b2, sm, sr = baseline_samples('30618_27e994fc')
    axs[2].plot(t, L, '.', ms=1, color='k', label='baseline length')
    axs[2].plot(t, 12.0 + 0.2 * (sm == 2), '.', ms=1, color='#1f77b4', label='master status (12.2 = RTK/2, 12.0 = 0)')
    axs[2].set_ylim(11.8, 13.0); axs[2].set_xlabel('bag time [s]'); axs[2].set_title('30618_27e994fc: baseline breaks when master drops to status 0', fontsize=10)
    axs[2].legend(fontsize=8, markerscale=8)
    for a in axs:
        a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'antenna_baseline.png', dpi=85)


def fig_quality(name='30618_27e994fc'):
    bag = T.load_bag(name)
    ref = T.reference_trajectory(bag)
    t = ref.t - bag.t0
    tv, ve, vn = T._dedup_sorted(*(lambda v: (v.t_hdr, v.ve, v.vn))(T.gnss_vel(bag)))
    vx, vy = np.interp(ref.t, tv, ve), np.interp(ref.t, tv, vn)
    dt = np.diff(ref.t)
    innov = np.hypot(np.diff(ref.x) - 0.5 * (vx[1:] + vx[:-1]) * dt, np.diff(ref.y) - 0.5 * (vy[1:] + vy[:-1]) * dt)
    st = ref.status
    fig, axs = plt.subplots(3, 1, figsize=(15, 8.5), sharex=True)
    for s, c in ((2, '#1f77b4'), (0, '#d62728')):
        m = st[1:] == s
        axs[0].semilogy(t[1:][m], np.maximum(innov[m], 1e-4), '.', ms=1.5, color=c, label=f'status {s}')
        m2 = st == s
        axs[1].plot(t[m2], ref.z[m2], '.', ms=1.5, color=c, label=f'status {s}')
    axs[0].set_ylabel('|dp - v dt| per epoch [m]'); axs[0].legend(markerscale=8, fontsize=8)
    axs[0].set_title(f'{name}: fix innovations (position step vs GNSS velocity) and altitude, coloured by status', fontsize=10)
    axs[1].set_ylabel('ENU up [m]')
    axs[2].plot(t, ref.speed, lw=.6, color='k', label='GNSS vel |v| (master)')
    axs[2].plot(t, ref.speed_pos, lw=.6, color='#2ca02c', alpha=.7, label='central-difference |dp/dt|')
    axs[2].set_ylim(-0.5, 17); axs[2].set_ylabel('m/s'); axs[2].set_xlabel('bag time [s]'); axs[2].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'gnss_quality.png', dpi=80)


def fig_delays():
    d = pd.read_csv(OUT / 'delays_per_bag.csv')
    d = d[d['skip'].isna()] if 'skip' in d else d
    good = d.front_pos_hh_rms < 0.06
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    cols = [('front_vel_hh_lag', 'wheel hdr vs\nGNSS vel hdr', None), ('rear_vel_hh_lag', 'rear hdr vs\nvel hdr', None),
            ('front_vel_bb_lag', 'wheel bag vs\nvel bag', None), ('front_vel_hb_lag', 'wheel hdr vs\nvel bag', None),
            ('front_vel_bh_lag', 'wheel bag vs\nvel hdr', None), ('front_pos_hh_lag', 'wheel hdr vs\n|dp/dt| hdr', good),
            ('front_pos_bb_lag', 'wheel bag vs\n|dp/dt| bag', good), ('posderiv_vs_vel_lag', '|dp/dt| vs\nvel (hdr)', good)]
    rng = np.random.default_rng(0)
    for i, (c, lab, m) in enumerate(cols):
        v = d[c][m] if m is not None else d[c]
        v = v.dropna()
        axs[0].scatter(i + rng.uniform(-.18, .18, len(v)), v * 1e3, s=9, alpha=.7)
        axs[0].hlines(np.median(v) * 1e3, i - .3, i + .3, color='k', lw=2)
        axs[0].text(i, 150, f'{np.median(v)*1e3:+.1f}', ha='center', fontsize=9)
    axs[0].set_xticks(range(len(cols))); axs[0].set_xticklabels([c[1] for c in cols], fontsize=8)
    axs[0].set_ylim(-150, 170); axs[0].set_ylabel('lag [ms]  (+ = wheel/first signal late)')
    axs[0].set_title('Per-bag lags (black bar = median; |dp/dt| columns only for 44 good-fix bags)', fontsize=10)
    axs[0].axhline(0, color='k', lw=.6)
    # cost curve example
    bag = T.load_bag('30618_e2dcf65f')
    vm = T.gnss_vel(bag); w = T.wheel(bag, 'front')
    for base, col in (('hdr', '#1f77b4'), ('bag', '#d62728')):
        tr = vm.t_hdr if base == 'hdr' else vm.t_bag
        ts = w.t_hdr if base == 'hdr' else w.t_bag
        o = np.argsort(ts)
        okr = ~T.stale_mask(vm.t_bag, vm.t_hdr)
        r = T.estimate_lag(tr[okr], vm.speed[okr], ts[o], w.v[o], lo=-0.4, hi=0.4, coarse=0.005)
        lags, cost = r['curve']
        axs[1].plot(lags * 1e3, np.sqrt(cost), color=col, label=f'{base}-{base}: lag {r["lag"]*1e3:+.1f} ms, rms {r["rms"]:.3f} m/s')
    axs[1].set_xlabel('candidate lag [ms]'); axs[1].set_ylabel('RMS(v_gnss - k*v_wheel) [m/s]')
    axs[1].set_title('30618_e2dcf65f: LS cost vs lag (header stamps give lower residual)', fontsize=10)
    axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'delays.png', dpi=85)


def fig_transient(name='30618_e2dcf65f'):
    bag = T.load_bag(name)
    ref = T.reference_trajectory(bag)
    w = T.wheel(bag, 'front')
    vm = T.gnss_vel(bag)
    k = 1.0004
    # find a strong acceleration
    tv, sp = ref.vel_t, ref.vel_speed
    acc = np.gradient(sp, tv)
    i = int(np.argmax(np.convolve(acc, np.ones(20) / 20, 'same')))
    a, b = tv[i] - 4, tv[i] + 4
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.5))
    for ax, (lo, hi) in zip(axs, ((a, b), (tv[i] - 0.8, tv[i] + 0.8))):
        m = (ref.t > lo) & (ref.t < hi)
        ax.plot(ref.t[m] - tv[i], ref.speed_pos[m], 'o-', ms=3, color='#2ca02c', label='|dp/dt| of master fix (central diff., hdr)')
        mv = (vm.t_hdr > lo) & (vm.t_hdr < hi)
        ax.plot(vm.t_hdr[mv] - tv[i], vm.speed[mv], 's-', ms=3, color='k', label='GNSS vel |v| (hdr)')
        mw = (w.t_hdr > lo) & (w.t_hdr < hi)
        ax.plot(w.t_hdr[mw] - tv[i], w.v[mw] * k, '^-', ms=3, color='#1f77b4', label='front wheel (hdr stamp)')
        mb = (w.t_bag > lo) & (w.t_bag < hi)
        ax.plot(w.t_bag[mb] - tv[i], w.v[mb] * k, 'v--', ms=3, color='#9467bd', alpha=.7, label='front wheel (bag time)')
        ax.set_xlabel(f'time rel. to {tv[i]:.1f} [s]'); ax.set_ylabel('m/s'); ax.grid(alpha=.3)
    axs[0].legend(fontsize=8)
    axs[0].set_title(f'{name}: strongest acceleration - wheel(hdr) coincides with GNSS vel(hdr)', fontsize=10)
    axs[1].set_title('zoom: both are ~48 ms behind the position derivative; bag time adds 0-100 ms', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'transient_zoom.png', dpi=85)


if __name__ == '__main__':
    warnings.simplefilter('ignore', RuntimeWarning)
    fig_baseline()
    fig_quality()
    fig_delays()
    fig_transient()
