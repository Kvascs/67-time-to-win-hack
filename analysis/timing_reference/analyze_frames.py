"""Part 3: which local metric frame does the judge use, and what does a wrong guess cost (x/y/z)?

Reference hypothesis F0 = exact WGS-84 ENU tangent plane at the first master fix (by header stamp).
Alternatives are evaluated on the *same* master fixes; the table gives the component differences
(alternative - F0) that a solution would suffer if it used the alternative while the judge uses F0
(or vice versa - the numbers are symmetric).

Also: altitude behaviour (profile range, stand-still noise, jumps, reproducibility of a route altitude map).

Outputs: frames_per_bag.csv, frames_summary.csv, frames_error_vs_distance.png, altitude.png
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
from pyproj import Transformer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing as T  # noqa: E402

OUT = Path(__file__).resolve().parent
R_SPHERE = 6371000.0
TR_UTM = Transformer.from_crs('EPSG:4326', 'EPSG:32637', always_xy=True)
TR_WM = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)


def radii(lat_deg):
    s = np.sin(np.radians(lat_deg))
    w = np.sqrt(1 - T.WGS84_E2 * s * s)
    N = T.WGS84_A / w
    M = T.WGS84_A * (1 - T.WGS84_E2) / w ** 3
    return N, M


def alt_frames(lat, lon, alt, lat0, lon0, h0, heading0, rover_xyz=None):
    """Dict name -> (x, y, z) arrays for every alternative frame."""
    e, n, u = T.llh_to_enu(lat, lon, alt, lat0, lon0, h0)
    out = {'ENU (judge hyp.)': (e, n, u)}
    X, Y = TR_UTM.transform(lon, lat)
    X0, Y0 = TR_UTM.transform(lon0, lat0)
    dz = alt - h0
    out['UTM37N minus origin'] = (X - X0, Y - Y0, dz)
    N0, M0 = radii(lat0)
    dlam, dphi = np.radians(lon - lon0), np.radians(lat - lat0)
    out['equirect. WGS84 radii'] = (N0 * np.cos(np.radians(lat0)) * dlam, M0 * dphi, dz)
    out['equirect. sphere R=6371km'] = (R_SPHERE * np.cos(np.radians(lat0)) * dlam, R_SPHERE * dphi, dz)
    out['ENU, z = alt - alt0'] = (e, n, dz)
    Xw, Yw = TR_WM.transform(lon, lat)
    Xw0, Yw0 = TR_WM.transform(lon0, lat0)
    out['WebMercator minus origin'] = (Xw - Xw0, Yw - Yw0, dz)
    c, s = np.cos(-heading0), np.sin(-heading0)
    out['odom (x = initial heading)'] = (c * e - s * n, s * e + c * n, u)
    if rover_xyz is not None:
        out['rover antenna as reference'] = rover_xyz
    return out


def analyze_bag(name, good_pos: bool):
    bag = T.load_bag(name)
    ref = T.reference_trajectory(bag)
    f = T.gnss_fix(bag)
    t, lat, lon, alt, st = T._dedup_sorted(f.t_hdr, f.lat, f.lon, f.alt, f.status)
    lat0, lon0, h0 = ref.origin
    # initial heading: dual-antenna baseline over the first stationary seconds, else first course
    fr = T.gnss_fix(bag, 'rover')
    tr, latr, lonr, altr = T._dedup_sorted(fr.t_hdr, fr.lat, fr.lon, fr.alt)
    er, nr, ur = T.llh_to_enu(latr, lonr, altr, lat0, lon0, h0)
    em, nm_, um = T.llh_to_enu(lat, lon, alt, lat0, lon0, h0)
    k = np.clip(np.searchsorted(tr, t[:30]), 0, len(tr) - 1)
    heading0 = float(np.arctan2(np.median(nr[k] - nm_[:30]), np.median(er[k] - em[:30])))
    # rover antenna trajectory at master stamps (for the 'wrong antenna' case)
    rover_xyz = (np.interp(t, tr, er), np.interp(t, tr, nr), np.interp(t, tr, ur))
    frames = alt_frames(lat, lon, alt, lat0, lon0, h0, heading0, rover_xyz)
    x0, y0, z0 = frames['ENU (judge hyp.)']
    dist = np.hypot(x0, y0)
    rows = []
    for fname, (x, y, z) in frames.items():
        ex, ey, ez = x - x0, y - y0, z - z0
        e3 = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
        rows.append(dict(bag=name, frame=fname, good_pos=good_pos, max_dist=float(dist.max()),
                         rms_x=float(np.sqrt(np.mean(ex ** 2))), rms_y=float(np.sqrt(np.mean(ey ** 2))),
                         rms_z=float(np.sqrt(np.mean(ez ** 2))), rms_3d=float(np.sqrt(np.mean(e3 ** 2))),
                         max_x=float(np.max(np.abs(ex))), max_y=float(np.max(np.abs(ey))),
                         max_z=float(np.max(np.abs(ez))), max_3d=float(e3.max()),
                         end_3d=float(e3[-1]), end_dist=float(dist[-1])))
    # altitude
    alt_info = dict(bag=name, z_rms=float(np.sqrt(np.mean(z0 ** 2))), z_min=float(z0.min()), z_max=float(z0.max()),
                    z_end=float(z0[-1]), curv_drop_at_max=float(-(dist.max() ** 2) / (2 * 6.371e6)))
    return rows, alt_info, (dist, frames), (lat, lon, alt, st)


def altitude_map_test(samples, good_bags):
    """Leave-half-out test: median altitude of status-2 fixes in 2x2 m UTM cells, evaluated on the other half."""
    names = sorted(good_bags)
    train, test = names[::2], names[1::2]

    def cells(lat, lon):
        X, Y = TR_UTM.transform(lon, lat)
        return np.floor(X / 2).astype(np.int64) * 10_000_000 + np.floor(Y / 2).astype(np.int64)

    keys, alts = [], []
    for b in train:
        lat, lon, alt, st = samples[b]
        m = st == 2
        keys.append(cells(lat[m], lon[m])); alts.append(alt[m])
    keys, alts = np.concatenate(keys), np.concatenate(alts)
    o = np.argsort(keys)
    keys, alts = keys[o], alts[o]
    uk, idx = np.unique(keys, return_index=True)
    med = np.array([np.median(a) for a in np.split(alts, idx[1:])])
    res = {}
    # per-run vertical offset vs the map and within-run spread after removing it
    per = []
    for b in test:
        lat, lon, alt, st = samples[b]
        m = st == 2
        if m.sum() < 500:
            continue
        kk = cells(lat[m], lon[m])
        j = np.clip(np.searchsorted(uk, kk), 0, len(uk) - 1)
        hit = uk[j] == kk
        r = alt[m][hit] - med[j[hit]]
        off = np.median(r)
        per.append(dict(bag=b, offset=off, within_mad=1.4826 * np.median(np.abs(r - off)),
                        within_p95=np.percentile(np.abs(r - off), 95)))
    per = pd.DataFrame(per)
    per.to_csv(OUT / 'altitude_run_offsets.csv', index=False, float_format='%.4g')
    res['run_offset'] = dict(n=len(per), abs_offset_median=float(per.offset.abs().median()),
                             abs_offset_p90=float(per.offset.abs().quantile(.9)),
                             within_mad_median=float(per.within_mad.median()),
                             within_p95_median=float(per.within_p95.median()))
    for lab, stat in (('status2', 2), ('status0', 0)):
        r = []
        for b in test:
            lat, lon, alt, st = samples[b]
            m = st == stat
            if not m.any():
                continue
            kk = cells(lat[m], lon[m])
            j = np.searchsorted(uk, kk)
            j = np.clip(j, 0, len(uk) - 1)
            hit = uk[j] == kk
            r.append(alt[m][hit] - med[j[hit]])
        r = np.concatenate(r) if r else np.array([np.nan])
        res[lab] = dict(n=len(r), rms=float(np.sqrt(np.nanmean(r ** 2))), med_abs=float(np.nanmedian(np.abs(r))),
                        p95_abs=float(np.nanpercentile(np.abs(r), 95)), bias=float(np.nanmedian(r)))
    return res


def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    d = pd.read_csv(OUT / 'delays_per_bag.csv')
    good = set(d[(d['front_pos_hh_rms'] < 0.06)]['bag'])
    g = pd.read_csv(OUT / 'gnss_per_bag.csv')
    bags = g[(g['skip'].isna()) & (g.duration > 600)]['bag'].tolist()
    rows, alts, curves, samples = [], [], {}, {}
    for b in bags:
        r, a, c, smp = analyze_bag(b, b in good)
        rows += r; alts.append(a); curves[b] = c; samples[b] = smp
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'frames_per_bag.csv', index=False, float_format='%.4g')
    gp = df[df.good_pos]
    summ = gp.groupby('frame')[['max_dist', 'rms_x', 'rms_y', 'rms_z', 'rms_3d', 'max_x', 'max_y', 'max_z', 'max_3d']].median()
    summ = summ.sort_values('rms_3d')
    summ.to_csv(OUT / 'frames_summary.csv', float_format='%.4g')
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(f'median over {gp.bag.nunique()} good-position bags (errors in metres vs exact ENU at first master fix):')
    print(summ.round(3).to_string())
    al = pd.DataFrame(alts)
    al.to_csv(OUT / 'altitude_per_bag.csv', index=False, float_format='%.4g')
    print('\naltitude (ENU up, relative to first fix): ')
    print(al[al.bag.isin(good)].describe().round(3).to_string())
    mt = altitude_map_test(samples, good & set(samples))
    print('\naltitude-map leave-half-out test (alt - map median in 2x2 m cells):', mt)
    pd.DataFrame(mt).T.to_csv(OUT / 'altitude_map_test.csv', float_format='%.4g')
    # utm factors along the route
    for lat, lon in ((55.8104, 37.4623), (55.8127, 37.5360)):
        print('UTM convergence / scale at', lat, lon, T.utm_convergence_scale(lat, lon))
    plot_curves(curves, good)
    plot_altitude(samples, good)


def plot_curves(curves, good):
    names = [n for n in curves if n in good][:12]
    good = good & set(curves)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    show = ['UTM37N minus origin', 'equirect. sphere R=6371km', 'equirect. WGS84 radii', 'ENU, z = alt - alt0']
    cols = dict(zip(show, ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']))
    for b in names:
        dist, frames = curves[b]
        x0, y0, z0 = frames['ENU (judge hyp.)']
        o = np.argsort(dist)
        for fname in show:
            x, y, z = frames[fname]
            axs[0].plot(dist[o], np.hypot(x - x0, y - y0)[o], '.', ms=0.5, color=cols[fname], label=fname if b == names[0] else None)
            axs[1].plot(dist[o], np.abs(z - z0)[o], '.', ms=0.5, color=cols[fname], label=fname if b == names[0] else None)
    axs[0].set_yscale('log'); axs[0].set_ylim(1e-3, 300)
    axs[0].set_xlabel('horizontal distance from origin [m]'); axs[0].set_ylabel('|horizontal difference| vs exact ENU [m]')
    axs[0].set_title('Horizontal: UTM (grid convergence 1.2-1.3 deg) is catastrophic', fontsize=10)
    axs[1].set_xlabel('horizontal distance from origin [m]'); axs[1].set_ylabel('|z difference| vs ENU up [m]')
    axs[1].set_title('Vertical: "alt - alt0" differs from ENU up by d^2/2R', fontsize=10)
    for a in axs[:2]:
        a.grid(alpha=.3); a.legend(markerscale=20, fontsize=8)
    # z profile of the reference itself (what z=0 output would cost)
    for b in names:
        dist, frames = curves[b]
        z0 = frames['ENU (judge hyp.)'][2]
        axs[2].plot(frames['ENU (judge hyp.)'][0], z0, lw=0.6)
    axs[2].set_xlabel('ENU east [m]'); axs[2].set_ylabel('ENU up [m]')
    axs[2].set_title('Reference z along the route (12 runs): outputting z=0 costs ~10 m RMS', fontsize=10)
    axs[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / 'frames_error_vs_distance.png', dpi=85)


def plot_altitude(samples, good):
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    for i, b in enumerate(sorted(good & set(samples))[:20]):
        lat, lon, alt, st = samples[b]
        X, _ = TR_UTM.transform(lon, lat)
        ax.plot(X[st == 2], alt[st == 2], '.', ms=0.4, color='#1f77b4', label='status 2' if i == 0 else None)
        ax.plot(X[st == 0], alt[st == 0], '.', ms=0.8, color='#d62728', label='status 0' if i == 0 else None)
    ax.set_xlabel('UTM easting [m]'); ax.set_ylabel('NavSatFix altitude [m]')
    ax.set_title('Altitude vs easting, 20 runs: consistent profile (~30 m range); status-0 fixes scatter by metres')
    ax.grid(alpha=.3); ax.legend(markerscale=15)
    fig.tight_layout()
    fig.savefig(OUT / 'altitude.png', dpi=85)


if __name__ == '__main__':
    main()
