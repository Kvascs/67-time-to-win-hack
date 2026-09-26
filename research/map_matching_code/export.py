"""Export the GNSS-derived route-10 map (both directions) + landmark DB to CSV (frame-independent: WGS84)."""
import sys, pickle, os, json
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
import numpy as np
from scipy.ndimage import gaussian_filter1d
from common import TR, LAT0, LON0
import evaldr2 as E

OUT = 'C:/MosTransHack/research/map_matching_data/'
os.makedirs(OUT, exist_ok=True)
M = E.M
C = E.C
VP = pickle.load(open(E.SP + 'vprof.pkl', 'rb'))
meta = {'source': 'GNSS master antenna RTK (status==2) traces, 30618 runs, iterative lateral-median centerline, 1 m resampling',
        'local_tm': '+proj=tmerc +lat_0=%.2f +lon_0=%.2f +k=1 +x_0=0 +y_0=0 +ellps=WGS84' % (LAT0, LON0),
        'altitude': 'NavSatFix altitude (WGS84 ellipsoidal per ROS spec), median over runs, gaussian-smoothed',
        'antenna': 'polyline = path of /sensing/gnss/master antenna (rover is ~12.2 m ahead along the car axis)',
        'edges': {}}
for dirn in ('AB', 'BA'):
    P = M[dirn]['P']
    z = M[dirn]['z']
    n = len(P)
    s = np.arange(n).astype(float)
    lon, lat = TR.transform(P[:, 0], P[:, 1], direction='INVERSE')
    psi = np.unwrap(np.arctan2(*np.gradient(P, axis=0)[:, ::-1].T))  # ENU yaw (from +x=East, CCW)
    kap = gaussian_filter1d(np.gradient(gaussian_filter1d(psi, 2)), 2)
    grade = np.gradient(gaussian_filter1d(z, 10))
    # odometric chainage: wheel distance per map metre (30618 runs only)
    ratios = []
    for nm, c in C.items():
        if c['dir'] != dirn or not nm.startswith('30618'):
            continue
        ok = np.isfinite(c['Sg']) & (np.abs(np.nan_to_num(c['Eg'], nan=99)) < 1.0)
        t = c['t'][ok]
        sg = c['Sg'][ok]
        vw = 0.5 * (c['vf'] + c['vr']) / 3.6
        odo = np.r_[0, np.cumsum(0.5 * (vw[1:] + vw[:-1]) * np.diff(c['tw']))]
        od = np.interp(t, c['tw'], odo)
        grid = np.arange(0, n, 10.0)
        # first passage times of each 10 m mark
        mono = np.maximum.accumulate(sg)
        keep = np.r_[True, np.diff(mono) > 0]
        o_at = np.interp(grid, mono[keep], od[keep], left=np.nan, right=np.nan)
        r = np.diff(o_at) / 10.0
        ratios.append(r)
    R = np.array(ratios)
    rmed = np.nanmedian(np.where((R > 0.9) & (R < 1.1), R, np.nan), axis=0)
    rmed = np.where(np.isfinite(rmed), rmed, 1.0)
    rfull = np.interp(s, np.arange(len(rmed)) * 10 + 5, rmed)
    s_odo = np.r_[0, np.cumsum(rfull[:-1])]
    vp = VP[dirn]
    sb = vp['sb']
    cols = dict(s=s, lat=lat, lon=lon, h=z, x_tm=P[:, 0], y_tm=P[:, 1], yaw_enu=np.arctan2(np.sin(psi), np.cos(psi)),
                kappa=kap, grade=grade, s_odo=s_odo,
                v_p10_kmh=np.interp(s, sb, np.nan_to_num(vp['p10'])), v_p50_kmh=np.interp(s, sb, np.nan_to_num(vp['p50'])),
                v_p90_kmh=np.interp(s, sb, np.nan_to_num(vp['p90'])), v_max_kmh=np.interp(s, sb, np.nan_to_num(vp['cap'])))
    hdr = ','.join(cols.keys())
    arr = np.c_[tuple(cols.values())]
    np.savetxt(OUT + 'route10_%s_centerline.csv' % dirn, arr, delimiter=',', header=hdr, comments='',
               fmt=['%.1f', '%.8f', '%.8f', '%.3f', '%.3f', '%.3f', '%.5f', '%.6f', '%.5f', '%.2f', '%.1f', '%.1f', '%.1f', '%.1f'])
    D = E.dbs(dirn, 'none')
    with open(OUT + 'route10_%s_landmarks.csv' % dirn, 'w') as f:
        f.write('type,s,sigma_m,occurrence_frac\n')
        for typ, name in (('stop', 'stop'), ('up', 'zone_exit_accel_22kmh'), ('dn', 'zone_entry_decel_22kmh')):
            for row in D[typ]:
                f.write('%s,%.1f,%.2f,%.2f\n' % (name, row[0], row[1], row[2]))
    meta['edges'][dirn] = dict(length_m=float(s[-1]), s_odo_total=float(s_odo[-1]), n_landmarks={k: int(len(v)) for k, v in D.items()},
                               start_latlon=[float(lat[0]), float(lon[0])], end_latlon=[float(lat[-1]), float(lon[-1])],
                               h_range=[float(z.min()), float(z.max())], max_abs_grade=float(np.abs(grade).max()))
    print(dirn, 'length %.0f m, odometric length %.1f m (ratio %.5f)' % (s[-1], s_odo[-1], s_odo[-1] / s[-1]))
json.dump(meta, open(OUT + 'route10_map_meta.json', 'w'), indent=2, ensure_ascii=False)
print('exported to', OUT)
