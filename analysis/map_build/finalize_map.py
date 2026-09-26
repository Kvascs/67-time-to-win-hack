"""Merge stop landmarks, sections, antenna offsets and calibration hints into <map_dir>/track_map.json.

Run after build_map.py and stops_analysis.py:  python finalize_map.py map   (or map_train)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import data_io as D

OUT = D.OUT


def sections_from_double_track(L, dbl):
    """Named s-ranges of 'main' (tram travels in increasing s, s=0 at the east platform)."""
    (a1, b1), (a2, b2) = sorted(dbl, key=lambda r: r[0])[:2]
    return [
        dict(name='east_loop_departure', s0=0.0, s1=a1, note='east platform -> east terminal loop -> WB start'),
        dict(name='wb_main', s0=a1, s1=b1, note='westbound double-track main line (E -> W)'),
        dict(name='west_yard_loop', s0=b1, s1=a2, note='west junction -> arrival stop -> west loop -> fan track F1'),
        dict(name='eb_main', s0=a2, s1=b2, note='eastbound double-track main line (W -> E)'),
        dict(name='east_loop_arrival', s0=b2, s1=L, note='east approach -> east terminal platform (s = L = 0)'),
    ]


def main(map_dir='map'):
    md = OUT / map_dir
    meta = json.loads((md / 'track_map.json').read_text(encoding='utf-8'))
    L = [e for e in meta['edges'] if e['id'] == 'main'][0]['length']
    dbl = [e for e in meta['edges'] if e['id'] == 'main'][0]['double_track_ranges']
    meta['sections'] = sections_from_double_track(L, dbl)
    cl = pd.read_csv(md / 'stops.csv')
    stops = []
    for _, c in cl[cl.n_runs >= 2].sort_values(['edge', 's_median']).iterrows():
        lm = bool(c.n_runs >= 3 and c.s_robust_std <= 1.0)
        stops.append(dict(id=f"{c.edge}:{c.s_median:.1f}", site=int(c.site), edge=c.edge, s=round(float(c.s_median), 2),
                          x=round(float(c.x), 3), y=round(float(c.y), 3), z=round(float(c.z), 3), cls=c.cls,
                          direction=c.dir, landmark=lm, s_robust_std=round(float(c.s_robust_std), 3),
                          s_std=round(float(c.s_std), 3), s_p10=round(float(c.s_p10), 2), s_p90=round(float(c.s_p90), 2),
                          n_runs=int(c.n_runs), n_stops=int(c.n_stops),
                          n_passes=None if pd.isna(c.n_passes) else int(c.n_passes),
                          p_stop=None if pd.isna(c.p_stop) else round(float(c.p_stop), 3),
                          dwell_med=round(float(c.dwell_med), 1), dwell_p10=round(float(c.dwell_p10), 1),
                          dwell_p90=round(float(c.dwell_p90), 1), frac_wheel_detected=round(float(c.frac_wheel_detected), 3)))
    meta['stops'] = stops
    meta['stops_note'] = ('stop = GNSS speed < 0.1 m/s for >= 5 s; s = median master-antenna position of the cluster; '
                          'cls: platform (P(stop|pass)>=0.5, robust std<=3 m, dwell>=8 s), signal (repeated, >=3 runs), '
                          'terminal (run start/end layover), random. landmark = n_runs>=3 and robust std<=1 m -> usable '
                          'as along-track fix when the wheels report standstill near it.')
    meta['antenna'] = dict(reference='master', rover_ahead_m=12.44, rover_ahead_tight_curves_m=12.55,
                           rover_lateral_m=-0.03,
                           note='rover path = master path shifted by +12.44 m in s (lateral |offset| < 5 cm median); '
                                'rover pose ~ pose(s + 12.44); midpoint ~ pose(s + 6.22)')
    wf = OUT / 'cache' / 'wheel_curv_fit.json'
    if wf.exists():
        W = json.loads(wf.read_text())
        a = W['all_cruise']
        meta['wheel_calibration_hint'] = dict(
            model='ds_map = ds_wheel * (1 + c0 + c_abs*|k(s)| + c_signed*k(s)), ds_wheel = wheel km/h / 3.6 * dt',
            front=a['dwf'], rear=a['dwr'], doppler_check=a['dd'],
            per_run_straight_scale=W['per_run_straight_scale'],
            data=f"cruising 10 s windows (|a|<0.15 m/s^2), RTK fixes on main, train+val: {W['n_cruise']} windows, "
                 f"{W['km_cruise']:.1f} km")
    vr = OUT / 'cache' / 'val_results.json'
    ir = OUT / 'cache' / 'init_results.json'
    meta['validation'] = dict(
        note='val runs vs train-only map (map_train); see REPORT.md',
        cross_track=json.loads(vr.read_text())['cross_track'] if vr.exists() else None,
        along_track=json.loads(vr.read_text())['along_track'] if vr.exists() else None,
        init=json.loads(ir.read_text()) if ir.exists() else None,
        drift_sim=json.loads((OUT / 'cache' / 'drift_sim.json').read_text())['summary']
        if (OUT / 'cache' / 'drift_sim.json').exists() else None)
    (md / 'track_map.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    write_geojson(md, meta)
    print(f'{map_dir}: {len(stops)} stops ({sum(s["landmark"] for s in stops)} landmarks), sections:',
          [(s['name'], round(s['s0'], 1), round(s['s1'], 1)) for s in meta['sections']])


def write_geojson(md, meta, step=5):
    """Edges (LineString, lon/lat/h every ~2.5 m) and stops (Point) as GeoJSON for QGIS / geojson.io."""
    feats = []
    for em in meta['edges']:
        c = pd.read_csv(md / em['file'])
        c = c.iloc[::step]
        coords = [[round(lo, 8), round(la, 8), round(h, 3)] for lo, la, h in zip(c.lon, c.lat, c.h)]
        if em['closed']:
            coords.append(coords[0])
        props = {k: v for k, v in em.items() if k in ('id', 'length', 'closed', 'n_runs')}
        props['from'] = json.dumps(em.get('from'))
        props['to'] = json.dumps(em.get('to'))
        feats.append(dict(type='Feature', geometry=dict(type='LineString', coordinates=coords), properties=props))
    import geo
    for st in meta.get('stops', []):
        la, lo, h = geo.enu_to_geodetic(st['x'], st['y'], st['z'], *D.MAP_ORIGIN)
        feats.append(dict(type='Feature', geometry=dict(type='Point', coordinates=[round(float(lo), 8), round(float(la), 8)]),
                          properties={k: st[k] for k in ('id', 'edge', 's', 'cls', 'landmark', 's_robust_std', 'n_runs', 'p_stop', 'dwell_med')}))
    (md / 'track_map.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=feats)), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'map')
