"""Self-test of the runtime module track_map.py against pyproj / geo.py and for internal consistency.
python selftest_track_map.py [map_dir]"""
import sys
import time

import numpy as np

import data_io as D
import geo
import track_map as TM


def main(map_dir='map'):
    tm = TM.TrackMap(D.OUT / map_dir)
    ok = True
    # 1. frames
    E = tm.edges['main']
    lat, lon, h = tm.map_to_geodetic(E.x[::97], E.y[::97], E.z[::97])
    x2, y2, z2 = tm.geodetic_to_map(lat, lon, h)
    err = np.max(np.abs(np.r_[x2 - E.x[::97], y2 - E.y[::97], z2 - E.z[::97]]))
    print(f'[frames] map->geodetic->map roundtrip max err {err:.2e} m'); ok &= err < 1e-6
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32637', always_xy=True)
    Er, Nr = tr.transform(lon, lat)
    Eu, Nu, hu = tm.map_to_frame(E.x[::97], E.y[::97], E.z[::97], 'utm')
    err = max(np.max(np.abs(Eu - Er)), np.max(np.abs(Nu - Nr)))
    print(f'[frames] UTM vs pyproj max err {err:.2e} m'); ok &= err < 1e-6
    csv = TM._read_csv(D.OUT / map_dir / 'edge_main.csv')
    err = max(np.max(np.abs(csv['utm_e'][::97] - Er)), np.max(np.abs(csv['utm_n'][::97] - Nr)))
    print(f'[frames] CSV utm columns vs pyproj max err {err:.2e} m'); ok &= err < 1e-3
    lat2, lon2, h2 = geo.enu_to_geodetic(E.x[::97], E.y[::97], E.z[::97], *D.MAP_ORIGIN)
    err = max(np.max(np.abs(lat2 - lat)), np.max(np.abs(lon2 - lon)))
    print(f'[frames] vs geo.py max err {err:.2e} deg'); ok &= err < 1e-10
    o2 = (lat[5], lon[5], h[5])
    xe, ye, ze = tm.map_to_frame(E.x[::97], E.y[::97], E.z[::97], 'enu', o2)
    print(f'[frames] ENU at another origin: point 5 -> ({xe[5]:.2e}, {ye[5]:.2e}, {ze[5]:.2e}) (should be 0)')
    # 2. projection / pose consistency on every edge
    rng = np.random.default_rng(1)
    for eid, Ed in tm.edges.items():
        ss = rng.uniform(0.5, Ed.length - 0.5, 200)
        x, y, z, yaw = Ed.pose(ss)
        s2, d2, dist, _ = Ed.project(x, y, yaw)
        e1 = np.max(np.abs(s2 - ss)); e2 = np.max(np.abs(d2))
        # offset laterally by 1 m -> d = +1
        s3, d3, _, _ = Ed.project(x - np.sin(yaw), y + np.cos(yaw), yaw)
        e3 = np.max(np.abs(d3 - 1.0))
        print(f'[proj] {eid:15s} L={Ed.length:9.2f}  |s err| {e1:.2e}  |d| {e2:.2e}  left-offset err {e3:.2e}')
        ok &= e1 < 1e-3 and e2 < 1e-3 and e3 < 0.02
    # 3. routes: continuity through branch merges and wrap-around
    for eid, Ed in tm.edges.items():
        if Ed.closed:
            continue
        rt = tm.route(eid, 0.0, 400.0)
        rr = np.arange(0, rt.length, 0.5)
        P = np.array([rt.pose(r)[:2] for r in rr])
        step = np.hypot(*np.diff(P, axis=0).T)
        print(f'[route] from {eid:15s}: pieces={[(p[1], round(p[2], 1), round(p[3], 1)) for p in rt.pieces]} '
              f'max step {step.max():.3f} m (0.5 m sampling)')
        ok &= step.max() < 0.6
    rt = tm.route('main', E.length - 50.0, 200.0)
    P = np.array([rt.pose(r)[:2] for r in np.arange(0, 200, 0.5)])
    print(f'[route] wrap-around main: max step {np.hypot(*np.diff(P, axis=0).T).max():.3f} m')
    # 4. timing
    t0 = time.perf_counter()
    for _ in range(2000):
        rt.pose(123.4)
    t1 = time.perf_counter()
    for _ in range(200):
        tm.locate(1000.0, 250.0, 3.0)
    t2 = time.perf_counter()
    print(f'[timing] route.pose {1e6 * (t1 - t0) / 2000:.1f} us/call; tm.locate (all edges) {1e3 * (t2 - t1) / 200:.2f} ms/call')
    # 5. init from real geodetic GNSS (first 2 s of a val run)
    b = D.split('val')[2]
    g = D.gnss_fix(b, 'master'); gr = D.gnss_fix(b, 'rover'); v = D.gnss_vel(b)
    m = g.th <= g.th[0] + 2.0
    mr = gr.th <= g.th[0] + 2.0
    mv = v[:, 1] <= g.th[0] + 2.0
    res = tm.init_from_gnss(g.th[m], g.lat[m], g.lon[m], g.alt[m], rover=(gr.lat[mr], gr.lon[mr], gr.alt[mr]),
                            vel=(v[mv, 2], v[mv, 3]), status=g.status[m])
    print(f'[init] {b}: {res}')
    print('SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else 'map') else 1)
