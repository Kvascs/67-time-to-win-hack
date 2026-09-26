"""How much do candidate 'local metric frames' differ on our route?  (competitions_prior_art.md, section 2.3)

Compares, for the master-antenna track of a bag (origin = first valid fix):
  * ENU tangent plane at the origin (exact, WGS84)            -- REP-103 default, GeographicLib::LocalCartesian
  * UTM zone 37N minus UTM(origin)                             -- Autoware LocalCartesianUTM / robot_localization UTM
  * 'flat earth' (dlon*R_N*cos(lat0), dlat*R_M)                 -- typical quick script
  * z: ENU-up vs (alt - alt0)
Run: python frames_check.py [npz ...]   (defaults to two bags from data/npz)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'analysis', 'map_build'))
import geo  # noqa: E402

A, F = 6378137.0, 1 / 298.257223563
E2 = F * (2 - F)


def analyze(path):
    d = np.load(path)
    fx = d['sensing__gnss__master__fix']
    lat, lon, alt = fx[:, 2], fx[:, 3], fx[:, 4]
    m = np.isfinite(lat) & (lat > 50)
    lat, lon, alt = lat[m], lon[m], alt[m]
    lat0, lon0, h0 = lat[0], lon[0], alt[0]
    e, n, u = geo.geodetic_to_enu(lat, lon, alt, lat0, lon0, h0)
    E, N, _, _ = geo.latlon_to_utm(lat, lon, 37, True)
    E0, N0, g0, k0 = geo.latlon_to_utm(np.array([lat0]), np.array([lon0]), 37, True)
    dx, dy = E - E0[0], N - N0[0]
    r = np.hypot(e, n)
    # best similarity UTM-rel = s * R * ENU  (to show it is just grid convergence + scale)
    Amat, Bmat = np.vstack([e, n]).T, np.vstack([dx, dy]).T
    U, S, Vt = np.linalg.svd(Amat.T @ Bmat)
    R = (U @ Vt).T
    ang = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    s = S.sum() / (Amat ** 2).sum()
    s0 = np.sin(np.radians(lat0))
    RN = A / np.sqrt(1 - E2 * s0 ** 2)
    RM = A * (1 - E2) / (1 - E2 * s0 ** 2) ** 1.5
    xe = np.radians(lon - lon0) * RN * np.cos(np.radians(lat0))
    ye = np.radians(lat - lat0) * RM
    print(os.path.basename(path), 'origin %.5f %.5f h0=%.1f' % (lat0, lon0, h0))
    print('  max distance from origin            : %.0f m' % r.max())
    print('  ENU vs UTM-relative, max |diff|     : %.1f m  (rotation %.3f deg = grid convergence %.3f deg, scale %.6f, k0=%.6f)'
          % (np.hypot(dx - e, dy - n).max(), ang, np.degrees(g0[0]), s, k0[0]))
    print('  ENU vs flat-earth, max |diff|       : %.2f m' % np.hypot(xe - e, ye - n).max())
    print('  ENU-up vs (alt-alt0), max |diff|    : %.2f m  (d^2/2R = %.2f m)' % (np.abs(u - (alt - h0)).max(), r.max() ** 2 / 2 / 6371000))
    print('  altitude range in run              : %.1f .. %.1f m' % (alt.min(), alt.max()))


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'npz')
    paths = sys.argv[1:] or [os.path.join(root, '30618_0652866c.npz'), os.path.join(root, '30639_d601d28f.npz')]
    for p in paths:
        analyze(p)
