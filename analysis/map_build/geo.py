"""Pure-numpy WGS84 geodesy: geodetic <-> ECEF <-> local ENU, and UTM (Krueger series).

No pyproj dependency so the same code can run inside the ROS 2 node. Accuracy:
  * geodetic<->ECEF<->ENU: exact closed form / iterative (sub-mm);
  * UTM forward/inverse: Krueger n-series to 6th order (Karney 2011), sub-mm inside a zone.
Verified against pyproj in selftest() (run `python geo.py`).
"""
from __future__ import annotations

import numpy as np

A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)
B = A * (1.0 - F)


# --------------------------------------------------------------------------- ECEF / ENU
def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h = np.asarray(h, dtype=float)
    sl, cl = np.sin(lat), np.cos(lat)
    n = A / np.sqrt(1.0 - E2 * sl * sl)
    x = (n + h) * cl * np.cos(lon)
    y = (n + h) * cl * np.sin(lon)
    z = (n * (1.0 - E2) + h) * sl
    return x, y, z


def ecef_to_geodetic(x, y, z):
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - E2))
    for _ in range(6):
        sl = np.sin(lat)
        n = A / np.sqrt(1.0 - E2 * sl * sl)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - E2 * n / (n + h)))
    sl = np.sin(lat)
    n = A / np.sqrt(1.0 - E2 * sl * sl)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def _enu_rot(lat0_deg, lon0_deg):
    la, lo = np.radians(lat0_deg), np.radians(lon0_deg)
    sla, cla, slo, clo = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array([[-slo, clo, 0.0],
                     [-sla * clo, -sla * slo, cla],
                     [cla * clo, cla * slo, sla]])


def geodetic_to_enu(lat, lon, h, lat0, lon0, h0):
    x, y, z = geodetic_to_ecef(lat, lon, h)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    R = _enu_rot(lat0, lon0)
    d = np.stack([np.asarray(x) - x0, np.asarray(y) - y0, np.asarray(z) - z0])
    e, n, u = np.tensordot(R, d, axes=1)
    return e, n, u


def enu_to_geodetic(e, n, u, lat0, lon0, h0):
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    R = _enu_rot(lat0, lon0)
    d = np.tensordot(R.T, np.stack([np.asarray(e, float), np.asarray(n, float), np.asarray(u, float)]), axes=1)
    return ecef_to_geodetic(d[0] + x0, d[1] + y0, d[2] + z0)


def enu_to_enu(e, n, u, origin_from, origin_to):
    """Exact re-expression of ENU coordinates in another ENU frame (origins are (lat, lon, h))."""
    lat, lon, h = enu_to_geodetic(e, n, u, *origin_from)
    return geodetic_to_enu(lat, lon, h, *origin_to)


# --------------------------------------------------------------------------- UTM (Krueger)
_K0 = 0.9996
_N = F / (2.0 - F)
_AA = A / (1.0 + _N) * (1.0 + _N ** 2 / 4.0 + _N ** 4 / 64.0 + _N ** 6 / 256.0)
_ALPHA = np.array([
    _N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16 + 41 * _N ** 4 / 180 - 127 * _N ** 5 / 288 + 7891 * _N ** 6 / 37800,
    13 * _N ** 2 / 48 - 3 * _N ** 3 / 5 + 557 * _N ** 4 / 1440 + 281 * _N ** 5 / 630 - 1983433 * _N ** 6 / 1935360,
    61 * _N ** 3 / 240 - 103 * _N ** 4 / 140 + 15061 * _N ** 5 / 26880 + 167603 * _N ** 6 / 181440,
    49561 * _N ** 4 / 161280 - 179 * _N ** 5 / 168 + 6601661 * _N ** 6 / 7257600,
    34729 * _N ** 5 / 80640 - 3418889 * _N ** 6 / 1995840,
    212378941 * _N ** 6 / 319334400,
])
_BETA = np.array([
    _N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96 - _N ** 4 / 360 - 81 * _N ** 5 / 512 + 96199 * _N ** 6 / 604800,
    _N ** 2 / 48 + _N ** 3 / 15 - 437 * _N ** 4 / 1440 + 46 * _N ** 5 / 105 - 1118711 * _N ** 6 / 3870720,
    17 * _N ** 3 / 480 - 37 * _N ** 4 / 840 - 209 * _N ** 5 / 4480 + 5569 * _N ** 6 / 90720,
    4397 * _N ** 4 / 161280 - 11 * _N ** 5 / 504 - 830251 * _N ** 6 / 7257600,
    4583 * _N ** 5 / 161280 - 108847 * _N ** 6 / 3991680,
    20648693 * _N ** 6 / 638668800,
])
_E = np.sqrt(E2)


def utm_zone_lon0(zone: int) -> float:
    return -183.0 + 6.0 * zone


def latlon_to_utm(lat_deg, lon_deg, zone: int = 37, north: bool = True):
    """Returns (easting, northing, convergence_rad, scale_factor)."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    dlon = np.radians(np.asarray(lon_deg, dtype=float) - utm_zone_lon0(zone))
    t = np.sinh(np.arctanh(np.sin(lat)) - _E * np.arctanh(_E * np.sin(lat)))
    xi_p = np.arctan2(t, np.cos(dlon))
    eta_p = np.arctanh(np.sin(dlon) / np.sqrt(1.0 + t * t))
    xi, eta = xi_p.copy(), eta_p.copy()
    p, q = 1.0, 0.0
    for j in range(1, 7):
        a = _ALPHA[j - 1]
        xi = xi + a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta = eta + a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
        p = p + 2 * j * a * np.cos(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        q = q + 2 * j * a * np.sin(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    e = 500000.0 + _K0 * _AA * eta
    n = _K0 * _AA * xi + (0.0 if north else 10000000.0)
    # meridian convergence (grid north relative to true north, CCW-positive) and point scale,
    # Karney (2011) eqs. (11)-(12) and (26)-(27)
    gamma = np.arctan(t * np.tan(dlon) / np.sqrt(1 + t * t)) + np.arctan2(q, p)
    sl = np.sin(lat)
    k = (_K0 * _AA / A * np.sqrt(1 - E2 * sl * sl) / np.cos(lat)
         / np.sqrt(t * t + np.cos(dlon) ** 2) * np.hypot(p, q))
    return e, n, gamma, k


def utm_to_latlon(easting, northing, zone: int = 37, north: bool = True):
    xi = (np.asarray(northing, float) - (0.0 if north else 10000000.0)) / (_K0 * _AA)
    eta = (np.asarray(easting, float) - 500000.0) / (_K0 * _AA)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j in range(1, 7):
        b = _BETA[j - 1]
        xi_p = xi_p - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    tau_p = np.sin(xi_p) / np.sqrt(np.sinh(eta_p) ** 2 + np.cos(xi_p) ** 2)
    dlon = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    tau = tau_p.copy()
    for _ in range(6):  # Newton for tau from tau'
        sig = np.sinh(_E * np.arctanh(_E * tau / np.sqrt(1 + tau * tau)))
        tp_i = tau * np.sqrt(1 + sig * sig) - sig * np.sqrt(1 + tau * tau)
        dt = (tau_p - tp_i) / np.sqrt(1 + tp_i * tp_i) * (1 + (1 - E2) * tau * tau) / ((1 - E2) * np.sqrt(1 + tau * tau))
        tau = tau + dt
    lat = np.degrees(np.arctan(tau))
    lon = np.degrees(dlon) + utm_zone_lon0(zone)
    return lat, lon


# --------------------------------------------------------------------------- self test
def selftest():
    from pyproj import Transformer
    rng = np.random.default_rng(0)
    lat = 55.80 + rng.uniform(-0.05, 0.05, 1000)
    lon = 37.45 + rng.uniform(-0.1, 0.1, 1000)
    h = 150 + rng.uniform(-20, 20, 1000)
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32637', always_xy=True)
    E_ref, N_ref = tr.transform(lon, lat)
    E, N, gam, k = latlon_to_utm(lat, lon)
    print('UTM fwd max err [m]:', np.max(np.abs(E - E_ref)), np.max(np.abs(N - N_ref)))
    la2, lo2 = utm_to_latlon(E, N)
    print('UTM inv max err [deg]:', np.max(np.abs(la2 - lat)), np.max(np.abs(lo2 - lon)))
    # convergence / scale check vs pyproj factors
    from pyproj import Proj
    fac = Proj('EPSG:32637').get_factors(lon[:5], lat[:5])
    print('scale  ours', k[:3], 'pyproj', np.array(fac.parallel_scale[:3]))
    print('conv   ours(deg)', np.degrees(gam[:3]), 'pyproj', np.array(fac.meridian_convergence[:3]))
    tr3 = Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True)
    X_ref, Y_ref, Z_ref = tr3.transform(lon, lat, h)
    X, Y, Z = geodetic_to_ecef(lat, lon, h)
    print('ECEF max err [m]:', np.max(np.abs(X - X_ref)), np.max(np.abs(Y - Y_ref)), np.max(np.abs(Z - Z_ref)))
    la3, lo3, h3 = ecef_to_geodetic(X, Y, Z)
    print('ECEF inv max err:', np.max(np.abs(la3 - lat)), np.max(np.abs(lo3 - lon)), np.max(np.abs(h3 - h)))
    e, n, u = geodetic_to_enu(lat, lon, h, 55.8, 37.45, 150.0)
    la4, lo4, h4 = enu_to_geodetic(e, n, u, 55.8, 37.45, 150.0)
    print('ENU roundtrip max err:', np.max(np.abs(la4 - lat)), np.max(np.abs(lo4 - lon)), np.max(np.abs(h4 - h)))


if __name__ == '__main__':
    selftest()
