"""Generate geodesy test vectors (pyproj) for the C++ core unit tests."""
from pyproj import Transformer, CRS
import numpy as np

pts = [
    (55.8100, 37.4620, 168.4),   # route east terminal area
    (55.8000, 37.3900, 174.3),   # route west terminal area
    (55.8050, 37.4200, 170.0),
    (55.7558, 37.6173, 150.0),   # Moscow centre
    (60.0000, 30.0000, 10.0),    # other zone (36)
    (-33.9, 18.4, 5.0),          # southern hemisphere zone 34
]
to_utm = {}
print('// lat, lon, h, zone, E, N, X, Y, Z')
for lat, lon, h in pts:
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    epsg = (32700 if south else 32600) + zone
    t = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    E, N = t.transform(lon, lat)
    ecef = Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True)
    X, Y, Z = ecef.transform(lon, lat, h)
    print(f'{{{lat:.10f}, {lon:.10f}, {h:.4f}, {zone}, {E:.6f}, {N:.6f}, {X:.6f}, {Y:.6f}, {Z:.6f}}},')

# ENU of point B relative to origin A via pyproj topocentric
A = (55.8100, 37.4620, 168.4)
B = (55.8000, 37.3900, 174.3)
enu = Transformer.from_pipeline(
    f'+proj=pipeline +step +proj=cart +ellps=WGS84 +step +proj=topocentric +ellps=WGS84 '
    f'+lat_0={A[0]} +lon_0={A[1]} +h_0={A[2]}')
e, n, u = enu.transform(B[1], B[0], B[2])
print(f'// ENU of B wrt A: {e:.6f}, {n:.6f}, {u:.6f}')
