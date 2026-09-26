"""Runtime track map (pathgraph) for the tram backup-odometry node.  Pure numpy -- no scipy/pyproj.

Map = directed edges (polylines with arc length s):
  * 'main'  : closed cycle through both terminal loops, s in [0, L) increasing in the travel direction
              (the tram is unidirectional; s=0 is the east terminal platform).
  * branches: open polylines (alternative tracks: yard/fan tracks, detours). A branch may start/end on
              'main' (from_s / to_s) or in the yard (None = end of observed data).

Coordinates: MAP ENU (x east, y north, z up, metres) = local tangent plane at MAP origin (lat0, lon0, h0)
of WGS84. Horizontal distances in it are true ground distances (<1e-6 rel.), so s is directly
comparable with wheel odometry.  Conversions to geodetic / UTM 37N / ENU at any other origin are exact.

Typical use in the estimator:
    tm = TrackMap('path/to/map')                       # directory with track_map.json + edge CSVs
    init = tm.init_from_gnss(t, lat, lon, alt, rover=(lat_r, lon_r, alt_r), vel=(ve, vn))
    route = tm.route(init.edge, init.s)                # 1-D chain following default successors
    ... integrate distance r along the route (wheel odometry / model) ...
    x, y, z, yaw = route.pose(r)                       # position for /result/position (MAP ENU)
    X, Y, Z = tm.map_to_frame(x, y, z, frame)          # judge frame (see map_to_frame)
"""
from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ============================================================================ geodesy (WGS84)
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def geodetic_to_ecef(lat, lon, h):
    lat = np.radians(np.asarray(lat, float)); lon = np.radians(np.asarray(lon, float)); h = np.asarray(h, float)
    n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    return ((n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon), (n * (1 - _E2) + h) * np.sin(lat))


def ecef_to_geodetic(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - _E2))
    for _ in range(6):
        n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - _E2 * n / (n + h)))
    n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - n


def _rot(lat0, lon0):
    la, lo = math.radians(lat0), math.radians(lon0)
    return np.array([[-math.sin(lo), math.cos(lo), 0.0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def geodetic_to_enu(lat, lon, h, origin):
    x, y, z = geodetic_to_ecef(lat, lon, h)
    x0, y0, z0 = geodetic_to_ecef(*origin)
    d = np.stack([np.asarray(x) - x0, np.asarray(y) - y0, np.asarray(z) - z0])
    return tuple(np.tensordot(_rot(origin[0], origin[1]), d, axes=1))


def enu_to_geodetic(e, n, u, origin):
    x0, y0, z0 = geodetic_to_ecef(*origin)
    d = np.tensordot(_rot(origin[0], origin[1]).T, np.stack([np.asarray(e, float), np.asarray(n, float),
                                                              np.asarray(u, float)]), axes=1)
    return ecef_to_geodetic(d[0] + x0, d[1] + y0, d[2] + z0)


# UTM (Krueger 6th order, Karney 2011), sub-mm
_K0 = 0.9996
_N = _F / (2.0 - _F)
_AA = _A / (1.0 + _N) * (1.0 + _N ** 2 / 4.0 + _N ** 4 / 64.0 + _N ** 6 / 256.0)
_AL = [_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16 + 41 * _N ** 4 / 180 - 127 * _N ** 5 / 288 + 7891 * _N ** 6 / 37800,
       13 * _N ** 2 / 48 - 3 * _N ** 3 / 5 + 557 * _N ** 4 / 1440 + 281 * _N ** 5 / 630 - 1983433 * _N ** 6 / 1935360,
       61 * _N ** 3 / 240 - 103 * _N ** 4 / 140 + 15061 * _N ** 5 / 26880 + 167603 * _N ** 6 / 181440,
       49561 * _N ** 4 / 161280 - 179 * _N ** 5 / 168 + 6601661 * _N ** 6 / 7257600,
       34729 * _N ** 5 / 80640 - 3418889 * _N ** 6 / 1995840,
       212378941 * _N ** 6 / 319334400]
_BE = [_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96 - _N ** 4 / 360 - 81 * _N ** 5 / 512 + 96199 * _N ** 6 / 604800,
       _N ** 2 / 48 + _N ** 3 / 15 - 437 * _N ** 4 / 1440 + 46 * _N ** 5 / 105 - 1118711 * _N ** 6 / 3870720,
       17 * _N ** 3 / 480 - 37 * _N ** 4 / 840 - 209 * _N ** 5 / 4480 + 5569 * _N ** 6 / 90720,
       4397 * _N ** 4 / 161280 - 11 * _N ** 5 / 504 - 830251 * _N ** 6 / 7257600,
       4583 * _N ** 5 / 161280 - 108847 * _N ** 6 / 3991680,
       20648693 * _N ** 6 / 638668800]
_EC = math.sqrt(_E2)


def latlon_to_utm(lat, lon, zone=37):
    lat = np.radians(np.asarray(lat, float))
    dl = np.radians(np.asarray(lon, float) - (6.0 * zone - 183.0))
    t = np.sinh(np.arctanh(np.sin(lat)) - _EC * np.arctanh(_EC * np.sin(lat)))
    xp = np.arctan2(t, np.cos(dl)); ep = np.arctanh(np.sin(dl) / np.sqrt(1 + t * t))
    xi, eta = xp.copy(), ep.copy()
    for j, a in enumerate(_AL, 1):
        xi = xi + a * np.sin(2 * j * xp) * np.cosh(2 * j * ep)
        eta = eta + a * np.cos(2 * j * xp) * np.sinh(2 * j * ep)
    return 500000.0 + _K0 * _AA * eta, _K0 * _AA * xi


def utm_to_latlon(e, n, zone=37):
    xi = np.asarray(n, float) / (_K0 * _AA); eta = (np.asarray(e, float) - 500000.0) / (_K0 * _AA)
    xp, ep = xi.copy(), eta.copy()
    for j, b in enumerate(_BE, 1):
        xp = xp - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        ep = ep - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    taup = np.sin(xp) / np.sqrt(np.sinh(ep) ** 2 + np.cos(xp) ** 2)
    dl = np.arctan2(np.sinh(ep), np.cos(xp))
    tau = taup.copy()
    for _ in range(6):
        sig = np.sinh(_EC * np.arctanh(_EC * tau / np.sqrt(1 + tau * tau)))
        tpi = tau * np.sqrt(1 + sig * sig) - sig * np.sqrt(1 + tau * tau)
        tau = tau + (taup - tpi) / np.sqrt(1 + tpi * tpi) * (1 + (1 - _E2) * tau * tau) / ((1 - _E2) * np.sqrt(1 + tau * tau))
    return np.degrees(np.arctan(tau)), np.degrees(dl) + (6.0 * zone - 183.0)


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ============================================================================ edges
class Edge:
    """Polyline edge sampled uniformly in s (MAP ENU)."""

    def __init__(self, eid: str, cols: dict, closed: bool, meta: dict):
        self.id = eid
        self.closed = closed
        self.meta = meta
        self.s = np.asarray(cols['s'], float)
        self.x = np.asarray(cols['x'], float)
        self.y = np.asarray(cols['y'], float)
        self.z = np.asarray(cols['z'], float)
        self.yaw = np.asarray(cols['yaw'], float)
        self.curv = np.asarray(cols['curvature'], float)
        self.grade = np.asarray(cols['grade'], float)
        self.length = float(meta['length'])
        # segment arrays for projection (closed: last segment back to vertex 0)
        if closed:
            x1, y1 = np.r_[self.x[1:], self.x[:1]], np.r_[self.y[1:], self.y[:1]]
            s1 = np.r_[self.s[1:], self.length]
        else:
            x1, y1, s1 = self.x[1:], self.y[1:], self.s[1:]
        n = len(x1)
        self._ax, self._ay = self.x[:n], self.y[:n]
        dx, dy = x1 - self._ax, y1 - self._ay
        self._len = np.maximum(np.hypot(dx, dy), 1e-9)
        self._tx, self._ty = dx / self._len, dy / self._len
        self._s0 = self.s[:n]
        self._ds = s1 - self._s0
        self._hdg = np.arctan2(dy, dx)
        self.bbox = (self.x.min(), self.x.max(), self.y.min(), self.y.max())
        self.from_ = meta.get('from')
        self.to = meta.get('to')
        # extended (wrap-closed) arrays for interpolation, precomputed once
        if closed:
            self._se = np.r_[self.s, self.length]
            ext = lambda a: np.r_[a, a[:1]]
        else:
            self._se = self.s
            ext = lambda a: a
        self._xe, self._ye, self._ze = ext(self.x), ext(self.y), ext(self.z)
        self._ce, self._sne = ext(np.cos(self.yaw)), ext(np.sin(self.yaw))
        self._ke, self._ge = ext(self.curv), ext(self.grade)
        # block bounding boxes (64 segments) for fast projection pruning
        B = 64
        nb = (n + B - 1) // B
        self._blk = []
        for k in range(nb):
            a, b = k * B, min((k + 1) * B, n)
            xs = np.r_[self._ax[a:b], x1[a:b]]
            ys = np.r_[self._ay[a:b], y1[a:b]]
            self._blk.append((a, b, xs.min(), xs.max(), ys.min(), ys.max()))
        self._blk_arr = np.array([bb[2:] for bb in self._blk])
        # python lists for the scalar fast path (bisect + one weight for all channels)
        self._l_s = self._se.tolist()
        self._l = [a.tolist() for a in (self._xe, self._ye, self._ze, self._ce, self._sne, self._ke, self._ge)]

    def sample(self, s: float):
        """Scalar fast path: (x, y, z, yaw, curvature, grade) at arc length s (~3 us)."""
        L = self.length
        if self.closed:
            s = s % L
        else:
            s = 0.0 if s < 0.0 else (self._l_s[-1] if s > self._l_s[-1] else s)
        ls = self._l_s
        j = bisect.bisect_right(ls, s) - 1
        if j < 0:
            j = 0
        elif j > len(ls) - 2:
            j = len(ls) - 2
        w = (s - ls[j]) / (ls[j + 1] - ls[j])
        x, y, z, c, sn, k, g = (a[j] + w * (a[j + 1] - a[j]) for a in self._l)
        return x, y, z, math.atan2(sn, c), k, g

    def _norm_s(self, s):
        return (s % self.length) if self.closed else min(max(s, 0.0), self.s[-1])

    def _interp(self, arr_e, s):
        s = np.asarray(s, float)
        if s.ndim == 0:  # fast scalar path, O(log n)
            sv = self._norm_s(float(s))
            j = int(np.searchsorted(self._se, sv, side='right')) - 1
            j = min(max(j, 0), len(self._se) - 2)
            w = (sv - self._se[j]) / (self._se[j + 1] - self._se[j])
            return arr_e[j] + w * (arr_e[j + 1] - arr_e[j])
        if self.closed:
            s = np.mod(s, self.length)
        return np.interp(s, self._se, arr_e)

    def pose(self, s):
        """(x, y, z, yaw) at arc length s (scalar or array)."""
        c, sn = self._interp(self._ce, s), self._interp(self._sne, s)
        return self._interp(self._xe, s), self._interp(self._ye, s), self._interp(self._ze, s), np.arctan2(sn, c)

    def curvature(self, s):
        return self._interp(self._ke, s)

    def grade_at(self, s):
        return self._interp(self._ge, s)

    def project(self, px, py, heading=None, max_d=np.inf, max_dpsi=math.radians(60)):
        """Nearest admissible point for each query point. Returns s, d (left +), dist, dpsi (NaN if none).

        Blocks of 64 segments are pruned by bounding box (distance lower bound) -> ~50-200 us per query."""
        px = np.atleast_1d(np.asarray(px, float)); py = np.atleast_1d(np.asarray(py, float))
        n = len(px)
        out_s = np.full(n, np.nan); out_d = np.full(n, np.nan); out_dist = np.full(n, np.nan)
        out_psi = np.full(n, np.nan)
        hd = None if heading is None else np.atleast_1d(np.asarray(heading, float))
        bx0, bx1, by0, by1 = self._blk_arr.T
        for i in range(n):
            # lower bound of the distance to each block
            lb = np.hypot(np.maximum(0.0, np.maximum(bx0 - px[i], px[i] - bx1)),
                          np.maximum(0.0, np.maximum(by0 - py[i], py[i] - by1)))
            order = np.argsort(lb)
            best = (np.inf, -1, 0.0, 0.0, 0.0)
            for k in order:
                if lb[k] > min(best[0], max_d):
                    break
                a, b = self._blk[k][0], self._blk[k][1]
                vx, vy = px[i] - self._ax[a:b], py[i] - self._ay[a:b]
                u = np.clip(vx * self._tx[a:b] + vy * self._ty[a:b], 0.0, self._len[a:b])
                qx = self._ax[a:b] + u * self._tx[a:b] - px[i]
                qy = self._ay[a:b] + u * self._ty[a:b] - py[i]
                dist = np.hypot(qx, qy)
                if hd is not None and np.isfinite(hd[i]):
                    dist = np.where(np.abs(_wrap(hd[i] - self._hdg[a:b])) <= max_dpsi, dist, np.inf)
                j = int(np.argmin(dist))
                if dist[j] < best[0]:
                    best = (float(dist[j]), a + j, float(u[j]), float(vx[j]), float(vy[j]))
            dmin, j, u, vx, vy = best
            if j < 0 or dmin > max_d:
                continue
            out_s[i] = self._s0[j] + u / self._len[j] * self._ds[j]
            out_d[i] = math.copysign(dmin, self._tx[j] * vy - self._ty[j] * vx)
            out_dist[i] = dmin
            if hd is not None:
                out_psi[i] = float(_wrap(hd[i] - self._hdg[j]))
        return out_s, out_d, out_dist, out_psi


@dataclass
class Candidate:
    edge: str
    s: float
    d: float
    dist: float
    dpsi: float


@dataclass
class InitResult:
    ok: bool
    edge: str | None
    s: float
    d: float
    heading: float
    heading_src: str      # 'doppler' | 'baseline' | 'map' | 'none'
    n_fixes: int
    moving: bool
    ambiguous: bool
    alternatives: list
    msg: str = ''
    t_ref: float = float('nan')   # time the (edge, s) refers to (= last fix time)
    speed: float = float('nan')   # along-track speed estimate at t_ref [m/s] (NaN if unknown)
    sigma_s: float = float('nan') # rough 1-sigma of s [m] (RTK ~0.02-0.05, non-RTK ~1-2)
    rtk: bool = False


# ============================================================================ route
class Route:
    """1-D chain of edge pieces starting at (edge, s0), following default successors.

    r (metres along the route from the start) -> (edge, s) -> pose. Built long enough to cover >= 2 laps."""

    def __init__(self, tm: 'TrackMap', edge: str, s0: float, length: float):
        self.tm = tm
        self.pieces = []  # (r_start, edge, s_start, s_end)
        r = 0.0
        e, s = edge, float(s0)
        guard = 0
        while r < length and guard < 20:
            guard += 1
            E = tm.edges[e]
            if E.closed:
                take = length - r
                self.pieces.append((r, e, s, s + take))
                r += take
                break
            end = E.length
            self.pieces.append((r, e, s, end))
            r += end - s
            if E.to is None:
                break  # dead end (yard) -- route stops here
            e, s = E.to['edge'], float(E.to['s'])
        self.length = r
        self._r0 = np.array([p[0] for p in self.pieces])
        self._r0l = self._r0.tolist()

    def edge_s(self, r):
        r = 0.0 if r < 0.0 else (self.length if r > self.length else float(r))
        k = bisect.bisect_right(self._r0l, r) - 1
        r0, e, s0, s1 = self.pieces[max(k, 0)]
        return e, s0 + (r - r0)

    def sample(self, r):
        """(x, y, z, yaw, curvature, grade) at route coordinate r -- the per-cycle call of the estimator."""
        e, s = self.edge_s(r)
        return self.tm.edges[e].sample(s)

    def pose(self, r):
        x, y, z, yaw, k, g = self.sample(r)
        return x, y, z, yaw

    def curvature(self, r):
        return self.sample(r)[4]

    def grade(self, r):
        return self.sample(r)[5]

    def landmarks(self, only_landmarks=True):
        """Stops along this route as (r, stop dict), sorted by r (closed edges: every lap copy)."""
        out = []
        for r0, e, s0, s1 in self.pieces:
            E = self.tm.edges[e]
            for st in self.tm.stops:
                if st['edge'] != e or (only_landmarks and not st.get('landmark')):
                    continue
                if E.closed:
                    k0 = math.floor((s0 - st['s']) / E.length)
                    for k in range(k0, k0 + 4):
                        sv = st['s'] + k * E.length
                        if s0 <= sv <= s1:
                            out.append((r0 + sv - s0, st))
                elif s0 <= st['s'] <= s1:
                    out.append((r0 + st['s'] - s0, st))
        out.sort(key=lambda a: a[0])
        return out

    def locate(self, x, y, heading=None, r_hint=None, window=200.0):
        """Route coordinate r of a point (optionally restricted to |r - r_hint| < window)."""
        best = None
        for r0, e, s0, s1 in self.pieces:
            E = self.tm.edges[e]
            s, d, dist, _ = E.project([x], [y], None if heading is None else [heading])
            if not np.isfinite(s[0]):
                continue
            ss = s[0]
            if E.closed:  # choose the lap copy consistent with this piece
                cands = [ss + k * E.length for k in range(-1, 3)]
                cands = [c for c in cands if s0 - 1e-6 <= c <= s1 + 1e-6]
                if not cands:
                    continue
                ss = cands[0] if r_hint is None else min(cands, key=lambda c: abs(r0 + c - s0 - r_hint))
            elif not (s0 - 1e-6 <= ss <= s1 + 1e-6):
                continue
            r = r0 + ss - s0
            if r_hint is not None and abs(r - r_hint) > window:
                continue
            if best is None or dist[0] < best[1]:
                best = (r, dist[0], d[0])
        return best  # (r, dist, d) or None


# ============================================================================ map
class TrackMap:
    def __init__(self, map_dir):
        map_dir = Path(map_dir)
        self.meta = json.loads((map_dir / 'track_map.json').read_text(encoding='utf-8'))
        fr = self.meta['frame']
        self.origin = (fr['origin_lat'], fr['origin_lon'], fr['origin_h'])
        self.utm_zone = self.meta.get('utm', {}).get('zone', 37)
        self.edges: dict[str, Edge] = {}
        for em in self.meta['edges']:
            cols = _read_csv(map_dir / em['file'])
            self.edges[em['id']] = Edge(em['id'], cols, em['closed'], em)
        self.stops = self.meta.get('stops', [])

    # ------------------------------------------------------------------ frames
    def map_to_geodetic(self, x, y, z):
        return enu_to_geodetic(x, y, z, self.origin)

    def geodetic_to_map(self, lat, lon, h):
        return geodetic_to_enu(lat, lon, h, self.origin)

    def map_to_frame(self, x, y, z, frame='map', frame_origin=None):
        """Convert MAP ENU to the output frame.

        frame='map'      : MAP ENU as is (fixed origin, see track_map.json)
        frame='enu'      : ENU tangent plane at frame_origin=(lat, lon, h) (e.g. first GNSS fix of the run)
        frame='utm'      : (E, N, h) in UTM zone 37N (EPSG:32637)
        frame='utm_local': UTM minus frame_origin projected to UTM, z = h - h0 (no Earth-curvature drop)
        frame='mgrs'     : Autoware-style MGRS grid: (E mod 100 km, N mod 100 km, h); square 37U DB here
        NB: ENU (true north) and UTM/MGRS (grid north) differ by the meridian convergence (~-1.28 deg here),
        i.e. ~110 m at 5 km from the origin -- the judge's frame must be matched exactly.
        """
        if frame == 'map':
            return x, y, z
        lat, lon, h = self.map_to_geodetic(x, y, z)
        if frame == 'enu':
            return geodetic_to_enu(lat, lon, h, frame_origin)
        e, n = latlon_to_utm(lat, lon, self.utm_zone)
        if frame == 'utm':
            return e, n, h
        if frame == 'utm_local':
            e0, n0 = latlon_to_utm(frame_origin[0], frame_origin[1], self.utm_zone)
            return e - e0, n - n0, h - frame_origin[2]
        if frame == 'mgrs':
            return np.mod(e, 100000.0), np.mod(n, 100000.0), h
        raise ValueError(frame)

    # ------------------------------------------------------------------ queries
    def candidates(self, x, y, heading=None, max_d=15.0, max_dpsi=math.radians(60)):
        out = []
        for eid, E in self.edges.items():
            x0, x1, y0, y1 = E.bbox
            if x < x0 - max_d or x > x1 + max_d or y < y0 - max_d or y > y1 + max_d:
                continue
            s, d, dist, dpsi = E.project([x], [y], None if heading is None else [heading], max_d, max_dpsi)
            if np.isfinite(s[0]):
                out.append(Candidate(eid, float(s[0]), float(d[0]), float(dist[0]), float(dpsi[0])))
        out.sort(key=lambda c: c.dist)
        return out

    def locate(self, x, y, heading=None, max_d=15.0):
        c = self.candidates(x, y, heading, max_d)
        return c[0] if c else None

    def pose(self, edge, s):
        return self.edges[edge].pose(s)

    def route(self, edge, s0, length=None):
        if length is None:
            length = 2.2 * self.edges['main'].length
        return Route(self, edge, s0, length)

    # ------------------------------------------------------------------ initial alignment from GNSS
    def init_from_gnss(self, t, lat, lon, alt, rover=None, vel=None, status=None, **kw):
        """Determine (edge, s) from the first seconds of GNSS (geodetic input).

        t, lat, lon, alt : master fixes (arrays, >= 1 fix)
        rover            : optional (lat, lon, alt) arrays of the rover antenna, same epochs as master
        vel              : optional (ve, vn) arrays of master Doppler velocity (ENU, m/s)
        status           : optional NavSatFix status per master fix (2 = RTK/GBAS); RTK fixes are preferred
        Returns InitResult valid at the time of the LAST fix."""
        x, y, z = self.geodetic_to_map(lat, lon, alt)
        rxy = None
        if rover is not None and len(rover[0]):
            xr, yr, _ = self.geodetic_to_map(*rover)
            rxy = (np.atleast_1d(xr), np.atleast_1d(yr))
        return self.init_from_map_xy(t, np.atleast_1d(x), np.atleast_1d(y), rover_xy=rxy, vel=vel, status=status, **kw)

    def init_from_map_xy(self, t, x, y, rover_xy=None, vel=None, status=None, v_moving=0.5, amb_margin=1.5,
                         max_d=15.0, baseline=(12.43, 0.3)):
        """Core of init_from_gnss with master (and rover) positions already in MAP ENU.

        Heading priority: Doppler (moving) > displacement of fixes (> 2 m) > master->rover baseline
        (tram is unidirectional, rover 12.43 m ahead) > the matched edge's own direction.
        Position: RTK fixes (status 2) preferred; median if stationary, last fix if moving.
        Ambiguity flag: another edge within amb_margin of the best that is laterally distinct (> 1 m)."""
        t = np.atleast_1d(np.asarray(t, float))
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        n = len(t)
        if n == 0:
            return InitResult(False, None, np.nan, np.nan, np.nan, 'none', 0, False, True, [], 'no fixes')
        use = np.ones(n, bool)
        if status is not None:
            st = np.atleast_1d(np.asarray(status))
            if np.any(st == 2):
                use = st == 2
        heading, src, moving = np.nan, 'none', False
        if vel is not None and len(vel[0]):
            ve, vn = np.asarray(vel[0], float), np.asarray(vel[1], float)
            sp = np.hypot(ve, vn)
            k = np.flatnonzero(np.isfinite(sp) & (sp > v_moving))
            if len(k):
                moving = True
                heading = float(np.angle(np.mean(np.exp(1j * np.arctan2(vn[k], ve[k])))))
                src = 'doppler'
        xu, yu = x[use], y[use]
        if not np.isfinite(heading) and len(xu) >= 2:
            dx, dy = xu[-1] - xu[0], yu[-1] - yu[0]
            if math.hypot(dx, dy) > 2.0:
                moving = True
                heading, src = math.atan2(dy, dx), 'displacement'
        if not np.isfinite(heading) and rover_xy is not None:
            xr, yr = rover_xy
            m = min(len(xr), n)
            bx, by = xr[-m:] - x[-m:], yr[-m:] - y[-m:]
            L = np.hypot(bx, by)
            ok = np.isfinite(L) & (np.abs(L - baseline[0]) < baseline[1])
            if ok.any():
                heading = float(np.angle(np.mean(np.exp(1j * np.arctan2(by[ok], bx[ok])))))
                src = 'baseline'
        tu = t[use]
        px, py = float(np.median(xu)), float(np.median(yu))
        if moving:
            px, py = float(xu[-1]), float(yu[-1])
        cands = self.candidates(px, py, heading if np.isfinite(heading) else None, max_d)
        if not cands:
            cands = self.candidates(px, py, None, 50.0)
            if not cands:
                return InitResult(False, None, np.nan, np.nan, heading, src, n, moving, True, [], 'off map')
        best = cands[0]
        alts = [c for c in cands[1:] if c.dist < best.dist + amb_margin and abs(c.d - best.d) > 1.0]
        E = self.edges[best.edge]
        rtk = status is not None and bool(np.all(np.atleast_1d(status)[use] == 2))
        s_ref, speed, sig = best.s, np.nan, (0.03 if rtk else 1.5)
        # speed along track: Doppler (last sample) or displacement of used fixes
        if vel is not None and len(vel[0]):
            spv = np.hypot(np.asarray(vel[0], float), np.asarray(vel[1], float))
            fin = np.flatnonzero(np.isfinite(spv))
            if len(fin):
                speed = float(spv[fin[-1]])
        if not np.isfinite(speed) and moving and len(tu) >= 2 and tu[-1] > tu[0]:
            speed = float(math.hypot(xu[-1] - xu[0], yu[-1] - yu[0]) / (tu[-1] - tu[0]))
        if not moving:
            # stationary: median projection of all used fixes (robust against multipath outliers)
            ss, dd, dist, _ = E.project(xu, yu, None, max_d + 5.0)
            okp = np.isfinite(ss)
            if okp.sum():
                sv = ss[okp]
                if E.closed:
                    sv = best.s + ((sv - best.s + E.length / 2) % E.length) - E.length / 2
                s_ref = float(np.median(sv))
                if not rtk:
                    sig = float(max(0.3, 1.4826 * np.median(np.abs(sv - s_ref))))
            speed = 0.0
        else:
            # moving: last used fix (best.s) advanced to the last fix time
            if np.isfinite(speed):
                s_ref = best.s + speed * (t[-1] - tu[-1])
        if E.closed:
            s_ref = s_ref % E.length
        if not np.isfinite(heading):
            src = 'map'
            heading = float(E.pose(s_ref)[3])
        return InitResult(True, best.edge, float(s_ref), best.d, heading, src, n, moving, len(alts) > 0,
                          [(c.edge, round(c.s, 1), round(c.d, 2)) for c in cands[:4]], '', float(t[-1]),
                          speed, sig, rtk)


def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [list(map(float, r)) for r in rd if r and not r[0].startswith('#')]
    arr = np.array(rows)
    return {h: arr[:, i] for i, h in enumerate(header)}
