"""Shared helpers for the data-product analyses ("one model, many products").

Every product in this folder is computed from the published outputs of ONE estimator run per bag:
build_core/tbo_replay_fix5.exe (frozen snapshot of the C++ core) replayed on the real bags with the
TRAIN-only map, branches, stop landmarks, cut-offs and disturbance field (analysis/validation_maps)
and the package traction table, GNSS only in the first 5 s (like the jury's test bags). GNSS beyond that window is used only as an independent reference to check the
products, never as their input.

Layout:
  cache/<bag>.parquet      estimator outputs (subset of columns), one row per published output
  cache/<bag>_run.parquet  the same, enriched: map arc length, direction, notch, raw bogie speeds
  cache/<bag>_ref.parquet  GNSS reference (master fix/vel after the init window), GNSS bags only
  out/*.csv                small result tables quoted in docs/PRODUCTS.md
  fig/*.png                figures
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge  # noqa: E402
from cpp_bridge import NPZ, PKG  # noqa: E402

# The bridge defaults to the live build; the products must come from the frozen snapshot.
EXE = ROOT / 'build_core' / 'tbo_replay_fix5.exe'
cpp_bridge.REPLAY_EXE = EXE

# train-only maps (the package ships an all-data map that contains the validation traces)
MAPS = ROOT / 'analysis' / 'validation_maps'
DFIELD = MAPS / 'dfield.csv'
MAP_CSV = MAPS / 'track_map.csv'
LANDMARKS = MAPS / 'landmarks.csv'
CUTOFFS = MAPS / 'cutoffs.csv'
BRANCHES = ','.join(str(b) for b in sorted(MAPS.glob('branch_*.csv')))
TRACTION = PKG / 'config' / 'traction_lut.csv'

CACHE = HERE / 'cache'
FIG = HERE / 'fig'
OUT = HERE / 'out'
for _d in (CACHE, FIG, OUT):
    _d.mkdir(parents=True, exist_ok=True)

MSK = timezone(timedelta(hours=3))  # Moscow local time (UTC+3, no DST)

# ---- health flag bits, copied from core/include/tbo/types.hpp ----
F_FRONT_SLIP = 1 << 0
F_REAR_SLIP = 1 << 1
F_FRONT_SLIDE = 1 << 2
F_REAR_SLIDE = 1 << 3
F_FRONT_DROPOUT = 1 << 4
F_REAR_DROPOUT = 1 << 5
F_CMD_DROPOUT = 1 << 6
F_FRONT_STUCK = 1 << 7
F_REAR_STUCK = 1 << 8
F_FRONT_INVALID = 1 << 9
F_REAR_INVALID = 1 << 10
F_MODEL_ONLY = 1 << 11
F_STANDSTILL = 1 << 12
F_NOT_INIT = 1 << 13
F_NO_MAP = 1 << 14
F_RECOVERED = 1 << 15
F_LATE = 1 << 16
F_UNMODELED = 1 << 17
F_CMD_BAD = 1 << 18
F_LANDMARK = 1 << 19

# estimator geometry (params.hpp defaults compiled into the snapshot): positions relative to antenna 1
FRONT_BOGIE_ALONG = 9.9
REAR_BOGIE_ALONG = 2.35
WHEEL_KMH_TO_MS = 1.00037 / 3.6   # params.hpp wheel_kmh_to_ms (straight-track scale from train bags)

KEEP_COLS = ['stamp_ns', 'trigger', 'v', 'v_var', 'x', 'y', 'z', 'yaw', 's', 's_var', 'mu0', 'mu1', 'mu2',
             'mu3', 'mu4', 'slip_f', 'slip_r', 'd', 'k', 'g', 'a_model', 'accel', 'flags', 'pos_valid', 's_map',
             'a_ext']


# ------------------------------------------------------------------------------------------
# bags and metadata
# ------------------------------------------------------------------------------------------
def splits() -> dict:
    return json.load(open(ROOT / 'data' / 'splits.json'))


def long_bags() -> list[str]:
    """All unique long runs (duplicates and <5 min bags excluded)."""
    sp = splits()
    bags = sorted(sp['train'] + sp['val'] + sp['no_gnss_long'])
    if os.environ.get('PRODUCTS_CACHED_ONLY'):  # quick iterations while the replays still run
        bags = [b for b in bags if (CACHE / f'{b}_run.parquet').exists() and
                ((CACHE / f'{b}_refv.parquet').exists() or b in sp['no_gnss_long'])]
    return bags


def bag_meta() -> pd.DataFrame:
    sp = splits()
    split_of = {b: k for k in ('train', 'val', 'no_gnss_long', 'short') for b in sp[k]}
    rows = []
    for r in sp['info']:
        t0 = datetime.fromtimestamp(r['t0'], tz=timezone.utc).astimezone(MSK)
        rows.append({'bag': r['bag'], 'vehicle': r['vehicle'], 'split': split_of.get(r['bag'], '?'),
                     'dur_s': r['dur'], 'has_gnss': r['gnss_master'] > 0, 't0_utc': r['t0'],
                     'local_start': t0.strftime('%Y-%m-%d %H:%M'), 'date': t0.strftime('%m-%d'),
                     'hour': t0.hour + t0.minute / 60.0})
    return pd.DataFrame(rows).set_index('bag')


# ------------------------------------------------------------------------------------------
# geodesy (same formulas as tools/replay/quick_eval.py)
# ------------------------------------------------------------------------------------------
_A, _F = 6378137.0, 1 / 298.257223563
_E2 = _F * (2 - _F)


def _ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - _E2) + h) * np.sin(la)], -1)


def enu(lat, lon, h, lat0, lon0, h0):
    d = _ecef(lat, lon, h) - _ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))
    la, lo = np.radians(lat0), np.radians(lon0)
    r = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return d @ r.T


# ------------------------------------------------------------------------------------------
# track map (main cycle): arc length exactly as core/src/track_map.cpp computes it
# ------------------------------------------------------------------------------------------
class TrackMap:
    def __init__(self, path: Path = MAP_CSV):
        meta = {}
        with open(path) as fh:
            for line in fh:
                if not line.startswith('#'):
                    break
                for tok in line[1:].split():
                    if '=' in tok:
                        k, v = tok.split('=', 1)
                        meta[k] = float(v)
        df = pd.read_csv(path, comment='#')
        self.origin = (meta['origin_lat'], meta['origin_lon'], meta['origin_h'])
        xy = df[['x', 'y']].to_numpy()
        keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 1e-6]
        df = df[keep].reset_index(drop=True)
        xy = df[['x', 'y']].to_numpy()
        ds = np.hypot(*np.diff(xy, axis=0).T)
        s = np.r_[0.0, np.cumsum(ds)]
        close = float(np.hypot(*(xy[0] - xy[-1])))
        self.L = float(s[-1] + close)
        self.s, self.x, self.y = s, xy[:, 0], xy[:, 1]
        self.z = df.z.to_numpy()
        self.grade = df.grade.to_numpy()
        self.curv = df.curvature.to_numpy()
        self._tree = cKDTree(xy)
        # west end of the line = the westmost point; WB = east terminal -> west loop, EB = back
        self.s_west = float(s[np.argmin(self.x)])
        # common route axis for both directions: distance from the east terminal measured along
        # the westbound track (EB points are mapped onto the nearest WB point)
        wb = s <= self.s_west
        wtree = cKDTree(xy[wb])
        _, j = wtree.query(xy)
        self.route = np.where(wb, s, s[wb][j])

    def wrap(self, s):
        return np.mod(s, self.L)

    def project(self, x, y):
        """Nearest point on the polyline: returns (s, distance)."""
        pts = np.column_stack([x, y])
        _, i = self._tree.query(pts)
        n = len(self.s)
        best_s = np.full(len(pts), np.nan)
        best_d = np.full(len(pts), np.inf)
        for off in (-1, 0):  # the segments before and after the nearest vertex
            a = (i + off) % n
            b = (a + 1) % n
            ax, ay = self.x[a], self.y[a]
            bx, by = self.x[b], self.y[b]
            seg = np.column_stack([bx - ax, by - ay])
            seglen2 = np.maximum((seg ** 2).sum(1), 1e-12)
            u = np.clip(((pts[:, 0] - ax) * seg[:, 0] + (pts[:, 1] - ay) * seg[:, 1]) / seglen2, 0, 1)
            px, py = ax + u * seg[:, 0], ay + u * seg[:, 1]
            d = np.hypot(pts[:, 0] - px, pts[:, 1] - py)
            sa = self.s[a]
            sl = np.where(b == 0, self.L - sa, self.s[b] - sa)  # closing segment
            s = sa + u * sl
            better = d < best_d
            best_d = np.where(better, d, best_d)
            best_s = np.where(better, s, best_s)
        return self.wrap(best_s), best_d

    def direction(self, s):
        """'WB' (east terminal -> west loop) or 'EB'."""
        return np.where(self.wrap(s) <= self.s_west, 'WB', 'EB')

    def route_at(self, s):
        """Distance from the east terminal along the WB track (common axis for both directions)."""
        return np.interp(self.wrap(s), self.s, self.route)

    def at(self, s, field):
        return np.interp(self.wrap(s), self.s, getattr(self, field))

    def xy_at(self, s):
        ss = self.wrap(s)
        return np.interp(ss, self.s, self.x), np.interp(ss, self.s, self.y)


_MAP: TrackMap | None = None


def track_map() -> TrackMap:
    global _MAP
    if _MAP is None:
        _MAP = TrackMap()
    return _MAP


def landmarks() -> pd.DataFrame:
    return pd.read_csv(LANDMARKS, comment='#')


def platforms() -> pd.DataFrame:
    lm = landmarks()
    return lm[lm.cls.isin(['platform', 'terminal'])].reset_index(drop=True)


# ------------------------------------------------------------------------------------------
# estimator replay (cached)
# ------------------------------------------------------------------------------------------
def replay_sets() -> dict:
    # Same configuration as tools/replay/quick_eval.eval_bag, but the output frame is the map's own
    # ENU frame so x, y can be projected onto track_map.csv; the published point is antenna 1
    # (base_link offsets 0), which is also the point the stop landmarks refer to.
    return {'output_frame': 'map', 'base_link_along_m': 0, 'base_link_height_m': 0,
            'landmark_file': str(LANDMARKS), 'cutoff_file': str(CUTOFFS), 'dfield_file': str(DFIELD)}


def run_estimator(bag: str, force: bool = False) -> Path:
    pq = CACHE / f'{bag}.parquet'
    if pq.exists() and not force:
        return pq
    tmp = CACHE / 'tmp'
    tmp.mkdir(exist_ok=True)
    ev, out = tmp / f'{bag}_events.csv', tmp / f'{bag}_out.csv'
    cpp_bridge.export_events(bag, ev)  # GNSS only in the first 5 s (jury-like)
    o = cpp_bridge.run_replay(ev, out, map_csv=MAP_CSV, traction_csv=TRACTION, sets=replay_sets(),
                              branches=BRANCHES)
    o = o[KEEP_COLS].rename(columns={'s_map': 's_map_est'})
    for c in o.columns:
        if c in ('stamp_ns', 'flags', 'pos_valid', 'trigger', 'x', 'y', 's', 's_map_est'):
            continue
        o[c] = o[c].astype('float32')
    o.to_parquet(pq, index=False)
    ev.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return pq


def _asof(t_query, t_src, val, max_age=None):
    """Last value of (t_src, val) at or before each t_query (NaN if none or older than max_age)."""
    order = np.argsort(t_src, kind='stable')
    t_src, val = t_src[order], val[order]
    i = np.searchsorted(t_src, t_query, side='right') - 1
    ok = i >= 0
    res = np.full(len(t_query), np.nan)
    res[ok] = val[i[ok]]
    if max_age is not None:
        age = np.full(len(t_query), np.inf)
        age[ok] = t_query[ok] - t_src[i[ok]]
        res[age > max_age] = np.nan
    return res


def load_run(bag: str, force: bool = False) -> pd.DataFrame:
    """Estimator outputs of one bag, deduplicated by stamp and enriched with map position
    (s_map, route, direction), notch and raw bogie speeds at the output stamp."""
    pq = CACHE / f'{bag}_run.parquet'
    if pq.exists() and not force:
        return pd.read_parquet(pq)
    o = pd.read_parquet(run_estimator(bag))
    # one row per stamp (outputs are published on cmd, wheel and grid triggers): keep the last
    o = o.sort_values('stamp_ns', kind='stable').drop_duplicates('stamp_ns', keep='last').reset_index(drop=True)
    t = o.stamp_ns.to_numpy() * 1e-9
    o['t'] = t - t[0]
    m = track_map()
    fl = o["flags"].to_numpy()
    anchored = (fl & F_NO_MAP) == 0
    # location = the estimator's own main-cycle arc length of antenna 1 (-1 when not anchored or on a
    # branch); the projection of the published x, y (which include the 45 ms position lead) is kept
    # as a cross-check
    s_est = o.s_map_est.to_numpy()
    ok = anchored & (s_est >= 0)
    s_proj, dist = m.project(o.x.to_numpy(), o.y.to_numpy())
    o['s_map'] = np.where(ok, s_est, np.nan)
    o['s_map_proj'] = np.where(anchored & (dist < 0.5), s_proj, np.nan)
    o['map_dist'] = np.where(anchored, dist, np.nan)
    o['route'] = np.where(ok, m.route_at(np.nan_to_num(s_est)), np.nan)
    o['dir'] = np.where(ok, m.direction(np.nan_to_num(s_est)), '')
    d = np.load(NPZ / f'{bag}.npz')
    cmd = d['vehicle__driver_position_cmd']
    o['notch'] = _asof(t, cmd[:, 1], cmd[:, 2], max_age=1.0)
    for key, col in (('vehicle__front_bogie_velocity', 'wf'), ('vehicle__rear_bogie_velocity', 'wr')):
        a = d[key]
        o[col] = _asof(t, a[:, 1], a[:, 2] * WHEEL_KMH_TO_MS, max_age=0.6) if len(a) else np.nan
    o.to_parquet(pq, index=False)
    return o


def load_ref(bag: str, force: bool = False) -> pd.DataFrame | None:
    """GNSS reference of one bag in the map frame: master fix projected onto the main cycle
    (s_ref, cross distance) and master horizontal speed, at the GNSS header stamps."""
    pq = CACHE / f'{bag}_ref.parquet'
    if pq.exists() and not force:
        return pd.read_parquet(pq)
    d = np.load(NPZ / f'{bag}.npz')
    mf = d['sensing__gnss__master__fix']
    mv = d['sensing__gnss__master__vel']
    if len(mf) == 0 or len(mv) == 0:
        return None
    m = track_map()
    mf = mf[np.isfinite(mf[:, 2])]
    p = enu(mf[:, 2], mf[:, 3], mf[:, 4], *m.origin)
    s_ref, dist = m.project(p[:, 0], p[:, 1])
    fix = pd.DataFrame({'t_fix': mf[:, 1], 's_ref': s_ref, 'dist_ref': dist, 'status': mf[:, 5]})
    vel = pd.DataFrame({'t_vel': mv[:, 1], 'v_ref': np.hypot(mv[:, 2], mv[:, 3])})
    fix.to_parquet(pq, index=False)
    vel.to_parquet(CACHE / f'{bag}_refv.parquet', index=False)
    return fix


def load_refv(bag: str) -> pd.DataFrame | None:
    pq = CACHE / f'{bag}_refv.parquet'
    if not pq.exists():
        if load_ref(bag) is None:
            return None
    return pd.read_parquet(pq)


def nearest(t_query: np.ndarray, t_src: np.ndarray, tol: float):
    """Index of the nearest t_src sample for each t_query (t_src sorted) and a within-tol mask."""
    idx = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    left, right = t_src[idx - 1], t_src[idx]
    pick = np.where(np.abs(t_query - left) <= np.abs(right - t_query), idx - 1, idx)
    return pick, np.abs(t_src[pick] - t_query) <= tol


def ref_aligned(bag: str) -> pd.DataFrame | None:
    """GNSS reference at the master-vel header stamps: speed, map arc length of the nearest master
    fix (only fixes within 1.5 m of the main cycle; the parallel track is ~3.5 m away) and the
    nearest estimator output (within 0.05 s, like the jury's matching)."""
    fix, vel = load_ref(bag), load_refv(bag)
    if fix is None or vel is None:
        return None
    o = load_run(bag)
    t_o = o.stamp_ns.to_numpy() * 1e-9
    tv = vel.t_vel.to_numpy()
    fix = fix.sort_values('t_fix')
    jf, okf = nearest(tv, fix.t_fix.to_numpy(), 0.06)
    good_fix = okf & (fix.dist_ref.to_numpy()[jf] < 1.5)
    r = pd.DataFrame({'t': tv - t_o[0], 'v_ref': vel.v_ref.to_numpy(),
                      's_ref': np.where(good_fix, fix.s_ref.to_numpy()[jf], np.nan),
                      'rtk': np.where(okf, fix.status.to_numpy()[jf] == 2, False)})
    io, oko = nearest(tv, t_o, 0.05)
    for c in ('v', 's_map', 'dir', 'notch', 'wf', 'wr', 'k', 'g', 'flags', 'mu3'):
        col = o[c].to_numpy()[io]
        if col.dtype.kind == 'f':
            col = np.where(oko, col, np.nan)
        r[c] = col
    r['matched'] = oko
    return r


def accel_centered(t: np.ndarray, v: np.ndarray, w: float = 1.0) -> np.ndarray:
    """Acceleration as the centred difference of speed over a window of w seconds."""
    return (np.interp(t + w / 2, t, v) - np.interp(t - w / 2, t, v)) / w


def circ_diff(a, b, L: float):
    """Signed difference a - b on a loop of length L, in (-L/2, L/2]."""
    return (np.asarray(a) - np.asarray(b) + L / 2) % L - L / 2


def load_all(bags=None, cols=None) -> pd.DataFrame:
    frames = []
    for b in bags or long_bags():
        o = load_run(b)
        if cols:
            o = o[cols]
        o = o.copy()
        o.insert(0, 'bag', b)
        frames.append(o)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------
def rising_edges(mask: np.ndarray) -> np.ndarray:
    """Indices where a boolean series switches from False to True."""
    m = np.asarray(mask, bool)
    return np.flatnonzero(m & ~np.r_[False, m[:-1]])


def segments(mask: np.ndarray, t: np.ndarray, merge_gap: float = 0.0, min_dur: float = 0.0):
    """Contiguous True segments of `mask` as (i_start, i_end) index pairs (inclusive); segments
    separated by less than `merge_gap` seconds are merged; shorter than `min_dur` dropped."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    ends = np.flatnonzero(m & ~np.r_[m[1:], False])
    segs = [[int(a), int(b)] for a, b in zip(starts, ends)]
    merged = [segs[0]]
    for a, b in segs[1:]:
        if t[a] - t[merged[-1][1]] < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if t[b] - t[a] >= min_dur]


def savefig(fig, name: str):
    path = FIG / name
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print(f'figure -> {path.relative_to(ROOT)}')


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 110,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt
