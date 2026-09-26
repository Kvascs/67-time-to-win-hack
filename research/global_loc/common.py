"""Shared helpers for the GNSS-free global localisation study (research/global_loc).

- paths to the frozen estimator snapshot, maps and raw data
- map / landmark / cut-off loading (main cycle, arc length s, closed loop)
- geodetic -> map ENU conversion (same ECEF tangent-plane formula as tools/replay/quick_eval.py)
- no-GNSS replay of a bag through build_core/tbo_replay_fix2.exe, cached as a small npz
- ground truth: master-antenna GNSS fixes projected onto the main cycle -> s_map(t)

Nothing here writes outside research/global_loc/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PKG = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry'
MAPS = ROOT / 'analysis' / 'validation_maps'  # TRAIN-only maps (honest validation)
NPZ = ROOT / 'data' / 'npz'
EXE = ROOT / 'build_core' / ('tbo_replay_fix2.exe' if os.name == 'nt' else 'tbo_replay_fix2')
CACHE = HERE / 'cache'
SPLITS = json.loads((ROOT / 'data' / 'splits.json').read_text(encoding='utf-8'))

sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge  # noqa: E402  (export_events only; tools/ is read-only for us)

# output flag bits (ros2_ws/src/tram_backup_odometry/core/include/tbo/types.hpp)
FLAG_MODEL_ONLY = 1 << 11
FLAG_STANDSTILL = 1 << 12
FLAG_NOT_INIT = 1 << 13
FLAG_NO_MAP = 1 << 14


# ----------------------------------------------------------------------------------------------
# map
@dataclass
class TrackMap:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    grade: np.ndarray
    curv: np.ndarray
    length: float
    origin: tuple

    def wrap(self, s):
        return np.mod(s, self.length)

    def interp(self, col: np.ndarray, s):
        """Periodic linear interpolation of a map column at arc length(s) s."""
        sw = self.wrap(np.asarray(s, float))
        return np.interp(sw, np.r_[self.s, self.length], np.r_[col, col[0]])


def _read_header(path: Path) -> dict:
    meta = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            for tok in line[1:].split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    meta[k] = v
    return meta


def load_map(path: Path = MAPS / 'track_map.csv') -> TrackMap:
    meta = _read_header(path)
    df = pd.read_csv(path, comment='#')
    s = df.s.to_numpy()
    # closed cycle: length = last s + distance from the last point back to the first one
    close = float(np.hypot(df.x.iloc[0] - df.x.iloc[-1], df.y.iloc[0] - df.y.iloc[-1]))
    length = float(s[-1] + close)
    origin = (float(meta['origin_lat']), float(meta['origin_lon']), float(meta['origin_h']))
    return TrackMap(s, df.x.to_numpy(), df.y.to_numpy(), df.z.to_numpy(), df.grade.to_numpy(),
                    df.curvature.to_numpy(), length, origin)


def load_places(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment='#')


def load_branches() -> list[dict]:
    out = []
    for p in sorted(MAPS.glob('branch_*.csv')):
        meta = _read_header(p)
        df = pd.read_csv(p, comment='#')
        out.append(dict(name=p.stem, join_s=float(meta['join_s']), df=df))
    return out


# ----------------------------------------------------------------------------------------------
# geodesy (WGS84 ECEF -> local ENU tangent plane at the map origin)
_A, _F = 6378137.0, 1 / 298.257223563
_E2 = _F * (2 - _F)


def _ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - _E2) + h) * np.sin(la)], -1)


def to_enu(lat, lon, h, origin):
    lat0, lon0, h0 = origin
    d = _ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(h, float)) - \
        _ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))
    la, lo = np.radians(lat0), np.radians(lon0)
    r = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return d @ r.T


# ----------------------------------------------------------------------------------------------
# no-GNSS replay through the frozen estimator snapshot
REPLAY_COLS = ['stamp_ns', 'trigger', 'v', 's', 'd', 'k', 'g', 'a_model', 'accel', 'flags', 'mu3',
               'slip_f', 'slip_r', 'pos_valid']


def replay_nognss(bag: str, force: bool = False) -> dict:
    """Replay `bag` with every GNSS message removed; cache the columns we need.

    Returns dict of arrays: t (s, header time), v, s (relative odometer), d, k, g, a_model, accel, flags,
    mu3, slip_f, slip_r, trig (0 cmd, 1 W0, 2 W1).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    out_npz = CACHE / f'replay_{bag}.npz'
    if out_npz.exists() and not force:
        z = np.load(out_npz)
        return {k: z[k] for k in z.files}
    tmp = CACHE / '_tmp'
    tmp.mkdir(exist_ok=True)
    ev = tmp / f'{bag}_events.csv'
    ev_ng = tmp / f'{bag}_events_nognss.csv'
    out = tmp / f'{bag}_out.csv'
    cpp_bridge.export_events(bag, ev, gnss_seconds=0)
    with open(ev, encoding='utf-8') as fi, open(ev_ng, 'w', newline='\n', encoding='utf-8') as fo:
        for line in fi:
            if not line.startswith('G'):  # drop GF / GV lines: no GNSS at all
                fo.write(line)
    ev.unlink()
    branches = ','.join(str(b) for b in sorted(MAPS.glob('branch_*.csv')))
    cmd = [str(EXE), '--in', str(ev_ng), '--out', str(out),
           '--map', str(MAPS / 'track_map.csv'), '--traction', str(PKG / 'config' / 'traction_lut.csv'),
           '--branches', branches,
           '--set', f"landmark_file={MAPS / 'landmarks.csv'}", '--set', f"cutoff_file={MAPS / 'cutoffs.csv'}",
           '--set', 'output_frame=map']
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f'replay failed for {bag}: {res.stderr}')
    df = pd.read_csv(out, usecols=REPLAY_COLS)
    ev_ng.unlink()
    out.unlink()
    trig = df.trigger.map({'C': 0, 'W0': 1, 'W1': 2}).fillna(3).astype(np.int8).to_numpy()
    arr = dict(t=df.stamp_ns.to_numpy() * 1e-9, v=df.v.to_numpy(), s=df.s.to_numpy(), d=df.d.to_numpy(),
               k=df.k.to_numpy(), g=df.g.to_numpy(), a_model=df.a_model.to_numpy(), accel=df.accel.to_numpy(),
               flags=df['flags'].to_numpy().astype(np.int64), mu3=df.mu3.to_numpy(), slip_f=df.slip_f.to_numpy(),
               slip_r=df.slip_r.to_numpy(), trig=trig, stderr=np.array(res.stderr.strip()))
    np.savez_compressed(out_npz, **arr)
    return arr


def notch_series(bag: str):
    """Controller notch (header stamp, value) from the raw bag, sorted by stamp."""
    a = np.load(NPZ / f'{bag}.npz')['vehicle__driver_position_cmd']
    o = np.argsort(a[:, 1], kind='stable')
    return a[o, 1], a[o, 2].astype(int)


def wheel_series(bag: str):
    d = np.load(NPZ / f'{bag}.npz')
    f = d['vehicle__front_bogie_velocity']
    r = d['vehicle__rear_bogie_velocity']
    return f[np.argsort(f[:, 1]), 1:], r[np.argsort(r[:, 1]), 1:]


# ----------------------------------------------------------------------------------------------
# ground truth s_map(t) of the master antenna
class Projector:
    """Nearest-point projection onto the main cycle with heading gating (the cycle carries both
    travel directions on parallel tracks and two terminal loops, so heading disambiguates)."""

    def __init__(self, m: TrackMap, step: float = 0.25):
        from scipy.spatial import cKDTree
        self.m = m
        s = np.arange(0.0, m.length, step)
        self.s = s
        self.x = m.interp(m.x, s)
        self.y = m.interp(m.y, s)
        dx = m.interp(m.x, s + 0.5) - m.interp(m.x, s - 0.5)
        dy = m.interp(m.y, s + 0.5) - m.interp(m.y, s - 0.5)
        self.h = np.arctan2(dy, dx)
        self.tree = cKDTree(np.c_[self.x, self.y])

    def candidates(self, px, py, r=12.0):
        return self.tree.query_ball_point([px, py], r)


def truth_track(bag: str, m: TrackMap, proj: Projector | None = None, gate: float = 12.0):
    """True main-cycle arc length of the master antenna at each master fix.

    Returns DataFrame t (header stamp), s_map (wrapped), s_unw (unwrapped, continuous), dist (m from the
    map), status, ok (projected onto the main cycle within the gate and consistent in heading).
    Heading = master -> rover baseline (points forward), falling back to the motion direction.
    """
    proj = proj or Projector(m)
    d = np.load(NPZ / f'{bag}.npz')
    mf = d['sensing__gnss__master__fix']
    rf = d['sensing__gnss__rover__fix']
    if len(mf) == 0:
        return None
    mf = mf[np.isfinite(mf[:, 2]) & np.isfinite(mf[:, 3])]
    mf = mf[np.argsort(mf[:, 1], kind='stable')]
    # drop duplicate header stamps
    keep = np.r_[True, np.diff(mf[:, 1]) > 1e-6]
    mf = mf[keep]
    p = to_enu(mf[:, 2], mf[:, 3], mf[:, 4], m.origin)
    t = mf[:, 1]
    # baseline heading (master -> rover) from the rover fix nearest in header time
    hd = np.full(len(t), np.nan)
    if len(rf):
        rf = rf[np.isfinite(rf[:, 2])]
        rf = rf[np.argsort(rf[:, 1], kind='stable')]
        pr = to_enu(rf[:, 2], rf[:, 3], rf[:, 4], m.origin)
        j = np.clip(np.searchsorted(rf[:, 1], t), 1, len(rf) - 1)
        j = np.where(np.abs(rf[j - 1, 1] - t) < np.abs(rf[j, 1] - t), j - 1, j)
        close = np.abs(rf[j, 1] - t) < 0.06
        bx, by = pr[j, 0] - p[:, 0], pr[j, 1] - p[:, 1]
        bl = np.hypot(bx, by)
        good = close & (bl > 8.0) & (bl < 16.0)
        hd[good] = np.arctan2(by[good], bx[good])
    # motion heading as a fallback
    dxm = np.gradient(p[:, 0])
    dym = np.gradient(p[:, 1])
    mv = np.hypot(dxm, dym) > 0.05
    hm = np.arctan2(dym, dxm)
    use_m = np.isnan(hd) & mv
    hd[use_m] = hm[use_m]
    s_out = np.full(len(t), np.nan)
    dist = np.full(len(t), np.nan)
    for i in range(len(t)):
        c = proj.candidates(p[i, 0], p[i, 1], gate)
        if not c:
            continue
        c = np.asarray(c)
        dd = np.hypot(proj.x[c] - p[i, 0], proj.y[c] - p[i, 1])
        if np.isfinite(hd[i]):
            dpsi = np.abs(np.angle(np.exp(1j * (proj.h[c] - hd[i]))))
            okh = dpsi < np.radians(50)
            if not okh.any():
                continue
            c, dd = c[okh], dd[okh]
        b = np.argmin(dd)
        s_out[i] = proj.s[c[b]]
        dist[i] = dd[b]
    df = pd.DataFrame(dict(t=t, s_map=s_out, dist=dist, status=mf[:, 5]))
    # continuity: unwrap and reject jumps inconsistent with the travelled distance
    ok = np.isfinite(df.s_map.to_numpy())
    s_unw = np.full(len(t), np.nan)
    L = m.length
    prev_s, prev_t = None, None
    for i in np.flatnonzero(ok):
        sm = df.s_map.iat[i]
        if prev_s is None:
            s_unw[i] = sm
        else:
            k = np.round((prev_s - sm) / L)
            cand = sm + k * L
            dt = t[i] - prev_t
            ds = cand - prev_s
            if ds < -3.0 - 0.5 * dt or ds > 3.0 + 25.0 * dt:  # tram never reverses; <= 25 m/s
                ok[i] = False
                continue
            s_unw[i] = cand
        prev_s, prev_t = s_unw[i], t[i]
    df['s_unw'] = s_unw
    df['ok'] = ok & np.isfinite(s_unw)
    return df
