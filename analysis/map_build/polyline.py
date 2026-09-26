"""Planar polyline utilities: arc-length resampling, smoothing, gated point projection, curvature.

A Polyline holds vertices (N,2) sampled (approximately) uniformly in arc length, optional
per-vertex z, and can be open or closed (closed: segment N-1 -> 0 exists).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

try:
    import fastproj
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cumlen(xy: np.ndarray, closed: bool = False) -> np.ndarray:
    d = np.hypot(*np.diff(xy, axis=0).T)
    s = np.r_[0.0, np.cumsum(d)]
    if closed:
        s = np.r_[s, s[-1] + np.hypot(*(xy[0] - xy[-1]))]
    return s


def resample(xy: np.ndarray, ds: float, closed: bool = False, extra: np.ndarray | None = None):
    """Uniform arc-length resampling (linear). For closed curves the last vertex != first."""
    pts = np.vstack([xy, xy[:1]]) if closed else xy
    ex = None
    if extra is not None:
        ex = np.concatenate([extra, extra[:1]]) if closed else extra
    s = cumlen(pts)
    keep = np.r_[True, np.diff(s) > 1e-9]
    pts, s = pts[keep], s[keep]
    if ex is not None:
        ex = ex[keep]
    L = s[-1]
    n = max(int(round(L / ds)), 2)
    if closed:
        sn = np.arange(n) * (L / n)
    else:
        sn = np.linspace(0.0, L, n + 1)
    out = np.column_stack([np.interp(sn, s, pts[:, 0]), np.interp(sn, s, pts[:, 1])])
    if ex is not None:
        return out, np.interp(sn, s, ex)
    return out


def smooth_xy(xy: np.ndarray, sigma_pts: float, closed: bool = False) -> np.ndarray:
    if sigma_pts <= 0:
        return xy.copy()
    mode = 'wrap' if closed else 'nearest'
    out = np.column_stack([gaussian_filter1d(xy[:, 0], sigma_pts, mode=mode),
                           gaussian_filter1d(xy[:, 1], sigma_pts, mode=mode)])
    if not closed:  # keep end points (avoid shrinking open ends)
        k = int(3 * sigma_pts) + 1
        w = np.clip(np.arange(len(xy)) / k, 0, 1)
        w = np.minimum(w, w[::-1])
        out = xy + (out - xy) * w[:, None]
    return out


class Polyline:
    def __init__(self, xy: np.ndarray, closed: bool = False, z: np.ndarray | None = None):
        self.xy = np.asarray(xy, dtype=float)
        self.closed = closed
        self.z = None if z is None else np.asarray(z, dtype=float)
        self._build()

    def _build(self):
        xy = self.xy
        nxt = np.roll(xy, -1, axis=0) if self.closed else np.vstack([xy[1:], xy[-1:]])
        seg = nxt - xy
        if not self.closed:
            seg[-1] = seg[-2]
        self.seglen = np.hypot(seg[:, 0], seg[:, 1])
        self.seglen[self.seglen < 1e-9] = 1e-9
        self.t = seg / self.seglen[:, None]  # unit tangent of segment i (i -> i+1)
        if not self.closed:
            self.seglen[-1] = 0.0
        self.s = np.r_[0.0, np.cumsum(self.seglen)[:-1]]
        self.length = self.s[-1] + (self.seglen[-1] if self.closed else 0.0)
        self.hdg_seg = np.arctan2(self.t[:, 1], self.t[:, 0])
        self.tree = cKDTree(xy)
        self.nseg = len(xy) if self.closed else len(xy) - 1
        self._grid = None

    # ------------------------------------------------------------------ geometry along s
    def vertex_heading(self):
        """Heading at vertices (average of adjacent segment tangents)."""
        t = self.t.copy()
        prev = np.roll(t, 1, axis=0)
        if not self.closed:
            prev[0] = t[0]
            t[-1] = t[-2]
        v = t + prev
        return np.arctan2(v[:, 1], v[:, 0])

    def curvature(self, sigma_m: float = 0.0):
        """Signed curvature at vertices (left turn positive): wrapped central heading differences
        (no 2*pi seam for closed curves), optionally Gaussian-smoothed over sigma_m metres."""
        h = self.vertex_heading()
        ds = float(np.median(self.seglen[:self.nseg]))
        if self.closed:
            k = wrap(np.roll(h, -1) - np.roll(h, 1)) / (2 * ds)
            mode = 'wrap'
        else:
            dh = wrap(np.diff(h))
            k = np.r_[dh[0], 0.5 * (dh[1:] + dh[:-1]), dh[-1]] / ds
            mode = 'nearest'
        if sigma_m > 0:
            k = gaussian_filter1d(k, sigma_m / ds, mode=mode)
        return k

    def interp(self, s, what='xy'):
        s = np.asarray(s, dtype=float)
        if self.closed:
            s = np.mod(s, self.length)
            sv = np.r_[self.s, self.length]
            xy = np.vstack([self.xy, self.xy[:1]])
        else:
            sv, xy = self.s, self.xy
        x = np.interp(s, sv, xy[:, 0])
        y = np.interp(s, sv, xy[:, 1])
        return x, y

    # ------------------------------------------------------------------ projection
    def project(self, px, py, heading=None, max_d: float = 10.0, max_dpsi: float = np.radians(60),
                k: int | None = None, chunk: int = 20000):
        """Project points onto the polyline.

        Returns s, d (signed, left of travel direction positive), dist, seg index, dpsi (heading
        difference, NaN if heading None). Points with no admissible segment get NaN.
        heading: travel heading of the point (rad, ENU); segments with |heading - seg heading| >
        max_dpsi are ignored (separates the two directions of a double track).
        """
        px = np.atleast_1d(np.asarray(px, float))
        py = np.atleast_1d(np.asarray(py, float))
        hd = None if heading is None else np.atleast_1d(np.asarray(heading, float))
        n = len(px)
        if _HAVE_NUMBA and k is None:
            return self._project_fast(px, py, hd, max_d, max_dpsi)
        if k is None and max_d > 3.0:
            # two-pass: points with an admissible segment within 3 m keep that (it is the global nearest
            # admissible one); only the rest are re-projected with the wide gate (expensive large k)
            res = list(self.project(px, py, hd, 3.0, max_dpsi, None, chunk))
            miss = ~np.isfinite(res[0])
            if miss.any():
                sub = self.project(px[miss], py[miss], None if hd is None else hd[miss], max_d, max_dpsi,
                                   int(min(max(12, 3 * (2 * max_d / float(np.median(self.seglen[:self.nseg])) + 2)), 160)),
                                   chunk)
                for j in range(5):
                    res[j] = res[j].copy()
                    res[j][miss] = sub[j]
            return tuple(res)
        if k is None:
            # enough nearest vertices to reach every track within max_d (several parallel tracks)
            ds = float(np.median(self.seglen[:self.nseg]))
            k = int(min(max(12, 3 * (2 * max_d / ds + 2)), 160))
        if n > chunk:
            outs = [self.project(px[i:i + chunk], py[i:i + chunk], None if hd is None else hd[i:i + chunk],
                                 max_d, max_dpsi, k, chunk) for i in range(0, n, chunk)]
            return tuple(np.concatenate([o[j] for o in outs]) for j in range(5))
        return self._project(px, py, hd, max_d, max_dpsi, k)

    def _project_fast(self, px, py, hd, max_d, max_dpsi, cell=4.0):
        if self._grid is None:
            ns = self.nseg
            ax, ay = self.xy[:ns, 0], self.xy[:ns, 1]
            bx = ax + self.t[:ns, 0] * self.seglen[:ns]
            by = ay + self.t[:ns, 1] * self.seglen[:ns]
            self._grid = fastproj.build_grid(ax, ay, bx, by, cell) + (cell,)
        gx0, gy0, nx, ny, start, items, cell = self._grid
        ns = self.nseg
        use = hd is not None
        h = hd if use else np.zeros(len(px))
        return fastproj.project_grid(px, py, h, use, float(max_d), float(max_dpsi),
                                     np.ascontiguousarray(self.xy[:ns, 0]), np.ascontiguousarray(self.xy[:ns, 1]),
                                     np.ascontiguousarray(self.t[:ns, 0]), np.ascontiguousarray(self.t[:ns, 1]),
                                     np.ascontiguousarray(self.seglen[:ns]), np.ascontiguousarray(self.s[:ns]),
                                     np.ascontiguousarray(self.hdg_seg[:ns]), gx0, gy0, nx, ny, cell, start, items)

    def _project(self, px, py, heading, max_d, max_dpsi, k):
        p = np.column_stack([px, py])
        n = len(p)
        kk = min(k, len(self.xy))
        _, idx = self.tree.query(p, k=kk, distance_upper_bound=max_d + 2 * np.max(self.seglen))
        idx = np.atleast_2d(idx)
        valid = idx < len(self.xy)
        idx = np.where(valid, idx, 0)
        # candidate segments: i and i-1 for each nearby vertex
        cand = np.concatenate([idx, idx - 1], axis=1)
        cvalid = np.concatenate([valid, valid], axis=1)
        if self.closed:
            cand = np.mod(cand, len(self.xy))
        else:
            cvalid &= (cand >= 0) & (cand < self.nseg)
            cand = np.clip(cand, 0, self.nseg - 1)
        a = self.xy[cand]                      # (n,K,2)
        tt = self.t[cand]
        L = self.seglen[cand]
        v = p[:, None, :] - a
        u = np.clip(np.einsum('nkj,nkj->nk', v, tt), 0.0, L)
        proj = a + tt * u[..., None]
        dvec = p[:, None, :] - proj
        dist = np.hypot(dvec[..., 0], dvec[..., 1])
        dsgn = tt[..., 0] * v[..., 1] - tt[..., 1] * v[..., 0]
        ok = cvalid & (dist <= max_d)
        dpsi = None
        if heading is not None:
            hd = np.asarray(heading, float)
            dpsi = wrap(hd[:, None] - self.hdg_seg[cand])
            ok &= np.abs(dpsi) <= max_dpsi
            ok &= np.isfinite(hd)[:, None]
        dist_m = np.where(ok, dist, np.inf)
        j = np.argmin(dist_m, axis=1)
        r = np.arange(n)
        good = np.isfinite(dist_m[r, j])
        seg = cand[r, j]
        s = self.s[seg] + u[r, j]
        d = np.where(np.abs(dsgn[r, j]) > 0, np.sign(dsgn[r, j]), 1.0) * dist[r, j]
        out_dpsi = dpsi[r, j] if dpsi is not None else np.full(n, np.nan)
        s[~good] = np.nan
        d[~good] = np.nan
        dist_r = np.where(good, dist[r, j], np.nan)
        out_dpsi[~good] = np.nan
        seg = np.where(good, seg, -1)
        return s, d, dist_r, seg, out_dpsi
