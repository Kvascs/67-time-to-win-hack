"""Numba-accelerated gated point-to-polyline projection with a uniform grid index (build-time only)."""
from __future__ import annotations

import numpy as np
from numba import njit


def build_grid(ax, ay, bx, by, cell):
    """CSR grid index of segments (a->b). Returns (x0, y0, nx, ny, start, items)."""
    x0 = min(ax.min(), bx.min()) - cell
    y0 = min(ay.min(), by.min()) - cell
    nx = int(np.ceil((max(ax.max(), bx.max()) - x0) / cell)) + 2
    ny = int(np.ceil((max(ay.max(), by.max()) - y0) / cell)) + 2
    i0 = np.floor((np.minimum(ax, bx) - x0) / cell).astype(np.int64)
    i1 = np.floor((np.maximum(ax, bx) - x0) / cell).astype(np.int64)
    j0 = np.floor((np.minimum(ay, by) - y0) / cell).astype(np.int64)
    j1 = np.floor((np.maximum(ay, by) - y0) / cell).astype(np.int64)
    cells, segs = [], []
    for s in range(len(ax)):
        for i in range(i0[s], i1[s] + 1):
            for j in range(j0[s], j1[s] + 1):
                cells.append(j * nx + i)
                segs.append(s)
    cells = np.array(cells, np.int64)
    segs = np.array(segs, np.int64)
    o = np.argsort(cells, kind='stable')
    cells, segs = cells[o], segs[o]
    start = np.searchsorted(cells, np.arange(nx * ny + 1)).astype(np.int64)
    return float(x0), float(y0), nx, ny, start, segs


@njit(cache=True)
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@njit(cache=True)
def project_grid(px, py, hd, use_hd, max_d, max_dpsi, ax, ay, tx, ty, seglen, s0, hdg,
                 gx0, gy0, nx, ny, cell, start, items):
    n = px.shape[0]
    out_s = np.full(n, np.nan)
    out_d = np.full(n, np.nan)
    out_dist = np.full(n, np.nan)
    out_seg = np.full(n, -1, np.int64)
    out_dpsi = np.full(n, np.nan)
    r = int(np.ceil(max_d / cell))
    for k in range(n):
        ci = int(np.floor((px[k] - gx0) / cell))
        cj = int(np.floor((py[k] - gy0) / cell))
        best = 1e300
        bs = -1
        bu = 0.0
        bsg = 0.0
        bdp = np.nan
        for j in range(cj - r, cj + r + 1):
            if j < 0 or j >= ny:
                continue
            for i in range(ci - r, ci + r + 1):
                if i < 0 or i >= nx:
                    continue
                c = j * nx + i
                for q in range(start[c], start[c + 1]):
                    sgi = items[q]
                    if use_hd:
                        dp = _wrap(hd[k] - hdg[sgi])
                        if abs(dp) > max_dpsi or not np.isfinite(hd[k]):
                            continue
                    else:
                        dp = np.nan
                    vx = px[k] - ax[sgi]
                    vy = py[k] - ay[sgi]
                    u = vx * tx[sgi] + vy * ty[sgi]
                    if u < 0.0:
                        u = 0.0
                    elif u > seglen[sgi]:
                        u = seglen[sgi]
                    qx = ax[sgi] + u * tx[sgi] - px[k]
                    qy = ay[sgi] + u * ty[sgi] - py[k]
                    dist = np.sqrt(qx * qx + qy * qy)
                    if dist < best:
                        best = dist
                        bs = sgi
                        bu = u
                        bsg = tx[sgi] * vy - ty[sgi] * vx
                        bdp = dp
        if bs >= 0 and best <= max_d:
            out_s[k] = s0[bs] + bu
            out_d[k] = best if bsg >= 0 else -best
            out_dist[k] = best
            out_seg[k] = bs
            out_dpsi[k] = bdp
    return out_s, out_d, out_dist, out_seg, out_dpsi
