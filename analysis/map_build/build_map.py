"""Build the tram track map (pathgraph) from GNSS traces (master antenna).

Stages
  1. seed cycle  -- splice clean RTK runs into one closed loop through both terminals ('main')
  2. refine      -- iterative robust re-centring (principal-curve style): project RTK fixes of all runs
                    (heading-gated so the opposite-direction track 3.5 m away never mixes in), per-run median
                    lateral offset per 1 m bin, quality-weighted median across runs, smooth, shift along the
                    normal, resample; gates shrink 2.0 -> 0.3 m
  3. branches    -- sustained departures (|d|>1.8 m) of RTK runs from 'main' are clustered; clusters whose
                    core offset is >= 3 m (a real parallel track, not a GNSS bias episode) become branch
                    edges built by the same robust averaging; attached to 'main' where they rejoin
  4. profiles    -- z(s) (robust across runs), yaw(s), curvature(s), grade(s), support, lateral spread
  5. export      -- CSV per edge (0.5 m) + track_map.json (frames, topology, sections, build info)

Run:  python build_map.py train   -> map_train/   (fit on train only; used for validation on val)
      python build_map.py all     -> map/         (train+val; deployment map)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import data_io as D
import geo
import runs as R
from polyline import Polyline, resample, smooth_xy, wrap

OUT = D.OUT
PLOTS = OUT / 'plots'

# ----------------------------------------------------------------------------- stage 1: seed
P_EAST = (2401.2, 846.9)          # east terminal platform (W->E runs end here, E->W runs start here)
SEED_PIECES = [
    # (bag, from_xy (None=run start), to_xy (None=run end), search window 'first'|'last')
    # RTK-grade days; WB main on the straight track (used on 4 of 5 recording dates; 27-28 Jul used a
    # parallel detour 4.2 m north over s~1124..1742, which becomes a branch)
    ('30618_dd8e6395', P_EAST, (-2100.0, -268.0), 'first'),  # east platform -> east loop -> WB main -> west junction
    ('30618_095a115b', (-2100.0, -268.0), None, 'last'),     # west junction -> west arrival stop
    ('30618_0652866c', (-2153.9, -314.1), None, 'last'),     # west arrival stop -> west loop -> fan track F1
    ('30618_e2dcf65f', None, (2300.0, 700.0), 'first'),      # fan track F1 -> EB main -> east approach
    ('30618_33bec73f', (2300.0, 700.0), P_EAST, 'last'),     # east approach -> east platform
]


def _nearest_idx(r, xy, window):
    d = np.hypot(r.x - xy[0], r.y - xy[1])
    if window == 'last':
        k0 = int(len(d) * 0.7)
        return k0 + int(np.argmin(d[k0:]))
    return int(np.argmin(d))


def decimate_path(x, y, min_step=0.5):
    keep = [0]
    lx, ly = x[0], y[0]
    for i in range(1, len(x)):
        if np.hypot(x[i] - lx, y[i] - ly) >= min_step:
            keep.append(i)
            lx, ly = x[i], y[i]
    return np.array(keep)


def seed_cycle(runs: dict, ds=1.0, sigma_m=2.0) -> Polyline:
    parts = []
    for bag, a, b, win in SEED_PIECES:
        r = runs[bag] if bag in runs else R.load_runs([bag])[bag]
        i0 = 0 if a is None else _nearest_idx(r, a, win)
        i1 = len(r) - 1 if b is None else _nearest_idx(r, b, win)
        sl = np.arange(i0, i1 + 1)
        sl = sl[r.good[sl]]
        k = decimate_path(r.x[sl], r.y[sl], 0.5)
        parts.append(np.column_stack([r.x[sl][k], r.y[sl][k]]))
    xy = np.vstack(parts)
    xy = resample(xy, 0.5, closed=True)
    xy = smooth_xy(xy, sigma_m / 0.5, closed=True)
    return Polyline(resample(xy, ds, closed=True), closed=True)


# ----------------------------------------------------------------------------- stage 2: refine
def collect_points(runs: dict, fields=('x', 'y', 'psi', 'z'), exclude: dict | None = None):
    """Concatenate good (RTK) fixes of all runs; returns dict of arrays incl. run index 'rid'.
    exclude: {bag: bool mask} of fixes to leave out (e.g. fixes on branch tracks)."""
    out = {f: [] for f in fields}
    out['rid'] = []
    for i, (b, r) in enumerate(runs.items()):
        m = r.good.copy()
        if exclude is not None and b in exclude:
            m &= ~exclude[b]
        for f in fields:
            out[f].append(getattr(r, f)[m])
        out['rid'].append(np.full(m.sum(), i))
    return {k: np.concatenate(v) for k, v in out.items()}


def _group_median(key, val):
    o = np.lexsort((val, key))
    k, v = key[o], val[o]
    uk, start, cnt = np.unique(k, return_index=True, return_counts=True)
    return uk, 0.5 * (v[start + (cnt - 1) // 2] + v[start + cnt // 2])


def robust_profile(s, val, rid, length, binw=1.0, closed=True, min_runs=2, run_w=None):
    """Per-bin weighted median over runs of per-run medians (one vote per run, weighted by run quality).

    Returns (bin centres, median, n_runs, MAD across runs)."""
    nb = int(np.ceil(length / binw))
    b = np.floor(s / binw).astype(np.int64)
    b = np.mod(b, nb) if closed else np.clip(b, 0, nb - 1)
    uk, med_rb = _group_median(rid.astype(np.int64) * nb + b, val)
    bins_rb, runs_rb = uk % nb, uk // nb
    w_rb = np.ones(len(uk)) if run_w is None else np.asarray(run_w, float)[runs_rb]
    o = np.lexsort((med_rb, bins_rb))
    bb, mm, ww = bins_rb[o], med_rb[o], w_rb[o]
    ub, st, cnt = np.unique(bb, return_index=True, return_counts=True)
    cw = np.cumsum(ww)
    half = np.r_[0.0, cw][st] + 0.5 * np.add.reduceat(ww, st)
    pos = np.searchsorted(cw, half - 1e-12)
    med = np.full(nb, np.nan)
    nrun = np.zeros(nb, int)
    med[ub] = mm[pos]
    nrun[ub] = cnt
    madv = np.full(nb, np.nan)
    ukk, mads = _group_median(bb, np.abs(mm - np.repeat(mm[pos], cnt)))
    madv[ukk] = mads
    med[nrun < min_runs] = np.nan
    return (np.arange(nb) + 0.5) * binw, med, nrun, madv


def fill_and_smooth(prof, sigma_bins, closed=True, med_k=5, max_gap_bins=100):
    """Fill NaN bins by linear interpolation across gaps <= max_gap_bins (zero beyond: keep current geometry
    in long unsupported stretches), median-filter then Gaussian-smooth."""
    n = len(prof)
    idx = np.arange(n)
    ok = np.isfinite(prof)
    if ok.sum() == 0:
        return np.zeros(n)
    if closed:
        f = np.interp(idx, np.r_[idx[ok] - n, idx[ok], idx[ok] + n], np.r_[prof[ok], prof[ok], prof[ok]])
    else:
        f = np.interp(idx, idx[ok], prof[ok])
    # gap length for every NaN bin
    if (~ok).any():
        lab = np.cumsum(np.r_[True, ok[1:] != ok[:-1]])
        if closed and not ok[0] and not ok[-1]:
            lab[lab == lab[-1]] = lab[0]
        big = np.zeros(n, bool)
        for l in np.unique(lab[~ok]):
            m = (lab == l) & ~ok
            if m.sum() > max_gap_bins:
                big |= m
        f[big] = 0.0
    mode = 'wrap' if closed else 'nearest'
    return gaussian_filter1d(median_filter(f, size=med_k, mode=mode), sigma_bins, mode=mode)


def refine(poly: Polyline, pts: dict, gates=(2.0, 1.5, 1.0, 0.6, 0.4, 0.3), sigma_m=2.0, sigma_xy=0.5, ds=1.0,
           max_dpsi=np.radians(45), verbose=True, closed=True, run_w=None, min_runs=1):
    """Iterative robust re-centring of a polyline on a point cloud (principal-curve style).

    Bins without data get the correction interpolated from their neighbours (gaps <= 100 m)."""
    hist = []
    for it, gate in enumerate(gates):
        s, d, dist, seg, dpsi = poly.project(pts['x'], pts['y'], pts['psi'], max_d=gate, max_dpsi=max_dpsi)
        ok = np.isfinite(s)
        c, med, nrun, mad = robust_profile(s[ok], d[ok], pts['rid'][ok], poly.length, binw=ds, closed=closed,
                                           run_w=run_w, min_runs=min_runs)
        corr_f = fill_and_smooth(med, sigma_m / ds, closed=closed)
        corr_c = fill_and_smooth(med, 4 * sigma_m / ds, closed=closed)
        # support-adaptive smoothing: fine where >= 6 runs vote, 4x coarser where <= 2 runs
        sup = gaussian_filter1d(nrun.astype(float), 5.0 / ds, mode='wrap' if closed else 'nearest')
        wf = np.clip((sup - 2.0) / 4.0, 0.0, 1.0)
        corr = wf * corr_f + (1 - wf) * corr_c
        corr_v = np.interp(poly.s, c, corr, period=poly.length if closed else None)
        h = poly.vertex_heading()
        xy = poly.xy + corr_v[:, None] * np.column_stack([-np.sin(h), np.cos(h)])
        xy = resample(xy, 0.5, closed=closed)
        xy = smooth_xy(xy, sigma_xy / 0.5, closed=closed)
        poly = Polyline(resample(xy, ds, closed=closed), closed=closed)
        st = dict(it=it, gate=gate, n_used=int(ok.sum()), med_abs_corr=round(float(np.nanmedian(np.abs(med))), 4),
                  p95_abs_corr=round(float(np.nanpercentile(np.abs(med), 95)), 4),
                  max_abs_corr=round(float(np.nanmax(np.abs(corr))), 4), length=round(poly.length, 3),
                  bins_nodata=int(np.sum(nrun < min_runs)))
        hist.append(st)
        if verbose:
            print('  refine', st, flush=True)
    return poly, hist


def run_quality(cyc: Polyline, runs: dict, exclude=((1100, 1760), (5350, 5800))):
    """Per-run GNSS quality vs the map: rigid 2-D shift (LS on lateral residuals) + residual MAD."""
    h_v = np.unwrap(cyc.vertex_heading())
    q = {}
    for b, r in runs.items():
        s, d, dist, seg, dpsi = cyc.project(r.x, r.y, r.psi, max_d=3.0, max_dpsi=np.radians(45))
        m = r.good & np.isfinite(d) & (np.abs(d) < 0.5)
        for a0, a1 in exclude:
            m &= ~((s > a0) & (s < a1))
        n_all = int(np.sum(r.good))
        if m.sum() < 100:
            q[b] = dict(dx=None, dy=None, mad=None, frac_near=float(m.sum() / max(n_all, 1)), n=int(m.sum()),
                        frac_rtk=float(np.mean(r.status == 2)))
            continue
        hh = np.interp(s[m], cyc.s, h_v)
        A = np.column_stack([-np.sin(hh), np.cos(hh)])
        w = np.ones(m.sum(), bool)
        for _ in range(3):
            delta, *_ = np.linalg.lstsq(A[w], d[m][w], rcond=None)
            res = d[m] - A @ delta
            w = np.abs(res) < 3 * np.median(np.abs(res)) + 0.02
        q[b] = dict(dx=float(delta[0]), dy=float(delta[1]), mad=float(np.median(np.abs(res))),
                    frac_near=float(m.sum() / max(n_all, 1)), n=int(m.sum()), frac_rtk=float(np.mean(r.status == 2)))
    return q


def quality_weights(runs: dict, q: dict, rtk_mad=0.03):
    """1.0 for RTK-grade runs (residual MAD < 3 cm), 0.1 otherwise."""
    return np.array([1.0 if (q[b]['mad'] is not None and q[b]['mad'] < rtk_mad) else 0.1 for b in runs])


# ----------------------------------------------------------------------------- stage 3: branches
def deviation_segments(cyc: Polyline, runs: dict, weights, d_on=1.8, d_off=0.15, min_len=15.0, min_n=20,
                       max_d=12.0, max_gap_s=3.0):
    """Sustained departures of RTK-grade runs from the cycle (time-contiguous good fixes with |d| > d_on,
    extended backwards/forwards until |d| < d_off = true divergence / merge point)."""
    segs = []
    for (b, r), w in zip(runs.items(), weights):
        if w < 1.0:
            continue
        s, d, dist, seg, dpsi = cyc.project(r.x, r.y, r.psi, max_d=max_d, max_dpsi=np.radians(60))
        g = np.flatnonzero(r.good)
        if len(g) < 10:
            continue
        dg, tg = d[g], r.th[g]
        off = ~np.isfinite(dg) | (np.abs(np.nan_to_num(dg, nan=99.0)) > d_on)
        n = len(g)
        k = 0
        while k < n:
            if not off[k]:
                k += 1
                continue
            k0 = k1 = k
            while k1 + 1 < n and off[k1 + 1] and tg[k1 + 1] - tg[k1] <= max_gap_s:
                k1 += 1
            e0, e1 = k0, k1
            while e0 - 1 >= 0 and tg[e0] - tg[e0 - 1] <= max_gap_s and np.isfinite(dg[e0 - 1]) and abs(dg[e0 - 1]) > d_off:
                e0 -= 1
            while e1 + 1 < n and tg[e1 + 1] - tg[e1] <= max_gap_s and np.isfinite(dg[e1 + 1]) and abs(dg[e1 + 1]) > d_off:
                e1 += 1
            # include one on-track fix at each end as anchor
            e0 = max(e0 - 1, 0)
            e1 = min(e1 + 1, n - 1)
            idx = g[e0:e1 + 1]
            xy = np.column_stack([r.x[idx], r.y[idx]])
            plen = float(np.sum(np.hypot(*np.diff(xy, axis=0).T))) if len(idx) > 1 else 0.0
            if (k1 - k0 + 1) >= min_n and plen >= min_len:
                fin = np.isfinite(s[idx])
                segs.append(dict(bag=b, idx=idx, start_run=(e0 == 0), end_run=(e1 == n - 1),
                                 s_first=float(s[idx][fin][0]) if fin.any() else None,
                                 s_last=float(s[idx][fin][-1]) if fin.any() else None,
                                 d_med=float(np.nanmedian(d[idx])) if fin.any() else None, n=len(idx), plen=plen))
            k = max(k1, e1) + 1
    return segs


def _seg_poly(r, idx, step=0.5):
    k = decimate_path(r.x[idx], r.y[idx], step)
    return np.column_stack([r.x[idx][k], r.y[idx][k]])


def cluster_segments(segs: list, runs: dict, link_d=1.2, min_overlap=15.0):
    """Union-find clustering of deviation segments whose paths run within link_d of each other."""
    n = len(segs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    polys = []
    for sg in segs:
        xy = _seg_poly(runs[sg['bag']], sg['idx'])
        polys.append(Polyline(resample(xy, 1.0), closed=False) if len(xy) >= 3 else None)
    for i in range(n):
        for j in range(i + 1, n):
            A, B = polys[i], polys[j]
            if A is None or B is None:
                continue
            ri = runs[segs[i]['bag']]
            idx = segs[i]['idx']
            s, d, dist, seg, _ = B.project(ri.x[idx], ri.y[idx], ri.psi[idx], max_d=5.0, max_dpsi=np.radians(45))
            ok = np.isfinite(s) & (s > 0.5) & (s < B.length - 0.5)
            if ok.sum() < 10:
                continue
            if np.nanmax(s[ok]) - np.nanmin(s[ok]) >= min_overlap and np.median(dist[ok]) < link_d:
                parent[find(i)] = find(j)
    _, labels = np.unique([find(i) for i in range(n)], return_inverse=True)
    return labels


def core_offset(members: list, runs: dict, cyc: Polyline, d_on=1.8):
    """Median |d| over the deviating core of all members (NaN = too far to project -> 99)."""
    vals = []
    for sg in members:
        rr = runs[sg['bag']]
        s, d, dist, seg, _ = cyc.project(rr.x[sg['idx']], rr.y[sg['idx']], rr.psi[sg['idx']], max_d=12.0,
                                         max_dpsi=np.radians(60))
        a = np.where(np.isfinite(d), np.abs(d), 99.0)
        vals.append(a[a > d_on])
    v = np.concatenate(vals) if vals else np.zeros(0)
    return float(np.median(v)) if len(v) else 0.0


def choose_branch_seed(members: list, runs: dict, cyc: Polyline, binw=2.0, max_score=0.3):
    """Seed = longest member whose lateral-offset profile along 'main' agrees with the cluster median
    (median |d - d_median| < max_score over bins shared with >= 2 other runs); fallback: most consistent."""
    prof = []
    for sg in members:
        rr = runs[sg['bag']]
        idx = sg['idx']
        s, d, dist, seg, _ = cyc.project(rr.x[idx], rr.y[idx], rr.psi[idx], max_d=12.0, max_dpsi=np.radians(60))
        ok = np.isfinite(s) & rr.good[idx]
        b = np.floor(s[ok] / binw).astype(int)
        dm = {}
        for bb in np.unique(b):
            dm[bb] = float(np.median(d[ok][b == bb]))
        prof.append(dm)
    allb = {}
    for i, dm in enumerate(prof):
        for bb, v in dm.items():
            allb.setdefault(bb, []).append((i, v))
    scores = []
    for i, dm in enumerate(prof):
        dev = []
        for bb, v in dm.items():
            others = [u for j, u in allb[bb] if j != i]
            if len(others) >= 2:
                dev.append(abs(v - np.median(others)))
        scores.append(np.median(dev) if len(dev) >= 5 else np.inf)
    ok = [i for i, sc in enumerate(scores) if sc < max_score]
    if ok:
        return max(ok, key=lambda i: members[i]['plen']), scores
    return int(np.argmin(scores)), scores


def build_branch(members: list, runs: dict, weights: dict, ds=1.0):
    """Robust average of member segments -> open polyline.

    Seed = member flagged '_seed' by stage3 (consensus-consistent, longest), else the longest member after a
    length cap; refine with shrinking gates on all member fixes; trim to the stretch supported by enough runs."""
    seeds = [sg for sg in members if sg.get('_seed')]
    if seeds:
        best = seeds[0]
    else:
        plen = np.array([sg['plen'] for sg in members])
        cap = 1.5 * np.percentile(plen, 75) if len(members) >= 3 else np.inf
        best = max([sg for sg in members if sg['plen'] <= cap] or members, key=lambda sg: sg['plen'])
    r = runs[best['bag']]
    xy = resample(_seg_poly(r, best['idx']), 0.5)
    poly = Polyline(resample(smooth_xy(xy, 1.0 / 0.5), ds), closed=False)
    pts = {'x': [], 'y': [], 'psi': [], 'z': [], 'rid': []}
    rw = []
    for i, sg in enumerate(members):
        rr = runs[sg['bag']]
        idx = sg['idx']
        m = rr.good[idx]
        for f in ('x', 'y', 'psi', 'z'):
            pts[f].append(getattr(rr, f)[idx][m])
        pts['rid'].append(np.full(m.sum(), i))
        rw.append(weights.get(sg['bag'], 1.0))
    pts = {k: np.concatenate(v) for k, v in pts.items()}
    poly, hist = refine(poly, pts, gates=(3.0, 2.0, 1.0, 0.6, 0.4), sigma_m=2.0, ds=ds, closed=False,
                        run_w=np.array(rw), min_runs=1, verbose=False)
    core, ext = support_ranges(poly, pts)
    return poly, pts, members, core, ext


def support_ranges(poly: Polyline, pts: dict, min_frac=0.2, min_runs=2, gate=0.6, max_hole_m=60.0):
    """Arc-length ranges of an open branch: core = longest stretch supported by >= max(2, 20%) member runs
    (holes <= max_hole_m closed); ext = core grown over contiguous 1-run support (used at yard ends)."""
    s, d, dist, seg, _ = poly.project(pts['x'], pts['y'], pts['psi'], max_d=gate, max_dpsi=np.radians(45))
    ok = np.isfinite(s)
    runs_all = np.unique(pts['rid'])
    L = poly.length
    if len(runs_all) < 2 or ok.sum() == 0:
        return (0.0, L), (0.0, L)
    need = max(min_runs, int(np.ceil(min_frac * len(runs_all))))
    nb = int(np.ceil(L)) + 1
    b = np.clip(s[ok].astype(int), 0, nb - 1)
    key = np.unique(pts['rid'][ok].astype(np.int64) * nb + b)
    cnt = np.bincount(key % nb, minlength=nb)

    def longest(mask):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return None
        m = mask.copy()
        for a0, a1 in zip(idx[:-1], idx[1:]):
            if 1 < a1 - a0 <= max_hole_m:
                m[a0:a1] = True
        lab = np.cumsum(np.r_[True, m[1:] != m[:-1]])
        best, bl = None, -1
        for l in np.unique(lab[m]):
            mm = np.flatnonzero((lab == l) & m)
            if len(mm) > bl:
                best, bl = mm, len(mm)
        return best
    core = longest(cnt >= need)
    if core is None:
        return (0.0, L), (0.0, L)
    c0, c1 = float(core[0]), float(min(core[-1] + 1, L))
    one = cnt >= 1
    e0 = core[0]
    while e0 - 1 >= 0 and one[e0 - 1]:
        e0 -= 1
    e1 = core[-1]
    while e1 + 1 < nb and one[e1 + 1]:
        e1 += 1
    return (c0, c1), (float(e0), float(min(e1 + 1, L)))


def cut(poly: Polyline, a: float, b: float) -> Polyline:
    keep = (poly.s >= a - 1e-9) & (poly.s <= b + 1e-9)
    xy = poly.xy[keep]
    return Polyline(resample(xy, 1.0), closed=False) if len(xy) >= 3 else poly


def attach(poly: Polyline, cyc: Polyline, tol=0.6):
    """Find where an open branch leaves / rejoins the cycle; trim the coincident ends.

    Returns (poly_trimmed, from_s, to_s); from_s/to_s = cycle s of divergence/merge or None (yard end)."""
    h = poly.vertex_heading()
    s, d, dist, seg, _ = cyc.project(poly.xy[:, 0], poly.xy[:, 1], h, max_d=3.0, max_dpsi=np.radians(30))
    near = np.isfinite(dist) & (dist < tol)
    n = len(poly.xy)
    i0, i1 = 0, n - 1
    from_s = to_s = None
    if near[:3].any():
        i0 = int(np.flatnonzero(near[:3])[0])
        while i0 + 1 < n and near[i0 + 1]:
            i0 += 1
        from_s = float(s[i0])
    if near[-3:].any():
        i1 = n - 3 + int(np.flatnonzero(near[-3:])[-1])
        while i1 - 1 > i0 and near[i1 - 1]:
            i1 -= 1
        to_s = float(s[i1])
    xy = poly.xy[i0:i1 + 1].copy()
    if from_s is not None:
        xy[0] = np.array(cyc.interp([from_s])).ravel()
    if to_s is not None:
        xy[-1] = np.array(cyc.interp([to_s])).ravel()
    return Polyline(resample(xy, 1.0), closed=False), from_s, to_s


def branch_exclusion(cyc: Polyline, runs: dict, segs: list, margin=40.0):
    """Fixes that must not vote for 'main': every deviation segment, plus all fixes of that run whose main-s
    lies within the segment's main-s range +-margin (runs that took a branch do not vote for main near it)."""
    L = cyc.length
    excl = {b: np.zeros(len(r), bool) for b, r in runs.items()}
    by_run = {}
    for sg in segs:
        by_run.setdefault(sg['bag'], []).append(sg)
    for b, lst in by_run.items():
        r = runs[b]
        s, d, dist, seg, _ = cyc.project(r.x, r.y, r.psi, max_d=6.0, max_dpsi=np.radians(60))
        for sg in lst:
            excl[b][sg['idx'][0]:sg['idx'][-1] + 1] = True
            sv = [v for v in (sg['s_first'], sg['s_last']) if v is not None]
            if not sv:
                continue
            a0, a1 = min(sv) - margin, max(sv) + margin
            if a1 - a0 > L / 2:  # wrap-around or degenerate
                continue
            ds0 = (s - a0) % L
            excl[b] |= np.isfinite(s) & (ds0 <= (a1 - a0))
    return excl


def stage3(cyc: Polyline, runs: dict, weights_arr, verbose=True, min_core=3.0, min_runs=2):
    segs = deviation_segments(cyc, runs, weights_arr)
    lab = cluster_segments(segs, runs)
    wdict = {b: float(w) for b, w in zip(runs, weights_arr)}
    branches = []
    for c in np.unique(lab):
        mem = [sg for sg, l in zip(segs, lab) if l == c]
        core = core_offset(mem, runs, cyc)
        if core < min_core:
            if verbose:
                print(f'  reject cluster {c}: {[m["bag"] for m in mem]} core |d|={core:.2f} m (GNSS bias episode)')
            continue
        n_mem_runs = len(set(m['bag'] for m in mem))
        if n_mem_runs < min_runs:
            if verbose:
                print(f'  skip cluster {c}: single run {[m["bag"] for m in mem]} (needs >= {min_runs} runs)')
            continue
        for sg in mem:
            sg.pop('_seed', None)
        if len(mem) >= 3:
            # drop members strongly inconsistent with the consensus (GNSS-biased tails), pick a consistent seed
            k, scores = choose_branch_seed(mem, runs, cyc)
            mem = [sg for sg, sc in zip(mem, scores) if not (np.isfinite(sc) and sc > 1.0)] or mem
            k2, _ = choose_branch_seed(mem, runs, cyc)
            mem[k2]['_seed'] = True
        full, pts, used, rng_core, rng_ext = build_branch(mem, runs, wdict)
        poly, fs, ts = attach(cut(full, *rng_core), cyc)
        if fs is None or ts is None:
            # yard (dead) ends keep all contiguous single-run RTK support beyond the core
            a = rng_ext[0] if fs is None else rng_core[0]
            b = rng_ext[1] if ts is None else rng_core[1]
            if (a, b) != tuple(rng_core):
                poly, fs, ts = attach(cut(full, a, b), cyc)
        bags = sorted(set(m['bag'] for m in used))
        branches.append(dict(poly=poly, from_s=fs, to_s=ts, bags=bags, n_runs=len(bags), core_offset=core,
                             starts_in_run=int(sum(m['start_run'] for m in used)),
                             ends_in_run=int(sum(m['end_run'] for m in used))))
        if verbose:
            print(f'  branch: L={poly.length:7.1f} from_s={fs} to_s={ts} runs={len(bags)} core|d|={core:.1f} '
                  f'start=({poly.xy[0, 0]:.1f},{poly.xy[0, 1]:.1f}) end=({poly.xy[-1, 0]:.1f},{poly.xy[-1, 1]:.1f})',
                  flush=True)
    return branches, segs, lab


def name_branch(br, cyc_len):
    """Human-readable ids by location/topology (route-specific)."""
    x0, y0 = br['poly'].xy[0]
    x1, y1 = br['poly'].xy[-1]
    fs, ts = br['from_s'], br['to_s']
    if fs is not None and ts is not None and 1000 < fs < 1300 and 1600 < ts < 1900:
        return 'wb_detour'
    if fs is not None and 5300 < fs < 5450 and ts is None:
        return 'west_arrival_2'
    if x0 < -2100 and ts is not None and 5600 < ts < 5800:
        return 'fan_F2' if ts < 5740 else 'fan_F3'   # F1 = default fan track inside 'main'
    if x0 > -2100 and x0 < -1700:
        return 'eb_parallel_west'
    return 'branch'


# ----------------------------------------------------------------------------- stage 4: profiles
def edge_profiles(poly: Polyline, runs: dict, run_w, closed: bool, ds_out=0.5, gate=0.6, exclude=None):
    """Resample to ds_out and compute z, yaw, curvature, grade, support and spreads along s."""
    xy = resample(poly.xy, ds_out, closed=closed)
    P = Polyline(xy, closed=closed)
    pts = collect_points(runs, exclude=exclude)
    s, d, dist, seg, dpsi = P.project(pts['x'], pts['y'], pts['psi'], max_d=gate, max_dpsi=np.radians(45))
    ok = np.isfinite(s)
    # lateral support / spread in 2 m bins
    c2, med_d, nrun2, mad_d = robust_profile(s[ok], d[ok], pts['rid'][ok], P.length, binw=2.0, closed=closed,
                                             run_w=run_w, min_runs=1)
    c1, med_z, nrun_z, mad_z = robust_profile(s[ok], pts['z'][ok], pts['rid'][ok], P.length, binw=1.0,
                                              closed=closed, run_w=run_w, min_runs=1)
    zf = fill_and_smooth(med_z, 3.0, closed=closed, med_k=7)
    z = np.interp(P.s, c1, zf, period=P.length if closed else None)
    yaw = P.vertex_heading()
    curv = P.curvature(sigma_m=2.0)
    # geodetic / UTM
    lat, lon, h = geo.enu_to_geodetic(P.xy[:, 0], P.xy[:, 1], z, *D.MAP_ORIGIN)
    E, N, gam, kk = geo.latlon_to_utm(lat, lon, D.UTM_ZONE)
    hh = gaussian_filter1d(h, 10.0 / ds_out, mode='wrap' if closed else 'nearest')
    if closed:
        grade = (np.roll(hh, -1) - np.roll(hh, 1)) / (2 * ds_out)
    else:
        grade = np.gradient(hh, P.s)
    nr = np.interp(P.s, c2, np.nan_to_num(nrun2.astype(float)), period=P.length if closed else None)
    lm = np.interp(P.s, c2, np.nan_to_num(mad_d, nan=-1), period=P.length if closed else None)
    zm = np.interp(P.s, c1, np.nan_to_num(mad_z, nan=-1), period=P.length if closed else None)
    return dict(s=P.s, x=P.xy[:, 0], y=P.xy[:, 1], z=z, yaw=yaw, curvature=curv, grade=grade,
                lat=lat, lon=lon, h=h, utm_e=E, utm_n=N, n_runs=np.round(nr), lat_mad=lm, z_mad=zm), P


def main_sections(prof: dict, P: Polyline):
    """Name s-ranges of 'main' from geometry: double-track (opposite-direction track within 6 m) or not."""
    s, x, y, yaw = prof['s'], prof['x'], prof['y'], prof['yaw']
    s2, d2, dist2, _, _ = P.project(x, y, yaw + np.pi, max_d=6.0, max_dpsi=np.radians(30))
    dbl = np.isfinite(dist2)
    dbl = median_filter(dbl.astype(int), size=61, mode='wrap').astype(bool)
    # contiguous ranges
    ranges = []
    cur = dbl[0]
    a = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or dbl[i] != cur:
            ranges.append((bool(cur), float(s[a]), float(s[i - 1])))
            if i < len(s):
                cur, a = dbl[i], i
    return ranges, dbl


# ----------------------------------------------------------------------------- stage 5: export
CSV_COLS = ['s', 'x', 'y', 'z', 'yaw', 'curvature', 'grade', 'lat', 'lon', 'h', 'utm_e', 'utm_n', 'n_runs',
            'lat_mad', 'z_mad']
FMT = {'s': '%.3f', 'x': '%.4f', 'y': '%.4f', 'z': '%.4f', 'yaw': '%.6f', 'curvature': '%.6f', 'grade': '%.6f',
       'lat': '%.9f', 'lon': '%.9f', 'h': '%.4f', 'utm_e': '%.4f', 'utm_n': '%.4f', 'n_runs': '%d',
       'lat_mad': '%.4f', 'z_mad': '%.4f'}


def write_edge_csv(path, prof):
    arr = np.column_stack([prof[c] for c in CSV_COLS])
    np.savetxt(path, arr, delimiter=',', header=','.join(CSV_COLS), comments='', fmt=[FMT[c] for c in CSV_COLS])


def build(split='train', verbose=True):
    t0 = time.time()
    bags = D.split('train') if split == 'train' else D.split('train') + D.split('val')
    out_dir = OUT / ('map_train' if split == 'train' else 'map')
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob('edge_*.csv'):
        f.unlink()  # stale edges of previous builds
    runs = R.load_runs(bags)
    # stage 1-2
    cyc0 = seed_cycle(runs)
    pts = collect_points(runs)
    if verbose:
        print(f'[{split}] runs={len(runs)} RTK fixes={len(pts["x"])} seed length={cyc0.length:.2f}', flush=True)
    cyc1, h1 = refine(cyc0, pts, gates=(2.0, 1.5, 1.0), verbose=verbose)
    q = run_quality(cyc1, runs)
    w = quality_weights(runs, q)
    if verbose:
        print(f'  RTK-grade runs: {int(np.sum(w == 1.0))}/{len(w)}', flush=True)
    cyc, h2 = refine(cyc1, pts, gates=(1.0, 0.6, 0.4, 0.3, 0.3), run_w=w, verbose=verbose)
    q = run_quality(cyc, runs)
    # stage 3 + EM step: fixes on branch tracks must not pull the main line (e.g. at the detour divergence)
    branches, segs, lab = stage3(cyc, runs, w, verbose=verbose)
    excl = branch_exclusion(cyc, runs, segs)
    pts_main = collect_points(runs, exclude=excl)
    if verbose:
        print(f'  EM: re-refine main from the seed without {len(pts["x"]) - len(pts_main["x"])} deviating fixes',
              flush=True)
    cyc, h3 = refine(cyc0, pts_main, gates=(2.0, 1.5, 1.0), run_w=w, verbose=verbose)
    cyc, h4 = refine(cyc, pts_main, gates=(1.0, 0.6, 0.4, 0.3, 0.3), run_w=w, verbose=verbose)
    h2 = h2 + h3 + h4
    q = run_quality(cyc, runs)
    branches, segs, lab = stage3(cyc, runs, w, verbose=verbose)
    # stage 4-5
    prof_main, Pm = edge_profiles(cyc, runs, w, closed=True, exclude=excl)
    sections, dbl = main_sections(prof_main, Pm)
    edges_meta = []
    write_edge_csv(out_dir / 'edge_main.csv', prof_main)
    edges_meta.append(dict(id='main', file='edge_main.csv', closed=True, length=round(Pm.length, 4),
                           description='closed cycle: east platform -> east loop -> WB main -> west yard/loop -> '
                                       'fan F1 -> EB main -> east platform', from_=None,
                           double_track_ranges=[[a, b] for dd, a, b in sections if dd]))
    names = {}
    wdict = {b: float(x) for b, x in zip(runs, w)}
    for br in branches:
        nm = name_branch(br, Pm.length)
        names[nm] = names.get(nm, 0) + 1
        eid = nm if names[nm] == 1 else f'{nm}_{names[nm]}'
        br['id'] = eid
        bw = np.array([wdict[b] for b in runs])
        prof, Pb = edge_profiles(br['poly'], runs, bw, closed=False)
        write_edge_csv(out_dir / f'edge_{eid}.csv', prof)
        edges_meta.append(dict(id=eid, file=f'edge_{eid}.csv', closed=False, length=round(Pb.length, 4),
                               **{'from': None if br['from_s'] is None else dict(edge='main', s=round(br['from_s'], 3)),
                                  'to': None if br['to_s'] is None else dict(edge='main', s=round(br['to_s'], 3))},
                               support_runs=br['bags'], n_runs=br['n_runs'], core_offset_m=round(br['core_offset'], 2),
                               starts_in_run=br['starts_in_run'], ends_in_run=br['ends_in_run']))
    for em in edges_meta:
        em.pop('from_', None)
    meta = dict(
        version=1, created=time.strftime('%Y-%m-%d %H:%M:%S'), antenna='master',
        frame=dict(type='ENU', origin_lat=D.MAP_ORIGIN[0], origin_lon=D.MAP_ORIGIN[1], origin_h=D.MAP_ORIGIN[2],
                   ellipsoid='WGS84', axes='x=east, y=north, z=up [m]; local tangent plane at origin',
                   s='horizontal arc length in the ENU plane = true ground distance (rel. error < 1e-6)',
                   yaw='rad, ENU convention (0 = east, CCW positive), direction of travel'),
        utm=dict(epsg=D.EPSG_UTM, zone=D.UTM_ZONE, hemisphere='N',
                 note='utm_e/utm_n columns; UTM grid scale ~0.99972 and convergence ~-1.28 deg here'),
        topology=dict(note='tram is unidirectional; every edge is traversed in increasing s. main is a closed '
                           'cycle; branches are alternatives attached to main at from.s (diverge) / to.s (merge); '
                           'from/to = null: yard end (end of observed data)',
                      default_route='main'),
        build=dict(split=split, bags=list(runs), n_rtk_fixes=int(len(pts['x'])),
                   rtk_grade_runs=[b for b, x in zip(runs, w) if x == 1.0], refine_history=h1 + h2,
                   seconds=round(time.time() - t0, 1)),
        run_quality=q, edges=edges_meta)
    (out_dir / 'track_map.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    if verbose:
        print(f'  exported {len(edges_meta)} edges to {out_dir} in {time.time() - t0:.0f} s', flush=True)
    return meta, cyc, branches, runs, w


if __name__ == '__main__':
    split = sys.argv[1] if len(sys.argv) > 1 else 'train'
    build(split)
