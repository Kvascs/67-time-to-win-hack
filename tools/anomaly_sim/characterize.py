"""Statistics of the *natural* anomalies in the clean dataset (used to calibrate the simulator).

Per bag: message rates, gaps, header-vs-bag latency, stamp glitches, value repeats, negative values,
front/rear disagreement episodes and wheel-vs-GNSS disagreement episodes (robust GNSS reference:
master and rover velocity must agree). Writes ``natural_stats.csv``, ``natural_events.json`` and
``natural_summary.json``.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from .constants import (CMD, FRONT, GNSS_MASTER_VEL, GNSS_ROVER_VEL, KMH_PER_MS, NPZ_DIR, REAR, SPLITS_JSON,
                        VEHICLE)
from .context import _series_on_grid, sanitize_stamps, zoh
from .run import load_run


def episodes(mask: np.ndarray, t: np.ndarray, min_len: float = 0.3, merge: float = 0.3) -> list[tuple[int, int]]:
    """Index ranges [a, b) where ``mask`` holds for >= ``min_len`` s (gaps < ``merge`` s are bridged)."""
    m = mask.astype(int)
    d = np.diff(np.r_[0, m, 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out: list[list[int]] = []
    for a, b in zip(st, en):
        if out and t[a] - t[out[-1][1] - 1] < merge:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if t[b - 1] - t[a] >= min_len]


def gnss_time(s) -> np.ndarray:
    """GNSS measurement time robust to the +-1 s header-stamp segments seen in ~6 bags:
    bag receive time minus the bag-wide median latency (equals header time in normal segments)."""
    lat = s.t_bag - s.t_hdr
    body = lat[min(60, len(lat) // 2):]
    return s.t_bag - float(np.median(body))


def reference_speed(run, grid: np.ndarray | None = None, agree: float = 0.3):
    """Robust GNSS speed: mean of master & rover |v| where they agree (else NaN), on robust GNSS time."""
    m, r = run.streams.get(GNSS_MASTER_VEL), run.streams.get(GNSS_ROVER_VEL)
    if m is None or r is None or len(m) < 50 or len(r) < 50:
        return None, None
    tm, vm = gnss_time(m), np.hypot(m.val[:, 0], m.val[:, 1])
    tr, vr = gnss_time(r), np.hypot(r.val[:, 0], r.val[:, 1])
    om, orr = np.argsort(tm, kind='stable'), np.argsort(tr, kind='stable')
    tm, vm, tr, vr = tm[om], vm[om], tr[orr], vr[orr]
    if grid is None:
        grid = np.arange(max(tm[0], tr[0]), min(tm[-1], tr[-1]), 0.1)
    a = np.interp(grid, tm, vm, left=np.nan, right=np.nan)
    b = np.interp(grid, tr, vr, left=np.nan, right=np.nan)
    for t_src, arr in ((tm, a), (tr, b)):  # no interpolation across GNSS gaps > 0.5 s
        j = np.clip(np.searchsorted(t_src, grid), 1, len(t_src) - 1)
        arr[np.minimum(grid - t_src[j - 1], t_src[j] - grid) > 0.5] = np.nan
    ok = np.abs(a - b) < agree
    return grid, np.where(ok, 0.5 * (a + b), np.nan)


def _timing(s, t0) -> dict:
    tb, th = s.t_bag, s.t_hdr
    if len(tb) < 3:
        return {'n': int(len(tb))}
    dtb = np.diff(tb)
    lat = tb - th
    med = median_filter(lat, size=min(31, len(lat) | 1), mode='nearest')
    glitch = np.abs(lat - med) > 0.5
    body = slice(min(60, len(lat) // 2), None)  # skip the start-of-bag burst
    gaps = dtb > 0.5
    return {
        'n': int(len(tb)), 'rate_hz': float((len(tb) - 1) / max(tb[-1] - tb[0], 1e-9)),
        'dt_med': float(np.median(dtb)), 'gaps_gt_0.5s': int(gaps.sum()), 'gaps_gt_2s': int((dtb > 2).sum()),
        'max_gap_s': float(dtb.max()), 'max_gap_at_rel': float(tb[int(np.argmax(dtb))] - t0),
        'lat_med': float(np.median(lat[body])), 'lat_p99': float(np.percentile(lat[body], 99)),
        'lat_max': float(lat[body].max()), 'lat_min': float(lat[body].min()),
        'lat_neg_frac': float(np.mean(lat[body] < 0)), 'lat_spikes_gt_0.3s': int((lat[body] - med[body] > 0.3).sum()),
        'stamp_glitches': int(glitch.sum()), 'hdr_nonmono': int((np.diff(th) <= 0).sum()),
        'zero_stamps': int((th <= 0).sum()), 'start_burst_msgs': int(np.argmax(lat < 0.2)),
    }


def analyse_bag(bag: str, npz_dir: str | Path = NPZ_DIR) -> dict:
    run = load_run(Path(npz_dir) / f'{bag}.npz', name=bag)
    t0 = run.t_start
    res: dict = {'bag': bag, 'vehicle': bag.split('_')[0], 'duration_s': run.duration,
                 'has_gnss': run.has_gnss(), 'timing': {}, 'values': {}, 'events': []}
    for key in VEHICLE:
        res['timing'][key.split('__')[1]] = _timing(run[key], t0)
    g = run.streams.get(GNSS_MASTER_VEL)
    if g is not None and len(g) > 100:
        lat = g.t_bag - g.t_hdr
        dev = np.abs(lat - np.median(lat[60:]))
        res['gnss_stamp_offset_frac'] = float(np.mean(dev[60:] > 0.5))
        res['gnss_stamp_offset_s'] = float(np.sum(dev[60:] > 0.5) * 0.1)
    F, R, C = run[FRONT], run[REAR], run[CMD]
    for key, s in ((FRONT, F), (REAR, R)):
        v = s.val[:, 0]
        rep = (np.diff(v) == 0) & (v[1:] > 1.0)
        res['values'][key.split('__')[1]] = {
            'min_kmh': float(np.nanmin(v)) if len(v) else None, 'max_kmh': float(np.nanmax(v)) if len(v) else None,
            'n_negative': int((v < 0).sum()), 'n_nan': int((~np.isfinite(v)).sum()),
            'repeat_frac_moving': float(rep.sum() / max((v[1:] > 1.0).sum(), 1)),
            'min_nonzero_kmh': float(v[v > 0].min()) if (v > 0).any() else None}
    # gaps as events (single sensor vs joint)
    for key, s in ((FRONT, F), (REAR, R), (CMD, C)):
        tb = s.t_bag
        for i in np.flatnonzero(np.diff(tb) > 1.0):
            res['events'].append({'type': 'gap', 'topic': key.split('__')[1], 't0_rel': float(tb[i] - t0),
                                  'duration': float(tb[i + 1] - tb[i]),
                                  'hdr_gap': float(s.t_hdr[i + 1] - s.t_hdr[i])})
    # +-1 s stamp glitches
    for key, s in ((FRONT, F), (CMD, C)):
        if len(s) < 50:
            continue
        lat = s.t_bag - s.t_hdr
        med = median_filter(lat, size=31, mode='nearest')
        for i in np.flatnonzero(np.abs(lat - med) > 0.5)[:50]:
            if i > 60:
                res['events'].append({'type': 'stamp_glitch', 'topic': key.split('__')[1],
                                      't0_rel': float(s.t_bag[i] - t0), 'offset_s': float(med[i] - lat[i])})
    # front/rear disagreement (they share header stamps)
    if len(F) > 50 and len(R) > 50:
        common, fi, ri = np.intersect1d(F.t_hdr, R.t_hdr, return_indices=True)
        f, r = F.val[fi, 0], R.val[ri, 0]
        n = zoh(C.t_hdr, C.val[:, 0], common)
        mx = np.maximum(f, r)
        mask = np.abs(f - r) > np.maximum(1.0, 0.05 * mx)
        tg, vref = reference_speed(run, common) if run.has_gnss() else (None, None)
        for a, b in episodes(mask, common, 0.3):
            j = a + int(np.argmax(np.abs(f[a:b] - r[a:b])))
            e = {'type': 'front_rear_mismatch', 't0_rel': float(common[a] - t0),
                 'duration': float(common[b - 1] - common[a]), 'front_ms': float(f[j] / KMH_PER_MS),
                 'rear_ms': float(r[j] / KMH_PER_MS), 'notch_min': float(n[a:b].min()),
                 'notch_max': float(n[a:b].max())}
            if vref is not None and np.isfinite(vref[j]):
                e['gnss_ms'] = float(vref[j])
                bad = 'front' if abs(f[j] / KMH_PER_MS - vref[j]) > abs(r[j] / KMH_PER_MS - vref[j]) else 'rear'
                w = (f[j] if bad == 'front' else r[j]) / KMH_PER_MS
                e.update(bogie=bad, kind='slip' if w > vref[j] else 'slide',
                         rel=float((w - vref[j]) / max(vref[j], 0.5)))
            else:
                e['bogie_guess'] = ('front' if f[j] > r[j] else 'rear') if n[j] > 0 else \
                    ('front' if f[j] < r[j] else 'rear')
                e['kind_guess'] = 'slip' if n[j] > 0 else ('slide' if n[j] < 0 else 'unknown')
            res['events'].append(e)
    # wheel vs GNSS (catches both-bogie slides)
    if run.has_gnss() and len(F) > 50:
        grid = np.arange(F.t_hdr[60], F.t_hdr[-1], 0.1)
        tg, vref = reference_speed(run, grid)
        if vref is not None and np.isfinite(vref).sum() > 100:
            n = zoh(C.t_hdr, C.val[:, 0], grid)
            fw = _series_on_grid(sanitize_stamps(F.t_bag, F.t_hdr), F.val[:, 0], grid)  # NaN inside gaps
            rw = _series_on_grid(sanitize_stamps(R.t_bag, R.t_hdr), R.val[:, 0], grid) if len(R) > 50 else fw
            ok = np.isfinite(vref) & (vref > 3)
            k = float(np.nanmedian(np.r_[fw[ok], rw[ok]] / np.r_[vref[ok], vref[ok]])) if ok.sum() > 50 else KMH_PER_MS
            res['kmh_per_ms'] = k
            with np.errstate(all='ignore'):
                both = np.nanmean(np.vstack([fw, rw]), axis=0) / k
            err = both - vref
            res['wheel_vs_gnss'] = {'rmse_ms': float(np.sqrt(np.nanmean(err[np.isfinite(err)] ** 2))),
                                    'p99_abs_ms': float(np.nanpercentile(np.abs(err), 99))}
            thr = np.maximum(0.4, 0.06 * np.nan_to_num(vref))
            for key, w in (('front', fw / k), ('rear', rw / k)):
                e_b = w - vref
                for a, b in episodes(np.nan_to_num(np.abs(e_b)) > thr, grid, 0.3):
                    j = a + int(np.nanargmax(np.abs(e_b[a:b])))
                    res['events'].append({'type': 'wheel_vs_gnss', 'bogie': key, 't0_rel': float(grid[a] - t0),
                                          'duration': float(grid[b - 1] - grid[a]), 'err_ms': float(e_b[j]),
                                          'rel': float(e_b[j] / max(vref[j], 0.5)), 'gnss_ms': float(vref[j]),
                                          'notch_min': float(n[a:b].min()), 'notch_max': float(n[a:b].max())})
    return res


def _flat(res: dict) -> dict:
    row = {'bag': res['bag'], 'vehicle': res['vehicle'], 'duration_s': round(res['duration_s'], 1),
           'has_gnss': res['has_gnss'], 'kmh_per_ms': res.get('kmh_per_ms'),
           'gnss_stamp_offset_s': res.get('gnss_stamp_offset_s')}
    for topic, st in res['timing'].items():
        for k in ('rate_hz', 'max_gap_s', 'gaps_gt_0.5s', 'lat_med', 'lat_p99', 'lat_min', 'lat_neg_frac',
                  'stamp_glitches', 'hdr_nonmono', 'start_burst_msgs'):
            row[f'{topic}.{k}'] = st.get(k)
    for topic, st in res['values'].items():
        for k in ('n_negative', 'n_nan', 'repeat_frac_moving', 'min_nonzero_kmh', 'max_kmh'):
            row[f'{topic}.{k}'] = st.get(k)
    ev = res['events']
    row['n_front_rear_mismatch'] = sum(e['type'] == 'front_rear_mismatch' for e in ev)
    row['n_wheel_vs_gnss'] = sum(e['type'] == 'wheel_vs_gnss' for e in ev)
    row['n_gaps_gt_1s'] = sum(e['type'] == 'gap' for e in ev)
    row['n_stamp_glitch'] = sum(e['type'] == 'stamp_glitch' for e in ev)
    if 'wheel_vs_gnss' in res:
        row['wheel_vs_gnss_rmse'] = res['wheel_vs_gnss']['rmse_ms']
    return row


def characterize_dataset(out_dir: str | Path, bags=('all',), workers: int = 6, npz_dir=NPZ_DIR) -> dict:
    from .scenario import bags_from_spec
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = bags_from_spec(list(bags))
    splits = json.loads(Path(SPLITS_JSON).read_text()) if Path(SPLITS_JSON).exists() else {}
    dups = set(splits.get('duplicates', {}))
    names = [b for b in names if b not in dups]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(analyse_bag, names, [npz_dir] * len(names)))
    rows = [_flat(r) for r in results]
    import csv
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 'bag', k))
    with open(out_dir / 'natural_stats.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    catalogue = [{'bag': r['bag'], **e} for r in results for e in r['events']]
    (out_dir / 'natural_events.json').write_text(json.dumps(catalogue, indent=1), encoding='utf-8')
    summary = summarize_natural(results)
    (out_dir / 'natural_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    return summary


def summarize_natural(results: list[dict]) -> dict:
    ev = [dict(e, bag=r['bag']) for r in results for e in r['events']]
    lat_med = [r['timing']['front_bogie_velocity'].get('lat_med') for r in results
               if 'lat_med' in r['timing']['front_bogie_velocity']]
    gaps = [e for e in ev if e['type'] == 'gap']
    frm = [e for e in ev if e['type'] == 'front_rear_mismatch']
    wvg = [e for e in ev if e['type'] == 'wheel_vs_gnss']
    total_h = sum(r['duration_s'] for r in results) / 3600
    return {
        'n_bags': len(results), 'hours': total_h,
        'wheel_rate_hz_median': float(np.median([r['timing']['front_bogie_velocity'].get('rate_hz', np.nan)
                                                 for r in results])),
        'cmd_rate_hz_median': float(np.median([r['timing']['driver_position_cmd'].get('rate_hz', np.nan)
                                               for r in results])),
        'wheel_latency_median_s': float(np.median(lat_med)),
        'bags_with_negative_latency': int(sum(r['timing']['front_bogie_velocity'].get('lat_min', 0) < -0.05
                                              for r in results)),
        'bags_with_stamp_glitch': len({e['bag'] for e in ev if e['type'] == 'stamp_glitch'}),
        'gnss_stamp_offset': {'bags': [r['bag'] for r in results if r.get('gnss_stamp_offset_s', 0) > 1.0],
                              'seconds': {r['bag']: round(r['gnss_stamp_offset_s'], 1) for r in results
                                          if r.get('gnss_stamp_offset_s', 0) > 1.0}},
        'gaps_gt_1s': {
            'count': len(gaps), 'bags': len({e['bag'] for e in gaps}),
            'max_s': max((e['duration'] for e in gaps), default=0.0),
            'by_topic': {t: sum(e['topic'] == t for e in gaps) for t in ('front_bogie_velocity',
                                                                         'rear_bogie_velocity',
                                                                         'driver_position_cmd')},
            'durations_s': sorted(round(e['duration'], 1) for e in gaps)},
        'front_rear_mismatch': {'count': len(frm), 'bags': len({e['bag'] for e in frm}),
                                'per_hour': len(frm) / max(total_h, 1e-9),
                                'durations_s': sorted(round(e['duration'], 1) for e in frm)},
        'wheel_vs_gnss': {'count': len(wvg), 'bags': len({e['bag'] for e in wvg}),
                          'rel_p50': float(np.median([abs(e['rel']) for e in wvg])) if wvg else None,
                          'rel_max': float(max((abs(e['rel']) for e in wvg), default=0.0))},
        'negative_value_bags': [r['bag'] for r in results
                                if any(v.get('n_negative', 0) for v in r['values'].values())],
        'repeat_frac_moving_median': float(np.median([r['values']['front_bogie_velocity']['repeat_frac_moving']
                                                      for r in results])),
    }
