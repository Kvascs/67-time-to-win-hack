"""Part 2: effective delays of the wheel-speed sensors w.r.t. the GNSS reference.

For every bag with GNSS we estimate (robust LS, scale re-fitted, sub-ms parabolic refinement) the lag of
front / rear bogie speed w.r.t.
  * GNSS master |v| (vel topic)                  -> what the judge compares our /result/velocity to
  * |d p/dt| of the master fix (central diff.)   -> what our integrated position is compared to
in four time-base combinations: (wheel stamp, GNSS stamp) in {hdr, bag} x {hdr, bag}.
Positive lag = the wheel signal is LATE (it shows at t+lag what the reference had at t).

Outputs: delays_per_bag.csv, delays.png
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing as T  # noqa: E402

OUT = Path(__file__).resolve().parent


def good_clock_mask(bag: T.Bag, t_hdr_query: np.ndarray, margin: float = 3.0) -> np.ndarray:
    """False within +/- margin s of GNSS-vs-vehicle clock anomalies (1-s steps / slews)."""
    tg, rel, anom = T.clock_offsets(bag)
    ok = np.ones(len(t_hdr_query), bool)
    if anom.any():
        ta = np.sort(tg[anom])
        k = np.clip(np.searchsorted(ta, t_hdr_query), 1, len(ta) - 1)
        d = np.minimum(np.abs(t_hdr_query - ta[k - 1]), np.abs(t_hdr_query - ta[k]))
        if len(ta) == 1:
            d = np.abs(t_hdr_query - ta[0])
        ok &= d > margin
    return ok


def analyze_bag(name: str) -> dict:
    warnings.simplefilter('ignore', RuntimeWarning)
    bag = T.load_bag(name)
    res = dict(bag=name, vehicle=bag.vehicle, t0=bag.t0)
    fm = T.gnss_fix(bag, 'master')
    if len(fm) < 600:
        res['skip'] = 'short/no gnss'
        return res
    ref = T.reference_trajectory(bag)
    vm = T.gnss_vel(bag, 'master')
    # ---- reference signals (drop stale start-up rows for bag-time use) ----
    vel_ok = ~T.stale_mask(vm.t_bag, vm.t_hdr)
    tvh, tvb, vsp = vm.t_hdr[vel_ok], vm.t_bag[vel_ok], vm.speed[vel_ok]
    o = np.argsort(tvh); tvh, tvb, vsp = tvh[o], tvb[o], vsp[o]
    clk_ok_v = good_clock_mask(bag, tvh)
    res['clock_anom_frac'] = float(1 - clk_ok_v.mean())
    # position-derivative reference (central differences; only fix-quality epochs)
    t, st = ref.t, ref.status
    dt = np.diff(t)
    cd_ok = np.r_[False, (dt[1:] < 0.15) & (dt[:-1] < 0.15), False]
    q2 = st == 2
    use_pos = (cd_ok & q2).sum() > 2000
    pm = cd_ok & q2 if use_pos else cd_ok
    res['posderiv_status2'] = bool(use_pos)
    fb = np.interp(t, np.sort(fm.t_hdr), fm.t_bag[np.argsort(fm.t_hdr)])  # bag time of fix rows
    clk_ok_p = good_clock_mask(bag, t)
    for which in ('front', 'rear'):
        w = T.wheel(bag, which)
        wok = ~T.stale_mask(w.t_bag, w.t_hdr)
        wth, wtb, wv = w.t_hdr[wok], w.t_bag[wok], w.v[wok]
        o = np.argsort(wth); wth_s, wv_h = wth[o], wv[o]
        o2 = np.argsort(wtb); wtb_s, wv_b = wtb[o2], wv[o2]
        combos = {
            'hh': (tvh, wth_s, wv_h), 'bb': (tvb, wtb_s, wv_b),
            'hb': (tvb, wth_s, wv_h),   # wheel hdr stamp vs GNSS bag time
            'bh': (tvh, wtb_s, wv_b),   # wheel bag time vs GNSS hdr stamp
        }
        for lab, (tr, ts, vs) in combos.items():
            r = T.estimate_lag(tr, vsp, ts, vs, lo=-0.8, hi=1.2, coarse=0.005, mask=clk_ok_v)
            res[f'{which}_vel_{lab}_lag'] = r['lag']
            res[f'{which}_vel_{lab}_rms'] = r['rms']
            if lab == 'hh':
                res[f'{which}_k'] = r['k']
                res[f'{which}_n'] = r['n']
        # position-derivative reference (hdr-hdr and bag-bag)
        r = T.estimate_lag(t[pm], ref.speed_pos[pm], wth_s, wv_h, lo=-0.8, hi=1.2, coarse=0.005, mask=clk_ok_p[pm])
        res[f'{which}_pos_hh_lag'] = r['lag']; res[f'{which}_pos_hh_rms'] = r['rms']
        r = T.estimate_lag(fb[pm], ref.speed_pos[pm], wtb_s, wv_b, lo=-0.8, hi=1.2, coarse=0.005, mask=clk_ok_p[pm])
        res[f'{which}_pos_bb_lag'] = r['lag']; res[f'{which}_pos_bb_rms'] = r['rms']
        # stability: lag in 4 consecutive chunks (hdr-hdr, vel reference)
        edges = np.linspace(tvh[0], tvh[-1], 5)
        chunk = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = clk_ok_v & (tvh >= a) & (tvh < b)
            rr = T.estimate_lag(tvh, vsp, wth_s, wv_h, lo=-0.8, hi=1.2, coarse=0.005, mask=m)
            chunk.append(rr['lag'])
        res[f'{which}_vel_hh_lag_chunks'] = ' '.join(f'{c:.4f}' for c in chunk)
        res[f'{which}_vel_hh_lag_chunk_std'] = float(np.nanstd(chunk))
        # accel vs decel phases separately (hdr-hdr, vel reference)
        acc = np.gradient(np.interp(tvh, tvh, vsp), tvh)
        from scipy.ndimage import uniform_filter1d
        acc = uniform_filter1d(acc, 5)
        for lab, m in (('acc', acc > 0.15), ('dec', acc < -0.15)):
            # widen masks by +/-1 s so that the lag search sees whole transients
            mm = np.convolve(m.astype(float), np.ones(21), 'same') > 0
            rr = T.estimate_lag(tvh, vsp, wth_s, wv_h, lo=-0.8, hi=1.2, coarse=0.005, mask=clk_ok_v & mm)
            res[f'{which}_vel_hh_lag_{lab}'] = rr['lag']
    # front vs rear directly (same header stamps -> lag should be ~0 unless the sensors differ)
    f, r_ = T.wheel(bag, 'front'), T.wheel(bag, 'rear')
    rr = T.estimate_lag(f.t_hdr, f.v, r_.t_hdr, r_.v, lo=-0.5, hi=0.5, coarse=0.005)
    res['rear_vs_front_lag'] = rr['lag']
    # GNSS vel vs position derivative (hdr), for reference
    rr = T.estimate_lag(tvh, vsp, t[pm], ref.speed_pos[pm], lo=-0.4, hi=0.4, coarse=0.005, mask=clk_ok_v)
    res['posderiv_vs_vel_lag'] = rr['lag']
    # nominal stamp offsets (bag - hdr) in ms, excluding start-up burst
    for key, lab in (('front', 'front'), ('cmd', 'cmd'), ('master_fix', 'mfix'), ('master_vel', 'mvel'), ('rover_fix', 'rfix')):
        a = bag.arr(key)
        off = a[:, 0] - a[:, 1]
        off = off[off < 0.5]
        res[f'off_{lab}_med_ms'] = float(np.median(off) * 1e3) if len(off) else np.nan
        res[f'off_{lab}_p05_ms'] = float(np.percentile(off, 5) * 1e3) if len(off) else np.nan
        res[f'off_{lab}_p95_ms'] = float(np.percentile(off, 95) * 1e3) if len(off) else np.nan
    return res


def safe(name):
    try:
        return analyze_bag(name)
    except Exception as e:
        return dict(bag=name, skip=f'error {e!r}')


def main():
    bags = T.list_bags('train', 'val', 'short')
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(safe, bags))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'delays_per_bag.csv', index=False, float_format='%.5g')
    pd.set_option('display.width', 250, 'display.max_columns', 80, 'display.max_rows', 200)
    ok = df[df['skip'].isna()] if 'skip' in df else df
    cols = [c for c in ok.columns if c.endswith('_lag') or c.endswith('_rms') or c.endswith('_k') or c.startswith('off_')
            or c.endswith('chunk_std') or c.endswith('_acc') or c.endswith('_dec')]
    print(ok[cols].describe().T.to_string())


if __name__ == '__main__':
    main()
