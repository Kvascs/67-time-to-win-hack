"""Calibrate the cue models of the global localiser on TRAIN bags only (honest for val).

Writes cache/calib_train.npz:
  lm_s, lm_sigma, lm_p        stop landmarks (landmarks.csv positions) with stop probabilities re-estimated
                              for the online stop detector (standstill flag, dwell >= DWELL)
  stop_rate                   stops away from any landmark, per metre
  cut_s, cut_sigma, cut_q     cut-off places with P(cut-off | pass with notch >= 4)
  cut_rate                    cut-offs away from the known places, per metre (conservative)
  vmax_s, vmax                speed upper profile (max estimator speed over train passes, 5 m bins)
  dprof_s, dprof              median estimator disturbance d per 10 m bin (no-GNSS replay)
  start_s, start_w            where train runs start at rest (for the optional start prior)
"""
from __future__ import annotations

import numpy as np

import common as C
import data as D

DWELL = 1.5


def cyc(a, L):
    return (a + 0.5 * L) % L - 0.5 * L


def main(split='train'):
    m = C.load_map()
    L = m.length
    lm = C.load_places(C.MAPS / 'landmarks.csv')
    cu = C.load_places(C.MAPS / 'cutoffs.csv')
    lm_s, lm_sig = lm.s.to_numpy(), lm.sigma.to_numpy()
    passes, stops = np.zeros(len(lm)), np.zeros(len(lm))
    cpass, cev = np.zeros(len(cu)), np.zeros(len(cu))
    n_rand, dist_tot, n_cut_rand = 0, 0.0, 0
    vbin, dbin = 5.0, 10.0
    nv, nd = int(np.ceil(L / vbin)), int(np.ceil(L / dbin))
    vmax = np.zeros(nv)
    vcnt = np.zeros(nv)
    dacc = [[] for _ in range(nd)]
    starts = []
    for name in C.SPLITS[split]:
        b = D.load_bag(name)
        if not b.has_truth:
            continue
        ok = b.truth_ok & np.isfinite(b.truth_s)
        s0, s1 = np.nanmin(b.truth_s[ok]), np.nanmax(b.truth_s[ok])
        dist_tot += s1 - s0
        for i in range(len(lm)):
            passes[i] += max(0, np.floor((s1 - 5 - lm_s[i]) / L) - np.ceil((s0 + 5 - lm_s[i]) / L) + 1)
        for st in D.stop_events(b, DWELL):
            ss = b.truth_s[st['i0']:st['i1'] + 1]
            ss = ss[np.isfinite(ss)]
            if len(ss) == 0:
                continue
            sw = float(np.median(ss)) % L
            dd = cyc(lm_s - sw, L)
            j = int(np.argmin(np.abs(dd)))
            assoc = abs(dd[j]) <= max(3 * np.hypot(lm_sig[j], 0.3), 1.5)
            if st['initial']:
                starts.append(sw)
                continue
            if st['final']:
                continue
            if assoc:
                stops[j] += 1
            else:
                n_rand += 1
        # cut-offs
        evs = []
        for e in D.cutoff_events(b):
            st_ = np.interp(e['t'], b.t, b.truth_s) + e['v'] * D.GNSS_LEAD_S
            if not np.isfinite(st_):
                continue
            dd = cyc(cu.s.to_numpy() - st_ % L, L)
            j = int(np.argmin(np.abs(dd)))
            if abs(dd[j]) < 3 * cu.sigma.iat[j] + 6.0:
                evs.append((j, st_))
            else:
                n_cut_rand += 1
        for i, r in cu.iterrows():
            for n in range(int(np.ceil((s0 + 30 - r.s) / L)), int(np.floor((s1 - 5 - r.s) / L)) + 1):
                sc = r.s + n * L
                k = np.flatnonzero(ok & (b.truth_s >= sc - 15) & (b.truth_s <= sc - 3))
                if len(k) == 0:
                    continue
                nt = b.notch[np.clip(np.searchsorted(b.notch_t, b.t[k[-1]]) - 1, 0, len(b.notch) - 1)]
                if nt >= 4:
                    cpass[i] += 1
                    cev[i] += any(j == i and abs(se - sc) < 3 * r.sigma + 6.0 for j, se in evs)
        # profiles
        sw = np.mod(b.truth_s[ok], L)
        iv = (sw / vbin).astype(int) % nv
        np.maximum.at(vmax, iv, b.v[ok])
        np.add.at(vcnt, iv, 1)
        mv = ok & (b.v > 3.0) & ((b.flags & C.FLAG_STANDSTILL) == 0) & (b.mu3 < 0.5)
        k = np.flatnonzero(mv)
        k = k[np.unique(np.floor(b.t[k]), return_index=True)[1]]
        for i, v in zip((np.mod(b.truth_s[k], L) / dbin).astype(int) % nd, b.d[k]):
            dacc[i].append(v)
    lm_p = (stops + 0.5) / (passes + 1.0)
    lm_p[passes < 3] = 0.3                      # terminals: few mid-run passes, keep a neutral value
    lm_p = np.clip(lm_p, 0.03, 0.95)
    cut_q = np.clip((cev + 0.5) / (cpass + 1.0), 0.05, 0.95)
    dprof = np.array([np.median(a) if len(a) >= 3 else np.nan for a in dacc])
    dprof = np.where(np.isfinite(dprof), dprof, np.nanmedian(dprof))
    vmax[vcnt < 20] = 30.0                      # too few samples: no constraint
    starts = np.array(starts)
    out = dict(lm_s=lm_s, lm_sigma=lm_sig, lm_p=lm_p, lm_passes=passes, lm_stops=stops,
               stop_rate=n_rand / dist_tot, cut_s=cu.s.to_numpy(), cut_sigma=cu.sigma.to_numpy(), cut_q=cut_q,
               cut_rate=max(n_cut_rand, 1) / dist_tot, vmax_s=np.arange(nv) * vbin, vmax=vmax,
               dprof_s=np.arange(nd) * dbin + 0.5 * dbin, dprof=dprof, start_s=starts, dist_tot=dist_tot)
    np.savez(C.CACHE / f'calib_{split}.npz', **out)
    print(f'distance {dist_tot / 1000:.1f} km; random stops {n_rand} ({1000 * n_rand / dist_tot:.2f}/km); '
          f'random cut-offs {n_cut_rand}')
    print('landmark p:', np.round(lm_p, 2))
    print('cut-off q:', np.round(cut_q, 2), 'passes', cpass, 'events', cev)
    print('run starts at rest:', len(starts), np.round(np.sort(starts), 1))


if __name__ == '__main__':
    main()
