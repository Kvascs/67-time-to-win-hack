"""Global localisation from the bogie ratio alone: match a window of held-out (val) data against
a ratio map of the whole 11 km cycle.

Map models (per 1 m bin of master-antenna arc s):
  'kappa'     : mu = M6 kappa regression (train fit, v>1.5), sigma = train residual std by kappa/speed class
  'empirical' : mu, sigma = mean / std of y over the TRAIN passes in that bin (sigma floored at 0.15 %)
  'emp_mean'  : place-specific mean mu(s) only; sigma = speed-only straight-track noise sigma_v(v)
  'emp_z'     : y normalised by sigma_v(v) (removes the "typical speed at this place" effect on the
                noise level), then place-specific mean and std of z per bin
Window: the last W metres of wheel-odometry travel (both bogies' mean speed integrated), y binned per 1 m;
window ends every 40 m along each val run.
Candidate end positions s0 = every 1 m of the cycle; LL(s0) = (1/tau) sum log N(y; mu(s0+d), sigma(s0+d)).
Success: |argmax - true s| <= 5 m (true s from RTK). 'Confident' = best LL exceeds the best LL outside
+-10 m of the argmax by ln(20).
Output: global_loc.txt, global_loc_windows.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C
import regress as R
import stub_test as T

_lines = []
MODELS = ('kappa', 'empirical', 'emp_mean', 'emp_z')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def main():
    pl = C.load_main()
    L = pl.L
    nb = int(np.ceil(L))
    mj = json.load(open(C.HERE / 'model_v1.5.json'))
    tau = mj['tau']
    b6 = np.array(mj['models']['M6 M3+kf^2,kr^2']['beta'])
    kb, vb, tab = T.noise_table()
    # ---- empirical map from train ----
    Tr, _ = R.load('train', 1.5)
    Tr = R.add_features(Tr)
    Tr['bin'] = np.floor(np.mod(Tr.s, L)).astype(int) % nb
    g = Tr.groupby('bin').y
    cnt = g.size().reindex(range(nb), fill_value=0).values
    mu_e = g.mean().reindex(range(nb)).to_numpy(copy=True)
    sd_e = g.std().reindex(range(nb)).to_numpy(copy=True)
    # bins with few samples -> neutral (mu 0, sigma 0.4 %)
    few = (cnt < 10) | ~np.isfinite(sd_e)
    mu_e[few] = 0.0
    sd_e[few] = 0.004
    sd_e = np.maximum(sd_e, 0.0015)
    say(f'empirical map: {nb} bins, {int((~few).sum())} with >=10 train samples')
    # ---- speed-only noise sigma_v(v) (straight track, train) and speed-normalised map z = y / sigma_v ----
    vgrid = np.arange(1.5, 16.01, 0.5)
    stt = Tr[Tr.kmax < 0.003]
    sv = np.array([stt.y[(stt.v >= a) & (stt.v < a + 0.5)].std() if ((stt.v >= a) & (stt.v < a + 0.5)).sum() > 200 else np.nan
                   for a in vgrid])
    ok = np.isfinite(sv)
    vc_, sv = vgrid[ok] + 0.25, sv[ok]

    def sigma_v(v):
        return np.interp(v, vc_, sv)
    say('speed-only sigma_v on straight [%] at v=2,3,5,8,12 m/s: ' +
        ' '.join(f'{100 * x:.2f}' for x in sigma_v(np.array([2, 3, 5, 8, 12.0]))))
    Tr['z'] = Tr.y / sigma_v(Tr.v.values)
    gz = Tr.groupby('bin').z
    mu_z = gz.mean().reindex(range(nb)).to_numpy(copy=True)
    sd_z = gz.std().reindex(range(nb)).to_numpy(copy=True)
    mu_z[few] = 0.0
    sd_z[few] = 1.0
    sd_z = np.maximum(sd_z, 0.5)
    # ---- kappa map ----
    sg = np.arange(nb) + 0.5
    kf, kr = pl.k_at(sg + C.D_FRONT), pl.k_at(sg + C.D_REAR)
    mu_k = np.c_[kf, kr, np.abs(kf), np.abs(kr), kf ** 2, kr ** 2] @ b6
    kmx = np.maximum(np.abs(kf), np.abs(kr))
    sd_k = [T.sigma_of(kmx, np.full(nb, 0.5 * (vb[i] + min(vb[i + 1], 12.0))), kb, vb, tab) for i in range(len(vb) - 1)]
    # ---- val runs ----
    V, _ = R.load('val', 1.5)
    V = R.add_features(V)
    rows = []
    for W in (50.0, 100.0, 200.0):
        for bag, idx in V.groupby('bag').indices.items():
            q = V.iloc[idx].sort_values('t')
            t = q.t.values
            # wheel odometry within contiguous stretches (gap < 1 s)
            dt = np.diff(t, prepend=t[0])
            brk = dt > 1.0
            dt[brk] = 0.0
            odo = np.cumsum(q.v.values * dt)
            seg = np.cumsum(brk)
            s_true = q.s.values
            for end in np.arange(W, odo[-1], 40.0):
                i_end = np.searchsorted(odo, end)
                if i_end >= len(odo):
                    break
                m = (odo > odo[i_end] - W) & (odo <= odo[i_end]) & (seg == seg[i_end])
                if m.sum() < 20:
                    continue
                d = odo[m] - odo[i_end]               # <= 0
                if d.min() > -0.8 * W:
                    continue
                y = q.y.values[m]
                v = q.v.values[m]
                j = np.floor(d).astype(int)            # bin offsets
                truth = np.mod(s_true[i_end], L)
                # which kappa class does the window cover (truth)?
                sw = np.mod(s_true[m], L)
                kwin = np.max(np.maximum(np.abs(pl.k_at(sw + C.D_FRONT)), np.abs(pl.k_at(sw + C.D_REAR))))
                res = dict(bag=bag, W=W, truth=truth, n=int(m.sum()), kmax_win=float(kwin), v_med=float(np.median(v)))
                # sufficient statistics per 1 m bin of the window
                uj, inv = np.unique(j, return_inverse=True)
                n_b = np.bincount(inv).astype(float)
                s1 = np.bincount(inv, weights=y)
                s2 = np.bincount(inv, weights=y * y)
                vbin = np.bincount(inv, weights=v) / n_b
                vc = np.clip(np.searchsorted(vb, np.maximum(vbin, 1.5), side='right') - 1, 0, len(vb) - 2)
                z = y / sigma_v(v)
                z1 = np.bincount(inv, weights=z)
                z2 = np.bincount(inv, weights=z * z)
                svb = sigma_v(vbin)
                for nm in MODELS:
                    LL = np.zeros(nb)
                    for b_i, jj in enumerate(uj):
                        a1, a2 = s1[b_i], s2[b_i]
                        if nm == 'empirical':
                            mu, sd = np.roll(mu_e, -jj), np.roll(sd_e, -jj)
                        elif nm == 'kappa':
                            mu, sd = np.roll(mu_k, -jj), np.roll(sd_k[vc[b_i]], -jj)
                        elif nm == 'emp_mean':          # place-specific mean, speed-only sigma
                            mu, sd = np.roll(mu_e, -jj), svb[b_i]
                        else:                           # 'emp_z': speed-normalised y, place-specific mean and sigma
                            mu, sd = np.roll(mu_z, -jj), np.roll(sd_z, -jj)
                            a1, a2 = z1[b_i], z2[b_i]
                        LL += -0.5 * (a2 - 2 * mu * a1 + n_b[b_i] * mu * mu) / (sd * sd) - n_b[b_i] * np.log(sd)
                    LL /= tau
                    best = int(np.argmax(LL))
                    circ = np.abs((np.arange(nb) - best + nb / 2) % nb - nb / 2)
                    second = np.max(LL[circ > 10])
                    err = (best + 0.5 - truth + L / 2) % L - L / 2
                    res[f'{nm}_err'] = float(err)
                    res[f'{nm}_margin'] = float(LL[best] - second)
                rows.append(res)
    D = pd.DataFrame(rows)
    D.to_csv(C.HERE / 'global_loc_windows.csv', index=False)
    say(f'val windows: {len(D)} (runs {D.bag.nunique()}), tau = {tau:.2f}')
    for W in (50.0, 100.0, 200.0):
        q = D[D.W == W]
        for sub, msk in (('all windows', np.ones(len(q), bool)), ('window has a pivot on |k|>0.02', q.kmax_win > 0.02),
                         ('window straight-ish (max|k|<0.01)', q.kmax_win < 0.01)):
            w = q[msk]
            if not len(w):
                continue
            parts = []
            for nm in MODELS:
                ok = np.abs(w[f'{nm}_err']) <= 5
                conf = w[f'{nm}_margin'] > np.log(20)
                prec = np.mean(ok[conf]) if conf.any() else np.nan
                parts.append(f'{nm}: within 5 m {ok.mean():.2f}, confident {conf.mean():.2f}, correct among confident {prec:.2f}')
            say(f'W={W:.0f} m, {sub:34s} n={len(w):4d} | ' + ' | '.join(parts))
    (C.HERE / 'global_loc.txt').write_text('\n'.join(_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
