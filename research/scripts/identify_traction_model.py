"""Offline identification of the notch -> acceleration (traction / brake) model of the 71-911EM tram.

Reproduces the numbers in research/tram_traction_model.md.

Input : data/npz/*.npz  (produced by tools/extract_bags.py)
Output: research/traction_model_params.json

Model (Hammerstein: static nonlinearity + first-order lag, per unit mass):
    a_cmd(n, v) =  s(n) * min(A0, Pm / max(v, 0.5))           n > 0   (traction envelope scaled by notch)
                = -d(|n|)                                        n < 0   (deceleration demand, speed independent)
                =  0                                             n = 0
    tau * da_drv/dt = a_cmd - a_drv
    dv/dt = a_drv - (c0 + c1 v + c2 v^2) + b                    (b = grade / unmodelled bias)
Two traction parametrisations are fitted:
    'env'  : s(n) = min(1, n / n_sat) ** gamma                   (4 params)
    'table': s(n) free per notch, power limit scaled by n/15     (16 params)
Brake: d(k), k = 1..15 free.
Target: GNSS speed derivative (Savitzky-Golay, centred -> non-causal, fine offline).
Duplicated bags (identical wheel-speed payload) are removed; bags without GNSS are skipped.
"""
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter, savgol_filter

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / 'data' / 'npz'
OUT = ROOT / 'research' / 'traction_model_params.json'
DT = 0.05  # 20 Hz grid


def load_bags():
    seen, data = set(), []
    for fn in sorted(glob.glob(str(NPZ / '*.npz'))):
        f = np.load(fn)
        n = f['vehicle__driver_position_cmd']
        gv = f['sensing__gnss__master__vel']
        fb = f['vehicle__front_bogie_velocity']
        rb = f['vehicle__rear_bogie_velocity']
        if len(gv) < 600 or len(n) < 300 or len(fb) < 300 or len(rb) < 300:
            continue
        key = hashlib.md5(np.round(fb[:, 2], 5).tobytes()).hexdigest()
        if key in seen:  # duplicated recording under another hash
            continue
        seen.add(key)
        t = gv[:, 1]  # header stamps
        tg = np.arange(t[0] + 4, t[-1] - 1, DT)
        gs = np.interp(tg, t, np.hypot(gv[:, 2], gv[:, 3]))
        idx = np.clip(np.searchsorted(n[:, 1], tg, side='right') - 1, 0, len(n) - 1)
        data.append(dict(
            name=os.path.basename(fn)[:-4],
            v=savgol_filter(gs, 21, 2),
            a=savgol_filter(gs, 21, 2, deriv=1, delta=DT),
            n=n[idx, 2].astype(int),
            fv=np.interp(tg, fb[:, 1], fb[:, 2]) / 3.6,  # wheel sensors are km/h
            rv=np.interp(tg, rb[:, 1], rb[:, 2]) / 3.6,
        ))
    return data


def unpack(p, model):
    tau, c, brk = p[0], p[1:4], p[4:19]
    if model == 'env':
        return tau, c, brk, ('env',) + tuple(p[19:23])
    return tau, c, brk, ('table', p[19:34], p[34])


def a_cmd(n, v, brk, tr):
    a = np.zeros_like(v)
    nn = np.clip(n, 0, 15)
    vv = np.maximum(v, 0.5)
    pos = n > 0
    if tr[0] == 'env':
        _, nsat, gam, A0, Pm = tr
        a[pos] = (np.minimum(1.0, nn / nsat) ** gam * np.minimum(A0, Pm / vv))[pos]
    else:
        _, s, Pm = tr
        g = np.concatenate([[0.0], s])[nn]
        a[pos] = np.minimum(g, nn / 15.0 * Pm / vv)[pos]
    neg = n < 0
    a[neg] = -np.concatenate([[0.0], brk])[np.clip(-n[neg], 0, 15)]
    return a


def simulate(p, d, model):
    tau, c, brk, tr = unpack(p, model)
    al = DT / (abs(tau) + DT)
    af = lfilter([al], [1, -(1 - al)], a_cmd(d['n'], d['v'], brk, tr))
    v = d['v']
    return af - (c[0] + c[1] * v + c[2] * v * v)


def residuals(p, model, bags):
    out = []
    for d in bags:
        m = (d['v'] > 0.3) & np.isfinite(d['a'])
        out.append((simulate(p, d, model) - d['a'])[m][::2])
    return np.concatenate(out)


def transient_table(bags):
    """Median acceleration 0.6-2.0 s after entering a notch (runs <= 6 s, v > 1.5 m/s)."""
    acc = {k: [] for k in range(-15, 16)}
    for d in bags:
        n, v, a = d['n'], d['v'], d['a']
        chg = np.r_[True, n[1:] != n[:-1]]
        rid = np.cumsum(chg)
        run_len = np.bincount(rid)[rid] * DT
        start = np.maximum.accumulate(np.where(chg, np.arange(len(n)), 0))
        age = (np.arange(len(n)) - start) * DT
        m = (age >= 0.6) & (age <= 2.0) & (run_len <= 6) & (v > 1.5)
        for k in acc:
            acc[k].append(a[m & (n == k)])
    return {k: float(np.median(np.concatenate(x))) for k, x in acc.items() if sum(len(y) for y in x) > 30}


def main():
    bags = load_bags()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(bags))
    test = [bags[i] for i in perm[:14]]
    train = [bags[i] for i in perm[14:]]
    base = [0.35, 0.03, 0.003, 0.0003] + [0.15, 0.4, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 1.0, 1.2, 1.3, 1.35, 1.4, 1.45, 1.7]
    init = {
        'env': base + [9.0, 1.0, 0.95, 8.0],
        'table': base + list(np.minimum(0.1 * np.arange(1, 16), 0.9)) + [8.0],
    }
    result = {'n_unique_bags_with_gnss': len(bags), 'models': {}}
    for model, p0 in init.items():
        sol = least_squares(residuals, np.array(p0, float), args=(model, train),
                            loss='soft_l1', f_scale=0.3, max_nfev=60)
        rtr, rte = residuals(sol.x, model, train), residuals(sol.x, model, test)
        tau, c, brk, tr = unpack(sol.x, model)
        entry = dict(tau_s=float(tau), resist_c=[float(x) for x in c], brake_decel=[float(x) for x in brk],
                     train_rmse=float(np.sqrt(np.mean(rtr ** 2))), test_rmse=float(np.sqrt(np.mean(rte ** 2))),
                     test_mae=float(np.mean(np.abs(rte))))
        if model == 'env':
            entry.update(n_sat=float(tr[1]), gamma=float(tr[2]), A0=float(tr[3]), Pm=float(tr[4]))
        else:
            entry.update(s=[float(x) for x in tr[1]], Pm=float(tr[2]))
        result['models'][model] = entry
        print(model, json.dumps(entry, indent=1))
    result['transient_notch_accel'] = transient_table(bags)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('saved', OUT)


if __name__ == '__main__':
    main()
