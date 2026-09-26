"""Experiment (a): can a quasi-Newton method tune continuous filter parameters against a replay metric?

Parameters (log-multipliers x_i = ln(p_i / p_i0) of the final2 defaults):
    sigma_accel 0.12, sigma_wheel 0.05, q_disturbance 0.004, landmark_assoc_q 0.008
Objective on a TRAIN subset (6 bags, train-only maps, jury-like base_link reference):
    J(x) = 0.5 * mean_v_rmse(x) / mean_v_rmse(0) + 0.5 * mean_along_rmse(x) / mean_along_rmse(0)
    (speed averaged over the bags with a good Doppler reference; J(0) = 1)

    python a_tune.py scan      # coordinate scans at h = +-0.03, +-0.1, +-0.35 (FD gradients at 3 steps)
    python a_tune.py lbfgsb    # L-BFGS-B, forward differences h = 0.1, bounds |x| <= 1.1, budget maxfun
    python a_tune.py val X1 X2 X3 X4   # baseline vs point on the 17 VAL bags, paired bootstrap
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qn_harness as H  # noqa: E402

PARAMS = [('sigma_accel', 0.12), ('sigma_wheel', 0.05), ('q_disturbance', 0.004), ('landmark_assoc_q', 0.008)]
SUBSET = ['30618_e9a34502', '30618_3e9f4952', '30639_927002c2',   # normal runs
          '30618_2050d396', '30618_1cc230fa', '30618_27e994fc']   # landmark-association sensitive
# Doppler reference is bad on 27e994fc (ref_good False in eval_par): speed is averaged over the others
V_BAGS = SUBSET[:5]
LOG = HERE / 'a_tune.log'


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(s + '\n')


def sets_of(x) -> dict:
    return {n: p0 * float(np.exp(xi)) for (n, p0), xi in zip(PARAMS, x)}


class Objective:
    def __init__(self, ev: H.Evaluator, bags=SUBSET, vbags=V_BAGS):
        self.ev, self.bags, self.vbags = ev, bags, vbags
        b = self.parts(np.zeros(len(PARAMS)))
        self.v0, self.a0 = b['V'], b['A']

    def parts_many(self, xs):
        out = []
        for df in self.ev.evaluate_many(self.bags, [sets_of(x) for x in xs]):
            V = df.set_index('bag').loc[self.vbags, 'v_rmse'].mean()
            A = df['along_rmse'].mean()
            out.append({'V': V, 'A': A, 'df': df})
        return out

    def parts(self, x):
        return self.parts_many([x])[0]

    def J(self, p):
        return 0.5 * p['V'] / self.v0 + 0.5 * p['A'] / self.a0


def scan(obj: Objective):
    hs = [0.03, 0.1, 0.35]
    pts, meta = [np.zeros(len(PARAMS))], [('base', 0.0)]
    for i, (n, _) in enumerate(PARAMS):
        for h in hs:
            for sgn in (-1, 1):
                x = np.zeros(len(PARAMS))
                x[i] = sgn * h
                pts.append(x)
                meta.append((n, sgn * h))
    t0 = time.time()
    res = obj.parts_many(pts)
    rows = []
    for (n, h), p in zip(meta, res):
        per = p['df'].set_index('bag')
        rows.append({'param': n, 'x': h, 'mult': np.exp(h), 'J': obj.J(p), 'V': p['V'], 'A': p['A'],
                     **{f'al_{b[6:]}': per.loc[b, 'along_rmse'] for b in SUBSET},
                     **{f'v_{b[6:]}': per.loc[b, 'v_rmse'] for b in SUBSET}})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'a_scan.csv', index=False)
    log(f'[scan] {len(pts)} points x {len(SUBSET)} bags, replays run {obj.ev.n_replays}, '
        f'wall {time.time() - t0:.0f} s, mean replay {obj.ev.replay_wall / max(obj.ev.n_replays, 1):.1f} s')
    log(df[['param', 'x', 'mult', 'J', 'V', 'A']].round(5).to_string(index=False))
    # finite-difference gradients (central) of J at the three step sizes, per parameter
    g = []
    for n, _ in PARAMS:
        d = df[df.param == n].set_index('x')
        g.append({'param': n, **{f'g_h{h}': (d.loc[h, 'J'] - d.loc[-h, 'J']) / (2 * h) for h in hs},
                  **{f'fwd_h{h}': (d.loc[h, 'J'] - 1.0) / h for h in hs},
                  **{f'bwd_h{h}': (1.0 - d.loc[-h, 'J']) / h for h in hs}})
    g = pd.DataFrame(g)
    g.to_csv(HERE / 'a_scan_grad.csv', index=False)
    log('[scan] finite-difference dJ/dx (x = ln multiplier):')
    log(g.round(4).to_string(index=False))


def lbfgsb(obj: Objective, h=0.1, maxfun=7, scale=0.3):
    """Optimiser variable y = x / scale (a unit step = x +-0.3, i.e. x1.35), forward differences with step h in x,
    box |x| <= 1.05 (multipliers 0.35..2.86). Every call costs 1 + 4 parameter points (L-BFGS-B's line search
    needs f and g at every trial point)."""
    from scipy.optimize import minimize
    trace = []
    t0 = time.time()

    def fg(y):
        x = scale * np.array(y, float)
        xs = [x] + [x + h * np.eye(len(PARAMS))[i] for i in range(len(PARAMS))]
        ps = obj.parts_many(xs)
        J = [obj.J(p) for p in ps]
        grad = np.array([(J[i + 1] - J[0]) / h for i in range(len(PARAMS))])
        trace.append({'call': len(trace), 'x': np.round(x, 4).tolist(), 'x_exact': [float(v) for v in x],
                      'mult': np.round(np.exp(x), 4).tolist(),
                      'J': J[0], 'V': ps[0]['V'], 'A': ps[0]['A'], 'grad_x': np.round(grad, 4).tolist(),
                      'replays': obj.ev.n_replays, 'wall_s': round(time.time() - t0)})
        log('[lbfgsb]', json.dumps(trace[-1]))
        return J[0], grad * scale

    r = minimize(fg, np.zeros(len(PARAMS)), jac=True, method='L-BFGS-B', bounds=[(-1.05 / scale, 1.05 / scale)] * len(PARAMS),
                 options={'maxfun': maxfun, 'maxiter': maxfun, 'maxls': 5})
    best = min(trace, key=lambda t: t['J'])
    log(f'[lbfgsb] status {r.status} "{r.message}", nfev {r.nfev}, nit {r.nit}; best J {best["J"]:.5f} at x={best["x"]} '
        f'(mult {best["mult"]}); replays {obj.ev.n_replays}, wall {time.time() - t0:.0f} s')
    pd.DataFrame(trace).to_csv(HERE / 'a_lbfgsb_trace.csv', index=False)
    (HERE / 'a_lbfgsb_best.json').write_text(json.dumps(best))
    return best


def val(ev: H.Evaluator, x):
    bags = H.splits()['val']
    t0 = time.time()
    d0, d1 = ev.evaluate_many(bags, [sets_of(np.zeros(len(PARAMS))), sets_of(np.array(x, float))])
    m = d0.set_index('bag')[['v_rmse', 'along_rmse', 'p3_rmse']].join(
        d1.set_index('bag')[['v_rmse', 'along_rmse', 'p3_rmse']], rsuffix='_new')
    m.to_csv(HERE / f'a_val_{"_".join(f"{v:+.3f}" for v in x)}.csv')
    rng = np.random.default_rng(0)
    log(f'[val] x={list(x)} mult={np.round(np.exp(x), 4).tolist()} bags={len(bags)} wall {time.time() - t0:.0f} s')
    for c in ('v_rmse', 'along_rmse', 'p3_rmse'):
        a, b = m[c].to_numpy(), m[c + '_new'].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        rel = b.mean() / a.mean() - 1
        bs = []
        for _ in range(4000):
            i = rng.integers(0, len(a), len(a))
            bs.append(b[i].mean() / a[i].mean() - 1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        log(f'  {c:10s} mean {a.mean():.4f} -> {b.mean():.4f} ({100 * rel:+.2f} %, 95 % CI {100 * lo:+.2f}..{100 * hi:+.2f} %)'
            f'  median {np.median(a):.4f} -> {np.median(b):.4f}; better {int((b < a - 1e-9).sum())} / worse '
            f'{int((b > a + 1e-9).sum())} of {len(a)}')


def main():
    what = sys.argv[1]
    ev = H.Evaluator()
    try:
        if what == 'val':
            if sys.argv[2] == 'best':
                x = json.loads((HERE / 'a_lbfgsb_best.json').read_text())['x_exact']
            elif sys.argv[2] == 'scanbest':  # best single coordinate move of the scan
                sc = pd.read_csv(HERE / 'a_scan.csv')
                r = sc.loc[sc.J.idxmin()]
                x = [float(r.x) if n == r.param else 0.0 for n, _ in PARAMS]
            else:
                x = [float(v) for v in sys.argv[2:6]]
            val(ev, x)
            return
        obj = Objective(ev)
        log(f'[{what}] base V={obj.v0:.5f} A={obj.a0:.4f}')
        if what == 'scan':
            scan(obj)
        elif what == 'lbfgsb':
            lbfgsb(obj, h=float(sys.argv[2]) if len(sys.argv) > 2 else 0.1,
                   maxfun=int(sys.argv[3]) if len(sys.argv) > 3 else 7)
    finally:
        log(f'[{what}] replays run in this call: {ev.n_replays}, replay wall sum {ev.replay_wall:.0f} s')
        ev.close()


if __name__ == '__main__':
    main()
