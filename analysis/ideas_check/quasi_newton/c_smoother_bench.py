"""Experiment (c), cost part: a 1-D fixed-lag MAP smoother over [s, v, d] per wheel epoch + wheel scale k,
solved by Gauss-Newton (sparse Cholesky/LU of J^T W J) vs L-BFGS on the same cost. Synthetic but with the
estimator's noise levels, so the numbers are about the solver, not about accuracy.

Factors (window of T seconds at f Hz, h = 1/f):
  s_{i+1} - s_i - h v_i                      sigma 1 cm          (kinematics, near-hard)
  v_{i+1} - v_i - h (a_i + d_i)             sigma_a sqrt(h), 0.12  (model acceleration a_i known from the notch)
  d_{i+1} - d_i                              sqrt(q_d h), q_d 0.004
  (1 + k) v_i - z_i  (two bogies)            sigma_w 0.05          (bilinear -> Gauss-Newton needs >1 iteration)
  s_j - L_j at stops (Cauchy loss, c = 1 m)  sigma 0.4 m, one of them a wrong association 3 m off
  prior on s_0, v_0, d_0, k                  (marginal of the part that left the window)

    python c_smoother_bench.py
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

rng = np.random.default_rng(0)
SA, QD, SW, SS, SL, CAU = 0.12, 0.004, 0.05, 1e-2, 0.4, 1.0


def make(T, f):
    h = 1.0 / f
    n = int(T * f)
    t = np.arange(n) * h
    a = 1.0 * np.sin(2 * np.pi * t / 60.0)                       # accelerate / brake cycles
    d_true = 0.05 * np.sin(2 * np.pi * t / 23.0)
    v = np.maximum(np.cumsum((a + d_true) * h) + 8.0, 0.0)
    s = np.cumsum(v * h)
    k_true = 0.01
    z = np.stack([(1 + k_true) * v + SW * rng.standard_normal(n) for _ in range(2)])
    stops = np.linspace(n // 6, n - 1, 4).astype(int)
    L = s[stops] + SL * rng.standard_normal(len(stops))
    L[1] += 3.0                                                   # wrong association (queue 3 m before a platform)
    return dict(n=n, h=h, a=a, z=z, stops=stops, L=L, s=s, v=v, k=k_true)


def residuals(x, P):
    n, h = P['n'], P['h']
    s, v, d, k = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n]
    r = [(s[1:] - s[:-1] - h * v[:-1]) / SS,
         (v[1:] - v[:-1] - h * (P['a'][:-1] + d[:-1])) / (SA * np.sqrt(h)),
         (d[1:] - d[:-1]) / np.sqrt(QD * h),
         ((1 + k) * v - P['z'][0]) / SW, ((1 + k) * v - P['z'][1]) / SW,
         np.array([(s[0] - P['s'][0]) / 0.5, (v[0] - P['v'][0]) / 0.3, d[0] / 0.1, k / 0.015]),
         (s[P['stops']] - P['L']) / SL]
    return r


def jacobian(x, P):
    n, h = P['n'], P['h']
    v, k = x[n:2 * n], x[3 * n]
    rows, cols, vals = [], [], []
    r0 = 0

    def add(rr, cc, vv):
        rows.append(rr), cols.append(cc), vals.append(vv)

    i = np.arange(n - 1)
    add(r0 + i, i + 1, np.full(n - 1, 1 / SS)); add(r0 + i, i, np.full(n - 1, -1 / SS)); add(r0 + i, n + i, np.full(n - 1, -h / SS))
    r0 += n - 1
    w = 1 / (SA * np.sqrt(h))
    add(r0 + i, n + i + 1, np.full(n - 1, w)); add(r0 + i, n + i, np.full(n - 1, -w)); add(r0 + i, 2 * n + i, np.full(n - 1, -h * w))
    r0 += n - 1
    w = 1 / np.sqrt(QD * h)
    add(r0 + i, 2 * n + i + 1, np.full(n - 1, w)); add(r0 + i, 2 * n + i, np.full(n - 1, -w))
    r0 += n - 1
    j = np.arange(n)
    for _ in range(2):
        add(r0 + j, n + j, np.full(n, (1 + k) / SW)); add(r0 + j, np.full(n, 3 * n), v / SW)
        r0 += n
    add(np.array([r0, r0 + 1, r0 + 2, r0 + 3]), np.array([0, n, 2 * n, 3 * n]), np.array([1 / 0.5, 1 / 0.3, 1 / 0.1, 1 / 0.015]))
    r0 += 4
    m = len(P['stops'])
    add(r0 + np.arange(m), P['stops'], np.full(m, 1 / SL))
    r0 += m
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, 3 * n + 1))


def cost_weights(r):
    """Cauchy on the landmark block (last), squared elsewhere; IRLS weights."""
    rl = r[-1]
    wl = 1.0 / (1.0 + (rl * SL / CAU) ** 2)
    c = 0.5 * sum(float(np.sum(q ** 2)) for q in r[:-1]) + 0.5 * float(np.sum((CAU / SL) ** 2 * np.log1p((rl * SL / CAU) ** 2)))
    w = np.concatenate([np.ones(sum(len(q) for q in r[:-1])), wl])
    return c, w


def perm_interleaved(n):
    """Variable order [s_0, v_0, d_0, s_1, ...] -> J^T W J is banded (block-tridiagonal); k is kept last."""
    return np.r_[np.stack([np.arange(n), n + np.arange(n), 2 * n + np.arange(n)], 1).ravel(), 3 * n]


def banded_solve(A, g, n):
    """Bordered banded solve: A = [[B, c], [c^T, e]] with B banded (interleaved [s, v, d]), k = last variable."""
    from scipy.linalg import solveh_banded
    p = perm_interleaved(n)
    Ap = A[p][:, p].tocsr()
    gp = g[p]
    m = 3 * n
    B = Ap[:m, :m].tocoo()
    bw = int(np.max(np.abs(B.row - B.col)))
    ab = np.zeros((bw + 1, m))
    up = B.col >= B.row
    ab[bw + B.row[up] - B.col[up], B.col[up]] = B.data[up]
    c = Ap[:m, m].toarray().ravel()
    e = Ap[m, m]
    y = solveh_banded(ab, np.stack([gp[:m], c], 1))
    dk = (gp[m] - c @ y[:, 0]) / (e - c @ y[:, 1])
    dxp = np.r_[y[:, 0] - y[:, 1] * dk, dk]
    dx = np.empty_like(dxp)
    dx[p] = dxp
    return dx, bw


def gauss_newton(P, x0, iters=30):
    x = x0.copy()
    t0 = time.perf_counter()
    t_solve = 0.0
    c_prev = np.inf
    for it in range(iters):
        r = residuals(x, P)
        c, w = cost_weights(r)
        if abs(c_prev - c) < 1e-9 * max(c, 1.0):
            break
        c_prev = c
        J = jacobian(x, P)
        rv = np.concatenate(r)
        A = (J.T @ sp.diags(w) @ J).tocsc()
        g = J.T @ (w * rv)
        t1 = time.perf_counter()
        dx, bw = banded_solve(A, -g, P['n'])
        t_solve += time.perf_counter() - t1
        x = x + dx
    return x, it, time.perf_counter() - t0, cost_weights(residuals(x, P))[0], t_solve / max(it, 1), bw


def lbfgs(P, x0, x_ref, maxiter=5000, precond=False):
    """L-BFGS on the same cost; precond=True: Jacobi scaling x = D^-1/2 y with D = diag(J^T W J) at x0."""
    n = P['n']
    hist = []
    if precond:
        J = jacobian(x0, P)
        D = np.asarray((J.multiply(J)).sum(0)).ravel()
        sc = 1.0 / np.sqrt(D)
    else:
        sc = np.ones_like(x0)

    def fg(y):
        x = y * sc
        r = residuals(x, P)
        c, w = cost_weights(r)
        J = jacobian(x, P)
        return c, (J.T @ (w * np.concatenate(r))) * sc

    t0 = time.perf_counter()
    res = minimize(fg, x0 / sc, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'maxfun': 3 * maxiter,
                                                                       'maxcor': 20, 'ftol': 1e-15, 'gtol': 1e-10},
                   callback=lambda yk: hist.append(np.max(np.abs(yk[:n] * sc[:n] - x_ref[:n]))))
    ds = np.array(hist)
    it_1cm = int(np.argmax(ds < 0.01)) + 1 if np.any(ds < 0.01) else None
    return res, it_1cm, time.perf_counter() - t0, float(ds[-1]) if len(ds) else np.nan


def main():
    for T, f in ((60, 10), (120, 10), (120, 50)):
        P = make(T, f)
        n = P['n']
        x0 = np.concatenate([P['s'][0] + np.cumsum(np.full(n, P['h']) * P['z'].mean(0)), P['z'].mean(0), np.zeros(n), [0.0]])
        x_gn, it, t_gn, c_gn, t_it, bw = gauss_newton(P, x0)
        line = (f'T={T:3d} s f={f:2d} Hz: {3 * n + 1} unknowns, band {bw} | GN+IRLS: {it} it, {1e3 * t_gn:.0f} ms total '
                f'(Python), banded solve {1e3 * t_it:.2f} ms/it; k={x_gn[3 * n]:.4f} (true 0.0100), weight of the wrong '
                f'landmark {1 / (1 + ((x_gn[P["stops"][1]] - P["L"][1]) / CAU) ** 2):.2f}, cost {c_gn:.2f}')
        print(line, flush=True)
        if n <= 1200:
            for pc in (False, True):
                res, it_1cm, t_lb, ds_last = lbfgs(P, x0, x_gn, precond=pc)
                print(f'    L-BFGS{" (Jacobi-scaled)" if pc else ""}: {res.nit} it, {res.nfev} f+g, {1e3 * t_lb:.0f} ms; '
                      f'to 1 cm of the GN solution after {it_1cm} it; final max|ds - ds_GN| {ds_last:.3f} m; '
                      f'cost {res.fun:.2f}', flush=True)


if __name__ == '__main__':
    main()
