"""Self-tests of the harness (run: ``python -m harness.tests.test_harness`` from tools/, or pytest).

Oracle estimators that replay the reference itself (optionally perturbed) must produce exactly
predictable metrics: this validates matching, sign conventions and the along/cross decomposition.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from harness.loader import load_bag
from harness.reference import build_reference, RefConfig, LocalFrame, TrackPath
from harness.replay import EvalConfig, Output, evaluate_bag, replay
from harness import metrics as M
from harness.faults import apply_faults

BAG = '30618_e3d94878'      # clean RTK bag


class RefOracle:
    """Publishes the reference samples themselves (stamp = reference stamp) from a 50 Hz tick.
    Perturbations: dv (speed bias), d_along / d_cross (metres along / left of the path tangent),
    dt (stamp shift)."""
    needs_oracle_ref = True

    def __init__(self, oracle_ref=None, frame='enu', dv=0.0, d_along=0.0, d_cross=0.0, dt=0.0):
        r = oracle_ref
        self.t = r.t_pos
        # speed at the position stamps (both topics share the same clock in the test)
        self.v = np.interp(r.t_pos, r.t_vel, r.v)
        tan = r.tan_pos
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        self.p = r.xyz.copy()
        self.p[:, :2] += d_along * tan + d_cross * nrm
        self.tv, self.vv = r.t_vel, r.v
        self.dv, self.dt = dv, dt
        self.i = 0
        self.j = 0
        # latency between the reference clock and the bag clock
        self.lag = float(np.median(r.tbag_pos - r.t_pos))

    def on_tick(self, t_bag):
        out = []
        while self.i < len(self.t) and self.t[self.i] + self.lag <= t_bag:
            k = self.i
            out.append(Output(self.t[k] + self.dt, self.v[k] + self.dv, *self.p[k], 'map'))
            self.i += 1
        # speed-only messages at the vel stamps (to make speed metrics exact)
        while self.j < len(self.tv) and self.tv[self.j] + self.lag <= t_bag:
            k = self.j
            out.append(Output(self.tv[k] + self.dt, self.vv[k] + self.dv, math.nan, math.nan, math.nan, 'map'))
            self.j += 1
        return out


def _eval(**params):
    cfg = EvalConfig(tick_hz=50.0)
    r = evaluate_bag(BAG, RefOracle, params, cfg)
    assert 'error' not in r, r.get('traceback')
    return r


def test_exact_oracle():
    r = _eval()
    s = r['summary']
    assert s['match_v'] > 0.999 and s['match_pos'] > 0.999, s
    assert s['v_rmse'] < 1e-9 and abs(s['v_bias']) < 1e-9, s
    assert s['pos_rmse3d'] < 1e-6 and s['along_rmse'] < 0.05 and s['drift_pct_3d'] < 1e-6, s
    return s


def test_speed_bias():
    s = _eval(dv=0.1)['summary']
    assert abs(s['v_bias'] - 0.1) < 1e-9 and abs(s['v_rmse'] - 0.1) < 1e-9, s
    return s


def test_along_cross_signs():
    s = _eval(d_along=10.0)['summary']
    r = _eval(d_along=10.0)
    assert abs(r['pos']['along_tan']['mean'] - 10.0) < 1e-6, r['pos']['along_tan']
    assert abs(r['pos']['along_arc']['mean'] - 10.0) < 0.5, r['pos']['along_arc']       # curvature
    assert abs(r['pos']['cross_tan']['mean']) < 1e-6
    r = _eval(d_cross=3.0)
    assert abs(r['pos']['cross_tan']['mean'] - 3.0) < 1e-6, r['pos']['cross_tan']
    assert abs(r['pos']['cross_map']['mean'] - 3.0) < 0.3, r['pos']['cross_map']
    assert abs(r['pos']['along_arc']['mean']) < 0.5, r['pos']['along_arc']
    return s


def test_matching_unit():
    t_ref = np.array([0.0, 0.1, 0.2, 0.3])
    t_out = np.array([0.26, 0.03, 0.12])          # unsorted on purpose
    iref, iout, n = M.pair(t_ref, t_out, np.ones(3, bool), 0.05, 'ref2out')
    assert n == 4 and list(iref) == [0, 1, 3] and list(iout) == [1, 2, 0], (iref, iout)
    iref, iout, n = M.pair(t_ref, t_out, np.ones(3, bool), 0.05, 'out2ref')
    assert n == 3 and sorted(zip(iout, iref)) == [(0, 3), (1, 0), (2, 1)], (iref, iout)


def test_stamp_shift_breaks_matching():
    s = _eval(dt=1e5)['summary']        # stamps from the wrong clock -> nothing matches
    assert s['match_v'] < 0.01 and s['match_pos'] < 0.01, s
    return s['match_v']


def test_frames_roundtrip():
    for kind in ('enu', 'utm'):
        f = LocalFrame(kind, 55.81, 37.46, 160.0)
        lat, lon, alt = 55.83, 37.52, 150.0
        x, y, z = f.forward(lat, lon, alt)
        la, lo, al = f.inverse(x, y, z)
        assert abs(la - lat) < 1e-9 and abs(lo - lon) < 1e-9 and abs(al - alt) < 1e-4, (kind, la, lo, al)
    # ENU vs UTM rotation ~ grid convergence (~1.3 deg here)
    a = LocalFrame('utm', 55.81, 37.46, 160.0).enu_rotation()
    assert 0.018 < abs(a) < 0.028, a


def test_path_projection():
    xy = np.array([[0, 0], [10, 0], [10, 10]], float)
    p = TrackPath(xy)
    s, lat, d = p.project(np.array([[5, 1], [11, 5], [5, -2]]), s_hint=np.array([5, 15, 5]), window=20)
    assert np.allclose(s, [5, 15, 5]) and np.allclose(lat, [1, -1, -2]) and np.allclose(d, [1, 1, 2]), (s, lat, d)
    # numpy fallback == numba (if available)
    lo = np.zeros(3, np.int64)
    hi = np.full(3, 1, np.int64)
    s2, lat2, d2 = p._project_numpy(np.array([[5, 1], [11, 5], [5, -2]], float), lo, hi, 1000)
    assert np.allclose(s, s2) and np.allclose(lat, lat2)


def test_faults_and_robust_estimator_does_not_crash():
    bag = load_bag(BAG)
    fb = apply_faults(bag, ['suite:basic', 'suite:garbage'], seed=1)
    from harness.baselines import B0
    log = replay(fb, B0(), EvalConfig())
    assert len(log) > 1000
    return len(log)


def main():
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            out = t()
            print(f'PASS {t.__name__}' + (f'  {out}' if isinstance(out, (int, float)) else ''))
        except AssertionError as ex:
            failed += 1
            print(f'FAIL {t.__name__}: {ex}')
    print('all passed' if not failed else f'{failed} failed')
    return failed


if __name__ == '__main__':
    sys.exit(main())
