import numpy as np
import pytest

from anomaly_sim import physics as P

DT = 0.004
T = np.arange(0, 12, DT)


def _traction(v0=3.0, acc=0.8, notch=10):
    return v0 + acc * T, np.full_like(T, acc), np.full_like(T, float(notch))


def test_notch_demand_signs_and_monotonic():
    f = P.notch_demand(np.arange(-15, 16))
    assert np.all(f[:15] < 0) and f[15] == 0 and np.all(f[16:] > 0)
    assert np.all(np.diff(f[16:]) >= 0)
    assert 0.08 < f[-1] < 0.12  # ~0.9 m/s^2 measured at full traction


def test_no_patch_gives_only_micro_creep():
    v, a, n = _traction()
    adh = P.AdhesionParams()
    sim = P.simulate_wheel(T, v, a, P.notch_demand(n), np.full_like(T, adh.mu_dry),
                           P.ControllerParams(kind='cutoff'), adh, 90.0)
    creep = sim.w / v
    assert np.all(np.abs(creep[100:]) < 0.02)  # natural creep of a few tenths of a percent


def test_cutoff_slip_hump_and_recovery():
    v, a, n = _traction()
    adh = P.AdhesionParams()
    mu = P.patch_profile(T, 2.0, 7.0, adh.mu_dry, 0.7 * P.notch_demand(n)[0])
    ctrl = P.ControllerParams(kind='cutoff', w_on=1.0, w_off=0.3, u_min=0.3)
    sim = P.simulate_slip_delta(T, v, a, n, mu, ctrl, adh, 90.0)
    assert 1.0 <= sim.w.max() <= 2.0            # triggers at w_on, limited overshoot
    assert np.all(np.abs(sim.w[T < 1.9]) < 1e-6)  # nothing before the patch
    assert abs(np.interp(9.0, T, sim.w)) < 0.05    # re-adhesion after the patch


def test_creep_controller_plateau():
    v, a, n = _traction()
    adh = P.AdhesionParams()
    mu = P.patch_profile(T, 2.0, 8.0, adh.mu_dry, 0.7 * P.notch_demand(n)[0])
    ctrl = P.ControllerParams(kind='creep', s_target=0.15, u_min=0.02)
    sim = P.simulate_slip_delta(T, v, a, n, mu, ctrl, adh, 90.0)
    s = sim.w / v
    plateau = np.median(s[(T > 5) & (T < 7.5)])
    assert plateau == pytest.approx(0.15, rel=0.1)
    assert s.max() < 0.15 * 1.5


def test_braking_lock_and_release():
    v = np.maximum(10 - 1.0 * T, 0)
    a = np.where(v > 0, -1.0, 0.0)
    n = np.full_like(T, -9.0)
    adh = P.AdhesionParams()
    mu = P.patch_profile(T, 2.0, 5.5, adh.mu_dry, 0.3 * abs(P.notch_demand(n)[0]))
    ctrl = P.ControllerParams(kind='none', lock_hold=1.5, w_off=0.3, r_up=0.8)
    sim = P.simulate_slip_delta(T, v, a, n, mu, ctrl, adh, 90.0)
    locked_s = sim.locked.sum() * DT
    assert 1.4 <= locked_s <= 3.0
    assert np.all(v + sim.w >= -1e-9)             # a braked wheel never turns backwards
    assert abs(np.interp(8.0, T, sim.w)) < 0.1     # spins back up after release


def test_time_step_convergence():
    v, a, n = _traction()
    adh = P.AdhesionParams()
    peaks = []
    for dt in (0.004, 0.001):
        t = np.arange(0, 12, dt)
        vv, aa, nn = 3 + 0.8 * t, np.full_like(t, 0.8), np.full_like(t, 10.0)
        mu = P.patch_profile(t, 2.0, 7.0, adh.mu_dry, 0.7 * P.notch_demand(nn)[0])
        sim = P.simulate_slip_delta(t, vv, aa, nn, mu, P.ControllerParams(kind='cutoff', w_on=1.0, w_off=0.3,
                                                                             u_min=0.3), adh, 90.0)
        peaks.append(sim.w.max())
    assert peaks[0] == pytest.approx(peaks[1], rel=0.1)
