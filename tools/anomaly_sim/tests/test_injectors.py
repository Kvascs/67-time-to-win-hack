import numpy as np
import pytest

from anomaly_sim.constants import CMD, FRONT, GNSS, REAR, VEHICLE, WHEELS, Label
from anomaly_sim.context import Context
from anomaly_sim.injectors import REGISTRY, build_injector
from anomaly_sim.scenario import Scenario, apply_scenario


def apply(clean, spec, seed=3, gnss=None):
    return apply_scenario(clean, Scenario('T', seed, injectors=[spec] if isinstance(spec, dict) else spec,
                                          gnss_keep_s=gnss))


def has(s, flag):
    return (s.label & np.uint32(flag)) > 0


def test_registry_complete():
    assert set(REGISTRY) == {'scale_drift', 'slip', 'slide', 'noise', 'frozen', 'notch_fault', 'outliers', 'dropout',
                             'duplicates', 'reorder', 'stamp_jitter', 'zero_stamps', 'stamp_glitch', 'clock_offset',
                             'gnss_cut'}


def test_unknown_parameter_rejected():
    with pytest.raises(ValueError):
        build_injector({'type': 'slip', 'peak': 0.2})
    with pytest.raises(ValueError):
        build_injector({'type': 'no_such_type'})


@pytest.mark.parametrize('spec', [{'type': 'scale_drift'}, {'type': 'slip', 'count': 3},
                                  {'type': 'slide', 'count': 3, 'lock_prob': 0.5}, {'type': 'noise', 'count': 2},
                                  {'type': 'frozen'}, {'type': 'outliers', 'rate': 0.02}, {'type': 'notch_fault'}],
                         ids=lambda s: s['type'])
def test_unlabelled_rows_are_untouched(clean, spec):
    """Invariant used by the evaluation: label == 0  =>  value identical to the clean recording."""
    bad = apply(clean, spec)
    for key in VEHICLE:
        s = bad[key]
        u = s.label == 0
        assert np.array_equal(s.val[u, 0], s.clean[u, 0])
        assert np.array_equal(s.t_hdr[u], s.t_hdr0[u])


def test_clean_input_not_modified(clean):
    before = {k: s.to_array().copy() for k, s in clean.streams.items()}
    apply(clean, [{'type': 'slip', 'count': 2}, {'type': 'dropout'}, {'type': 'outliers', 'rate': 0.05}], gnss=5.0)
    for k, s in clean.streams.items():
        assert np.array_equal(s.to_array(), before[k], equal_nan=True)


def test_slip_over_reads_only_in_traction(clean):
    bad = apply(clean, {'type': 'slip', 'count': 3})
    ctx = Context.build(clean)
    ev = [e for e in bad.events if e['type'] == 'slip']
    assert ev, 'no traction phase found for slip'
    for key in WHEELS:
        s = bad[key]
        slip = has(s, Label.SLIP)
        untouched = s.label == 0
        assert np.array_equal(s.val[untouched, 0], s.clean[untouched, 0])
        assert np.all(s.val[slip, 0] >= s.clean[slip, 0] - 1e-3)
    for e in ev:
        assert ctx.notch_at([e['stats']['patch_t0']])[0] > 0
        for k, st in e['stats'].items():
            if k in WHEELS and st.get('n_rows'):
                assert 0.0 < st['peak_rel'] < 1.0


def test_slide_and_lock(clean):
    bad = apply(clean, {'type': 'slide', 'count': 3, 'lock_prob': 1.0, 'lock_duration': [1.0, 2.0]})
    locked = 0
    for key in WHEELS:
        s = bad[key]
        sl = has(s, Label.SLIDE)
        assert np.all(s.val[sl, 0] <= s.clean[sl, 0] + 1e-3)
        lk = has(s, Label.LOCK)
        assert np.all(s.val[lk, 0] == 0.0)
        locked += lk.sum()
    assert locked >= 5


def test_kinematic_step_model_exact_magnitude(clean):
    bad = apply(clean, {'type': 'slip', 'model': 'step', 'count': 2, 'bogie': 'both', 'peak_rel': 0.2,
                        'duration': 4.0, 'min_hold': 0.5})
    s = bad[FRONT]
    m = has(s, Label.SLIP) & (s.clean[:, 0] > 5)
    ratio = s.val[m, 0] / s.clean[m, 0]
    assert np.median(ratio) == pytest.approx(1.2, abs=0.03)


def test_scale_drift_bounds(clean):
    bad = apply(clean, {'type': 'scale_drift', 'topics': 'front', 'start': 0.0, 'end': -0.02})
    s = bad[FRONT]
    m = s.clean[:, 0] > 5
    r = s.val[m, 0] / s.clean[m, 0]
    assert r.min() > 0.98 - 1e-4 and r.max() <= 1.0 + 1e-6
    assert r[-1] < r[0]
    assert np.array_equal(bad[REAR].val, clean[REAR].val)


def test_dropout_removes_messages(clean):
    bad = apply(clean, {'type': 'dropout', 'count': 3, 'duration': [2.0, 4.0], 'topics': 'wheels'})
    ev = [e for e in bad.events if e['type'] == 'dropout']
    assert len(ev) == 3
    for key in WHEELS:
        s = bad[key]
        assert len(s) < len(clean[key])
        for e in ev:
            inside = (s.t_hdr0 > e['t0'] + 0.1) & (s.t_hdr0 < e['t1'] - 0.1)
            assert not inside.any()
        assert has(s, Label.RESUME).sum() >= 1
    assert len(bad[CMD]) == len(clean[CMD])


def test_stall_delays_but_keeps_messages(clean):
    bad = apply(clean, {'type': 'dropout', 'mode': 'stall', 'count': 2, 'duration': [1.0, 1.5], 'topics': 'vehicle'})
    for key in VEHICLE:
        s = bad[key]
        assert len(s) == len(clean[key])
        st = has(s, Label.STALL)
        assert st.sum() > 5
        lat = s.t_bag - s.t_hdr
        assert lat[st].max() > 0.8
        assert np.all(np.diff(s.t_bag) >= 0)


@pytest.mark.parametrize('kind,flag', [('nan', Label.NAN), ('inf', Label.INF), ('absurd', Label.ABSURD),
                                       ('zero', Label.ZERO), ('negative', Label.NEGATIVE), ('spike', Label.SPIKE)])
def test_outlier_kinds(clean, kind, flag):
    bad = apply(clean, {'type': 'outliers', 'rate': 0.02, 'kinds': {kind: 1.0}})
    s = bad[FRONT]
    m = has(s, flag)
    assert m.sum() > 3
    v = s.val[m, 0]
    if kind == 'nan':
        assert np.isnan(v).all()
    elif kind == 'inf':
        assert np.isinf(v).all()
    elif kind == 'zero':
        assert (v == 0).all()
    elif kind == 'negative':
        assert (v < 0).all()
    assert np.array_equal(s.val[s.label == 0, 0], s.clean[s.label == 0, 0])


def test_cmd_outliers_stay_int8(clean):
    bad = apply(clean, {'type': 'outliers', 'topics': 'cmd', 'rate': 0.02, 'kinds': {'nan': 0.5, 'absurd': 0.5}})
    v = bad[CMD].val[:, 0]
    assert np.all(np.isfinite(v)) and v.min() >= -128 and v.max() <= 127 and np.all(v == np.round(v))
    assert has(bad[CMD], Label.NOTCH_FAULT).sum() > 3


def test_frozen_holds_value(clean):
    bad = apply(clean, {'type': 'frozen', 'count': 2, 'topics': 'front', 'mode': 'hold', 'duration': 5.0})
    s = bad[FRONT]
    for e in [e for e in bad.events if e['type'] == 'frozen']:
        m = (s.t_hdr0 >= e['t0']) & (s.t_hdr0 <= e['t1']) & has(s, Label.FROZEN)
        assert m.sum() >= 30 and np.unique(s.val[m, 0]).size == 1


def test_noise_level(clean):
    bad = apply(clean, {'type': 'noise', 'count': 0, 'sigma_kmh': 2.0, 'topics': 'rear'})
    s = bad[REAR]
    m = s.clean[:, 0] > 10  # away from the zero clip
    assert np.std(s.val[m, 0] - s.clean[m, 0]) == pytest.approx(2.0, rel=0.15)


def test_duplicates_and_reorder(clean):
    bad = apply(clean, [{'type': 'duplicates', 'rate': 0.05}, {'type': 'reorder', 'rate': 0.02}])
    s = bad[FRONT]
    d = has(s, Label.DUPLICATE)
    assert len(s) == len(clean[FRONT]) + d.sum() and d.sum() > 5
    assert np.any(np.diff(s.t_hdr) < -0.05)  # out-of-order in arrival order
    for i in np.flatnonzero(d)[:10]:
        orig = np.flatnonzero((s.src == s.src[i]) & ~d)
        assert len(orig) == 1 and s.t_hdr[orig[0]] == s.t_hdr[i] and s.val[orig[0], 0] == s.val[i, 0]


def test_stamp_faults(clean):
    bad = apply(clean, [{'type': 'stamp_jitter', 'sigma_ms': 20, 'tail_prob': 0.0},
                        {'type': 'zero_stamps', 'rate': 0.01},
                        {'type': 'stamp_glitch', 'count': 3},
                        {'type': 'clock_offset', 'count': 1, 'offset_s': 0.8, 'drift': 0.0, 'duration': 10.0}])
    s = bad[FRONT]
    others = np.uint32(Label.ZERO_STAMP | Label.STAMP_GLITCH | Label.CLOCK_OFFSET)
    jit_only = has(s, Label.STAMP_JITTER) & ((s.label & others) == 0)
    assert np.std(s.t_hdr[jit_only] - s.t_hdr0[jit_only]) == pytest.approx(0.02, rel=0.2)
    assert np.all(s.t_hdr[has(s, Label.ZERO_STAMP)] == 0.0)
    for key in VEHICLE:  # glitches hit all vehicle topics together
        g = has(bad[key], Label.STAMP_GLITCH) & ~has(bad[key], Label.ZERO_STAMP) & ~has(bad[key], Label.CLOCK_OFFSET)
        assert g.sum() >= 3
        off = bad[key].t_hdr[g] - bad[key].t_hdr0[g]
        assert np.all(np.abs(np.abs(off) - 1.0) < 0.2)
    co = has(s, Label.CLOCK_OFFSET) & ~has(s, Label.ZERO_STAMP)
    assert np.median(s.t_hdr[co] - s.t_hdr0[co]) == pytest.approx(0.8, abs=0.05)


def test_gnss_cut(clean):
    bad = apply(clean, [], gnss=5.0)
    for key in GNSS:
        s = bad[key]
        assert len(s) > 0 and s.t_bag.max() <= bad.t_start + 5.0 + 1e-9
    for key in VEHICLE:
        assert np.array_equal(bad[key].to_array(), clean[key].to_array())


def test_notch_fault(clean):
    bad = apply(clean, {'type': 'notch_fault', 'count': 4, 'kinds': {'stuck': 0.5, 'invalid': 0.5}})
    s = bad[CMD]
    assert has(s, Label.NOTCH_FAULT).sum() > 0
    assert s.val.min() >= -128 and s.val.max() <= 127


def test_explicit_times(clean):
    bad = apply(clean, {'type': 'dropout', 'at': [50.0, 120.0], 'duration': 1.0, 'topics': 'front'})
    ev = [e for e in bad.events if e['type'] == 'dropout']
    assert [round(e['t0_rel'], 3) for e in ev] == [50.0, 120.0]
