import numpy as np
import pytest

from anomaly_sim.constants import DEFAULT_SUITE, GNSS
from anomaly_sim.run import check_npz, save_run
from anomaly_sim.scenario import apply_scenario, get_scenario, load_suite, summarize

SUITE = load_suite(DEFAULT_SUITE)


def test_suite_well_formed():
    assert len(SUITE) >= 15
    assert len({s.name for s in SUITE}) == len(SUITE)
    assert len({s.seed for s in SUITE}) == len(SUITE)
    for s in SUITE:
        assert s.description
        assert any(x['type'] == 'gnss_cut' for x in s.specs()), 'every scenario must cut GNSS'
    covered = {x['type'] for s in SUITE for x in s.specs()}
    assert covered == {'scale_drift', 'slip', 'slide', 'noise', 'frozen', 'notch_fault', 'outliers', 'dropout',
                       'duplicates', 'reorder', 'stamp_jitter', 'zero_stamps', 'stamp_glitch', 'clock_offset',
                       'gnss_cut'}


@pytest.mark.parametrize('sc', SUITE, ids=[s.name for s in SUITE])
def test_every_scenario_runs_and_saves(sc, clean, tmp_path):
    bad = apply_scenario(clean, sc)
    p = save_run(bad, tmp_path / f'{sc.name}.npz')
    assert check_npz(p) == []
    summ = summarize(bad)
    assert bad.meta['scenario'] == sc.name and bad.meta['seed'] == sc.seed
    for key in GNSS:
        assert bad[key].t_bag.max() <= bad.t_start + sc.gnss_keep_s + 1e-9
    if sc.name != 'S00_baseline_gnss_cut':
        assert sum(summ['events'].values()) > 1, f'{sc.name} injected nothing on the test crop'


def test_deterministic_and_seed_sensitive(clean):
    sc = get_scenario(SUITE, 'S17')
    a = apply_scenario(clean, sc)
    b = apply_scenario(clean, sc)
    c = apply_scenario(clean, sc, seed=sc.seed + 1)
    for k in a.streams:
        assert np.array_equal(a[k].to_array(), b[k].to_array(), equal_nan=True)
    assert [e['t0'] for e in a.events] == [e['t0'] for e in b.events]
    assert [e['t0'] for e in a.events] != [e['t0'] for e in c.events]


def test_bag_name_changes_realisation(clean):
    sc = get_scenario(SUITE, 'S06')
    other = clean.copy()
    other.name = 'another_bag'
    a = apply_scenario(clean, sc)
    b = apply_scenario(other, sc)
    assert [round(e['t0'], 3) for e in a.events] != [round(e['t0'], 3) for e in b.events]
