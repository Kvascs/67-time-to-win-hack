import json

import numpy as np

from anomaly_sim.constants import ALL_TOPICS, COLUMNS, FRONT, GNSS_MASTER_FIX, Label
from anomaly_sim.run import check_npz, load_pair, load_run, save_run
from anomaly_sim.scenario import Scenario, apply_scenario


def test_clean_run_layout(clean_full):
    assert set(ALL_TOPICS) <= set(clean_full.streams)
    for k, s in clean_full.streams.items():
        assert s.val.shape == (len(s), len(COLUMNS[k]))
        assert np.all(s.src == np.arange(len(s)))
        assert not s.label.any()
        assert np.all(np.diff(s.t_bag) >= 0)
    # wheel speeds are km/h (max ~52 km/h on this line), front/rear share header stamps
    assert 40 < np.nanmax(clean_full[FRONT].v) < 60


def test_crop_keeps_source_index(clean_full, clean):
    s = clean[FRONT]
    assert len(s) < len(clean_full[FRONT])
    assert np.array_equal(clean_full[FRONT].val[s.src], s.val)
    assert clean.meta['crop'] == [100.0, 420.0]


def test_save_load_roundtrip(tmp_path, clean):
    sc = Scenario('T', 5, injectors=[{'type': 'slip', 'count': 2}, {'type': 'dropout', 'count': 2},
                                     {'type': 'outliers', 'rate': 0.01}])
    bad = apply_scenario(clean, sc)
    p = save_run(bad, tmp_path / 'x.npz')
    assert check_npz(p) == []
    back = load_run(p)
    assert back.meta['scenario'] == 'T' and back.meta['seed'] == 5
    assert len(back.events) == len(bad.events)
    for k, s in bad.streams.items():
        b = back.streams[k]
        assert np.array_equal(b.to_array(), s.to_array(), equal_nan=True)
        assert np.array_equal(b.label, s.label)
        assert np.array_equal(b.src, s.src)
    # the side arrays make the file self-describing
    with np.load(p) as d:
        meta = json.loads(str(d['_meta_json']))
        assert meta['bag'] == clean.name and 'events' in meta
        assert d[FRONT].shape[1] == 3 and d[GNSS_MASTER_FIX].shape[1] == 9


def test_load_pair_recovers_clean_reference(tmp_path, clean):
    bad = apply_scenario(clean, Scenario('T', 1, injectors=[{'type': 'noise', 'count': 1}]))
    p = save_run(bad, tmp_path / 'y.npz')
    cr, cl = load_pair(p)
    assert cl.meta.get('crop') == [100.0, 420.0]
    assert len(cl[FRONT]) == len(clean[FRONT])
    # GNSS was cut in the corrupted run but is complete in the reference
    assert len(cr[GNSS_MASTER_FIX]) < len(cl[GNSS_MASTER_FIX])


def test_label_names():
    assert Label.names(int(Label.SLIP | Label.NAN)) == ['SLIP', 'NAN']
