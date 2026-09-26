import sqlite3

import numpy as np
import pytest
import yaml

pytest.importorskip('rosbags')

from anomaly_sim.bagio import SourceBag, compare_schema, read_bag, verify_bag, write_bag  # noqa: E402
from anomaly_sim.constants import CMD, FRONT, GNSS, VEHICLE, topic_name  # noqa: E402
from anomaly_sim.scenario import apply_scenario, get_scenario, load_suite  # noqa: E402


def _rows(db, name):
    con = sqlite3.connect(db)
    tid = con.execute('SELECT id FROM topics WHERE name=?', (name,)).fetchone()[0]
    rows = con.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp, id', (tid,)).fetchall()
    con.close()
    return rows


def test_identity_copy_is_byte_exact(tmp_path, clean, clean_full, src_bag):
    """An uncorrupted crop must reproduce the original CDR bytes and timestamps exactly."""
    rep = write_bag(clean, tmp_path / 'ident', src_bag=src_bag, clean=clean_full)
    assert rep['verify'] == 'ok'
    for st in rep['topics'].values():
        assert st['encoded'] == 0
    src = SourceBag.open(src_bag)
    out_db = tmp_path / 'ident' / 'ident_0.db3'
    for key in (FRONT, CMD, GNSS[0]):
        name = topic_name(key)
        got = _rows(out_db, name)
        ts, data = src.raw[name]
        idx = clean[key].src
        assert [g[0] for g in got] == ts[idx].tolist()
        assert all(bytes(g[1]) == data[i] for g, i in zip(got, idx))


@pytest.mark.parametrize('fmt', ['humble', 'rosbags'])
def test_corrupted_roundtrip(tmp_path, clean, clean_full, src_bag, fmt):
    sc = get_scenario(load_suite(), 'S18')
    bad = apply_scenario(clean, sc)
    rep = write_bag(bad, tmp_path / f'b_{fmt}', src_bag=src_bag, clean=clean_full, fmt=fmt)
    assert rep['verify'] == 'ok'
    assert sum(st['encoded'] for st in rep['topics'].values()) > 100
    back = read_bag(tmp_path / f'b_{fmt}')
    for key in VEHICLE:
        v = back[key].val[:, 0]
        assert len(v) == len(bad[key])
    assert np.isnan(back[FRONT].val[:, 0]).any() or np.isnan(back['vehicle__rear_bogie_velocity'].val[:, 0]).any()
    assert (back[FRONT].t_hdr == 0).any() or (back[CMD].t_hdr == 0).any()
    assert verify_bag(tmp_path / f'b_{fmt}', bad) == []
    if fmt == 'humble':
        assert compare_schema(tmp_path / f'b_{fmt}', src_bag) == []


def test_humble_metadata_contract(tmp_path, clean, clean_full, src_bag):
    """Emulate what ROS 2 Humble's rosbag2 parses from metadata.yaml and the QoS strings."""
    bad = apply_scenario(clean, get_scenario(load_suite(), 'S06'))
    out = tmp_path / 'h'
    write_bag(bad, out, src_bag=src_bag, clean=clean_full, fmt='humble')
    info = yaml.safe_load((out / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    assert info['version'] == 5 and info['storage_identifier'] == 'sqlite3'
    for k in ('duration', 'starting_time', 'message_count', 'topics_with_message_count', 'compression_format',
              'compression_mode', 'relative_file_paths', 'files'):
        assert k in info
    assert (out / info['relative_file_paths'][0]).exists()
    total = 0
    for t in info['topics_with_message_count']:
        qos = yaml.safe_load(t['topic_metadata']['offered_qos_profiles'])
        for q in qos:  # Humble's Rosbag2QoS decoder calls as<int>() on these
            for f in ('history', 'depth', 'reliability', 'durability', 'liveliness'):
                assert isinstance(q[f], int)
            assert isinstance(q['deadline']['sec'], int)
        total += t['message_count']
    assert total == info['message_count'] == info['files'][0]['message_count']
    con = sqlite3.connect(out / info['relative_file_paths'][0])
    assert con.execute('SELECT schema_version, ros_distro FROM schema').fetchall() == [(3, 'humble')]
    assert con.execute('SELECT COUNT(*) FROM messages').fetchone()[0] == total
    ts = [r[0] for r in con.execute('SELECT timestamp FROM messages ORDER BY id')]
    assert ts == sorted(ts), 'messages must be stored in playback order'
    con.close()
