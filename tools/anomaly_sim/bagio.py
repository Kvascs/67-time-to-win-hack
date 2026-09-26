"""Write (corrupted) runs back to rosbag2 / sqlite3 so a ROS 2 node can be tested with ``ros2 bag play``.

Copy-on-write: every message whose payload and timing are unchanged keeps the *original CDR bytes*
and the original int64 bag timestamp; only corrupted / synthetic messages are re-serialised with the
rosbags typestore (custom ``tram_vehicle_msgs`` types registered from the .msg files). GNSS messages
are only ever dropped (GNSS cut), never re-encoded.

Two storage back-ends:

* ``fmt='humble'`` (default): a clone of the Humble recorder layout used by the dataset -
  identical sqlite DDL (``schema`` v3 / ``ros_distro=humble``, empty ``metadata`` table),
  ``<name>_0.db3`` and a version-5 ``metadata.yaml`` with the original QoS strings.
  This is byte-for-byte the format ROS 2 Humble's ``ros2 bag play`` already reads for the originals.
* ``fmt='rosbags'``: the rosbags ``Writer`` (rosbag2 v8 = Jazzy-era metadata/schema v4, integer QoS
  encoding). rosbags >= 0.10 cannot write v5; Humble should read v8 (unknown YAML keys are ignored),
  but that is not verified here - prefer ``humble`` for the jury-like replay.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .constants import (BAG_DIR, CMD, COLUMNS, FRAME_IDS, GNSS, INT8_MAX, INT8_MIN, MSG_DIR, MSGTYPES, VEHICLE,
                        topic_key, topic_name)
from .run import Run, Stream, load_run

_TYPESTORE = None


def make_typestore():
    """ROS 2 Humble typestore with ``tram_vehicle_msgs`` registered from the dataset .msg files."""
    global _TYPESTORE
    if _TYPESTORE is None:
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
        ts = get_typestore(Stores.ROS2_HUMBLE)
        add = {}
        for name in ('VelocitySensor', 'DriverControllerCommand'):
            add.update(get_types_from_msg((MSG_DIR / f'{name}.msg').read_text(encoding='utf-8'),
                                          f'tram_vehicle_msgs/msg/{name}'))
        ts.register(add)
        _TYPESTORE = ts
    return _TYPESTORE


# ------------------------------------------------------------------------------------ source bag

@dataclass
class SourceBag:
    path: Path
    db_path: Path
    meta: dict                      # parsed metadata.yaml
    ddl: list[tuple[str, str, str]]  # (type, name, sql) from sqlite_master
    schema_rows: list[tuple]
    topics: list[tuple]              # (id, name, type, serialization_format, offered_qos_profiles)
    raw: dict[str, tuple[np.ndarray, list[bytes]]]  # topic name -> (timestamps ns, raw CDR)

    @classmethod
    def open(cls, path: str | Path) -> 'SourceBag':
        path = Path(path)
        meta = yaml.safe_load((path / 'metadata.yaml').read_text(encoding='utf-8'))
        info = meta['rosbag2_bagfile_information']
        rel = info['relative_file_paths']
        if len(rel) != 1:
            raise NotImplementedError('multi-file bags are not supported')
        db_path = path / rel[0]
        con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        try:
            ddl = [r for r in con.execute('SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL')]
            tables = {r[1] for r in ddl if r[0] == 'table'}
            schema_rows = con.execute('SELECT * FROM schema').fetchall() if 'schema' in tables else []
            topics = con.execute('SELECT id, name, type, serialization_format, offered_qos_profiles '
                                 'FROM topics ORDER BY id').fetchall()
            raw = {}
            for tid, name, *_ in topics:
                rows = con.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp, id',
                                   (tid,)).fetchall()
                raw[name] = (np.array([r[0] for r in rows], dtype=np.int64), [bytes(r[1]) for r in rows])
        finally:
            con.close()
        return cls(path, db_path, meta, ddl, schema_rows, topics, raw)


def _hdr_from_cdr(data: list[bytes]) -> np.ndarray:
    """header.stamp [s] of CDR messages that start with std_msgs/Header (little endian)."""
    if not data:
        return np.zeros(0)
    buf = np.frombuffer(b''.join(d[4:12] for d in data), dtype=np.dtype([('sec', '<i4'), ('ns', '<u4')]))
    return buf['sec'].astype(np.float64) + buf['ns'].astype(np.float64) * 1e-9


def _split_stamp(t: float) -> tuple[int, int]:
    if not np.isfinite(t) or t <= 0:
        return 0, 0
    sec = int(np.floor(t))
    ns = int(round((t - sec) * 1e9))
    if ns >= 1_000_000_000:
        sec, ns = sec + 1, ns - 1_000_000_000
    return sec, ns


class _Serializer:
    def __init__(self):
        self.ts = make_typestore()
        self.T = self.ts.types

    def encode(self, key: str, t_hdr: float, value: float, frame_id: str) -> bytes:
        sec, ns = _split_stamp(t_hdr)
        header = self.T['std_msgs/msg/Header'](stamp=self.T['builtin_interfaces/msg/Time'](sec=sec, nanosec=ns),
                                               frame_id=frame_id)
        msgtype = MSGTYPES[key]
        if key == CMD:
            v = 0 if not np.isfinite(value) else int(np.clip(np.rint(value), INT8_MIN, INT8_MAX))
            msg = self.T[msgtype](header=header, position=v)
        else:
            msg = self.T[msgtype](header=header, velocity=float(value))
        return bytes(self.ts.serialize_cdr(msg, msgtype))

    def frame_id(self, key: str, data: bytes | None) -> str:
        if data is None:
            return FRAME_IDS.get(key, '')
        msg = self.ts.deserialize_cdr(data, MSGTYPES[key])
        return msg.header.frame_id


def build_messages(run: Run, src: SourceBag, clean: Run) -> tuple[dict[str, tuple[np.ndarray, list[bytes]]], dict]:
    """Per topic name: (bag timestamps ns, CDR payloads) for the corrupted run (copy-on-write)."""
    ser = _Serializer()
    out: dict[str, tuple[np.ndarray, list[bytes]]] = {}
    stats: dict[str, dict] = {}
    for tid, name, mtype, *_ in src.topics:
        key = topic_key(name)
        ts_src, data_src = src.raw[name]
        if key not in run.streams:
            out[name] = (ts_src, data_src)
            stats[key] = {'copied': len(data_src), 'encoded': 0}
            continue
        s: Stream = run.streams[key]
        c: Stream = clean.streams[key]
        if len(data_src) != len(c):
            raise ValueError(f'{run.name} {name}: bag has {len(data_src)} messages but npz has {len(c)}')
        if len(c):
            hdr = _hdr_from_cdr(data_src)
            if np.max(np.abs(hdr - c.t_hdr)) > 1e-5:
                raise ValueError(f'{run.name} {name}: npz rows do not align with bag messages')
        src_idx = s.src
        valid = src_idx >= 0
        same_hdr = np.zeros(len(s), bool)
        same_val = np.zeros(len(s), bool)
        same_t = np.zeros(len(s), bool)
        if valid.any():
            si = src_idx[valid]
            same_hdr[valid] = s.t_hdr[valid] == c.t_hdr[si]
            a, b = s.val[valid], c.val[si]
            same_val[valid] = np.all((a == b) | (np.isnan(a) & np.isnan(b)), axis=1)
            same_t[valid] = s.t_bag[valid] == c.t_bag[si]
        reuse_data = valid & same_hdr & same_val
        if key in GNSS and not reuse_data.all():
            raise NotImplementedError(f'{name}: modified GNSS messages cannot be re-encoded (only dropping is supported)')
        frame = ser.frame_id(key, data_src[0] if len(data_src) else None) if key in VEHICLE else ''
        stamps = np.empty(len(s), dtype=np.int64)
        payload: list[bytes] = []
        n_enc = 0
        for i in range(len(s)):
            j = src_idx[i]
            stamps[i] = ts_src[j] if (j >= 0 and same_t[i]) else int(round(s.t_bag[i] * 1e9))
            if reuse_data[i]:
                payload.append(data_src[j])
            else:
                payload.append(ser.encode(key, s.t_hdr[i], s.val[i, 0], frame))
                n_enc += 1
        out[name] = (stamps, payload)
        stats[key] = {'copied': int(reuse_data.sum()), 'encoded': n_enc}
    return out, stats


# ------------------------------------------------------------------------------------ writers

def _metadata_v5(src_meta: dict, db_name: str, counts: dict[str, int], start: int, duration: int) -> str:
    info = src_meta['rosbag2_bagfile_information']
    total = sum(counts.values())
    lines = ['rosbag2_bagfile_information:', '  version: 5', '  storage_identifier: sqlite3', '  duration:',
             f'    nanoseconds: {duration}', '  starting_time:', f'    nanoseconds_since_epoch: {start}',
             f'  message_count: {total}', '  topics_with_message_count:']
    for t in info['topics_with_message_count']:
        tm = t['topic_metadata']
        lines += ['    - topic_metadata:', f"        name: {tm['name']}", f"        type: {tm['type']}",
                  f"        serialization_format: {tm['serialization_format']}",
                  f"        offered_qos_profiles: {json.dumps(tm.get('offered_qos_profiles', ''))}",
                  f"      message_count: {counts.get(tm['name'], 0)}"]
    lines += ['  compression_format: ""', '  compression_mode: ""', '  relative_file_paths:', f'    - {db_name}',
              '  files:', f'    - path: {db_name}', '      starting_time:',
              f'        nanoseconds_since_epoch: {start}', '      duration:', f'        nanoseconds: {duration}',
              f'      message_count: {total}']
    return '\n'.join(lines) + '\n'


def _sorted_messages(msgs: dict[str, tuple[np.ndarray, list[bytes]]], topic_ids: dict[str, int]):
    allm = []
    for name, (stamps, datas) in msgs.items():
        tid = topic_ids[name]
        allm.extend(zip(stamps.tolist(), [tid] * len(datas), datas, [name] * len(datas)))
    allm.sort(key=lambda r: r[0])  # stable: equal stamps keep topic order
    return allm


def _write_humble(out_dir: Path, src: SourceBag, msgs) -> dict:
    db_name = f'{out_dir.name}_0.db3'
    db = out_dir / db_name
    con = sqlite3.connect(db)
    try:
        tables = [sql for typ, _, sql in src.ddl if typ == 'table']
        others = [sql for typ, _, sql in src.ddl if typ != 'table']
        for sql in tables:
            con.execute(sql)
        if src.schema_rows:
            ph = ','.join('?' * len(src.schema_rows[0]))
            con.executemany(f'INSERT INTO schema VALUES ({ph})', src.schema_rows)
        con.executemany('INSERT INTO topics (id, name, type, serialization_format, offered_qos_profiles) '
                        'VALUES (?,?,?,?,?)', src.topics)
        ids = {name: tid for tid, name, *_ in src.topics}
        allm = _sorted_messages(msgs, ids)
        con.executemany('INSERT INTO messages (topic_id, timestamp, data) VALUES (?,?,?)',
                        [(tid, ts, data) for ts, tid, data, _ in allm])
        for sql in others:  # indices after bulk insert
            con.execute(sql)
        con.commit()
    finally:
        con.close()
    counts = {name: len(d) for name, (_, d) in msgs.items()}
    start = allm[0][0] if allm else 0
    duration = (allm[-1][0] - allm[0][0]) if allm else 0
    (out_dir / 'metadata.yaml').write_text(_metadata_v5(src.meta, db_name, counts, start, duration), encoding='utf-8')
    return {'db': str(db), 'messages': len(allm)}


def _write_rosbags(out_dir: Path, src: SourceBag, msgs) -> dict:
    from rosbags.rosbag2 import Writer
    from rosbags.rosbag2.metadata import parse_qos
    ts = make_typestore()
    with Writer(out_dir, version=8) as w:
        conns = {}
        for tid, name, mtype, fmt, qos in src.topics:
            conns[name] = w.add_connection(name, mtype, typestore=ts, serialization_format=fmt,
                                           offered_qos_profiles=parse_qos(qos))
        ids = {name: tid for tid, name, *_ in src.topics}
        allm = _sorted_messages(msgs, ids)
        for stamp, _, data, name in allm:
            w.write(conns[name], int(stamp), data)
    return {'db': str(out_dir / f'{out_dir.name}.db3'), 'messages': len(allm)}


def write_bag(run: Run, out_dir: str | Path, src_bag: str | Path | None = None, clean: Run | None = None,
              fmt: str = 'humble', overwrite: bool = False, verify: bool = True) -> dict:
    """Write ``run`` as a rosbag2 directory ``out_dir`` (created). Returns a small report dict."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    src_bag = Path(src_bag) if src_bag else BAG_DIR / run.name
    if clean is None:
        srcnpz = run.meta.get('source_npz')
        clean = load_run(srcnpz, name=run.name) if srcnpz and Path(srcnpz).exists() else load_run(run.name)
    src = SourceBag.open(src_bag)
    msgs, stats = build_messages(run, src, clean)
    if fmt == 'humble':
        out_dir.mkdir(parents=True)
        rep = _write_humble(out_dir, src, msgs)
    elif fmt == 'rosbags':
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        rep = _write_rosbags(out_dir, src, msgs)
    else:
        raise ValueError(f'fmt must be humble or rosbags, not {fmt!r}')
    rep.update({'bag': str(out_dir), 'fmt': fmt, 'topics': stats})
    (out_dir / 'anomaly_events.json').write_text(json.dumps({'meta': run.meta, 'events': run.events}, indent=1,
                                                            default=float), encoding='utf-8')
    if verify:
        problems = verify_bag(out_dir, run)
        rep['verify'] = problems or 'ok'
        if problems:
            raise RuntimeError(f'bag verification failed: {problems[:5]}')
    return rep


# ------------------------------------------------------------------------------------ reading back

def read_bag(path: str | Path, name: str | None = None) -> Run:
    """Read any rosbag2 of this dataset into a Run (same arrays as tools/extract_bags.py)."""
    from rosbags.highlevel import AnyReader
    path = Path(path)
    ts = make_typestore()
    rows: dict[str, list] = {}
    with AnyReader([path], default_typestore=ts) as r:
        for c in r.connections:
            key = topic_key(c.topic)
            lst = rows.setdefault(key, [])
            for conn, t, raw in r.messages(connections=[c]):
                m = r.deserialize(raw, conn.msgtype)
                st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                tt = t * 1e-9
                if c.msgtype.endswith('VelocitySensor'):
                    lst.append((tt, st, m.velocity))
                elif c.msgtype.endswith('DriverControllerCommand'):
                    lst.append((tt, st, m.position))
                elif c.msgtype.endswith('NavSatFix'):
                    lst.append((tt, st, m.latitude, m.longitude, m.altitude, m.status.status,
                                m.position_covariance[0], m.position_covariance[4], m.position_covariance[8]))
                elif c.msgtype.endswith('TwistStamped'):
                    lst.append((tt, st, m.twist.linear.x, m.twist.linear.y, m.twist.linear.z, m.twist.angular.z))
    streams = {}
    for key, lst in rows.items():
        arr = np.array(lst, dtype=np.float64) if lst else np.zeros((0, 2 + len(COLUMNS.get(key, (0,)))))
        if len(arr):
            arr = arr[np.argsort(arr[:, 0], kind='stable')]
        streams[key] = Stream.from_array(key, arr)
    return Run(name=name or path.name, streams=streams, meta={'bag_path': str(path)})


def verify_bag(path: str | Path, run: Run, tol: float = 2e-6) -> list[str]:
    """Compare a written bag with the run it was written from. Returns problems (empty = OK)."""
    back = read_bag(path)
    problems = []
    for key, s in run.streams.items():
        b = back.streams.get(key)
        if b is None:
            if len(s):
                problems.append(f'{key}: missing in bag')
            continue
        if len(b) != len(s):
            problems.append(f'{key}: {len(b)} msgs in bag vs {len(s)} in run')
            continue
        if not len(s):
            continue
        # equal bag stamps may be reordered between topics but not within a topic
        if np.max(np.abs(b.t_bag - s.t_bag)) > tol:
            problems.append(f'{key}: bag time mismatch {np.max(np.abs(b.t_bag - s.t_bag)):.3g}s')
        hs = np.where(np.isfinite(s.t_hdr) & (s.t_hdr > 0), s.t_hdr, 0.0)
        if np.max(np.abs(b.t_hdr - hs)) > tol:
            problems.append(f'{key}: header stamp mismatch {np.max(np.abs(b.t_hdr - hs)):.3g}s')
        a = s.val
        if key == CMD:
            a = np.clip(np.rint(np.nan_to_num(a)), INT8_MIN, INT8_MAX)
        bv = b.val
        same = (a == bv) | (np.isnan(a) & np.isnan(bv))
        if key not in (CMD,) and key not in GNSS:
            same |= np.isclose(a, bv, rtol=0, atol=0)
        if not same.all():
            problems.append(f'{key}: {int((~same).sum())} payload mismatches')
    return problems


def compare_schema(bag_a: str | Path, bag_b: str | Path) -> list[str]:
    """Differences between the sqlite DDL / schema rows / QoS strings of two single-file bags."""
    a, b = SourceBag.open(bag_a), SourceBag.open(bag_b)
    diffs = []
    if sorted(x[2] for x in a.ddl) != sorted(x[2] for x in b.ddl):
        diffs.append('DDL differs')
    if a.schema_rows != b.schema_rows:
        diffs.append(f'schema rows {a.schema_rows} vs {b.schema_rows}')
    ta = {t[1]: t[2:] for t in a.topics}
    tb = {t[1]: t[2:] for t in b.topics}
    if ta != tb:
        diffs.append('topics table differs')
    va = a.meta['rosbag2_bagfile_information']['version']
    vb = b.meta['rosbag2_bagfile_information']['version']
    if va != vb:
        diffs.append(f'metadata version {va} vs {vb}')
    return diffs
