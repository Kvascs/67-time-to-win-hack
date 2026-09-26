"""Convert all rosbag2 recordings in data/bags into compact .npz files for offline analysis.

Each npz holds, per topic, arrays of bag receive time (t), header stamp (hs) and payload fields.
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

ROOT = Path(__file__).resolve().parents[1]
BAGS = ROOT / 'data' / 'bags'
MSGS = ROOT / 'data' / 'tram_vehicle_msgs' / 'msg'
OUT = ROOT / 'data' / 'npz'


def make_typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    add = {}
    for name in ('VelocitySensor', 'DriverControllerCommand'):
        add.update(get_types_from_msg((MSGS / f'{name}.msg').read_text(), f'tram_vehicle_msgs/msg/{name}'))
    ts.register(add)
    return ts


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def convert(bag: Path):
    out = OUT / f'{bag.name}.npz'
    if out.exists():
        return bag.name, 'skip'
    ts = make_typestore()
    d = {}
    with AnyReader([bag], default_typestore=ts) as r:
        for c in r.connections:
            rows = []
            for conn, t, raw in r.messages(connections=[c]):
                m = r.deserialize(raw, conn.msgtype)
                tt = t * 1e-9
                if c.msgtype.endswith('VelocitySensor'):
                    rows.append((tt, stamp(m.header), m.velocity))
                elif c.msgtype.endswith('DriverControllerCommand'):
                    rows.append((tt, stamp(m.header), m.position))
                elif c.msgtype.endswith('NavSatFix'):
                    rows.append((tt, stamp(m.header), m.latitude, m.longitude, m.altitude, m.status.status,
                                 m.position_covariance[0], m.position_covariance[4], m.position_covariance[8]))
                elif c.msgtype.endswith('TwistStamped'):
                    rows.append((tt, stamp(m.header), m.twist.linear.x, m.twist.linear.y, m.twist.linear.z,
                                 m.twist.angular.z))
                elif c.msgtype.endswith('Odometry'):  # reference localisation (/localization/kinematic_state)
                    p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist
                    rows.append((tt, stamp(m.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                                 v.linear.x, v.linear.y, v.linear.z, v.angular.z))
            key = c.topic.strip('/').replace('/', '__')
            arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 3))
            if key in d:  # multiple connections on same topic
                d[key] = np.vstack([d[key], arr])
                d[key] = d[key][np.argsort(d[key][:, 0], kind='stable')]
            else:
                d[key] = arr
    np.savez_compressed(out, **d)
    return bag.name, {k: v.shape for k, v in d.items()}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    bags = sorted(p for p in BAGS.iterdir() if p.is_dir())
    if len(sys.argv) > 1:
        bags = [b for b in bags if b.name in sys.argv[1:]]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for name, info in ex.map(convert, bags):
            print(name, info, flush=True)
