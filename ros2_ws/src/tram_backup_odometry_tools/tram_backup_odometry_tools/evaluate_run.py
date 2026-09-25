"""Judge-like accuracy check of a recorded run.

Reads the GNSS reference from the ORIGINAL input bag (master antenna: /sensing/gnss/master/fix
and /vel) and our outputs from a bag recorded during playback (/result/velocity,
/result/position). Pairs every reference sample with the output of the nearest header stamp
(tolerance 0.05 s) and prints speed RMSE/MAE/bias, position error (3-D, along/cross-track using
the published heading), z error and end drift in % of the travelled reference distance.
Reference frame: ENU tangent plane at the first master fix (same as the node's output_frame=enu).

Usage:
  ros2 bag record -o run_out /result/velocity /result/position /result/status   # during playback
  ros2 run tram_backup_odometry_tools evaluate_run --input-bag <dataset bag dir> --output-bag run_out
"""
from __future__ import annotations

import argparse
import math

import numpy as np

A, F = 6378137.0, 1 / 298.257223563
E2 = F * (2 - F)


def _ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


def enu(lat, lon, h, lat0, lon0, h0):
    d = _ecef(lat, lon, h) - _ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))
    la, lo = math.radians(lat0), math.radians(lon0)
    r = np.array([[-math.sin(lo), math.cos(lo), 0],
                  [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                  [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
    return d @ r.T


def read_bag(path: str, topics: set[str]) -> dict:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic in topics and topic in types:
            out[topic].append(deserialize_message(raw, get_message(types[topic])))
    return out


def st(h) -> float:
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def nearest(ref_t, out_t, tol=0.05):
    idx = np.clip(np.searchsorted(out_t, ref_t), 1, len(out_t) - 1)
    pick = np.where(np.abs(ref_t - out_t[idx - 1]) <= np.abs(out_t[idx] - ref_t), idx - 1, idx)
    return pick, np.abs(out_t[pick] - ref_t) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-bag', required=True)
    ap.add_argument('--output-bag', required=True)
    a = ap.parse_args()
    ref = read_bag(a.input_bag, {'/sensing/gnss/master/fix', '/sensing/gnss/master/vel'})
    res = read_bag(a.output_bag, {'/result/velocity', '/result/position'})
    vel = sorted(res['/result/velocity'], key=lambda m: st(m.header))
    pos = sorted(res['/result/position'], key=lambda m: st(m.header))
    if not vel or not pos:
        raise SystemExit('no /result/* messages in the output bag')
    # ---- speed
    rv = ref['/sensing/gnss/master/vel']
    rt = np.array([st(m.header) for m in rv])
    rspeed = np.array([math.hypot(m.twist.linear.x, m.twist.linear.y) for m in rv])
    ot = np.array([st(m.header) for m in vel])
    ov = np.array([m.velocity for m in vel])
    pick, ok = nearest(rt, ot)
    ev = ov[pick][ok] - rspeed[ok]
    print(f'speed : matched {ok.mean() * 100:.1f}% of {len(rt)} reference samples | RMSE {np.sqrt(np.mean(ev ** 2)):.3f} '
          f'MAE {np.mean(np.abs(ev)):.3f} bias {np.mean(ev):+.3f} m/s | moving RMSE '
          f'{np.sqrt(np.mean(ev[rspeed[ok] > 0.5] ** 2)):.3f} m/s')
    # ---- position
    fx = [m for m in ref['/sensing/gnss/master/fix'] if math.isfinite(m.latitude)]
    ft = np.array([st(m.header) for m in fx])
    lat = np.array([m.latitude for m in fx]); lon = np.array([m.longitude for m in fx]); alt = np.array([m.altitude for m in fx])
    p_ref = enu(lat, lon, alt, lat[0], lon[0], alt[0])
    pt = np.array([st(m.header) for m in pos])
    p_out = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for m in pos])
    yaw = np.array([2 * math.atan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w) for m in pos])
    pick, ok = nearest(ft, pt)
    e = p_out[pick][ok] - p_ref[ok]
    y = yaw[pick][ok]
    along = e[:, 0] * np.cos(y) + e[:, 1] * np.sin(y)
    cross = -e[:, 0] * np.sin(y) + e[:, 1] * np.cos(y)
    e3 = np.linalg.norm(e, axis=1)
    dist = float(np.sum(np.hypot(np.diff(p_ref[:, 0]), np.diff(p_ref[:, 1]))))
    print(f'pos   : matched {ok.mean() * 100:.1f}% | 3-D RMSE {np.sqrt(np.mean(e3 ** 2)):.2f} m, max {e3.max():.2f} m | '
          f'along RMSE {np.sqrt(np.mean(along ** 2)):.2f} m (max {np.abs(along).max():.2f}) | cross RMSE '
          f'{np.sqrt(np.mean(cross ** 2)):.2f} m | z RMSE {np.sqrt(np.mean(e[:, 2] ** 2)):.2f} m')
    print(f'drift : end error {e3[-1]:.2f} m over {dist:.0f} m = {100 * e3[-1] / max(dist, 1):.3f} %')
    rate = (len(ot) - 1) / (ot[-1] - ot[0]) if len(ot) > 1 else 0.0
    print(f'rate  : {rate:.1f} Hz of /result/velocity (by header stamps), stamps strictly increasing: '
          f'{bool(np.all(np.diff(ot) > 0))}')


if __name__ == '__main__':
    main()
