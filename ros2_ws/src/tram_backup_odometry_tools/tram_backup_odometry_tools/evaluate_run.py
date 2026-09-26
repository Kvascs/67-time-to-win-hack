"""Judge-like accuracy check of a recorded run.

Reads the GNSS reference from the ORIGINAL input bag (master antenna: /sensing/gnss/master/fix
and /vel) and our outputs from a bag recorded during playback (/result/velocity,
/result/position). Pairs every reference sample with the output of the nearest header stamp
(tolerance 0.05 s) and prints speed RMSE/MAE/bias, position error (3-D, along/cross-track using
the published heading), z error and end drift in % of the travelled reference distance.
Two reference frames:
  --frame mgrs (default, like the jury): base_link in MGRS 37U CB (x = UTM E - 300000, y = N - 6100000)
      built from BOTH antennas with the organisers' TF (master x = -9.873, rover x = +2.563, z = +3.0 in
      base_link): base_link = master + 9.873 * unit(rover - master), height - 3.0 m. Node at defaults.
  --frame enu: antenna 1 in the ENU tangent plane at the first master fix; run the node with
      output_frame:=enu base_link_along_m:=0 base_link_height_m:=0.

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


def utm(lat, lon, zone=37):
    """WGS84 -> UTM (northern hemisphere), Krueger series to n^4 (sub-millimetre)."""
    n = F / (2 - F)
    a_ = A / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    al = [n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180,
          13 * n ** 2 / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440,
          61 * n ** 3 / 240 - 103 * n ** 4 / 140,
          49561 * n ** 4 / 161280]
    la, lo = np.radians(lat), np.radians(lon) - math.radians(zone * 6 - 183)
    e = math.sqrt(E2)
    t = np.sinh(np.arctanh(np.sin(la)) - e * np.arctanh(e * np.sin(la)))
    xi, eta = np.arctan2(t, np.cos(lo)), np.arctanh(np.sin(lo) / np.sqrt(1 + t ** 2))
    x, y = eta.copy(), xi.copy()
    for j, aj in enumerate(al, 1):
        x += aj * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        y += aj * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return 500000.0 + 0.9996 * a_ * x, 0.9996 * a_ * y


def base_link_reference(ref: dict, along=9.873, height=3.0, e0=300000.0, n0=6100000.0):
    """Reference base_link in MGRS from both antennas (fixes paired within 20 ms)."""
    m = [f for f in ref['/sensing/gnss/master/fix'] if math.isfinite(f.latitude)]
    r = [f for f in ref['/sensing/gnss/rover/fix'] if math.isfinite(f.latitude)]
    if not m or not r:
        return None
    mt = np.array([st(f.header) for f in m])
    rt = np.array([st(f.header) for f in r])
    order = np.argsort(rt)
    rt, r = rt[order], [r[i] for i in order]
    j = np.clip(np.searchsorted(rt, mt), 1, len(rt) - 1)
    j = np.where(np.abs(rt[j - 1] - mt) < np.abs(rt[j] - mt), j - 1, j)
    ok = np.abs(rt[j] - mt) < 0.02
    me, mn = utm(np.array([f.latitude for f in m]), np.array([f.longitude for f in m]))
    re_, rn = utm(np.array([r[k].latitude for k in j]), np.array([r[k].longitude for k in j]))
    ma = np.array([f.altitude for f in m])
    ra = np.array([r[k].altitude for k in j])
    u = np.stack([re_ - me, rn - mn, ra - ma], -1)
    ln = np.linalg.norm(u, axis=1)
    ok &= (ln > 11.9) & (ln < 13.0)  # antenna baseline 12.44 m: both fixes consistent
    u = u / np.maximum(ln, 1e-9)[:, None]
    b = np.stack([me - e0, mn - n0, ma], -1) + along * u
    b[:, 2] -= height
    return mt[ok], b[ok], np.arctan2(u[ok, 1], u[ok, 0])


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
    ap.add_argument('--frame', choices=['mgrs', 'enu'], default='mgrs')
    a = ap.parse_args()
    ref = read_bag(a.input_bag, {'/sensing/gnss/master/fix', '/sensing/gnss/master/vel', '/sensing/gnss/rover/fix'})
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
    if a.frame == 'mgrs':
        br = base_link_reference(ref)
        if br is None or len(br[0]) == 0:
            raise SystemExit('no paired master/rover fixes for the base_link reference')
        ft, p_ref, _ = br
    else:
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
    print(f'pos   : [{a.frame}] matched {ok.mean() * 100:.1f}% | 3-D RMSE {np.sqrt(np.mean(e3 ** 2)):.2f} m, max {e3.max():.2f} m | '
          f'along RMSE {np.sqrt(np.mean(along ** 2)):.2f} m (max {np.abs(along).max():.2f}) | cross RMSE '
          f'{np.sqrt(np.mean(cross ** 2)):.2f} m | z RMSE {np.sqrt(np.mean(e[:, 2] ** 2)):.2f} m')
    print(f'drift : end error {e3[-1]:.2f} m over {dist:.0f} m = {100 * e3[-1] / max(dist, 1):.3f} %')
    rate = (len(ot) - 1) / (ot[-1] - ot[0]) if len(ot) > 1 else 0.0
    print(f'rate  : {rate:.1f} Hz of /result/velocity (by header stamps), stamps strictly increasing: '
          f'{bool(np.all(np.diff(ot) > 0))}')


if __name__ == '__main__':
    main()
