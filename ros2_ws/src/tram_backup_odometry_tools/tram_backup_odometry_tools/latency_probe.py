"""Live latency / rate / resource probe for the tram backup odometry node.

Latency definition (input -> result): for every published /result/velocity whose header stamp
equals the stamp of an input message (controller or bogie), latency = wall-clock arrival of the
result - wall-clock arrival of that input, both measured in this probe process. Outputs published
on the fixed stamp grid are matched to the most recent input that triggered them.
Also reports the output rate and, if psutil is available, CPU and RSS of the estimator process.

Usage:  ros2 run tram_backup_odometry_tools latency_probe [--duration 60] [--csv latency.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor


def stamp_ns(h) -> int:
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


class LatencyProbe(Node):
    def __init__(self, csv_path: str | None):
        super().__init__('tbo_latency_probe')
        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.inputs = collections.OrderedDict()   # stamp_ns -> arrival (monotonic ns)
        self.last_input_arrival = None
        self.lat_ms = []
        self.out_times = collections.deque(maxlen=2000)
        self.n_out = 0
        self.csv = open(csv_path, 'w', newline='') if csv_path else None
        self.writer = csv.writer(self.csv) if self.csv else None
        if self.writer:
            self.writer.writerow(['stamp_ns', 'latency_ms', 'matched'])
        self.create_subscription(DriverControllerCommand, '/vehicle/driver_position_cmd', self.on_input, qos)
        self.create_subscription(VelocitySensor, '/vehicle/front_bogie_velocity', self.on_input, qos)
        self.create_subscription(VelocitySensor, '/vehicle/rear_bogie_velocity', self.on_input, qos)
        self.create_subscription(VelocitySensor, '/result/velocity', self.on_output, qos)
        self.proc = self._find_estimator()
        self.cpu_samples, self.rss_samples = [], []
        self.create_timer(5.0, self.report)

    def _find_estimator(self):
        try:
            import psutil
        except ImportError:
            self.get_logger().warn('psutil not installed: CPU/RAM not measured (pip install psutil)')
            return None
        # the compiled node itself, not a `ros2 run` / launch wrapper whose command line mentions it
        best = None
        for p in psutil.process_iter(['name', 'cmdline']):
            name = p.info.get('name') or ''
            cmd = p.info.get('cmdline') or []
            exe = os.path.basename(cmd[0]) if cmd else ''
            if name == 'tbo_node' or exe == 'tbo_node':
                best = p
                break
        if best is None:
            self.get_logger().warn('tbo_node process not found: CPU/RAM not measured')
            return None
        best.cpu_percent(None)
        self.get_logger().info(f'measuring CPU/RAM of pid {best.pid} ({best.info.get("name")})')
        return best

    def on_input(self, msg):
        now = time.monotonic_ns()
        self.inputs[stamp_ns(msg.header)] = now
        self.last_input_arrival = now
        while len(self.inputs) > 5000:
            self.inputs.popitem(last=False)

    def on_output(self, msg):
        now = time.monotonic_ns()
        self.n_out += 1
        self.out_times.append(now)
        st = stamp_ns(msg.header)
        t_in = self.inputs.get(st)
        matched = t_in is not None
        if t_in is None:
            t_in = self.last_input_arrival  # grid-stamped output: triggered by the latest input
        if t_in is None:
            return
        lat = (now - t_in) / 1e6
        self.lat_ms.append(lat)
        if self.writer:
            self.writer.writerow([st, f'{lat:.3f}', int(matched)])

    def report(self):
        if self.proc is not None:
            try:
                self.cpu_samples.append(self.proc.cpu_percent(None))
                self.rss_samples.append(self.proc.memory_info().rss / 2**20)
            except Exception:
                self.proc = None
        if not self.lat_ms:
            self.get_logger().info('waiting for /result/velocity ...')
            return
        lat = sorted(self.lat_ms)
        q = lambda f: lat[min(len(lat) - 1, int(f * len(lat)))]
        rate = 0.0
        if len(self.out_times) > 1:
            rate = (len(self.out_times) - 1) / ((self.out_times[-1] - self.out_times[0]) / 1e9)
        res = f'outputs={self.n_out} rate={rate:.1f} Hz latency ms p50={q(0.5):.2f} p95={q(0.95):.2f} ' \
              f'p99={q(0.99):.2f} max={lat[-1]:.2f}'
        if self.cpu_samples:
            res += f' | node CPU {self.cpu_samples[-1]:.1f}% (max {max(self.cpu_samples):.1f}%)' \
                   f' RSS {self.rss_samples[-1]:.1f} MiB (max {max(self.rss_samples):.1f})'
        self.get_logger().info(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=0.0, help='stop after N seconds (0 = until Ctrl-C)')
    ap.add_argument('--csv', default=None, help='write per-output latency to this CSV')
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = LatencyProbe(args.csv)
    t_end = time.monotonic() + args.duration if args.duration > 0 else None
    try:
        while rclpy.ok() and (t_end is None or time.monotonic() < t_end):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.report()
    if node.csv:
        node.csv.close()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
