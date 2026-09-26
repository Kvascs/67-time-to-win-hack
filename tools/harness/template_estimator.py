"""Copy-paste starting point for a new estimator (full API: module docstring of harness/replay.py).

Run it:  python -m harness.run_eval --est harness.template_estimator:MyEstimator --split val --tick-hz 50

Rules the harness enforces the same way the hidden test does:
  * GNSS callbacks only fire during the first ``gnss_seconds`` (default 5 s) of the bag;
  * callbacks arrive in bag-time order, you can only use the past (causal);
  * whatever you return is 'published' at the current bag time (emit time) with YOUR stamp.
"""
from __future__ import annotations

import math

from harness.reference import LocalFrame
from harness.replay import Output


class MyEstimator:
    def __init__(self, frame: str = 'enu', kmh_div: float = 3.6):
        self.frame_kind = frame           # MUST match the judge's frame (enu vs utm differ ~110 m!)
        self.kmh_div = kmh_div
        self.frame = None                 # LocalFrame, origin = first valid MASTER fix
        self.pos = [0.0, 0.0, 0.0]        # tracked point = MASTER antenna (rear), rover is 12.44 m ahead
        self.heading = None
        self.v = 0.0
        self.vf = self.vr = None
        self.t_last = None                # bag clock of the last integration step
        self.last_hdr = None              # last input header stamp (for stamping)
        self._m = self._r = None

    # ---- GNSS: first seconds only (initial alignment) ----
    def on_gnss_fix(self, stamp, lat, lon, alt, status, t_bag, antenna):
        if status < 0:
            return None
        if self.frame is None and antenna == 'master':
            self.frame = LocalFrame(self.frame_kind, lat, lon, alt)
        if self.frame is None:
            return None
        p = [float(c) for c in self.frame.forward(lat, lon, alt)]
        if antenna == 'master':
            self._m, self.pos, self.t_last = (t_bag, p), p, t_bag
        else:
            self._r = (t_bag, p)
        if self._m and self._r and abs(self._m[0] - self._r[0]) < 0.25:
            dx, dy = self._r[1][0] - self._m[1][0], self._r[1][1] - self._m[1][1]
            if 8.0 < math.hypot(dx, dy) < 17.0:
                self.heading = math.atan2(dy, dx)     # master -> rover = direction of travel
        return None

    def on_gnss_vel(self, stamp, vx, vy, vz, t_bag, antenna):
        return None                                   # ENU [m/s]; useful for scale calibration

    # ---- inputs ----
    def on_front(self, stamp, v_kmh, t_bag):
        self._step(t_bag)
        if math.isfinite(v_kmh):
            self.vf = v_kmh / self.kmh_div
        self.last_hdr = stamp
        return None                                   # publish from on_tick instead

    def on_rear(self, stamp, v_kmh, t_bag):
        self._step(t_bag)
        if math.isfinite(v_kmh):
            self.vr = v_kmh / self.kmh_div
        self.last_hdr = stamp
        return None

    def on_cmd(self, stamp, notch, t_bag):
        return None

    # ---- publication timer (enable with --tick-hz 50) ----
    def on_tick(self, t_bag):
        self._step(t_bag)
        # stamp: bag clock (= sim time under `ros2 bag play --clock`). Header-consistent stamps
        # (bag clock - running wheel latency, see baselines.stamp_mode='hdr_extrap') are physically
        # aligned with GNSS header time. Whichever you pick: >= 20 Hz, distinct stamps, and predict
        # the state to the stamp time (holding the last wheel value adds a lag bias on transients).
        return Output(t_bag, self.v, self.pos[0], self.pos[1], self.pos[2], 'map',
                      cov=(4.0, 4.0, 9.0), v_var=0.01)

    # ---- model ----
    def _step(self, t_bag):
        vals = [v for v in (self.vf, self.vr) if v is not None]
        if self.t_last is not None and self.heading is not None:
            dt = min(max(t_bag - self.t_last, 0.0), 1.0)
            self.pos[0] += self.v * dt * math.cos(self.heading)
            self.pos[1] += self.v * dt * math.sin(self.heading)
        self.t_last = t_bag if self.t_last is None or t_bag > self.t_last else self.t_last
        if vals:
            self.v = sum(vals) / len(vals)
