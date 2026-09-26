"""Reference estimators for the harness.

B0  - naive: speed = mean(front, rear)/3.6, position = dead reckoning along the INITIAL heading
      (straight line). Heading from the GNSS antenna baseline master->rover (rover is 12.44 m
      ahead of master along the car body; works at standstill), fallback GNSS velocity direction.
B1  - oracle map: same speed, position = arc length s = s0 + integral(v dt) mapped onto the GNSS
      path of the SAME bag (cheating: upper bound of any 'integrate along a map' approach with
      naive speed; its error is purely the odometry distance error).
B2  - B1 + stale/NaN-aware and notch-aware bogie fusion (input hygiene only, no dynamics model).

All follow the estimator API documented in replay.py and emit one output per wheel message
(~20 Hz, but front/rear share header stamps -> only ~10 Hz of DISTINCT stamps), plus one per cmd
message if emit_on_cmd=True (-> ~40 Hz). They integrate on the bag clock (header stamps have
+-1 s glitches). Timing options (all measured in harness studies on val):
  stamp_mode  'header' (input header.stamp), 'bag' (bag clock), 'hdr_extrap' (bag clock minus the
              running-median wheel latency: header-consistent AND distinct at any rate)
  extrapolate predict v (and s) from the last wheel update to the output time (removes hold lag)
  integ       'zoh' (left Riemann, lags ~dt/2 = 50 ms) or 'trap' (trapezoid at wheel updates)
  pos_lead    extra prediction for POSITION only [s]; GNSS fix stamps lead wheel/vel time by ~45 ms
  lead        extra prediction for speed and position [s]
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from .reference import LocalFrame
from .replay import Output


class WheelSpeedDR:
    """Shared plumbing: GNSS initial alignment (first N s), wheel speed, bag-clock integration."""

    def __init__(self, frame: str = 'enu', stamp_mode: str = 'header', kmh_div: float = 3.6,
                 emit_on_cmd: bool = False, min_status: int = 0, extrapolate: bool = False, lead: float = 0.0,
                 integ: str = 'zoh', pos_lead: float = 0.0):
        self.extrapolate = extrapolate    # predict v (and s) from the last wheel update to the output time
        self.lead = lead                  # extra prediction horizon [s] (compensates sensor lag)
        self.integ = integ                # 'zoh': left Riemann on the bag clock (lags ~dt/2 = 50 ms);
        #                                   'trap': trapezoid at each wheel update + extrapolation between
        self.pos_lead = pos_lead          # extra horizon for POSITION only [s] (GNSS fix stamps lead ~45 ms)
        self._hist = deque(maxlen=12)     # (t_bag, fused v) at wheel updates -> acceleration estimate
        self.frame_kind = frame
        self.stamp_mode = stamp_mode
        self.kmh_div = kmh_div
        self.emit_on_cmd = emit_on_cmd
        self.min_status = min_status
        self.frame: Optional[LocalFrame] = None
        self.vf = None            # latest front speed [m/s]
        self.vr = None            # latest rear speed  [m/s]
        self.v = 0.0              # current speed estimate [m/s]
        self.t_int = None         # bag time up to which the state is integrated
        self.master = None        # (t_bag, xyz) latest master fix
        self.rover = None         # (t_bag, xyz) latest rover fix
        self.heading = None       # [rad] in frame x/y
        self.heading_src = None
        self.gnss_pos = None      # latest master position (reset source)
        self._lat = deque(maxlen=51)  # recent wheel latencies t_bag - header.stamp (for 'hdr_extrap')

    # ---- GNSS (only during the first seconds) ----
    def on_gnss_fix(self, stamp, lat, lon, alt, status, t_bag, antenna):
        if status < self.min_status or not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        if self.frame is None:
            if antenna != 'master':
                return None
            self.frame = LocalFrame(self.frame_kind, lat, lon, alt)
        x, y, z = self.frame.forward(lat, lon, alt)
        p = np.array([float(x), float(y), float(z)])
        if antenna == 'master':
            self.master = (t_bag, p)
            self._reset_to_fix(p, t_bag)
        else:
            self.rover = (t_bag, p)
        if self.master is not None and self.rover is not None and abs(self.master[0] - self.rover[0]) < 0.25:
            d = self.rover[1] - self.master[1]
            if 8.0 < math.hypot(d[0], d[1]) < 17.0:     # plausible 12.44 m baseline
                self.heading = math.atan2(d[1], d[0])
                self.heading_src = 'baseline'
        return None

    def on_gnss_vel(self, stamp, vx, vy, vz, t_bag, antenna):
        if self.frame is None or antenna != 'master':
            return None
        if self.heading_src != 'baseline' and math.hypot(vx, vy) > 1.0:
            ex, ey = self.frame.rotate_enu(vx, vy)
            self.heading = math.atan2(float(ey), float(ex))
            self.heading_src = 'vel'
        return None

    def _reset_to_fix(self, p, t_bag):
        self.gnss_pos = p
        self.t_int = t_bag

    # ---- wheels / cmd ----
    def _speed(self):
        vals = [v for v in (self.vf, self.vr) if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def on_front(self, stamp, v_kmh, t_bag):
        return self._on_wheel(stamp, v_kmh, t_bag, 'f')

    def on_rear(self, stamp, v_kmh, t_bag):
        return self._on_wheel(stamp, v_kmh, t_bag, 'r')

    def on_cmd(self, stamp, notch, t_bag):
        if self.emit_on_cmd:
            if self.integ == 'zoh':
                self._advance(t_bag)
            return self._output(stamp, t_bag)
        return None

    def _note_latency(self, stamp, t_bag):
        d = t_bag - stamp
        if 0.0 <= d < 0.5:                        # skip bag-start burst and +-1 s header glitches
            self._lat.append(d)

    def _stamp(self, stamp, t_bag):
        """'header': input header.stamp; 'bag': bag clock; 'hdr_extrap': bag clock minus the running
        median wheel latency = header-consistent time that is distinct for every output."""
        if self.stamp_mode == 'header':
            return stamp
        if self.stamp_mode == 'bag':
            return t_bag
        if self.stamp_mode == 'hdr_extrap':
            lat = sorted(self._lat)[len(self._lat) // 2] if self._lat else 0.05
            return t_bag - lat
        raise ValueError(self.stamp_mode)

    def _on_wheel(self, stamp, v_kmh, t_bag, which):
        if self.integ == 'zoh':
            self._advance(t_bag)                   # integrate with the previous speed (causal ZOH)
        self._note_latency(stamp, t_bag)
        v = v_kmh / self.kmh_div if math.isfinite(v_kmh) and v_kmh >= 0 else None
        if which == 'f':
            self.vf = v if v is not None else self.vf
        else:
            self.vr = v if v is not None else self.vr
        self._set_speed(self._speed(), t_bag)
        return self._output(stamp, t_bag)

    def _set_speed(self, v_new, t_bag):
        """New fused wheel speed. 'trap': integrate [t_int, t_bag] with the mean of old and new speed."""
        if self.integ == 'trap':
            if self.t_int is None:
                self.t_int = t_bag
            dt = t_bag - self.t_int
            if dt > 0:
                self._integrate(0.5 * (self.v + v_new) * min(dt, 1.0))
                self.t_int = t_bag
        self.v = v_new
        self._hist.append((t_bag, v_new))

    def _advance(self, t_bag):
        if self.t_int is None:
            self.t_int = t_bag
            return
        dt = t_bag - self.t_int
        if dt <= 0:
            return
        dt = min(dt, 1.0)                         # guard against long gaps
        self._integrate(self.v * dt)
        self.t_int = t_bag

    # to be specialised
    def _integrate(self, ds):
        raise NotImplementedError

    def _position(self, ds: float = 0.0):
        raise NotImplementedError

    def _accel(self):
        """Finite-difference acceleration of the fused wheel speed over >= 0.3 s (clamped)."""
        if len(self._hist) < 2:
            return 0.0
        t1, v1 = self._hist[-1]
        for t0, v0 in reversed(self._hist):
            if t1 - t0 >= 0.3:
                return max(-2.0, min(2.0, (v1 - v0) / (t1 - t0)))
        return 0.0

    def _output(self, stamp, t_bag):
        v = self.v
        ds = 0.0
        h = t_bag - self.t_int if (self.integ == 'trap' and self.t_int is not None) else 0.0
        if self.extrapolate and self._hist:
            a = self._accel()
            dt = (t_bag - self._hist[-1][0]) + self.lead
            v = max(0.0, v + a * dt)
            ds = self.v * self.lead + 0.5 * a * h * h
        ds += self.v * (h + self.pos_lead)        # 'trap' state is anchored at the last wheel update
        x, y, z = self._position(ds)
        return Output(self._stamp(stamp, t_bag), v, x, y, z, 'map', yaw=self._yaw())

    def _yaw(self):
        return self.heading


class B0(WheelSpeedDR):
    """Naive dead reckoning along the initial heading."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.pos = np.zeros(3)

    def _reset_to_fix(self, p, t_bag):
        super()._reset_to_fix(p, t_bag)
        self.pos = p.copy()

    def _integrate(self, ds):
        if self.heading is None:
            return
        self.pos[0] += ds * math.cos(self.heading)
        self.pos[1] += ds * math.sin(self.heading)

    def _position(self, ds: float = 0.0):
        if self.heading is None or ds == 0.0:
            return float(self.pos[0]), float(self.pos[1]), float(self.pos[2])
        return (float(self.pos[0] + ds * math.cos(self.heading)), float(self.pos[1] + ds * math.sin(self.heading)),
                float(self.pos[2]))


class B1(WheelSpeedDR):
    """Along-track integration on the oracle path of the same bag (upper bound for map-based)."""
    needs_oracle_ref = True

    def __init__(self, oracle_ref=None, **kw):
        super().__init__(**kw)
        if oracle_ref is None:
            raise ValueError('B1 needs oracle_ref (the harness passes it automatically)')
        if oracle_ref.frame.kind != self.frame_kind:
            raise ValueError('oracle path frame differs from estimator frame')
        self.path = oracle_ref.path
        self.s = 0.0
        self.s_known = False

    def _reset_to_fix(self, p, t_bag):
        super()._reset_to_fix(p, t_bag)
        hint = self.s if self.s_known else 0.0
        s, _, _ = self.path.project(p[None, :2], s_hint=hint, window=60.0)
        self.s = float(s[0])
        self.s_known = True

    def _integrate(self, ds):
        self.s += ds          # no clamp at the path end: knowing where the recording stops is cheating

    def _yaw(self):
        return self.path.heading_scalar(min(self.s, self.path.length))

    def _position(self, ds: float = 0.0):
        s = self.s + ds
        L = self.path.length
        if s <= L:
            return self.path.interp_scalar(s)
        x, y, z = self.path.interp_scalar(L)          # beyond the end: continue along the last tangent
        t = self.path.tangent(np.array([L - 5.0]))[0]
        return x + t[0] * (s - L), y + t[1] * (s - L), z


class B2(B1):
    """B1 + simple robust bogie fusion (still naive otherwise; oracle map for position).

    * a bogie is ignored when its last valid message is older than ``stale_s`` (dropout) or its
      value is NaN / negative / > 150 km/h;
    * if both bogies are fresh and disagree by more than max(0.3 m/s, 5 %): traction (notch > 0)
      -> take the lower one (spin reads high), braking (notch < 0) -> the higher one (slide reads
      low), coasting -> the mean;
    * no fresh bogie -> hold the last speed.
    Shows how much of the val error is plain input hygiene (stale bogie in 30639_* bags).
    """

    def __init__(self, stale_s: float = 0.3, **kw):
        super().__init__(**kw)
        self.stale_s = stale_s
        self.tf = -math.inf
        self.tr = -math.inf
        self.notch = 0

    def on_cmd(self, stamp, notch, t_bag):
        if -15 <= notch <= 15:
            self.notch = notch
        return super().on_cmd(stamp, notch, t_bag)

    def _on_wheel(self, stamp, v_kmh, t_bag, which):
        if self.integ == 'zoh':
            self._advance(t_bag)
        self._note_latency(stamp, t_bag)
        if math.isfinite(v_kmh) and 0.0 <= v_kmh <= 150.0:
            if which == 'f':
                self.vf, self.tf = v_kmh / self.kmh_div, t_bag
            else:
                self.vr, self.tr = v_kmh / self.kmh_div, t_bag
        self._set_speed(self._fused(t_bag), t_bag)
        return self._output(stamp, t_bag)

    def _fused(self, t):
        f = self.vf if (self.vf is not None and t - self.tf <= self.stale_s) else None
        r = self.vr if (self.vr is not None and t - self.tr <= self.stale_s) else None
        if f is not None and r is not None:
            if abs(f - r) <= max(0.3, 0.05 * max(f, r)) or self.notch == 0:
                return 0.5 * (f + r)
            return max(f, r) if self.notch < 0 else min(f, r)
        if f is not None:
            return f
        if r is not None:
            return r
        return self.v
