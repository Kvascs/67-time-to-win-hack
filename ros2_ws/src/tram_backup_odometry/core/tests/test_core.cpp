// Core behaviour tests: invariants that need no ground truth, robustness to garbage
// input, slip/dropout handling on a synthetic tram, and the "no GNSS after the init
// window" proof (outputs must be bit-identical with and without later GNSS).
#include <algorithm>
#include <cstdio>
#include <string>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mini_test.hpp"
#include "tbo/estimator.hpp"
#include "tbo/global_localizer.hpp"
#include "tbo/scheduler.hpp"
#include "tbo/traction_model.hpp"
#include "tbo/track_map.hpp"

using namespace tbo;

namespace {

constexpr double kKmh = 3.5965;  // km/h reading per m/s (matches default calibration)

struct In {
  int type;  // 0 front, 1 rear, 2 cmd, 3 gnss fix master, 4 gnss fix rover
  Stamp recv, stamp;
  double a, b, c;
};

struct Truth {
  std::vector<double> t, v, s;
  double vAt(double tt) const {
    if (tt <= t.front()) return v.front();
    if (tt >= t.back()) return v.back();
    const size_t i = static_cast<size_t>((tt - t.front()) / (t[1] - t[0]));
    return v[std::min(i, v.size() - 1)];
  }
  double sAt(double tt) const {
    if (tt <= t.front()) return s.front();
    if (tt >= t.back()) return s.back();
    const size_t i = static_cast<size_t>((tt - t.front()) / (t[1] - t[0]));
    return s[std::min(i, s.size() - 1)];
  }
};

uint32_t lcg(uint32_t& st) {
  st = st * 1664525u + 1013904223u;
  return st;
}
double noise(uint32_t& st) { return ((lcg(st) >> 8) / 16777216.0 - 0.5) * 2.0; }  // [-1, 1)

// Synthetic run: accelerate with notch 10, cruise, brake with notch -8, stop.
// wheel_fn(sensor, t, v_true) returns the km/h reading (or NaN to drop the message).
template <class WheelFn>
std::vector<In> makeRun(double t0, double dur, Truth& truth, WheelFn wheel_fn, double gnss_until = 3.0,
                        double (*extra_accel)(double) = nullptr) {
  std::vector<In> ev;
  const double dt = 0.01;
  double v = 0.0, s = 0.0;
  truth = Truth{};
  auto notchAt = [](double t) {
    if (t < 5.0) return 0;
    if (t < 20.0) return 10;
    if (t < 40.0) return 0;
    if (t < 52.0) return -8;
    return 0;
  };
  auto accelAt = [&](double t, double vv) {
    const int n = notchAt(t);
    if (n > 0) return std::min(0.9, 7.0 / std::max(vv, 1.0));
    if (n < 0) return vv > 0.0 ? -0.846 : 0.0;
    return vv > 0.0 ? -0.03 : 0.0;
  };
  for (double t = 0.0; t <= dur + 1e-9; t += dt) {
    truth.t.push_back(t0 + t);
    truth.v.push_back(v);
    truth.s.push_back(s);
    double a = accelAt(t, v);
    if (extra_accel && v > 0.0) a += extra_accel(t);
    const double v1 = std::max(0.0, v + a * dt);
    s += 0.5 * (v + v1) * dt;
    v = v1;
  }
  uint32_t st = 12345u;
  for (int k = 0; t0 + k * 0.05 <= t0 + dur; ++k) {  // controller 20 Hz
    const double t = k * 0.05 + 0.016;
    ev.push_back({2, fromSec(t0 + t + 0.001), fromSec(t0 + t), static_cast<double>(notchAt(t)), 0, 0});
  }
  for (int k = 0; k * 0.1 <= dur; ++k) {  // bogies 10 Hz, 46 ms transport delay
    const double t = k * 0.1 + 0.03;
    const double vt = truth.vAt(t0 + t);
    for (int sensor = 0; sensor < 2; ++sensor) {
      const double kmh = wheel_fn(sensor, t, vt) + 0.02 * noise(st);
      if (std::isnan(kmh)) continue;
      ev.push_back({sensor, fromSec(t0 + t + 0.046), fromSec(t0 + t), kmh, 0, 0});
    }
  }
  for (int k = 0; k * 0.1 <= std::min(gnss_until, dur); ++k) {  // GNSS: fixed position, master + rover
    const double t = k * 0.1;
    ev.push_back({3, fromSec(t0 + t + 0.04), fromSec(t0 + t), 55.81, 37.462, 168.4});
    ev.push_back({4, fromSec(t0 + t + 0.04), fromSec(t0 + t), 55.81, 37.462 + 0.0002, 168.4});
  }
  std::stable_sort(ev.begin(), ev.end(), [](const In& x, const In& y) { return x.recv < y.recv; });
  return ev;
}

struct RunResult {
  std::vector<Output> outs;
  Diagnostics diag;
};

RunResult run(const std::vector<In>& ev, const Config& cfg, const TractionModel& model,
              const TrackMap* map = nullptr, const TrackMap* stub = nullptr,
              const std::vector<WheelEpoch>* epochs = nullptr) {
  Estimator est(cfg, model, map);
  if (stub) est.setStub(*stub);
  if (epochs) est.setWheelEpochs(*epochs);
  OutputScheduler sched(cfg.p);
  RunResult r;
  Stamp buf[64];
  std::uint64_t resets = 0;
  for (const In& e : ev) {
    bool is_input = true, is_cmd = false;
    switch (e.type) {
      case 0: is_input = est.onWheel(Sensor::Front, e.stamp, e.a); break;
      case 1: is_input = est.onWheel(Sensor::Rear, e.stamp, e.a); break;
      case 2: is_input = est.onCmd(e.stamp, static_cast<int>(e.a)); is_cmd = true; break;
      case 3: est.onGnssFix(GnssSource::Master, e.stamp, e.a, e.b, e.c, 2); is_input = false; break;
      case 4: est.onGnssFix(GnssSource::Rover, e.stamp, e.a, e.b, e.c, 2); is_input = false; break;
    }
    if (!is_input || !est.started()) continue;
    if (est.diagnostics().resets != resets) {  // same as the node: new run -> new output stream
      resets = est.diagnostics().resets;
      sched.reset();
    }
    const int n = sched.onInput(is_cmd, e.stamp, est.latestStamp(), buf, 64);
    for (int k = 0; k < n; ++k) r.outs.push_back(est.query(buf[k]));
  }
  r.diag = est.diagnostics();
  return r;
}

bool sameOutputs(const RunResult& a, const RunResult& b) {
  if (a.outs.size() != b.outs.size()) return false;
  for (size_t i = 0; i < a.outs.size(); ++i)
    if (a.outs[i].stamp != b.outs[i].stamp || a.outs[i].v != b.outs[i].v || a.outs[i].s != b.outs[i].s ||
        a.outs[i].x != b.outs[i].x)
      return false;
  return true;
}

// Removes every message (all topics) whose stamp falls inside (t_from, t_to) seconds.
std::vector<In> cutAll(const std::vector<In>& ev, double t_from, double t_to) {
  std::vector<In> out;
  for (const In& e : ev) {
    const double t = toSec(e.stamp);
    if (t > t_from && t < t_to) continue;
    out.push_back(e);
  }
  return out;
}

void sortByArrival(std::vector<In>& ev) {
  std::stable_sort(ev.begin(), ev.end(), [](const In& x, const In& y) { return x.recv < y.recv; });
}

// Straight east-west test track through the GNSS test position.
TrackMap straightMap() {
  TrackMap map;
  map.setPoints({{-500, 0, 0, 0}, {5000, 0, 0, 0}}, false, {55.81, 37.462, 168.4});
  return map;
}

// GNSS master/rover fixes that follow the truth along the straight map (rover 12.4 m ahead).
void addMovingGnss(std::vector<In>& ev, const Truth& tr, double t0, double from, double to) {
  const geo::LocalCartesian lc({55.81, 37.462, 168.4});
  for (int k = 0; from + k * 0.1 <= to + 1e-9; ++k) {
    const double t = from + k * 0.1;
    const double s = tr.sAt(t0 + t);
    const geo::Geodetic m = lc.reverse({s, 0.0, 0.0}), r = lc.reverse({s + 12.4, 0.0, 0.0});
    ev.push_back({3, fromSec(t0 + t + 0.04), fromSec(t0 + t), m.lat_deg, m.lon_deg, m.h});
    ev.push_back({4, fromSec(t0 + t + 0.04), fromSec(t0 + t), r.lat_deg, r.lon_deg, r.h});
  }
  sortByArrival(ev);
}

auto cleanWheels = [](int, double, double v) { return v * kKmh; };

double maxSpeedError(const RunResult& r, const Truth& tr, double from = 0.0, double to = 1e9) {
  double m = 0.0;
  for (const Output& o : r.outs) {
    const double t = toSec(o.stamp);
    if (t < from || t > to) continue;
    m = std::max(m, std::abs(o.v - tr.vAt(t)));
  }
  return m;
}

}  // namespace

// ---------------------------------------------------------------- traction model & map

TEST_CASE("traction LUT: brake rows decelerate, traction limited by power") {
  TractionModel m;
  CHECK(m.target(-15, 5.0) < m.target(-1, 5.0));
  CHECK(m.target(0, 5.0) < 0.0);
  CHECK(m.target(15, 15.0) <= 7.0 / 15.0 + 1e-9);
  CHECK_NEAR(m.target(10, 0.0), 0.928, 1e-3);
  CHECK(m.target(99, 5.0) == m.target(15, 5.0));  // clamps notch
}

TEST_CASE("track map: arc length, interpolation, projection, cycle wrap") {
  TrackMap map;
  std::vector<MapPose> sq = {{0, 0, 0, 0}, {100, 0, 1, 0}, {100, 100, 2, 0}, {0, 100, 3, 0}};
  map.setPoints(sq, true, {55.8, 37.4, 170.0});
  CHECK_NEAR(map.length(), 400.0, 1e-9);
  const MapPose p = map.at(150.0);
  CHECK_NEAR(p.x, 100.0, 1e-9);
  CHECK_NEAR(p.y, 50.0, 1e-9);
  CHECK_NEAR(p.heading, 1.5707963, 1e-6);
  CHECK_NEAR(map.at(-10.0).y, 10.0, 1e-9);  // wraps to s = 390 (closing segment)
  const MapProjection pr = map.project(50.0, 3.0);
  CHECK_NEAR(pr.s, 50.0, 1e-9);
  CHECK_NEAR(pr.dist, 3.0, 1e-9);
  MapProjection c[4];
  CHECK(map.projectAll(50.0, 50.0, 60.0, c, 4) == 4);   // centre: all four sides within gate
  CHECK(map.projectAll(2.0, 50.0, 10.0, c, 4) == 1);    // near closing side only
}

// ---------------------------------------------------------------- estimator invariants

TEST_CASE("standstill: zero speed and zero position drift over 60 s") {
  Config cfg;
  TractionModel model;
  Truth tr;
  std::vector<In> ev;
  for (int k = 0; k < 600; ++k) {
    const double t = 1000.0 + k * 0.1;
    ev.push_back({0, fromSec(t + 0.046), fromSec(t), 0.0, 0, 0});
    ev.push_back({1, fromSec(t + 0.046), fromSec(t), 0.0, 0, 0});
    ev.push_back({2, fromSec(t + 0.001), fromSec(t), -3.0, 0, 0});
  }
  std::stable_sort(ev.begin(), ev.end(), [](const In& a, const In& b) { return a.recv < b.recv; });
  const RunResult r = run(ev, cfg, model);
  CHECK(!r.outs.empty());
  double maxv = 0.0, maxs = 0.0;
  for (const Output& o : r.outs) {
    maxv = std::max(maxv, o.v);
    maxs = std::max(maxs, std::abs(o.s));
  }
  CHECK(maxv < 1e-3);
  CHECK(maxs < 1e-3);
  CHECK((r.outs.back().flags & kFlagStandstill) != 0);
}

TEST_CASE("clean run: speed tracks truth, distance within 0.5 %") {
  Config cfg;
  TractionModel model;
  Truth tr;
  const auto ev = makeRun(5000.0, 70.0, tr, cleanWheels);
  const RunResult r = run(ev, cfg, model);
  CHECK(maxSpeedError(r, tr, 5001.0) < 0.15);
  const Output& last = r.outs.back();
  const double s_true = tr.sAt(toSec(last.stamp));
  CHECK(s_true > 150.0);
  CHECK(std::abs(last.s - s_true) / s_true < 0.005);
  for (size_t i = 1; i < r.outs.size(); ++i) CHECK(r.outs[i].stamp > r.outs[i - 1].stamp);
}

TEST_CASE("time-shift invariance: identical results for shifted stamps") {
  Config cfg;
  TractionModel model;
  Truth tr1, tr2;
  const RunResult a = run(makeRun(1000.0, 30.0, tr1, cleanWheels), cfg, model);
  const RunResult b = run(makeRun(1000.0 + 86400.0, 30.0, tr2, cleanWheels), cfg, model);
  CHECK(a.outs.size() == b.outs.size());
  double md = 0.0;
  for (size_t i = 0; i < std::min(a.outs.size(), b.outs.size()); ++i)
    md = std::max(md, std::abs(a.outs[i].v - b.outs[i].v) + std::abs(a.outs[i].s - b.outs[i].s));
  CHECK(md < 1e-6);
}

TEST_CASE("determinism: two replays are bit-identical") {
  Config cfg;
  TractionModel model;
  Truth tr;
  const auto ev = makeRun(2000.0, 40.0, tr, cleanWheels);
  const RunResult a = run(ev, cfg, model), b = run(ev, cfg, model);
  CHECK(a.outs.size() == b.outs.size());
  bool same = a.outs.size() == b.outs.size();
  for (size_t i = 0; same && i < a.outs.size(); ++i)
    same = a.outs[i].v == b.outs[i].v && a.outs[i].s == b.outs[i].s && a.outs[i].stamp == b.outs[i].stamp;
  CHECK(same);
}

TEST_CASE("out-of-order arrival inside the lag window does not change the estimate") {
  Config cfg;
  TractionModel model;
  Truth tr;
  auto ev = makeRun(3000.0, 40.0, tr, cleanWheels);
  auto shuffled = ev;
  // swap neighbouring bogie messages' arrival order (delay by up to 80 ms)
  for (size_t i = 0; i + 1 < shuffled.size(); i += 7)
    if (shuffled[i].type < 2) shuffled[i].recv += fromSec(0.08);
  std::stable_sort(shuffled.begin(), shuffled.end(), [](const In& a, const In& b) { return a.recv < b.recv; });
  const RunResult a = run(ev, cfg, model), b = run(shuffled, cfg, model);
  // Published streams differ in timing; the estimate at a common late stamp must agree.
  Estimator ea(cfg, model, nullptr), eb(cfg, model, nullptr);
  for (const In& e : ev)
    if (e.type < 2) ea.onWheel(e.type == 0 ? Sensor::Front : Sensor::Rear, e.stamp, e.a);
    else if (e.type == 2) ea.onCmd(e.stamp, static_cast<int>(e.a));
  for (const In& e : shuffled)
    if (e.type < 2) eb.onWheel(e.type == 0 ? Sensor::Front : Sensor::Rear, e.stamp, e.a);
    else if (e.type == 2) eb.onCmd(e.stamp, static_cast<int>(e.a));
  const Stamp tq = fromSec(3039.0);
  CHECK_NEAR(ea.query(tq).s, eb.query(tq).s, 1e-6);
  CHECK_NEAR(ea.query(tq).v, eb.query(tq).v, 1e-9);
  CHECK(!a.outs.empty() && !b.outs.empty());
}

TEST_CASE("garbage input never crashes and outputs stay finite") {
  Config cfg;
  TractionModel model;
  Estimator est(cfg, model, nullptr);
  const double bad[] = {NAN, INFINITY, -INFINITY, -50.0, 1e9, 0.0, 30.0};
  Stamp t = fromSec(100.0);
  for (int k = 0; k < 2000; ++k) {
    t += fromSec(0.05);
    est.onWheel(Sensor::Front, t, bad[k % 7]);
    est.onWheel(Sensor::Rear, (k % 13 == 0) ? 0 : t, bad[(k + 3) % 7]);  // zero stamps too
    est.onCmd(k % 17 == 0 ? t + fromSec(1e6) : t, (k % 5 == 0) ? 120 : (k % 31) - 15);  // future + bad notch
    est.onGnssFix(GnssSource::Master, t, NAN, 37.0, 150.0, 2);
    const Output o = est.query(t);
    CHECK(std::isfinite(o.v) && std::isfinite(o.s) && std::isfinite(o.x) && std::isfinite(o.v_var));
    if (!(std::isfinite(o.v) && std::isfinite(o.s))) break;
  }
  CHECK(est.diagnostics().invalid_wheel > 0);
  CHECK(est.diagnostics().rejected_stamps > 0);
}

TEST_CASE("traction slip on one bogie (+30 % for 3 s): flagged, speed error bounded") {
  Config cfg;
  TractionModel model;
  Truth tr;
  auto slipFront = [](int sensor, double t, double v) {
    const double k = (sensor == 0 && t > 10.0 && t < 13.0) ? 1.3 : 1.0;
    return v * kKmh * k;
  };
  const RunResult r = run(makeRun(7000.0, 60.0, tr, slipFront), cfg, model);
  CHECK(maxSpeedError(r, tr, 7010.0, 7014.0) < 0.3);
  bool flagged = false;
  for (const Output& o : r.outs)
    if (toSec(o.stamp) > 7010.5 && toSec(o.stamp) < 7013.0 && (o.flags & kFlagFrontSlip)) flagged = true;
  CHECK(flagged);
}

TEST_CASE("joint slip on both bogies (+25 % for 2 s): model bridges, error bounded") {
  Config cfg;
  TractionModel model;
  Truth tr;
  auto slipBoth = [](int, double t, double v) {
    const double k = (t > 12.0 && t < 14.0) ? 1.25 : 1.0;
    return v * kKmh * k;
  };
  const RunResult r = run(makeRun(8000.0, 60.0, tr, slipBoth), cfg, model);
  CHECK(maxSpeedError(r, tr, 8012.0, 8016.0) < 1.0);
  CHECK(maxSpeedError(r, tr, 8020.0, 8060.0) < 0.2);  // recovered afterwards
}

TEST_CASE("wheel lock-up to zero while braking (1.5 s): no false stop") {
  Config cfg;
  TractionModel model;
  Truth tr;
  auto lock = [](int, double t, double v) { return (t > 43.0 && t < 44.5) ? 0.0 : v * kKmh; };
  const RunResult r = run(makeRun(9000.0, 60.0, tr, lock), cfg, model);
  CHECK(maxSpeedError(r, tr, 9043.0, 9046.0) < 1.0);
}

TEST_CASE("unmodelled emergency brake (-2.8 m/s^2, notch 0): wheels trusted, flagged") {
  Config cfg;
  TractionModel model;
  Truth tr;
  auto brake = [](double t) { return (t > 30.0 && t < 33.0) ? -2.8 : 0.0; };
  const RunResult r = run(makeRun(9800.0, 60.0, tr, cleanWheels, 3.0, +brake), cfg, model);
  // onset needs ~2 samples to switch to the maneuver mode: <= 0.2 s x 2.8 m/s^2
  CHECK(maxSpeedError(r, tr, 9829.0, 9836.0) < 0.5);
  bool flagged = false;
  for (const Output& o : r.outs)
    if (toSec(o.stamp) > 9830.5 && toSec(o.stamp) < 9833.0 && (o.flags & kFlagUnmodeledAccel)) flagged = true;
  CHECK(flagged);
}

TEST_CASE("dropout of both bogies (4 s) bridged by the model, then recovers") {
  Config cfg;
  TractionModel model;
  Truth tr;
  auto drop = [](int, double t, double v) { return (t > 25.0 && t < 29.0) ? NAN : v * kKmh; };
  const RunResult r = run(makeRun(9500.0, 60.0, tr, drop), cfg, model);
  CHECK(maxSpeedError(r, tr, 9525.0, 9529.0) < 0.6);
  CHECK(maxSpeedError(r, tr, 9531.0, 9560.0) < 0.15);
  bool dropout_flag = false;
  for (const Output& o : r.outs)
    if (toSec(o.stamp) > 9026.0 + 500.0 && toSec(o.stamp) < 9028.5 + 500.0 &&
        (o.flags & kFlagFrontDropout) && (o.flags & kFlagRearDropout))
      dropout_flag = true;
  CHECK(dropout_flag);
}

TEST_CASE("no-GNSS proof: GNSS after the init window changes nothing (bit-identical)") {
  Config cfg;
  TractionModel model;
  TrackMap map;
  map.setPoints({{-500, 0, 0, 0}, {5000, 0, 0, 0}}, false, {55.81, 37.462, 168.4});
  Truth tr;
  // Run A streams GNSS for the whole run; run B only for the first 6 s (window is 5 s
  // from the first fix), i.e. B is what the jury's test bags look like.
  const auto with_all = makeRun(4000.0, 50.0, tr, cleanWheels, 1e9);
  const auto init_only = makeRun(4000.0, 50.0, tr, cleanWheels, 6.0);
  const RunResult a = run(with_all, cfg, model, &map), b = run(init_only, cfg, model, &map);
  bool same = a.outs.size() == b.outs.size();
  for (size_t i = 0; same && i < a.outs.size(); ++i)
    same = a.outs[i].x == b.outs[i].x && a.outs[i].y == b.outs[i].y && a.outs[i].v == b.outs[i].v;
  CHECK(same);
  CHECK(!a.outs.empty() && a.outs.back().map_matched);
}

TEST_CASE("distance never runs backwards without landmarks, even with an unmodelled grade") {
  // Regression: the wheel-update gain of s leaked through the wheel-scale channel (P_sk * v) and
  // moved s backwards by metres per sample when the model missed an acceleration (no map grade).
  // 4 km at 10 +- 2 m/s (30 s period) under a constant notch: the model misses every acceleration.
  // The joint slip monitor is off here: its roll-back moves s back by design; this test isolates the leak.
  Config cfg;
  cfg.p.cusum_h = 1e6;
  TractionModel model;
  std::vector<In> ev;
  const double t0 = 20000.0, dur = 400.0;
  auto vAt = [](double t) { return t < 10.0 ? t : 10.0 + 2.0 * std::sin(2.0 * 3.14159265 * (t - 10.0) / 30.0); };
  double s_true = 0.0;
  for (double t = 0.0; t < dur; t += 0.01) s_true += vAt(t + 0.005) * 0.01;
  for (int k = 0; k * 0.05 <= dur; ++k) {
    const double t = k * 0.05 + 0.016;
    ev.push_back({2, fromSec(t0 + t + 0.001), fromSec(t0 + t), 3.0, 0, 0});
  }
  for (int k = 0; k * 0.1 <= dur; ++k) {
    const double t = k * 0.1 + 0.03;
    for (int sensor = 0; sensor < 2; ++sensor)
      ev.push_back({sensor, fromSec(t0 + t + 0.046), fromSec(t0 + t), vAt(t) * kKmh, 0, 0});
  }
  sortByArrival(ev);
  const RunResult r = run(ev, cfg, model);
  double worst = 0.0;
  for (size_t i = 1; i < r.outs.size(); ++i) worst = std::min(worst, r.outs[i].s - r.outs[i - 1].s);
  CHECK(worst > -0.01);
  CHECK_NEAR(r.outs.back().s, s_true, 0.02 * s_true);  // leak: -3.0 %; now -1.4 % (model misses +-0.42 m/s^2)
}

TEST_CASE("GNSS-free localisation: stops at known places fix the place on a closed track") {
  // 2 km ring, six stop places with irregular spacing; the run starts at an unknown place (s = 500 m)
  std::vector<MapPose> ring;
  const double R = 2000.0 / (2.0 * 3.14159265358979);
  for (int i = 0; i < 400; ++i) {
    const double a = 2.0 * 3.14159265358979 * i / 400.0;
    ring.push_back({R * std::cos(a), R * std::sin(a), 0.0, 0.0});
  }
  TrackMap map;
  map.setPoints(ring, true, {55.81, 37.462, 168.4});
  const double L = map.length();
  std::vector<Landmark> stops;
  for (double s : {100.0, 350.0, 720.0, 1100.0, 1400.0, 1800.0}) stops.push_back({s * L / 2000.0, 0.3, 0.9});
  GlobalLocalizer gl;
  GlobalLocalizer::Params gp;
  CHECK(gl.init(map, stops, {}, TrackField{}, gp));
  const double s0 = 500.0 * L / 2000.0;
  double t = 0.0, r = 0.0;
  size_t next = 2;  // first stop place ahead of s0
  for (int lap_stop = 0; lap_stop < 8 && !gl.fixed(); ++lap_stop) {
    const double target = stops[next % stops.size()].s + (next >= stops.size() ? L : 0.0);
    while (s0 + r < target - 1e-6) {  // cruise at 10 m/s
      const double step = std::min(1.0, target - (s0 + r));
      r += step;
      t += step / 10.0;
      gl.step(t, r, 10.0, false, 0);
    }
    for (int k = 0; k < 40; ++k) {  // 4 s standstill
      t += 0.1;
      gl.step(t, r, 0.0, true, 0);
    }
    ++next;
  }
  CHECK(gl.fixed());
  if (gl.fixed()) {
    double err = std::fmod(gl.fix().s - (s0 + gl.lastOdometer()), L);
    if (err > 0.5 * L) err -= L;
    if (err < -0.5 * L) err += L;
    CHECK(std::abs(err) < 2.0);
  }
}

TEST_CASE("dead-end stub: a stop at its far part puts the tram on it, a stop near the switch does not") {
  Config cfg;
  cfg.output_frame = "map";
  TractionModel model;
  TrackMap map;
  map.setPoints({{-500, 0, 0, 0}, {5000, 0, 0, 0}}, false, {55.81, 37.462, 168.4});  // map s = x + 500
  Truth tr;
  // the synthetic brake leaves the tram creeping at ~0.3 m/s; an extra 0.4 m/s^2 from 49 s stops it
  const auto ev = makeRun(9300.0, 60.0, tr, cleanWheels, 3.0, [](double t) { return t > 49.0 ? -0.4 : 0.0; });
  const double s_stop = 500.0 + tr.s.back();  // antenna 1 on the main line at the final stop
  const std::string path = "tbo_test_stub.csv";
  auto stubAt = [&](double past) {  // stub leaving the main line `past` metres before the stop, 30 deg left
    const double join = s_stop - past;
    std::FILE* fh = std::fopen(path.c_str(), "wb");
    std::fprintf(fh, "# stub\n# origin_lat=55.81 origin_lon=37.462 origin_h=168.4 cyclic=0 join_s=%.3f\ns,x,y,z\n", join);
    for (int i = 0; i <= 125; ++i) std::fprintf(fh, "%d,%.4f,%.4f,0\n", i, join - 500.0 + i * 0.8660254, i * 0.5);
    std::fclose(fh);
    TrackMap stub;
    std::string err;
    CHECK(stub.loadCsv(path, &err));
    return stub;
  };
  const TrackMap far = stubAt(110.0), near = stubAt(50.0);
  std::remove(path.c_str());
  const RunResult a = run(ev, cfg, model, &map, &far), b = run(ev, cfg, model, &map, &near);
  CHECK(!a.outs.empty() && !b.outs.empty());
  if (a.outs.empty() || b.outs.empty()) return;
  // on the stub after the stop: base_link ~120 m along it, i.e. ~60 m off the main line
  CHECK(a.outs.back().y > 50.0 && a.outs.back().y < 70.0);
  CHECK(a.outs.back().s_map < 0.0);
  // a stop 50 m past the switch is a main-line stop (the recordings end there)
  CHECK(std::abs(b.outs.back().y) < 1.0);
  // before the stop both runs are on the main line
  for (const Output& o : a.outs)
    if (toSec(o.stamp) < 9300.0 + 40.0) CHECK(std::abs(o.y) < 1.0);
}

TEST_CASE("wheel scale from the speed quantum: taken from the matching epoch, not from a later one") {
  Config cfg;
  cfg.output_frame = "map";
  TractionModel model;
  const TrackMap map = straightMap();
  Truth tr;
  const double k_x = 0.01;  // readings 1 % high on top of the test calibration
  auto ev = makeRun(9300.0, 60.0, tr, [&](int, double, double v) { return v * kKmh * (1.0 + k_x); });
  const double q = 0.00504;  // the sensor reports whole steps of q km/h
  for (In& e : ev)
    if (e.type == 0 || e.type == 1) e.a = std::round(e.a / q) * q;
  const double k_eff = (1.0 + k_x) * kKmh * cfg.p.wheel_kmh_to_ms - 1.0;  // the filter's k of these readings
  const double c = q / (1.0 + k_eff);
  CHECK(dateMsk(fromSec(9300.0)) == 19700101);
  const std::vector<WheelEpoch> right = {{1, 19700101, c, c}, {2, 19700101, c * 1.05, c * 1.05}};
  const std::vector<WheelEpoch> later = {{1, 19700102, c, c}};  // an epoch after the run date: not used
  const RunResult a = run(ev, cfg, model, &map, nullptr, &right);
  const RunResult b = run(ev, cfg, model, &map, nullptr, &later);
  CHECK(!a.outs.empty() && !b.outs.empty());
  if (a.outs.empty() || b.outs.empty()) return;
  const Output& oa = a.outs.back();
  const Output& ob = b.outs.back();
  CHECK(std::abs(oa.scale - k_eff) < 0.001);  // no landmark here: only the quantum can tell k
  CHECK(std::abs(ob.scale) < 0.001);           // prior k = 0 kept
  // distance from standstill at 4 s to the final stop
  auto travelled = [&](const RunResult& r) {
    const Output* o0 = nullptr;
    for (const Output& o : r.outs)
      if (toSec(o.stamp) >= 9304.0) {
        o0 = &o;
        break;
      }
    return o0 ? r.outs.back().s - o0->s : 0.0;
  };
  const double s_true = tr.sAt(toSec(a.outs.back().stamp)) - tr.sAt(9304.0);
  CHECK(s_true > 200.0);
  CHECK(std::abs(travelled(a) - s_true) < 0.5);
  CHECK(std::abs(travelled(b) - s_true) > 2.0);  // ~0.9 % of the run without it
}

TEST_CASE("track field: linear interpolation and cyclic wrap (learned d(s), adhesion map)") {
  const std::string path = "tbo_test_field.csv";
  {
    std::FILE* fh = std::fopen(path.c_str(), "wb");
    CHECK(fh != nullptr);
    if (!fh) return;
    std::fprintf(fh, "# test field\ns,d,runs\n5,0.1,3\n15,0.3,3\n95,-0.1,3\n");
    std::fclose(fh);
  }
  TrackField f;
  std::string err;
  CHECK(f.loadCsv(path, 100.0, &err));
  CHECK_NEAR(f.at(10.0), 0.2, 1e-12);
  CHECK_NEAR(f.at(15.0), 0.3, 1e-12);
  CHECK_NEAR(f.at(100.0), 0.0, 1e-12);   // wrap segment 95 -> 105 (= 5): -0.1 .. 0.1
  CHECK_NEAR(f.at(-5.0), -0.1, 1e-12);   // negative s wraps too (-5 == 95)
  CHECK_NEAR(f.at(-10.0), -0.075, 1e-12); // 90: between 15 (0.3) and 95 (-0.1)
  CHECK_NEAR(f.at(210.0), 0.2, 1e-12);   // two laps later
  TrackField open;
  CHECK(open.loadCsv(path, 0.0, &err));
  CHECK_NEAR(open.at(1000.0), -0.1, 1e-12);  // open track: clamp
  std::remove(path.c_str());
}

// ---------------------------------------------------------------- time base and GNSS window

TEST_CASE("all inputs silent for 3 s and 8 s: no lock-up, the model bridges, then recovers") {
  Config cfg;
  TractionModel model;
  Truth tr;
  const auto ev = makeRun(9600.0, 60.0, tr, cleanWheels);
  for (const double gap : {3.0, 8.0}) {
    const RunResult r = run(cutAll(ev, 9625.0, 9625.0 + gap), cfg, model);
    size_t after = 0;
    for (const Output& o : r.outs)
      if (toSec(o.stamp) > 9636.0) ++after;
    CHECK(after > 400);                                        // outputs continue after the gap
    CHECK(maxSpeedError(r, tr, 9636.0, 9660.0) < 0.15);       // and track the wheels again
    CHECK(r.diag.resets == 0);
    CHECK(gap < 5.0 || r.diag.time_gaps == 1);                 // > max_future_s: confirmed jump
  }
}

TEST_CASE("lone garbage stamps (+-1000 s) are dropped: no reset, no output, bit-identical") {
  Config cfg;
  TractionModel model;
  Truth tr;
  const auto ev = makeRun(9700.0, 60.0, tr, cleanWheels);
  auto bad = ev;
  bad.push_back({0, fromSec(9730.0), fromSec(10730.0), 30.0, 0, 0});  // wheel far ahead
  bad.push_back({2, fromSec(9740.0), fromSec(8740.0), 5.0, 0, 0});    // controller far behind
  bad.push_back({1, fromSec(9745.0), fromSec(9745.0 - 20.0), 0.0, 0, 0});  // bogie 20 s old
  sortByArrival(bad);
  const RunResult a = run(ev, cfg, model), b = run(bad, cfg, model);
  CHECK(sameOutputs(a, b));
  CHECK(b.diag.resets == 0);
  CHECK(b.diag.rejected_stamps == a.diag.rejected_stamps + 3);
}

TEST_CASE("bag replayed from the start: new run; another bag 100 s later: new run") {
  Config cfg;
  TractionModel model;
  Truth tr, tr2;
  const auto first = makeRun(9900.0, 40.0, tr, cleanWheels);
  auto loop = first;
  for (In e : first) {  // the same bag once more, arriving after the first pass
    e.recv += fromSec(45.0);
    loop.push_back(e);
  }
  sortByArrival(loop);
  const RunResult r = run(loop, cfg, model);
  CHECK(r.diag.resets == 1);
  // the second pass publishes the same stamps again and tracks the truth
  size_t second = 0;
  double m = 0.0;
  for (size_t i = r.outs.size() / 2; i < r.outs.size(); ++i) {
    const double t = toSec(r.outs[i].stamp);
    if (t > 9905.0 && t < 9938.0) {
      ++second;
      m = std::max(m, std::abs(r.outs[i].v - tr.vAt(t)));
    }
  }
  CHECK(second > 300);
  CHECK(m < 0.3);
  auto two = first;
  const auto next = makeRun(9900.0 + 140.0, 40.0, tr2, cleanWheels);
  two.insert(two.end(), next.begin(), next.end());
  sortByArrival(two);
  const RunResult r2 = run(two, cfg, model);
  CHECK(r2.diag.resets == 1);
  CHECK(maxSpeedError(r2, tr2, 10045.0, 10078.0) < 0.3);
}

TEST_CASE("GNSS init window counts from the first fix (GNSS starts 3 s after the wheels)") {
  Config cfg;
  cfg.p.gnss_init_window_s = 1.0;
  TractionModel model;
  const TrackMap map = straightMap();
  Truth tr;
  auto ev = makeRun(9950.0, 30.0, tr, cleanWheels, -1.0);  // no GNSS from makeRun
  addMovingGnss(ev, tr, 9950.0, 3.0, 5.0);
  const RunResult r = run(ev, cfg, model, &map);
  CHECK(!r.outs.empty() && r.outs.back().map_matched);
  CHECK(r.diag.gnss_ignored_after_window > 0);  // fixes after 4 s are not used
}

TEST_CASE("anchor while moving inside a 10 s GNSS window: position follows the truth") {
  Config cfg;
  cfg.p.speed_output_delay_s = 0.0;  // judge-reference alignment is not what this test checks
  cfg.p.position_output_delay_s = 0.0;
  cfg.output_frame = "map";
  cfg.p.gnss_init_window_s = 10.0;
  cfg.p.position_lead_s = 0.0;
  cfg.p.base_link_along_m = 0.0;
  cfg.p.base_link_height_m = 0.0;
  TractionModel model;
  const TrackMap map = straightMap();
  Truth tr;
  auto ev = makeRun(9980.0, 40.0, tr, cleanWheels, -1.0);
  addMovingGnss(ev, tr, 9980.0, 0.0, 10.0);  // tram starts at 5 s: last fix ~11 m further
  const RunResult r = run(ev, cfg, model, &map);
  double m = 0.0;
  for (const Output& o : r.outs) {
    const double t = toSec(o.stamp);
    if (t > 9992.0 && t < 10000.0 && o.pos_valid) m = std::max(m, std::abs(o.x - tr.sAt(t)));
  }
  CHECK(m < 1.0);
}

#ifdef TBO_PARAMS_YAML
TEST_CASE("params.yaml: every key is a known parameter and every parameter is present") {
  Config cfg;
  std::string err, unknown;
  CHECK(loadFlatYaml(TBO_PARAMS_YAML, cfg, &err, &unknown));
  CHECK(unknown.empty());
  if (!unknown.empty()) std::printf("  unknown keys: %s\n", unknown.c_str());
  // every registry entry must appear in the file
  int n = 0;
  const ParamInfo* reg = paramRegistry(&n);
  std::FILE* fh = std::fopen(TBO_PARAMS_YAML, "rb");
  CHECK(fh != nullptr);
  std::string text;
  if (fh) {
    char b[4096];
    size_t r;
    while ((r = std::fread(b, 1, sizeof(b), fh)) > 0) text.append(b, r);
    std::fclose(fh);
  }
  for (int i = 0; i < n; ++i) {
    const bool present = text.find(std::string(reg[i].name) + ":") != std::string::npos;
    if (!present) std::printf("  missing in params.yaml: %s\n", reg[i].name);
    CHECK(present);
  }
}
#endif

TEST_CASE("scheduler: grid stamps hit exact multiples, stamps strictly increase") {
  Params p;
  OutputScheduler s(p);
  Stamp out[64];
  std::vector<Stamp> all;
  for (int k = 0; k < 100; ++k) {
    const Stamp t = fromSec(10.0 + k * 0.05 + 0.016);
    const int n = s.onInput(true, t, t, out, 64);
    for (int i = 0; i < n; ++i) all.push_back(out[i]);
  }
  bool inc = true;
  int grid_hits = 0;
  for (size_t i = 0; i < all.size(); ++i) {
    if (i > 0 && all[i] <= all[i - 1]) inc = false;
    if (all[i] % fromSec(0.05) == 0) ++grid_hits;
  }
  CHECK(inc);
  CHECK(grid_hits >= 99);
}
