// Core behaviour tests: invariants that need no ground truth, robustness to garbage
// input, slip/dropout handling on a synthetic tram, and the "no GNSS after the init
// window" proof (outputs must be bit-identical with and without later GNSS).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mini_test.hpp"
#include "tbo/estimator.hpp"
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
};

RunResult run(const std::vector<In>& ev, const Config& cfg, const TractionModel& model,
              const TrackMap* map = nullptr) {
  Estimator est(cfg, model, map);
  OutputScheduler sched(cfg.p);
  RunResult r;
  Stamp buf[64];
  for (const In& e : ev) {
    bool is_input = true, is_cmd = false;
    switch (e.type) {
      case 0: est.onWheel(Sensor::Front, e.stamp, e.a); break;
      case 1: est.onWheel(Sensor::Rear, e.stamp, e.a); break;
      case 2: est.onCmd(e.stamp, static_cast<int>(e.a)); is_cmd = true; break;
      case 3: est.onGnssFix(GnssSource::Master, e.stamp, e.a, e.b, e.c, 2); is_input = false; break;
      case 4: est.onGnssFix(GnssSource::Rover, e.stamp, e.a, e.b, e.c, 2); is_input = false; break;
    }
    if (!is_input || !est.started()) continue;
    const int n = sched.onInput(is_cmd, e.stamp, est.latestStamp(), buf, 64);
    for (int k = 0; k < n; ++k) r.outs.push_back(est.query(buf[k]));
  }
  return r;
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
  // from the first message), i.e. B is what the jury's test bags look like.
  const auto with_all = makeRun(4000.0, 50.0, tr, cleanWheels, 1e9);
  const auto init_only = makeRun(4000.0, 50.0, tr, cleanWheels, 6.0);
  const RunResult a = run(with_all, cfg, model, &map), b = run(init_only, cfg, model, &map);
  bool same = a.outs.size() == b.outs.size();
  for (size_t i = 0; same && i < a.outs.size(); ++i)
    same = a.outs[i].x == b.outs[i].x && a.outs[i].y == b.outs[i].y && a.outs[i].v == b.outs[i].v;
  CHECK(same);
  CHECK(!a.outs.empty() && a.outs.back().map_matched);
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
