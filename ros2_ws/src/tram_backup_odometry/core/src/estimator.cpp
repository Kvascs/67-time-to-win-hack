#include "tbo/estimator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace tbo {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kLog2Pi = 1.8378770664093453;
constexpr double kImplausiblePenalty = -4.0;  // log-likelihood penalty for trusting a jump

// Which bogie each IMM mode trusts: {front, rear}.
constexpr bool kTrust[kNumModes][2] = {
    {true, true}, {false, true}, {true, false}, {false, false}, {true, true}};

double wrapAngle(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a < -kPi) a += 2.0 * kPi;
  return a;
}

double medianOf(std::vector<double> v) {
  if (v.empty()) return 0.0;
  const size_t m = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<long>(m), v.end());
  double hi = v[m];
  if (v.size() % 2 == 1) return hi;
  const double lo = *std::max_element(v.begin(), v.begin() + static_cast<long>(m));
  return 0.5 * (lo + hi);
}

// One EKF measurement update with M bogie speeds; returns the Gaussian log-likelihood.
template <int M>
double kfUpdate(StateVec& x, StateCov& P, const double* z, const int* idx, const Params& p) {
  Mat<M, kNx> H;
  Vec<M> nu;
  Mat<M, M> R;
  const double v = x(kV, 0), k = x(kK, 0);
  for (int r = 0; r < M; ++r) {
    H(r, kV) = 1.0 + k;
    H(r, kK) = v;
    nu(r, 0) = z[idx[r]] - (1.0 + k) * v;
    const double rel = p.sigma_wheel_rel * v;
    R(r, r) = p.sigma_wheel * p.sigma_wheel + rel * rel;
  }
  const Mat<kNx, M> PHt = P * transpose(H);
  const Mat<M, M> S = H * PHt + R;
  Mat<M, M> Si;
  double det = 0.0;
  if (!invert(S, Si, det)) return -1e6;
  Mat<kNx, M> K = PHt * Si;
  // Schmidt ("consider") state: the wheel scale k is not observable from wheels + model
  // (only (1+k)*g is), so bogie updates must not move it; stop landmarks calibrate it.
  for (int r = 0; r < M; ++r) K(kK, r) = 0.0;
  x += K * nu;
  const StateCov IKH = StateCov::identity() - K * H;
  P = IKH * P * transpose(IKH) + K * R * transpose(K);  // Joseph form
  symmetrize(P);
  const double maha = (transpose(nu) * Si * nu)(0, 0);
  return -0.5 * maha - 0.5 * std::log(det) - 0.5 * M * kLog2Pi;
}

void clampState(StateVec& x, StateCov& P, const Params& p, double d_min, double d_max) {
  if (!(x(kV, 0) >= 0.0)) x(kV, 0) = 0.0;
  x(kG, 0) = std::clamp(x(kG, 0), p.gain_min, p.gain_max);
  x(kD, 0) = std::clamp(x(kD, 0), d_min, d_max);
  x(kK, 0) = std::clamp(x(kK, 0), -0.05, 0.05);
  conditionCovariance(P, 1e-12);
}

void pinVelocity(StateVec& x, StateCov& P, double v, double var) {
  x(kV, 0) = v;
  for (int i = 0; i < kNx; ++i) {
    P(kV, i) = 0.0;
    P(i, kV) = 0.0;
  }
  P(kV, kV) = var;
}

}  // namespace

// --------------------------------------------------------------------------------------

bool loadLandmarks(const std::string& path, std::vector<Landmark>& out, std::string* err) {
  std::ifstream in(path);
  if (!in) {
    if (err) *err = "cannot open landmarks " + path;
    return false;
  }
  out.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#' || line[0] == 's') continue;
    for (char& c : line)
      if (c == ',') c = ' ';
    std::istringstream ss(line);
    Landmark l;
    if (ss >> l.s >> l.sigma >> l.p_stop) out.push_back(l);
  }
  std::sort(out.begin(), out.end(), [](const Landmark& a, const Landmark& b) { return a.s < b.s; });
  return true;
}

Estimator::Estimator(const Config& cfg, const TractionModel& model, const TrackMap* map,
                     std::vector<const TrackMap*> branches)
    : cfg_(cfg), p_(cfg_.p), model_(model), map_(map && !map->empty() ? map : nullptr) {
  for (const TrackMap* b : branches)
    if (map_ && b && !b->empty() && b->hasJoin()) branches_.push_back(b);
  buf_.reserve(256);
  if (map_) map_lc_.reset(map_->origin());
  reset();
}

void Estimator::reset() {
  committed_ = FilterState{};
  buf_.clear();
  started_ = false;
  latest_ = 0;
  init_ = InitState{};
  init_.fixes.reserve(512);
  init_.other.reserve(512);
  frame_ready_ = false;
}

bool Estimator::acceptStamp(Stamp stamp) {
  if (stamp <= 0) {
    ++diag_.rejected_stamps;
    return false;
  }
  if (started_) {
    if (stamp > latest_ + fromSec(p_.max_future_s)) {
      ++diag_.rejected_stamps;
      return false;
    }
    if (stamp < latest_ - fromSec(p_.max_backjump_s)) {  // time base restarted (bag loop)
      ++diag_.resets;
      reset();
    }
  }
  return true;
}

void Estimator::onWheel(Sensor sensor, Stamp stamp, double speed_kmh) {
  ++diag_.wheel_msgs;
  if (!acceptStamp(stamp)) return;
  Event e;
  e.t = stamp - fromSec(p_.wheel_delay_s);
  e.seq = seq_++;
  const int i = sensor == Sensor::Front ? 0 : 1;
  e.has[i] = true;
  if (!std::isfinite(speed_kmh) || speed_kmh < -0.5 || speed_kmh > p_.wheel_max_kmh) {
    ++diag_.invalid_wheel;
    e.z[i] = std::numeric_limits<double>::quiet_NaN();
  } else {
    e.z[i] = std::max(0.0, speed_kmh) * p_.wheel_kmh_to_ms;
  }
  if (!init_.have_first) {
    init_.have_first = true;
    init_.t_first = stamp;
  }
  if (!started_) {
    started_ = true;
    latest_ = stamp;
    startFilter(committed_, e.t - fromSec(p_.lag_window_s));
  }
  latest_ = std::max(latest_, stamp);
  insert(e);
}

void Estimator::onCmd(Stamp stamp, int notch) {
  ++diag_.cmd_msgs;
  if (!acceptStamp(stamp)) return;
  if (notch < TractionModel::kNotchMin || notch > TractionModel::kNotchMax) {
    ++diag_.invalid_cmd;
    return;
  }
  Event e;
  e.is_cmd = true;
  e.notch = notch;
  e.t = stamp + fromSec(p_.cmd_delay_s);
  e.seq = seq_++;
  if (!init_.have_first) {
    init_.have_first = true;
    init_.t_first = stamp;
  }
  if (!started_) {
    started_ = true;
    latest_ = stamp;
    startFilter(committed_, stamp - fromSec(p_.lag_window_s));
  }
  latest_ = std::max(latest_, stamp);
  insert(e);
}

void Estimator::insert(const Event& e) {
  if (e.t < committed_.t) {  // older than the fixed-lag window
    if (e.is_cmd) {
      if (!committed_.have_cmd || e.t >= committed_.last_cmd) {
        committed_.notch = e.notch;
        committed_.have_cmd = true;
        committed_.last_cmd = committed_.t;
      }
    } else {
      ++diag_.late_dropped;
      committed_.late_t = committed_.t;
    }
    return;
  }
  if (!e.is_cmd) {  // merge front/rear samples that share a stamp into one measurement
    for (Event& b : buf_) {
      if (b.is_cmd || b.t != e.t) continue;
      bool clash = false;
      for (int i = 0; i < 2; ++i)
        if (e.has[i] && b.has[i]) clash = true;
      if (clash) continue;
      for (int i = 0; i < 2; ++i)
        if (e.has[i]) {
          b.has[i] = true;
          b.z[i] = e.z[i];
        }
      ++diag_.merged_pairs;
      commitOlderThan(latest_ - fromSec(p_.lag_window_s));
      return;
    }
  }
  auto pos = std::upper_bound(buf_.begin(), buf_.end(), e,
                              [](const Event& a, const Event& b) { return a.t < b.t; });
  buf_.insert(pos, e);
  diag_.max_buffer = std::max(diag_.max_buffer, static_cast<int>(buf_.size()));
  commitOlderThan(latest_ - fromSec(p_.lag_window_s));
  while (buf_.size() > 200) {  // hard bound on work per query
    applyEvent(committed_, buf_.front());
    buf_.erase(buf_.begin());
  }
}

void Estimator::commitOlderThan(Stamp t) {
  size_t n = 0;
  while (n < buf_.size() && buf_[n].t < t) {
    applyEvent(committed_, buf_[n]);
    ++n;
  }
  if (n > 0) buf_.erase(buf_.begin(), buf_.begin() + static_cast<long>(n));
}

void Estimator::startFilter(FilterState& f, Stamp t) const {
  f = FilterState{};
  f.started = true;
  f.t = t;
  f.t_mix = t;
  StateVec x0;
  x0(kG, 0) = 1.0;
  StateCov P0;
  P0(kS, kS) = 0.0;
  P0(kV, kV) = p_.init_sigma_v * p_.init_sigma_v;
  P0(kD, kD) = p_.init_sigma_d * p_.init_sigma_d;
  P0(kK, kK) = p_.init_sigma_scale * p_.init_sigma_scale;
  P0(kG, kG) = p_.init_sigma_gain * p_.init_sigma_gain;
  for (int j = 0; j < kNumModes; ++j) {
    f.x[j] = x0;
    f.P[j] = P0;
  }
  for (int j = 0; j < kNumModes; ++j) f.mu[j] = 1e-3;
  f.mu[kModeNominal] = 1.0 - 1e-3 * (kNumModes - 1);
}

double Estimator::combinedV(const FilterState& f) const {
  double v = 0.0;
  for (int j = 0; j < kNumModes; ++j) v += f.mu[j] * f.x[j](kV, 0);
  return std::max(0.0, v);
}

void Estimator::applyEvent(FilterState& f, const Event& e) const {
  if (!f.started) startFilter(f, e.t);
  advance(f, e.t);
  if (e.is_cmd) {
    f.notch = e.notch;
    f.have_cmd = true;
    f.last_cmd = e.t;
    return;
  }
  wheelUpdate(f, e);
}

void Estimator::advance(FilterState& f, Stamp t) const {
  if (!f.started || t <= f.t) return;
  const double total = toSec(t - f.t);
  int steps = static_cast<int>(std::ceil(total / p_.max_step_s - 1e-9));
  steps = std::clamp(steps, 1, 4000);
  const double h = total / steps;
  const Stamp cmd_timeout = fromSec(p_.cmd_timeout_s);
  const double alpha = p_.drive_tau_s > 1e-3 ? 1.0 - std::exp(-h / p_.drive_tau_s) : 1.0;
  for (int n = 0; n < steps; ++n) {
    const Stamp t_step = f.t + fromSec((n + 1) * h);
    int notch = f.notch;
    if (!f.have_cmd || t_step - f.last_cmd > cmd_timeout) notch = 0;  // silent controller -> neutral
    const double vc = combinedV(f);
    double at = model_.target(notch, vc);
    if (at < 0.0 && (f.standstill || vc < 0.05)) at = 0.0;  // brakes cannot push backwards
    if (f.standstill && notch <= 0) at = 0.0;
    f.a_target = at;
    f.a_drive += alpha * (at - f.a_drive);
    const double a_ext = trackAccel(f);
    for (int j = 0; j < kNumModes; ++j)
      predictMode(f.x[j], f.P[j], f.a_drive, a_ext, h, f.standstill,
                  j == kModeManeuver ? p_.sigma_accel_maneuver : p_.sigma_accel,
                  j == kModeManeuver ? p_.q_disturbance_maneuver : p_.q_disturbance);
  }
  f.t = t;
}

bool Estimator::routeAt(double s_rel, const TrackMap*& m, double& s) const {
  if (!map_ || !init_.map_matched) return false;
  if (init_.prefix) {
    if (s_rel < init_.prefix_len) {
      m = init_.prefix;
      s = init_.prefix_s0 + s_rel;
      return true;
    }
    m = map_;
    s = map_->wrap(init_.prefix->joinS() + (s_rel - init_.prefix_len));
    return true;
  }
  m = map_;
  s = map_->wrap(init_.s_offset + s_rel);
  return true;
}

double Estimator::trackAccel(const FilterState& f) const {
  double s = 0.0, v = 0.0;
  for (int j = 0; j < kNumModes; ++j) {
    s += f.mu[j] * f.x[j](kS, 0);
    v += f.mu[j] * f.x[j](kV, 0);
  }
  const TrackMap* m = nullptr;
  double sm = 0.0;
  if (!routeAt(s, m, sm) || !m->hasProfile()) return 0.0;
  const double kg = f.notch < 0 ? p_.kg_brake : (f.notch > 0 ? p_.kg_traction : p_.kg_coast);
  double a = -kg * p_.map_grade_gain * m->gradeAt(sm);
  if (p_.curve_resist_coef > 0.0 && v > 0.1) a -= p_.curve_resist_coef * std::abs(m->curvatureAt(sm));
  return a;
}

void Estimator::predictMode(StateVec& x, StateCov& P, double a, double a_ext, double h,
                            bool standstill, double sigma_accel, double q_d) const {
  if (standstill) {  // zero-velocity: position and speed frozen, parameters diffuse
    x(kV, 0) = 0.0;
    P(kD, kD) += p_.q_disturbance * h;
    P(kK, kK) += p_.q_scale * h;
    P(kG, kG) += p_.q_gain * h;
    return;
  }
  const double v0 = x(kV, 0), d = x(kD, 0), g = x(kG, 0);
  const double acc = g * a + d + a_ext;
  double v1 = v0 + acc * h;
  double ds;
  if (v1 < 0.0) {  // comes to rest inside the step
    const double tstop = acc < 0.0 ? v0 / -acc : 0.0;
    ds = 0.5 * v0 * tstop;
    v1 = 0.0;
  } else {
    ds = v0 * h + 0.5 * acc * h * h;
  }
  x(kS, 0) += ds;
  x(kV, 0) = v1;

  StateCov F = StateCov::identity();
  F(kS, kV) = h;
  F(kS, kD) = 0.5 * h * h;
  F(kS, kG) = 0.5 * a * h * h;
  F(kV, kD) = h;
  F(kV, kG) = a * h;
  P = F * P * transpose(F);
  const double qa = sigma_accel * sigma_accel;
  P(kS, kS) += qa * h * h * h / 3.0;
  P(kS, kV) += qa * h * h / 2.0;
  P(kV, kS) += qa * h * h / 2.0;
  P(kV, kV) += qa * h;
  P(kD, kD) += q_d * h;
  P(kK, kK) += p_.q_scale * h;
  P(kG, kG) += p_.q_gain * h;
}

void Estimator::wheelUpdate(FilterState& f, const Event& e) const {
  const double vc = combinedV(f);
  bool avail[2] = {false, false};
  bool implausible[2] = {false, false};
  double z[2] = {0.0, 0.0};
  for (int i = 0; i < 2; ++i) {
    if (!e.has[i]) continue;
    WheelTrack& w = f.wheel[i];
    if (!std::isfinite(e.z[i])) {
      w.invalid_t = e.t;
      continue;
    }
    const double zi = e.z[i];
    w.implausible = false;
    if (w.have) {
      const double dt = toSec(e.t - w.t);
      if (dt > 1e-3 && dt < 1.0 && std::abs(zi - w.z) / dt > p_.max_wheel_accel) w.implausible = true;
    }
    // frozen reading: identical non-zero value while the vehicle speed evolves
    if (w.have && zi == w.z && zi > 0.3) {
      if (w.same_since < 0) {
        w.same_since = w.t;
        w.v_at_same = vc;
      }
      if (toSec(e.t - w.same_since) >= p_.stuck_time_s &&
          std::abs(vc - w.v_at_same) >= p_.stuck_min_change)
        w.stuck = true;
    } else {
      w.same_since = -1;
      w.stuck = false;
    }
    w.have_prev = w.have;
    w.z_prev = w.z;
    w.t_prev = w.t;
    w.have = true;
    w.z = zi;
    w.t = e.t;
    if (w.stuck) continue;
    avail[i] = true;
    implausible[i] = w.implausible;
    z[i] = zi;
  }
  // ---- a bogie reading exactly zero while the other shows motion is dead (sensor stuck at
  // zero at motion start, or a locked axle): exclude it instead of blaming the good one ----
  for (int i = 0; i < 2; ++i) {
    const int o = 1 - i;
    if (!avail[i] || z[i] > 0.05) continue;
    const WheelTrack& other = f.wheel[o];
    const bool other_moving = (avail[o] && z[o] > p_.zero_stuck_other) ||
                              (other.have && e.t - other.t <= fromSec(0.3) && other.z > p_.zero_stuck_other);
    if (other_moving) {
      avail[i] = false;
      f.wheel[i].zero_stuck_t = e.t;
    }
  }
  if (!avail[0] && !avail[1]) return;

  // ---- wheels under-read in tight curves (inner/outer rail geometry): correct via map ----
  {
    double s = 0.0;
    for (int j = 0; j < kNumModes; ++j) s += f.mu[j] * f.x[j](kS, 0);
    const TrackMap* m = nullptr;
    double sm = 0.0;
    if (routeAt(s, m, sm) && m->hasProfile()) {
      const double k = m->curvatureAt(sm);
      const double corr = 1.0 + std::min(p_.wheel_curv_abs * std::abs(k), p_.wheel_curv_sat) +
                          p_.wheel_curv_signed * k;
      for (int i = 0; i < 2; ++i) z[i] *= corr;
    }
  }

  // ---- zero-velocity (standstill) detection ----
  const double thr = p_.standstill_kmh * p_.wheel_kmh_to_ms;
  bool all_low = true;
  for (int i = 0; i < 2; ++i)
    if (avail[i] && z[i] > thr) all_low = false;
  if (all_low && vc < p_.standstill_max_v) {
    if (f.still_since < 0) f.still_since = e.t;
    if (!f.standstill && toSec(e.t - f.still_since) >= p_.standstill_time_s) {
      f.standstill = true;
      for (int j = 0; j < kNumModes; ++j) pinVelocity(f.x[j], f.P[j], 0.0, 1e-6);
      if (f.notch <= 0) f.a_drive = 0.0;
    }
  } else if (!all_low) {
    f.still_since = -1;
    if (f.standstill) {  // leaving standstill: speed uncertainty must open up again
      f.standstill = false;
      f.lm_done = false;
      for (int j = 0; j < kNumModes; ++j)
        pinVelocity(f.x[j], f.P[j], 0.0, p_.init_sigma_v * p_.init_sigma_v);
    }
  }
  if (f.standstill) {
    if (!f.lm_done && toSec(e.t - f.still_since) >= p_.landmark_dwell_s) {
      f.lm_done = true;
      landmarkUpdate(f, e.t);
    }
    const double relax = 1.0 - std::exp(-toSec(e.t - f.t_mix) * p_.rate_recover);
    double rest = 0.0;
    for (int j = 1; j < kNumModes; ++j) {
      f.mu[j] *= (1.0 - relax);
      rest += f.mu[j];
    }
    f.mu[kModeNominal] = 1.0 - rest;
    f.t_mix = e.t;
    f.bad_since = f.agree_since = -1;
    f.latch = f.onset = false;
    for (int i = 0; i < 2; ++i) f.wheel[i].cusum_pos = f.wheel[i].cusum_neg = 0.0;
    return;
  }

  // ---- joint slip/slide monitor: may roll back to the model and latch model-only ----
  if (jointMonitor(f, e, avail, z)) return;

  // ---- IMM interaction (mixing) ----
  const double dt = std::clamp(toSec(e.t - f.t_mix), 0.0, 1.0);
  f.t_mix = e.t;
  const bool effort = std::abs(f.a_target) > 0.5 || std::abs(f.notch) >= 8;
  const double boost = effort ? p_.slip_context_boost : 1.0;
  const double pb = 1.0 - std::exp(-p_.rate_to_bad * boost * dt);
  const double pbb = 1.0 - std::exp(-p_.rate_to_both_bad * boost * dt);
  const double pr = 1.0 - std::exp(-p_.rate_recover * dt);
  const double pm = 1.0 - std::exp(-p_.rate_to_maneuver * dt);
  const double pme = 1.0 - std::exp(-p_.rate_maneuver_end * dt);
  const double eps = 0.1 * pb;
  // rows: from-mode, columns: to-mode (N, F, R, B, M)
  const double T[kNumModes][kNumModes] = {
      {1.0 - 2.0 * pb - pbb - pm, pb, pb, pbb, pm},
      {pr, 1.0 - pr - pb - eps, eps, pb, 0.0},
      {pr, eps, 1.0 - pr - pb - eps, pb, 0.0},
      {0.5 * pr, 0.2 * pr, 0.2 * pr, 1.0 - pr, 0.1 * pr},
      {pme, eps, eps, pb, 1.0 - pme - 2.0 * eps - pb}};
  // GPB1: all modes share the collapsed posterior of the previous update and were predicted
  // with their own process noise (see advance()), so the priors here are mode-specific.
  // (Full IMM mixing let the nominal filter keep tracking biased wheels during a joint
  // slip and re-lock onto them; our modes differ only in trust and process noise.)
  double cbar[kNumModes];
  StateVec x0[kNumModes];
  StateCov P0[kNumModes];
  for (int j = 0; j < kNumModes; ++j) {
    cbar[j] = 0.0;
    for (int i = 0; i < kNumModes; ++i) cbar[j] += T[i][j] * f.mu[i];
    x0[j] = f.x[j];
    P0[j] = f.P[j];
  }

  // ---- mode-conditioned updates ----
  // A faulty bogie is modelled by a broad uniform density. Its sign is informative: under
  // traction a slipping wheel over-reads, under braking a sliding wheel under-reads.
  const double log_outlier = -std::log(p_.outlier_range);
  const bool cmd_ok = !(f.cmd_fault_t >= 0 && toSec(e.t - f.cmd_fault_t) < p_.cmd_fault_hold_s);
  const int effort_sign = !cmd_ok ? 0 : (f.a_target > 0.15 ? 1 : (f.a_target < -0.15 ? -1 : 0));
  // Beyond-model acceleration is only a slip under traction; otherwise the wheels may lead.
  const double d_max_pos = (cmd_ok && f.notch > 0) ? p_.disturbance_max : p_.disturbance_max_free;
  double logL[kNumModes];
  for (int j = 0; j < kNumModes; ++j) {
    StateVec x = x0[j];
    StateCov P = P0[j];
    int idx[2];
    int m = 0;
    double ll = 0.0;
    for (int i = 0; i < 2; ++i) {
      if (!avail[i]) continue;
      if (kTrust[j][i]) {
        idx[m++] = i;
        if (implausible[i]) ll += kImplausiblePenalty;
      } else {
        ll += log_outlier;
        const double r = z[i] - (1.0 + x(kK, 0)) * x(kV, 0);
        if (effort_sign != 0 && r * effort_sign < -3.0 * p_.sigma_wheel) ll -= p_.wrong_sign_penalty;
      }
    }
    if (m == 1) ll += kfUpdate<1>(x, P, z, idx, p_);
    if (m == 2) ll += kfUpdate<2>(x, P, z, idx, p_);
    // Unmodelled deceleration (emergency / track brake) is physically possible; unmodelled
    // acceleration beyond the traction capability is not, so the disturbance is asymmetric.
    clampState(x, P, p_, -p_.disturbance_max_decel, d_max_pos);
    if (!allFinite(x) || !allFiniteM(P)) {  // numerical safety net: keep the prediction
      x = x0[j];
      P = P0[j];
    }
    f.x[j] = x;
    f.P[j] = P;
    logL[j] = ll;
  }
  double mx = -1e300;
  for (int j = 0; j < kNumModes; ++j) mx = std::max(mx, logL[j] + std::log(std::max(cbar[j], 1e-300)));
  double sum = 0.0;
  for (int j = 0; j < kNumModes; ++j) {
    f.mu[j] = std::exp(logL[j] + std::log(std::max(cbar[j], 1e-300)) - mx);
    sum += f.mu[j];
  }
  for (int j = 0; j < kNumModes; ++j) f.mu[j] = std::max(f.mu[j] / sum, p_.mode_prob_floor);
  sum = 0.0;
  for (int j = 0; j < kNumModes; ++j) sum += f.mu[j];
  for (int j = 0; j < kNumModes; ++j) f.mu[j] /= sum;

  // collapse the mode-conditioned posteriors (moment matching) into every mode
  StateVec xbar;
  for (int j = 0; j < kNumModes; ++j) xbar += f.mu[j] * f.x[j];
  StateCov Pbar;
  for (int j = 0; j < kNumModes; ++j) {
    const StateVec dx = f.x[j] - xbar;
    Pbar += f.mu[j] * (f.P[j] + dx * transpose(dx));
  }
  symmetrize(Pbar);
  for (int j = 0; j < kNumModes; ++j) {
    f.x[j] = xbar;
    f.P[j] = Pbar;
  }

  // ---- recovery: wheels agree with each other for long while the filter rejects them ----
  if (f.mu[kModeBothBad] > 0.5) {
    if (f.bad_since < 0) f.bad_since = e.t;
  } else {
    f.bad_since = -1;
  }
  const bool agree = avail[0] && avail[1] && std::abs(z[0] - z[1]) < p_.recover_agree &&
                     !implausible[0] && !implausible[1];
  if (agree) {
    if (f.agree_since < 0) f.agree_since = e.t;
  } else {
    f.agree_since = -1;
  }
  if (f.bad_since >= 0 && f.agree_since >= 0 && toSec(e.t - f.bad_since) >= p_.recover_min_bad_s &&
      toSec(e.t - f.agree_since) >= p_.recover_time_s) {
    const double zm = 0.5 * (z[0] + z[1]);
    const double vnow = combinedV(f);
    // never re-anchor onto wheels locked near zero while the model says we still move
    if (zm > 2.0 * thr || vnow < p_.standstill_max_v) {
      for (int j = 0; j < kNumModes; ++j) {
        const double k = f.x[j](kK, 0);
        pinVelocity(f.x[j], f.P[j], zm / (1.0 + k), p_.sigma_wheel * p_.sigma_wheel);
      }
      for (int j = 0; j < kNumModes; ++j) f.mu[j] = 0.01;
      f.mu[kModeNominal] = 1.0 - 0.01 * (kNumModes - 1);
      f.recovered_t = e.t;
      f.bad_since = f.agree_since = -1;
    }
  }
}

bool Estimator::jointMonitor(FilterState& f, const Event& e, const bool* avail, const double* z) const {
  StateVec xm;
  for (int j = 0; j < kNumModes; ++j) xm += f.mu[j] * f.x[j];
  const double vc = std::max(0.0, xm(kV, 0));
  const double k = xm(kK, 0);
  // Reference: the controller model with a grade-sized disturbance only. A large negative d
  // learned during an unmodelled brake must not make normal driving look like a slip.
  const double d_ref = std::clamp(xm(kD, 0), -p_.disturbance_max, p_.disturbance_max);
  const double a_model = xm(kG, 0) * f.a_drive + d_ref;
  auto setModelOnly = [&]() {
    for (int j = 0; j < kNumModes; ++j) f.mu[j] = p_.mode_prob_floor;
    f.mu[kModeBothBad] = 1.0 - p_.mode_prob_floor * (kNumModes - 1);
    f.t_mix = e.t;
  };
  auto setNominal = [&]() {
    for (int j = 0; j < kNumModes; ++j) f.mu[j] = 0.01;
    f.mu[kModeNominal] = 1.0 - 0.01 * (kNumModes - 1);
  };
  auto resetMonitor = [&]() {
    f.onset = false;
    for (int i = 0; i < 2; ++i) {
      f.wheel[i].cusum_pos = f.wheel[i].cusum_neg = 0.0;
      f.wheel[i].alarm = false;
    }
  };

  // ---- latched: the filter coasts on the model; release when wheels return to it ----
  if (f.latch) {
    const double gate = std::max(p_.latch_release_abs, p_.latch_release_rel * vc);
    int n = 0;
    bool ok = true;
    for (int i = 0; i < 2; ++i)
      if (avail[i]) {
        ++n;
        if (std::abs(z[i] - (1.0 + k) * vc) > gate) ok = false;
      }
    f.release_count = (n > 0 && ok) ? f.release_count + 1 : 0;
    if (f.release_count >= static_cast<int>(p_.latch_release_n)) {
      f.latch = false;
      setNominal();
      resetMonitor();
      f.snap_t = -1;
      return false;  // this sample updates the filter normally
    }
    if (toSec(e.t - f.latch_t) > p_.latch_max_s) {  // model drift now dominates: re-anchor
      const bool both = avail[0] && avail[1];
      if ((both && std::abs(z[0] - z[1]) < p_.recover_agree) || (avail[0] != avail[1])) {
        const double zm = both ? 0.5 * (z[0] + z[1]) : (avail[0] ? z[0] : z[1]);
        for (int j = 0; j < kNumModes; ++j)
          pinVelocity(f.x[j], f.P[j], zm / (1.0 + f.x[j](kK, 0)), p_.sigma_wheel * p_.sigma_wheel);
        f.latch = false;
        f.recovered_t = e.t;
        setNominal();
        resetMonitor();
        f.snap_t = -1;
        return false;
      }
    }
    setModelOnly();
    return true;
  }

  // ---- per-bogie CUSUM on (bogie acceleration over ~0.3 s - model acceleration) ----
  // Only the physically possible direction is monitored: slip under traction, slide under
  // braking. Consistent wheel motion of the opposite sign indicts the controller instead.
  const bool cmd_fault = f.cmd_fault_t >= 0 && toSec(e.t - f.cmd_fault_t) < p_.cmd_fault_hold_s;
  const bool traction = !cmd_fault && f.notch > 0 && f.a_target > 0.1;
  const bool braking = !cmd_fault && f.notch < 0;
  int n_av = 0, alarms = 0, sign = 0;
  double excess_sum = 0.0;
  int n_excess = 0;
  bool all_active = true;
  for (int i = 0; i < 2; ++i) {
    if (!avail[i]) continue;
    ++n_av;
    WheelTrack& w = f.wheel[i];
    const int slot = w.hhead;
    w.ht[slot] = e.t;
    w.hz[slot] = z[i];
    w.ham[slot] = a_model;
    w.hhead = (w.hhead + 1) % WheelTrack::kHist;
    w.hn = std::min(w.hn + 1, WheelTrack::kHist);
    int ref = -1;
    double best = 1e9;
    for (int q = 0; q < w.hn; ++q) {
      const double age = toSec(e.t - w.ht[q]);
      if (age >= 0.2 && age <= 0.5 && std::abs(age - 0.3) < best) {
        best = std::abs(age - 0.3);
        ref = q;
      }
    }
    const double dtc = w.t_cusum >= 0 ? std::clamp(toSec(e.t - w.t_cusum), 0.0, 0.5) : 0.0;
    w.t_cusum = e.t;
    if (ref < 0 || (vc < 0.5 && z[i] < 0.5)) {  // not enough history, or starting from rest
      w.cusum_pos = w.cusum_neg = 0.0;
    } else {
      const double age = toSec(e.t - w.ht[ref]);
      const double excess = (z[i] - w.hz[ref]) / age - 0.5 * (a_model + w.ham[ref]);
      excess_sum += excess;
      ++n_excess;
      w.cusum_pos = traction ? std::max(0.0, w.cusum_pos + (excess - p_.cusum_slip_accel) * dtc) : 0.0;
      w.cusum_neg = braking ? std::max(0.0, w.cusum_neg + (-excess - p_.cusum_slide_accel) * dtc) : 0.0;
    }
    w.alarm = w.cusum_pos > p_.cusum_h || w.cusum_neg > p_.cusum_h;
    if (w.alarm) {
      ++alarms;
      sign = w.cusum_pos > p_.cusum_h ? +1 : -1;
    }
    if (w.cusum_pos <= 0.0 && w.cusum_neg <= 0.0) all_active = false;
  }
  if (n_av == 0) return false;

  // ---- controller consistency: wheels accelerating without traction (or under braking), or
  // decelerating hard under traction, mean the notch signal is wrong -> trust the wheels ----
  if (n_excess > 0 && vc > 0.3) {
    const double ex = excess_sum / n_excess;
    double impossible = 0.0;
    if (f.notch <= 0) impossible = ex;                  // speeding up with no traction
    else if (f.a_target > 0.1) impossible = -ex - 1.0;  // strong braking under traction
    const double dtc = f.t_cmd_cusum >= 0 ? std::clamp(toSec(e.t - f.t_cmd_cusum), 0.0, 0.5) : 0.0;
    f.cmd_cusum = std::max(0.0, f.cmd_cusum + (impossible - p_.cmd_fault_accel) * dtc);
    if (f.cmd_cusum > p_.cmd_fault_h) {
      f.cmd_fault_t = e.t;
      f.cmd_cusum = p_.cmd_fault_h;  // keep refreshing while evidence persists
    }
  }
  f.t_cmd_cusum = e.t;

  // ---- onset bookkeeping: model trajectory from the last clean state ----
  if (!all_active) {
    f.onset = false;
    if (f.wheel[0].cusum_pos <= 0.0 && f.wheel[0].cusum_neg <= 0.0 &&
        f.wheel[1].cusum_pos <= 0.0 && f.wheel[1].cusum_neg <= 0.0) {
      f.snap_t = e.t;
      f.snap_v = vc;
      f.snap_s = xm(kS, 0);
      f.snap_d = d_ref;
      f.snap_g = xm(kG, 0);
    }
  } else if (!f.onset && f.snap_t >= 0) {
    f.onset = true;
    f.onset_t = f.mod_t = f.snap_t;
    f.mod_v = f.snap_v;
    f.mod_s = f.snap_s;
    f.onset_d = f.snap_d;
    f.onset_g = f.snap_g;
  }
  if (f.onset) {
    const double dt = toSec(e.t - f.mod_t);
    const double a = f.onset_g * f.a_drive + f.onset_d;
    double v1 = f.mod_v + a * dt;
    if (v1 < 0.0) v1 = 0.0;
    f.mod_s += 0.5 * (f.mod_v + v1) * dt;
    f.mod_v = v1;
    f.mod_t = e.t;
  }

  // ---- joint alarm: every available bogie misbehaves in the same direction; with a single
  // live bogie there is no cross-check, so demand stronger evidence ----
  if (n_av == 1 && alarms == 1) {
    int live = avail[0] ? 0 : 1;
    const WheelTrack& w = f.wheel[live];
    if (std::max(w.cusum_pos, w.cusum_neg) < p_.cusum_h * p_.single_bogie_latch_mult) alarms = 0;
  }
  if (alarms == n_av) {
    if (f.onset) {
      const double elapsed = toSec(e.t - f.onset_t);
      const double var_v = p_.sigma_wheel * p_.sigma_wheel + p_.sigma_accel * p_.sigma_accel * elapsed;
      for (int j = 0; j < kNumModes; ++j) {
        f.x[j](kS, 0) = f.mod_s;
        f.x[j](kD, 0) = f.onset_d;
        f.x[j](kG, 0) = f.onset_g;
        pinVelocity(f.x[j], f.P[j], f.mod_v, var_v);
      }
    }
    f.latch = true;
    f.latch_t = e.t;
    f.latch_sign = sign;
    f.release_count = 0;
    resetMonitor();
    setModelOnly();
    return true;
  }
  return false;
}

void Estimator::landmarkUpdate(FilterState& f, Stamp t) const {
  if (landmarks_.empty() || p_.landmark_enable < 0.5 || !map_) return;
  StateVec xm;
  for (int j = 0; j < kNumModes; ++j) xm += f.mu[j] * f.x[j];
  StateCov Pm;
  for (int j = 0; j < kNumModes; ++j) {
    const StateVec dx = f.x[j] - xm;
    Pm += f.mu[j] * (f.P[j] + dx * transpose(dx));
  }
  const TrackMap* m = nullptr;
  double sm = 0.0;
  if (!routeAt(xm(kS, 0), m, sm) || m != map_) return;  // landmarks exist on the main cycle only
  const double L = map_->length();
  const double extra2 = p_.landmark_sigma_extra * p_.landmark_sigma_extra;
  const double var_s = std::max(Pm(kS, kS), 0.0);
  const double g = p_.landmark_gate_sigma;
  double best_w = 0.0, sum_w = 0.0, best_delta = 0.0, best_r = 0.0;
  for (const Landmark& l : landmarks_) {
    double delta = l.s - sm;
    if (map_->cyclic()) {
      delta = std::fmod(delta, L);
      if (delta > 0.5 * L) delta -= L;
      if (delta < -0.5 * L) delta += L;
    }
    const double r = l.sigma * l.sigma + extra2;
    const double var = var_s + r;
    if (delta * delta > g * g * var) continue;
    const double w = std::max(l.p_stop, 0.02) * std::exp(-0.5 * delta * delta / var) / std::sqrt(2.0 * kPi * var);
    sum_w += w;
    if (w > best_w) {
      best_w = w;
      best_delta = delta;
      best_r = r;
    }
  }
  const double width = 2.0 * g * std::sqrt(var_s + extra2 + 0.25);
  const double w_random = p_.landmark_p_random / std::max(width, 1.0);
  static const bool dbg = std::getenv("TBO_DEBUG_LM") != nullptr;
  if (dbg)
    std::fprintf(stderr, "LM t=%.1f s_map=%.1f sd=%.2f best_delta=%.2f post=%.2f k=%.4f\n", toSec(t), sm,
                 std::sqrt(var_s), best_delta, best_w > 0 ? best_w / (sum_w + w_random) : 0.0, xm(kK, 0));
  if (best_w <= 0.0) return;
  if (best_w / (sum_w + w_random) < p_.landmark_min_prob) return;
  const double target = xm(kS, 0) + best_delta;
  for (int j = 0; j < kNumModes; ++j) {
    StateVec& x = f.x[j];
    StateCov& P = f.P[j];
    const double S = P(kS, kS) + best_r;
    if (!(S > 0.0)) continue;
    StateVec K;
    // a landmark fixes position and (through the s-k correlation) the wheel scale only
    K(kS, 0) = P(kS, kS) / S;
    K(kK, 0) = P(kK, kS) / S;
    const double innov = target - x(kS, 0);
    const double dk = K(kK, 0) * innov;
    if (std::abs(dk) > p_.landmark_max_dk && std::abs(K(kK, 0)) > 0.0) {
      // inflate the scale-position coupling so one fix cannot rewrite the wheel calibration
      const double shrink = p_.landmark_max_dk / std::abs(dk);
      K(kK, 0) *= shrink;
    }
    x += K * innov;
    Mat<1, kNx> H;
    H(0, kS) = 1.0;
    const StateCov IKH = StateCov::identity() - K * H;
    Mat<1, 1> R;
    R(0, 0) = best_r;
    P = IKH * P * transpose(IKH) + K * R * transpose(K);
    symmetrize(P);
    x(kK, 0) = std::clamp(x(kK, 0), -0.05, 0.05);
  }
  f.lm_t = t;
}

// --------------------------------------------------------------------------------------
// GNSS initialisation (only inside the init window) and output frame.

void Estimator::onGnssVel(GnssSource, Stamp, double, double, double) {
  // Velocity is not needed: every run starts at standstill and speed comes from wheels.
  ++diag_.gnss_msgs;
}

void Estimator::onGnssFix(GnssSource src, Stamp stamp, double lat, double lon, double alt, int status) {
  ++diag_.gnss_msgs;
  if (stamp <= 0 || !std::isfinite(lat) || !std::isfinite(lon) || !std::isfinite(alt)) return;
  if (std::abs(lat) > 90.0 || std::abs(lon) > 180.0 || status < 0) return;
  if (!init_.have_first) {
    init_.have_first = true;
    init_.t_first = stamp;
  }
  if (stamp > init_.t_first + fromSec(p_.gnss_init_window_s)) {
    ++diag_.gnss_ignored_after_window;
    return;
  }
  const bool primary = (cfg_.init_source == "rover") == (src == GnssSource::Rover);
  const Fix fx{stamp, lat, lon, alt};
  if (primary) {
    if (!init_.origin_set) {
      init_.origin = {lat, lon, alt};
      init_.origin_set = true;
      configureFrame();
    }
    if (init_.fixes.size() < 512) init_.fixes.push_back(fx);
  } else if (init_.other.size() < 512) {
    init_.other.push_back(fx);
  }
  if (init_.origin_set) updateAnchor();
}

void Estimator::configureFrame() {
  out_lc_.reset(init_.origin);
  if (map_) {
    const geo::Ecef o = map_lc_.toEcef({0.0, 0.0, 0.0});
    const geo::Enu t0 = out_lc_.fromEcef(o);
    trans_[0] = t0.e;
    trans_[1] = t0.n;
    trans_[2] = t0.u;
    for (int c = 0; c < 3; ++c) {
      geo::Enu unit{c == 0 ? 1.0 : 0.0, c == 1 ? 1.0 : 0.0, c == 2 ? 1.0 : 0.0};
      const geo::Enu q = out_lc_.fromEcef(map_lc_.toEcef(unit));
      rot_[0][c] = q.e - t0.e;
      rot_[1][c] = q.n - t0.n;
      rot_[2][c] = q.u - t0.u;
    }
  }
  frame_ready_ = true;
}

void Estimator::mapToOutput(double mx, double my, double mz, double& ox, double& oy, double& oz) const {
  if (cfg_.output_frame == "map") {
    ox = mx;
    oy = my;
    oz = mz;
    return;
  }
  if (cfg_.output_frame == "utm") {
    const geo::Geodetic g = map_lc_.reverse({mx, my, mz});
    const geo::Utm o = geo::geodeticToUtm(init_.origin.lat_deg, init_.origin.lon_deg);
    const geo::Utm u = geo::geodeticToUtm(g.lat_deg, g.lon_deg, o.zone);
    ox = u.easting - o.easting;
    oy = u.northing - o.northing;
    oz = g.h - init_.origin.h;
    return;
  }
  ox = rot_[0][0] * mx + rot_[0][1] * my + rot_[0][2] * mz + trans_[0];
  oy = rot_[1][0] * mx + rot_[1][1] * my + rot_[1][2] * mz + trans_[1];
  oz = rot_[2][0] * mx + rot_[2][1] * my + rot_[2][2] * mz + trans_[2];
}

void Estimator::updateAnchor() {
  if (init_.fixes.empty()) return;
  std::vector<double> la, lo, al;
  for (const Fix& fx : init_.fixes) {
    la.push_back(fx.lat);
    lo.push_back(fx.lon);
    al.push_back(fx.alt);
  }
  const geo::Geodetic med{medianOf(la), medianOf(lo), medianOf(al)};
  Stamp t_med = init_.fixes[init_.fixes.size() / 2].t;

  // Forward yaw from the antenna baseline (master -> rover points forward), in the
  // map frame when a map exists, else in the output ENU frame.
  const geo::LocalCartesian& lc = map_ ? map_lc_ : out_lc_;
  double sx = 0.0, sy = 0.0;
  int npairs = 0;
  for (const Fix& a : init_.fixes) {
    const Fix* best = nullptr;
    Stamp bd = fromSec(0.02);
    for (const Fix& b : init_.other)
      if (std::llabs(b.t - a.t) <= bd) {
        bd = std::llabs(b.t - a.t);
        best = &b;
      }
    if (!best) continue;
    const geo::Enu pa = lc.forward({a.lat, a.lon, a.alt});
    const geo::Enu pb = lc.forward({best->lat, best->lon, best->alt});
    double dx = pb.e - pa.e, dy = pb.n - pa.n;
    const double len = std::hypot(dx, dy);
    if (len < 3.0 || len > 40.0) continue;
    if (cfg_.init_source == "rover") {
      dx = -dx;
      dy = -dy;
    }
    sx += dx / len;
    sy += dy / len;
    ++npairs;
  }
  if (npairs > 0) {
    init_.yaw0 = std::atan2(sy, sx);
    init_.have_yaw = true;
  }

  const Output o = query(t_med);
  const double s_rel = o.valid ? o.s : 0.0;
  if (map_) {
    const geo::Enu m = map_lc_.forward(med);
    // candidates on the main cycle and on branches merging into it (e.g. a terminal fan track)
    constexpr int kMax = 24;
    MapProjection cand[kMax];
    const TrackMap* owner[kMax];
    int n = map_->projectAll(m.e, m.n, p_.map_gate_m, cand, 8);
    for (int i = 0; i < n; ++i) owner[i] = map_;
    for (const TrackMap* b : branches_) {
      const int k = b->projectAll(m.e, m.n, p_.map_gate_m, cand + n, std::min(4, kMax - n));
      for (int i = n; i < n + k; ++i) owner[i] = b;
      n += k;
    }
    int best = -1;
    const double gate = p_.map_heading_gate_deg * kPi / 180.0;
    for (int pass = 0; pass < 2 && best < 0; ++pass) {
      double bd = 1e300;
      for (int i = 0; i < n; ++i) {
        const bool heading_ok = !init_.have_yaw || p_.use_baseline_heading < 0.5 ||
                                std::abs(wrapAngle(cand[i].heading - init_.yaw0)) < gate;
        // a branch must be clearly closer than the main line to be preferred
        const double d = cand[i].dist + (owner[i] == map_ ? 0.0 : 1.0);
        if ((pass == 0 && !heading_ok) || d >= bd) continue;
        bd = d;
        best = i;
      }
    }
    init_.map_matched = best >= 0;
    if (best >= 0 && !init_.var_applied && committed_.started) {
      const double sig = p_.init_sigma_s + p_.init_sigma_s_per_m * cand[best].dist;
      for (int j = 0; j < kNumModes; ++j) committed_.P[j](kS, kS) += sig * sig;
      init_.var_applied = true;
    }
    init_.prefix = nullptr;
    if (best >= 0) {
      init_.match_dist = cand[best].dist;
      if (owner[best] == map_) {
        init_.s_offset = cand[best].s - s_rel;
      } else {
        init_.prefix = owner[best];
        init_.prefix_s0 = cand[best].s - s_rel;
        init_.prefix_len = owner[best]->length() - init_.prefix_s0;
      }
    }
  }
  // Dead-reckoning start in the output frame (used when not map-matched).
  if (cfg_.output_frame == "utm") {
    const geo::Utm oz = geo::geodeticToUtm(init_.origin.lat_deg, init_.origin.lon_deg);
    const geo::Utm u = geo::geodeticToUtm(med.lat_deg, med.lon_deg, oz.zone);
    init_.dr_x = u.easting - oz.easting;
    init_.dr_y = u.northing - oz.northing;
    init_.dr_z = med.h - init_.origin.h;
  } else {
    const geo::Enu q = out_lc_.forward(med);
    init_.dr_x = q.e;
    init_.dr_y = q.n;
    init_.dr_z = q.u;
  }
  init_.anchored = true;
}

// --------------------------------------------------------------------------------------

Output Estimator::query(Stamp t) const {
  if (!started_) {
    Output o;
    o.stamp = t;
    o.flags = kFlagNotInitialized | kFlagNoMap;
    return o;
  }
  FilterState f = committed_;
  for (const Event& e : buf_) {
    if (e.t > t) break;
    applyEvent(f, e);
  }
  advance(f, t);
  return makeOutput(f, t);
}

Output Estimator::makeOutput(const FilterState& f, Stamp t) const {
  Output o;
  o.stamp = t;
  o.valid = f.started;
  StateVec xm;
  for (int j = 0; j < kNumModes; ++j) xm += f.mu[j] * f.x[j];
  StateCov Pm;
  for (int j = 0; j < kNumModes; ++j) {
    const StateVec dx = f.x[j] - xm;
    Pm += f.mu[j] * (f.P[j] + dx * transpose(dx));
  }
  o.v = std::max(0.0, xm(kV, 0));
  o.v_var = Pm(kV, kV);
  o.s = xm(kS, 0);
  o.s_var = Pm(kS, kS);
  o.disturbance = xm(kD, 0);
  o.scale = xm(kK, 0);
  o.gain = xm(kG, 0);
  o.a_model = f.a_target;
  o.accel = f.standstill ? 0.0 : xm(kG, 0) * f.a_drive + xm(kD, 0);
  for (int j = 0; j < kNumModes; ++j) o.mode_prob[j] = f.mu[j];
  o.wheel_trust = 1.0 - f.mu[kModeBothBad];

  // ---- diagnostics flags ----
  std::uint32_t fl = 0;
  const Stamp wheel_to = fromSec(p_.wheel_timeout_s);
  const Stamp hold = fromSec(1.0);
  const double vref = std::max(o.v, 1.0);
  for (int i = 0; i < 2; ++i) {
    const WheelTrack& w = f.wheel[i];
    const bool fresh = w.have && t - w.t <= wheel_to;
    if (!fresh) fl |= i == 0 ? kFlagFrontDropout : kFlagRearDropout;
    if (w.stuck || (w.zero_stuck_t >= 0 && t - w.zero_stuck_t <= hold))
      fl |= i == 0 ? kFlagFrontStuck : kFlagRearStuck;
    if (w.invalid_t >= 0 && t - w.invalid_t <= hold) fl |= i == 0 ? kFlagFrontInvalid : kFlagRearInvalid;
    const double bad = f.mu[i == 0 ? kModeFrontBad : kModeRearBad] + f.mu[kModeBothBad];
    const double ratio = fresh ? (w.z - o.v * (1.0 + o.scale)) / vref : 0.0;
    if (i == 0) o.slip_front = ratio;
    else o.slip_rear = ratio;
    if (fresh && bad > 0.5 && !f.standstill) {
      if (ratio > 0) fl |= i == 0 ? kFlagFrontSlip : kFlagRearSlip;
      else fl |= i == 0 ? kFlagFrontSlide : kFlagRearSlide;
    }
  }
  if (!f.have_cmd || t - f.last_cmd > fromSec(p_.cmd_timeout_s)) fl |= kFlagCmdDropout;
  const bool both_out = (fl & kFlagFrontDropout) && (fl & kFlagRearDropout);
  if (f.mu[kModeBothBad] > 0.5 || both_out) fl |= kFlagModelOnly;
  if (f.standstill) fl |= kFlagStandstill;
  if (f.recovered_t >= 0 && t - f.recovered_t <= fromSec(2.0)) fl |= kFlagRecovered;
  if (f.mu[kModeManeuver] > 0.5 && !f.standstill) fl |= kFlagUnmodeledAccel;
  if (f.cmd_fault_t >= 0 && t - f.cmd_fault_t < fromSec(p_.cmd_fault_hold_s)) fl |= kFlagCmdInconsistent;
  if (f.lm_t >= 0 && t - f.lm_t < fromSec(3.0)) fl |= kFlagLandmark;
  if (committed_.late_t >= 0 && t - committed_.late_t <= hold) fl |= kFlagLateData;

  // ---- position ----
  const double sig_s2 = o.s_var + (init_.var_applied ? 0.0 : p_.init_sigma_s * p_.init_sigma_s);
  const TrackMap* rm = nullptr;
  double sm = 0.0;
  const double s_pub = o.s + o.v * p_.position_lead_s;
  if (init_.anchored && routeAt(s_pub, rm, sm)) {
    const TrackMap* rm2 = nullptr;
    double sm2 = 0.0;
    routeAt(s_pub + 1.0, rm2, sm2);
    const MapPose a = rm->at(sm), b = rm2->at(sm2);
    double ax, ay, az, bx, by, bz;
    mapToOutput(a.x, a.y, a.z, ax, ay, az);
    mapToOutput(b.x, b.y, b.z, bx, by, bz);
    o.x = ax;
    o.y = ay;
    o.z = az;
    o.yaw = std::atan2(by - ay, bx - ax);
    o.map_matched = true;
    const double sc2 = p_.map_sigma_cross * p_.map_sigma_cross;
    const double c = std::cos(o.yaw), s = std::sin(o.yaw);
    o.cov_xx = sig_s2 * c * c + sc2 * s * s;
    o.cov_yy = sig_s2 * s * s + sc2 * c * c;
    o.cov_xy = (sig_s2 - sc2) * s * c;
    o.cov_zz = p_.map_sigma_z * p_.map_sigma_z;
  } else if (init_.anchored) {
    const double yaw = init_.have_yaw ? init_.yaw0 : 0.0;
    o.x = init_.dr_x + o.s * std::cos(yaw);
    o.y = init_.dr_y + o.s * std::sin(yaw);
    o.z = init_.dr_z;
    o.yaw = yaw;
    const double cross = 1.0 + 0.05 * std::abs(o.s);  // heading unknown along curves
    o.cov_xx = sig_s2 + cross * cross;
    o.cov_yy = sig_s2 + cross * cross;
    o.cov_zz = 4.0 + (0.01 * o.s) * (0.01 * o.s);
    fl |= kFlagNoMap;
  } else {
    o.x = o.s;
    o.y = 0.0;
    o.z = 0.0;
    o.cov_xx = sig_s2;
    o.cov_yy = 1e4;
    o.cov_zz = 1e4;
    fl |= kFlagNotInitialized | kFlagNoMap;
  }
  o.flags = fl;
  return o;
}

}  // namespace tbo
