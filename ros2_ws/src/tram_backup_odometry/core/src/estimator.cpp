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
constexpr std::size_t kQuantMaxSteps = 20000;  // speed-quantum steps kept per bogie and run

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
  // Position is the integral of speed and is corrected by landmarks only. Its wheel-update gain,
  // (P_sv (1+k) + P_sk v) / S, is the small difference of two large terms (both ~ distance * P_kk)
  // that the linearised scale does not keep balanced when speed changes: without landmarks the
  // remainder moved s by 0.4-0.6 m per sample (up to 96 m backwards over a run without GNSS).
  for (int r = 0; r < M; ++r) K(kS, r) = 0.0;
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

// Imposes speed v with variance var (standstill, re-anchoring to the wheels, roll-back) as a
// scalar measurement of v on a widened prior, so the covariance stays consistent. Zeroing the
// v row/column instead left stale s-k / s-v cross terms behind; later wheel updates then moved s
// by metres per sample through the wheel-scale channel (P_sk * v / S), backwards as often as not.
void pinVelocity(StateVec& x, StateCov& P, double v, double var) {
  const double innov = v - x(kV, 0);
  P(kV, kV) += innov * innov + 1.0;  // the value is imposed: open the prior on v first
  const double S = P(kV, kV) + var;
  StateVec K;
  for (int i = 0; i < kNx; ++i) K(i, 0) = P(i, kV) / S;
  K(kK, 0) = 0.0;  // the wheel scale is calibrated by landmarks only (Schmidt state)
  x += K * innov;
  x(kV, 0) = v;
  Mat<1, kNx> H;
  H(0, kV) = 1.0;
  const StateCov IKH = StateCov::identity() - K * H;
  Mat<1, 1> R;
  R(0, 0) = var;
  P = IKH * P * transpose(IKH) + K * R * transpose(K);
  symmetrize(P);
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
    if (ss >> l.s >> l.sigma) {
      double p = 0.0;
      l.p_stop = (ss >> p) ? (p <= 1.0 ? p : 1.0) : 1.0;  // cut-off files carry a count instead
      out.push_back(l);
    }
  }
  std::sort(out.begin(), out.end(), [](const Landmark& a, const Landmark& b) { return a.s < b.s; });
  return true;
}

bool loadWheelEpochs(const std::string& path, std::vector<WheelEpoch>& out, std::string* err) {
  std::ifstream in(path);
  if (!in) {
    if (err) *err = "cannot open wheel epochs " + path;
    return false;
  }
  out.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#' || line[0] == 'v') continue;  // comment or header
    for (char& c : line)
      if (c == ',') c = ' ';
    std::istringstream ss(line);
    WheelEpoch e;
    if (ss >> e.vehicle >> e.date >> e.c_front >> e.c_rear && e.c_front > 0.0 && e.c_rear > 0.0) out.push_back(e);
  }
  return true;
}

bool loadRatioMap(const std::string& path, RatioMap& out, std::string* err) {
  std::ifstream in(path);
  if (!in) {
    if (err) *err = "cannot open ratio map " + path;
    return false;
  }
  out = RatioMap{};
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line.rfind("#sv,", 0) == 0) {  // speed table of the straight-track sd of log(front/rear)
      double v = 0.0, s = 0.0;
      if (std::sscanf(line.c_str() + 4, "%lf,%lf", &v, &s) == 2) {
        out.sv_v.push_back(v);
        out.sv_s.push_back(s);
      }
      continue;
    }
    if (line.rfind("# L=", 0) == 0) {
      out.L = std::atof(line.c_str() + 4);
      continue;
    }
    if (line[0] == '#' || line[0] == 'b') continue;
    int i = 0, r = 0;
    double mu = 0.0, sd = 1.0;
    if (std::sscanf(line.c_str(), "%d,%lf,%lf,%d", &i, &mu, &sd, &r) == 4) {
      out.mu.push_back(mu);
      out.sd.push_back(sd);
      out.rel.push_back(static_cast<unsigned char>(r != 0));
    }
  }
  if (out.mu.empty() || out.sv_v.size() < 2 || !(out.L > 0.0)) {
    if (err) *err = "bad ratio map " + path;
    out = RatioMap{};
    return false;
  }
  return true;
}

int dateMsk(Stamp t) {
  // civil date of the Unix day (Howard Hinnant's days-to-civil), Moscow time
  const long long sec = static_cast<long long>(std::floor(toSec(t))) + 3 * 3600;
  long long z = (sec >= 0 ? sec : sec - 86399) / 86400 + 719468;
  const long long era = (z >= 0 ? z : z - 146096) / 146097;
  const long long doe = z - era * 146097;
  const long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const long long mp = (5 * doy + 2) / 153;
  const long long d = doy - (153 * mp + 2) / 5 + 1;
  const long long m = mp < 10 ? mp + 3 : mp - 9;
  const long long y = yoe + era * 400 + (m <= 2 ? 1 : 0);
  return static_cast<int>(y * 10000 + m * 100 + d);
}

Estimator::Estimator(const Config& cfg, const TractionModel& model, const TrackMap* map,
                     std::vector<const TrackMap*> branches)
    : cfg_(cfg), p_(cfg_.p), model_(model), map_(map && !map->empty() ? map : nullptr) {
  for (const TrackMap* b : branches)
    if (map_ && b && !b->empty() && b->hasJoin()) branches_.push_back(b);
  buf_.reserve(256);
  for (QuantTrack& q : quant_) q.steps.reserve(kQuantMaxSteps);
  quant_scratch_.reserve(kQuantMaxSteps);
  rring_.resize(kRatioCap);
  rll_.reserve(1024);
  if (map_) map_lc_.reset(map_->origin());
  reset();
}

void Estimator::reset() {
  committed_ = FilterState{};
  buf_.clear();
  started_ = false;
  latest_ = 0;
  pending_jump_ = 0;
  s_hist_.clear();
  gl_.reset();
  gl_tried_ = false;
  init_ = InitState{};
  init_.fixes.reserve(512);
  init_.other.reserve(512);
  frame_ready_ = false;
  for (int i = 0; i < 2; ++i) {
    quant_[i].have = false;
    quant_[i].steps.clear();
    quant_cache_n_[i] = 0;
  }
  rhead_ = 0;
  rcount_ = 0;
  rlast_s_ = -1e18;
}

bool Estimator::acceptStamp(Stamp stamp) {
  if (stamp <= 0) {
    ++diag_.rejected_stamps;
    return false;
  }
  if (!started_) return true;
  const bool ahead = stamp > latest_ + fromSec(p_.max_future_s);
  const bool behind = stamp < latest_ - fromSec(p_.max_backjump_s);
  if (!ahead && !behind) {
    pending_jump_ = 0;
    return true;
  }
  // A lone stamp far from the current time is a glitch and is dropped. A second message close to
  // it means the time base itself jumped: every input was silent for a while (continue, the model
  // bridges the gap) or the bag was restarted / another bag is playing (new run).
  if (pending_jump_ <= 0 || std::llabs(stamp - pending_jump_) > fromSec(p_.jump_confirm_s)) {
    pending_jump_ = stamp;
    ++diag_.rejected_stamps;
    return false;
  }
  pending_jump_ = 0;
  if (behind || stamp - latest_ >= fromSec(p_.new_run_gap_s)) {
    ++diag_.resets;
    reset();
  } else {
    ++diag_.time_gaps;
  }
  return true;
}

bool Estimator::onWheel(Sensor sensor, Stamp stamp, double speed_kmh) {
  ++diag_.wheel_msgs;
  if (!acceptStamp(stamp)) return false;
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
    // smallest reading step (speed quantum): consecutive changes inside the search range
    QuantTrack& qt = quant_[i];
    if (qt.have) {
      const double dq = std::abs(speed_kmh - qt.last);
      if (dq > p_.quant_step_lo_kmh && dq < p_.quant_step_hi_kmh && qt.steps.size() < kQuantMaxSteps)
        qt.steps.push_back(dq);
    }
    qt.last = speed_kmh;
    qt.have = true;
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
  return true;
}

bool Estimator::onCmd(Stamp stamp, int notch) {
  ++diag_.cmd_msgs;
  if (notch < TractionModel::kNotchMin || notch > TractionModel::kNotchMax) {
    ++diag_.invalid_cmd;
    return false;
  }
  if (!acceptStamp(stamp)) return false;
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
  return true;
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
    commitOne(buf_.front());
    buf_.erase(buf_.begin());
  }
}

void Estimator::commitOlderThan(Stamp t) {
  size_t n = 0;
  while (n < buf_.size() && buf_[n].t < t) {
    commitOne(buf_[n]);
    ++n;
  }
  if (n > 0) buf_.erase(buf_.begin(), buf_.begin() + static_cast<long>(n));
}

void Estimator::commitOne(const Event& e) {
  const Stamp lm_before = committed_.lm_t;
  applyEvent(committed_, e);
  noteCommitted();
  feedGlobal();
  ratioStep(e, committed_.lm_t != lm_before);
}

void Estimator::ratioStep(const Event& e, bool place_fixed) {
  if (rmap_.empty() || p_.ratio_enable < 0.5 || !map_ || !init_.anchored || e.is_cmd) return;
  FilterState& f = committed_;
  if (!f.started) return;
  if (place_fixed || f.stub) {  // the arc jumped (place fix) or left the main cycle: start the window afresh
    rcount_ = 0;
    if (f.stub) return;
  }
  const bool both = e.has[0] && e.has[1] && std::isfinite(e.z[0]) && std::isfinite(e.z[1]);
  if (!both || e.z[0] < p_.ratio_vmin || e.z[1] < p_.ratio_vmin) return;
  // healthy bogies only: no joint anomaly, no CUSUM alarm, nothing stuck, the nominal hypothesis dominant
  if (f.latch || f.wheel[0].alarm || f.wheel[1].alarm || f.wheel[0].stuck || f.wheel[1].stuck ||
      f.mu[kModeNominal] < 0.9)
    return;
  const double y = std::log(e.z[0] / e.z[1]);
  if (!(std::abs(y) < p_.ratio_ymax)) return;
  double s = 0.0;
  for (int j = 0; j < kNumModes; ++j) s += f.mu[j] * f.x[j](kS, 0);
  const TrackMap* m = nullptr;
  double sm = 0.0;
  if (!routeAt(s, m, sm) || m != map_) {  // main cycle only (the map is learned there)
    rcount_ = 0;
    return;
  }
  auto sigmaV = [&](double v) {  // straight-track sd of log(front/rear) at speed v
    const auto& xv = rmap_.sv_v;
    const auto& ys = rmap_.sv_s;
    if (v <= xv.front()) return ys.front();
    if (v >= xv.back()) return ys.back();
    const auto it = std::upper_bound(xv.begin(), xv.end(), v);
    const size_t i = static_cast<size_t>(it - xv.begin());
    const double w = (v - xv[i - 1]) / (xv[i] - xv[i - 1]);
    return ys[i - 1] + w * (ys[i] - ys[i - 1]);
  };
  // append; drop what fell out of the window
  const int cap = static_cast<int>(rring_.size());
  if (rcount_ == cap) {
    rhead_ = (rhead_ + 1) % cap;
    --rcount_;
  }
  const double sv0 = sigmaV(0.5 * (e.z[0] + e.z[1]));
  rring_[(rhead_ + rcount_) % cap] = {s, y / sv0, 1.0 / sv0};
  ++rcount_;
  while (rcount_ > 0 && rring_[rhead_].s < s - p_.ratio_window_m - 5.0) {
    rhead_ = (rhead_ + 1) % cap;
    --rcount_;
  }
  if (s - rlast_s_ < p_.ratio_step_m) return;
  rlast_s_ = s;
  int n = 0;
  for (int k = 0; k < rcount_; ++k)
    if (rring_[(rhead_ + k) % cap].s >= s - p_.ratio_window_m) ++n;
  if (n < p_.ratio_min_samples) return;
  const double L = rmap_.L;
  const int nb = static_cast<int>(rmap_.mu.size());
  auto wrapL = [&](double q) {
    q = std::fmod(q, L);
    return q < 0.0 ? q + L : q;
  };
  {  // corrections only where the leave-one-out check never made the estimate worse
    const int b = std::clamp(static_cast<int>(wrapL(sm) * nb / L), 0, nb - 1);
    if (!rmap_.rel[b]) return;
  }
  ++diag_.ratio_tries;
  auto mapAt = [&](double q, double& mu, double& sd) {  // linear between bin centres, cyclic
    const double u = wrapL(q) * nb / L - 0.5;
    const double fl = std::floor(u);
    const double w = u - fl;
    int i0 = static_cast<int>(fl) % nb;
    if (i0 < 0) i0 += nb;
    const int i1 = (i0 + 1) % nb;
    mu = rmap_.mu[i0] + w * (rmap_.mu[i1] - rmap_.mu[i0]);
    sd = rmap_.sd[i0] + w * (rmap_.sd[i1] - rmap_.sd[i0]);
  };
  // log-likelihood of the window for every candidate correction g: y_i = b + sv_i (mu_z + sd_z eps), free b
  constexpr double kStep = 0.1;
  const int ng = 2 * static_cast<int>(std::lround(p_.ratio_search_m / kStep)) + 1;
  rll_.assign(static_cast<size_t>(ng), 0.0);
  for (int gi = 0; gi < ng; ++gi) {
    const double g = -p_.ratio_search_m + gi * kStep;
    double swc2 = 0.0, swcu = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
      const double b = pass == 0 ? 0.0 : (swc2 > 0.0 ? swcu / swc2 : 0.0);
      double ll = 0.0;
      for (int k = 0; k < rcount_; ++k) {
        const RatioSample& r = rring_[(rhead_ + k) % cap];
        if (r.s < s - p_.ratio_window_m) continue;
        double mu = 0.0, sd = 1.0;
        mapAt(sm + (r.s - s) + g, mu, sd);
        const double w = 1.0 / (sd * sd), c = r.c, u = r.u;
        if (pass == 0) {
          swc2 += w * c * c;
          swcu += w * c * (u - mu);
        } else {
          const double res = u - b * c - mu;
          ll += -0.5 * w * res * res - std::log(sd);
        }
      }
      if (pass == 1) rll_[static_cast<size_t>(gi)] = ll / p_.ratio_tau;
    }
  }
  // peak: argmax refined by a parabola, sharpness from a quadratic fit over +-0.5 m, margin to other maxima
  int kb = 0;
  for (int gi = 1; gi < ng; ++gi)
    if (rll_[static_cast<size_t>(gi)] > rll_[static_cast<size_t>(kb)]) kb = gi;
  const auto ll = [&](int gi) { return rll_[static_cast<size_t>(gi)]; };
  if (kb == 0 || kb == ng - 1) return;  // at the edge of the search: not a peak
  double d_hat = -p_.ratio_search_m + kb * kStep;
  {
    const double den = ll(kb - 1) - 2.0 * ll(kb) + ll(kb + 1);
    if (den < 0.0) d_hat += 0.5 * kStep * (ll(kb - 1) - ll(kb + 1)) / den;
  }
  double sx[5] = {0, 0, 0, 0, 0}, sy[3] = {0, 0, 0};  // sums of x^0..x^4 and y x^0..x^2
  for (int gi = std::max(0, kb - 5); gi <= std::min(ng - 1, kb + 5); ++gi) {
    const double x = (gi - kb) * kStep, yv = ll(gi);
    double xp = 1.0;
    for (int q = 0; q < 5; ++q) {
      sx[q] += xp;
      if (q < 3) sy[q] += yv * xp;
      xp *= x;
    }
  }
  // normal equations for yv = a x^2 + b x + c (Cramer's rule)
  const double A[3][3] = {{sx[4], sx[3], sx[2]}, {sx[3], sx[2], sx[1]}, {sx[2], sx[1], sx[0]}};
  const double B[3] = {sy[2], sy[1], sy[0]};
  auto det3 = [](const double M3[3][3]) {
    return M3[0][0] * (M3[1][1] * M3[2][2] - M3[1][2] * M3[2][1]) - M3[0][1] * (M3[1][0] * M3[2][2] - M3[1][2] * M3[2][0]) +
           M3[0][2] * (M3[1][0] * M3[2][1] - M3[1][1] * M3[2][0]);
  };
  const double D0 = det3(A);
  if (!(std::abs(D0) > 0.0)) return;
  double Aa[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) Aa[r][c] = c == 0 ? B[r] : A[r][c];
  const double curv = -2.0 * det3(Aa) / D0;
  const double sig_c = curv > 0.0 ? 1.0 / std::sqrt(curv) : 1e9;
  double second = -1e300;
  for (int gi = 0; gi < ng; ++gi) {
    if (std::abs(gi - kb) * kStep <= 2.0 + 1e-9) continue;
    const bool lmax = (gi == 0 || ll(gi) >= ll(gi - 1)) && (gi == ng - 1 || ll(gi) >= ll(gi + 1));
    if (lmax) second = std::max(second, ll(gi));
  }
  const double margin = second > -1e299 ? ll(kb) - second : 1e9;
  if (!(margin > p_.ratio_margin) || !(sig_c < p_.ratio_sigma_max)) return;
  const double sd_c = std::max(sig_c, p_.ratio_sigma_min);
  const double R = sd_c * sd_c;
  double pss = 0.0;
  for (int j = 0; j < kNumModes; ++j) {
    const double ds = f.x[j](kS, 0) - s;
    pss += f.mu[j] * (f.P[j](kS, kS) + ds * ds);
  }
  if (!(std::abs(d_hat) < p_.ratio_dmax) || !(std::abs(d_hat) < p_.ratio_gate_sd * std::sqrt(std::max(pss, 0.0) + R)))
    return;
  static const bool dbg = std::getenv("TBO_DEBUG_RATIO") != nullptr;
  if (dbg)
    std::fprintf(stderr, "RATIO t=%.1f s_map=%.1f d=%+.2f sig=%.2f margin=%.1f n=%d sd_s=%.2f\n", toSec(f.t), sm, d_hat,
                 sig_c, margin, n, std::sqrt(std::max(pss, 0.0)));
  for (int j = 0; j < kNumModes; ++j) {  // one scalar measurement of the path on every hypothesis
    StateVec& x = f.x[j];
    StateCov& P = f.P[j];
    const double S = P(kS, kS) + R;
    if (!(S > 0.0)) continue;
    // like a place fix it moves the path only (speed, disturbance and gain stay with the wheels and the model;
    // letting the s-v correlation act made the speed vs doppler worse on all 12 val runs)
    StateVec K;
    K(kS, 0) = P(kS, kS) / S;
    if (p_.ratio_update_k > 0.5 && std::abs(d_hat) < p_.ratio_k_dmax)  // k: normally calibrated by the quantum and places
      K(kK, 0) = p_.ratio_k_gain * P(kK, kS) / S;
    x += K * (s + d_hat - x(kS, 0));
    Mat<1, kNx> H;
    H(0, kS) = 1.0;
    const StateCov IKH = StateCov::identity() - K * H;
    Mat<1, 1> Rm;
    Rm(0, 0) = R;
    P = IKH * P * transpose(IKH) + K * Rm * transpose(K);
    symmetrize(P);
    x(kK, 0) = std::clamp(x(kK, 0), -0.05, 0.05);
    if (!(x(kV, 0) >= 0.0)) x(kV, 0) = 0.0;
  }
  double s_after = 0.0;
  for (int j = 0; j < kNumModes; ++j) s_after += f.mu[j] * f.x[j](kS, 0);
  for (int k = 0; k < rcount_; ++k) rring_[(rhead_ + k) % cap].s += s_after - s;  // keep the window consistent
  rlast_s_ += s_after - s;
  if (p_.ratio_reset_assoc > 0.5) f.lm_odo = f.odo;
  ++diag_.ratio_fixes;
}

void Estimator::noteCommitted() {
  if (!committed_.started || gnssWindowClosed()) return;  // only the anchor needs the history
  double s = 0.0;
  for (int j = 0; j < kNumModes; ++j) s += committed_.mu[j] * committed_.x[j](kS, 0);
  if (!s_hist_.empty() && s_hist_.back().first >= committed_.t) {
    s_hist_.back().second = s;  // several events at one stamp: keep the latest state
    return;
  }
  s_hist_.emplace_back(committed_.t, s);
  constexpr double kHistS = 30.0;  // longer than the init window plus any fix delay
  const Stamp keep = committed_.t - fromSec(kHistS);
  while (s_hist_.size() > 2 && (s_hist_[1].first <= keep || s_hist_.size() > 4096)) s_hist_.pop_front();
}

bool Estimator::globalAvailable() const {
  return p_.global_loc_enable > 0.5 && map_ && map_->cyclic() && !gl_stops_.empty();
}

void Estimator::feedGlobal() {
  if (init_.anchored || !committed_.started || !globalAvailable()) return;
  if (!gl_) {
    // only when no GNSS fix came at all within the wait (the normal path anchors on GNSS)
    if (gl_tried_ || init_.have_gnss || !init_.have_first) return;
    if (committed_.t - init_.t_first < fromSec(p_.gnss_wait_s)) return;
    gl_tried_ = true;
    GlobalLocalizer::Params gp;
    gp.curv_abs = p_.wheel_curv_abs;
    gp.curv_signed = p_.wheel_curv_signed;
    gp.curv_sat = p_.wheel_curv_sat;
    gp.front_along = p_.front_bogie_along_m;
    gp.rear_along = p_.rear_bogie_along_m;
    gl_ = std::make_unique<GlobalLocalizer>();
    if (!gl_->init(*map_, gl_stops_, gl_cutoffs_, vmax_env_, gp)) {
      gl_.reset();
      return;
    }
    gl_s0_ = 0.0;
    for (int j = 0; j < kNumModes; ++j) gl_s0_ += committed_.mu[j] * committed_.x[j](kS, 0);
  }
  double s = 0.0;
  for (int j = 0; j < kNumModes; ++j) s += committed_.mu[j] * committed_.x[j](kS, 0);
  gl_->step(toSec(committed_.t), s - gl_s0_, combinedV(committed_), committed_.standstill,
            committed_.have_cmd ? committed_.notch : 0);
  if (gl_->fixed()) handoverGlobal();
}

void Estimator::handoverGlobal() {
  const GlobalLocalizer::Estimate e = gl_->fix();
  double s_now = 0.0;
  for (int j = 0; j < kNumModes; ++j) s_now += committed_.mu[j] * committed_.x[j](kS, 0);
  static const bool dbg = std::getenv("TBO_DEBUG_GL") != nullptr;
  if (dbg)
    std::fprintf(stderr, "GL fix t=%.1f s_map=%.2f kappa=%.4f+-%.4f conf=%.4f cues=%d s_var=%.2f odo=%.1f\n",
                 toSec(committed_.t), e.s, e.kappa, e.kappa_sd, e.conf, e.cues, e.s_var, gl_->lastOdometer());
  // the fix is the current place of antenna 1 on the main cycle; from now on the map is used as with GNSS
  init_.anchored = true;
  init_.map_matched = true;
  init_.prefix = nullptr;
  init_.s_offset = e.s - s_now;
  init_.match_dist = 0.0;
  if (!init_.origin_set) {  // no GNSS origin: the map origin defines the ENU / UTM output frames
    init_.origin = map_->origin();
    init_.origin_set = true;
    configureFrame();
  }
  // Wheel scale: a few cues rarely pin kappa; a well-determined one is taken over, otherwise the
  // prior stays and the landmarks calibrate it (a wrong tight kappa drifted 3 % in dd8d0741).
  const bool use_kappa = e.kappa_sd < 0.003;
  for (int j = 0; j < kNumModes; ++j) {
    StateVec& x = committed_.x[j];
    StateCov& P = committed_.P[j];
    const double k_new = use_kappa ? e.kappa : x(kK, 0);
    x(kV, 0) *= (1.0 + x(kK, 0)) / (1.0 + k_new);
    x(kK, 0) = k_new;
    for (int i = 0; i < kNx; ++i) {
      P(kK, i) = 0.0;
      P(i, kK) = 0.0;
    }
    P(kK, kK) = use_kappa ? std::max(e.kappa_sd * e.kappa_sd, 1e-6) : p_.init_sigma_scale * p_.init_sigma_scale;
    P(kS, kS) += e.s_var + 1.0;
  }
  committed_.lm_odo = committed_.odo;  // the global fix is a place fix: landmark association starts afresh
  init_.var_applied = true;
  ++diag_.global_fixes;
  gl_.reset();
}

double Estimator::distanceAt(Stamp t) const {
  if (s_hist_.empty() || t >= s_hist_.back().first) {  // inside the lag window: replay the buffer
    const Output o = query(t);
    return o.valid ? o.s : 0.0;
  }
  if (t <= s_hist_.front().first) return s_hist_.front().second;
  const auto it = std::lower_bound(s_hist_.begin(), s_hist_.end(), t,
                                   [](const std::pair<Stamp, double>& a, Stamp b) { return a.first < b; });
  const auto& b = *it;
  const auto& a = *(it - 1);
  const double w = toSec(t - a.first) / std::max(1e-9, toSec(b.first - a.first));
  return a.second + w * (b.second - a.second);
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
  if (f.stub) {  // moved on beyond the stub end (or back before its start): it was not the stub after all
    double s = 0.0;
    for (int j = 0; j < kNumModes; ++j) s += f.mu[j] * f.x[j](kS, 0);
    if (s - f.stub_s0 > stub_.length() + 10.0 || s < f.stub_s0 - 5.0) f.stub = false;
  }
  if (e.is_cmd) {
    if (p_.cutoff_enable > 0.5 && !f.stub && f.have_cmd && e.notch == 0 && f.notch >= p_.cutoff_notch &&
        !cutoffs_.empty() && combinedV(f) > p_.cutoff_min_v) {
      if (placeUpdate(f, e.t, cutoffs_, p_.cutoff_p_random, p_.position_lead_s)) f.lm_t = e.t;
    }
    f.notch = e.notch;
    f.have_cmd = true;
    f.last_cmd = e.t;
    return;
  }
  wheelUpdate(f, e);
  stubRoughness(f, e);
  quantFuse(f);
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
    if (p_.standstill_exit_no_wheels > 0.5 && f.standstill && at > 0.1) {
      // standstill is only cleared by a bogie reading above threshold: with both bogies silent the
      // tram would stay pinned at 0 m/s under traction. Let the model drive once both are out.
      const Stamp to = fromSec(p_.wheel_timeout_s);
      bool silent = true;
      for (int i = 0; i < 2; ++i)
        if (f.wheel[i].have && t_step - f.wheel[i].t <= to) silent = false;
      if (silent) {
        f.standstill = false;
        f.still_since = -1;
        f.lm_done = false;
        for (int j = 0; j < kNumModes; ++j)
          pinVelocity(f.x[j], f.P[j], 0.0, p_.init_sigma_v * p_.init_sigma_v);
      }
    }
    f.a_target = at;
    f.a_drive += alpha * (at - f.a_drive);
    const double a_ext = trackAccel(f);
    double da_ds = 0.0;  // sensitivity of the map acceleration to position (grade changes along s)
    if (p_.grade_s_coupling > 0.5 && !f.standstill) {
      constexpr double kDs = 5.0;
      da_ds = (trackAccel(f, kDs) - trackAccel(f, -kDs)) / (2.0 * kDs);
    }
    for (int j = 0; j < kNumModes; ++j)
      predictMode(f.x[j], f.P[j], f.a_drive, a_ext, h, f.standstill,
                  j == kModeManeuver ? p_.sigma_accel_maneuver : p_.sigma_accel,
                  j == kModeManeuver ? p_.q_disturbance_maneuver : p_.q_disturbance, da_ds);
    f.odo += combinedV(f) * h;
  }
  f.t = t;
}

bool Estimator::stubCheck(FilterState& f) const {
  if (f.stub) return true;
  if (!has_stub_ || p_.stub_enable < 0.5 || !map_ || !init_.anchored) return false;
  double s = 0.0;
  for (int j = 0; j < kNumModes; ++j) s += f.mu[j] * f.x[j](kS, 0);
  const TrackMap* m = nullptr;
  double sm = 0.0;
  if (!routeAt(s, m, sm) || m != map_) return false;
  double d = sm - stub_.joinS();  // antenna path past the stub start
  if (map_->cyclic()) {
    const double L = map_->length();
    d = std::fmod(d, L);
    if (d > 0.5 * L) d -= L;
    if (d < -0.5 * L) d += L;
  }
  static const bool dbg = std::getenv("TBO_DEBUG_STUB") != nullptr;
  if (dbg) std::fprintf(stderr, "STUB t=%.1f s_map=%.1f past_start=%.1f\n", toSec(f.t), sm, d);
  if (d < p_.stub_stop_min_m || d > p_.stub_stop_max_m) return false;
  f.stub = true;
  f.stub_s0 = s - d;
  return true;
}

void Estimator::stubRoughness(FilterState& f, const Event& e) const {
  const bool both = e.has[0] && e.has[1] && std::isfinite(e.z[0]) && std::isfinite(e.z[1]);
  if (both && e.z[0] > 3.0 && e.z[1] > 3.0) {  // slow bogie noise level while moving
    const double dz = e.z[0] - e.z[1];
    if (f.noise_slow < 0.0) {
      f.noise_slow = dz * dz;
    } else {
      const double dt = f.t_noise_slow >= 0 ? std::clamp(toSec(e.t - f.t_noise_slow), 0.0, 1.0) : 0.0;
      f.noise_slow += (1.0 - std::exp(-dt / 60.0)) * (dz * dz - f.noise_slow);
    }
    f.t_noise_slow = e.t;
  }
  if (!has_stub_ || p_.stub_enable < 0.5 || p_.stub_rough_min <= 0.0 || !map_ || !init_.anchored || f.stub) return;
  double s = 0.0;
  for (int j = 0; j < kNumModes; ++j) s += f.mu[j] * f.x[j](kS, 0);
  const TrackMap* m = nullptr;
  double sm = 0.0;
  if (!routeAt(s, m, sm) || m != map_) return;
  double d = sm - stub_.joinS();
  if (map_->cyclic()) {
    const double L = map_->length();
    d = std::fmod(d, L);
    if (d > 0.5 * L) d -= L;
    if (d < -0.5 * L) d += L;
  }
  if (d < p_.stub_rough_from_m - 5.0 || d > p_.stub_rough_to_m + 100.0) {  // away from the switch
    f.stub_rs = 0.0;
    f.stub_rn = 0;
    f.stub_rdone = false;
    return;
  }
  if (f.stub_rdone || d < p_.stub_rough_from_m) return;
  if (d <= p_.stub_rough_to_m) {
    if (both && e.z[0] > 1.0 && e.z[1] > 1.0) {
      if (f.stub_rn == 0) f.stub_noise0 = f.noise_slow;
      const double y = std::log(e.z[0] / e.z[1]);
      f.stub_rs += y * y;
      ++f.stub_rn;
      // overwhelming evidence decides at once instead of at the window end
      if (p_.stub_rough_early > 0.0 && d >= p_.stub_rough_early_from_m && f.stub_rn >= 20) {
        const double rough = std::sqrt(f.stub_rs / f.stub_rn);
        const bool quiet = f.stub_noise0 >= 0.0 && f.stub_noise0 < p_.stub_rough_max_noise * p_.stub_rough_max_noise;
        if (quiet && rough > p_.stub_rough_early) {
          static const bool dbg = std::getenv("TBO_DEBUG_STUB") != nullptr;
          if (dbg) std::fprintf(stderr, "STUBR early t=%.1f d=%.1f rough=%.4f n=%d\n", toSec(e.t), d, rough, f.stub_rn);
          f.stub_rdone = true;
          f.stub = true;
          f.stub_s0 = s - d;
        }
      }
    }
    return;
  }
  f.stub_rdone = true;  // passed the window: decide once
  const double rough = f.stub_rn > 0 ? std::sqrt(f.stub_rs / f.stub_rn) : 0.0;
  const bool quiet = f.stub_noise0 >= 0.0 && f.stub_noise0 < p_.stub_rough_max_noise * p_.stub_rough_max_noise;
  static const bool dbg = std::getenv("TBO_DEBUG_STUB") != nullptr;
  if (dbg)
    std::fprintf(stderr, "STUBR t=%.1f rough=%.4f n=%d noise=%.4f\n", toSec(e.t), rough, f.stub_rn,
                 std::sqrt(std::max(f.stub_noise0, 0.0)));
  if (f.stub_rn >= 20 && quiet && rough > p_.stub_rough_min) {
    f.stub = true;
    f.stub_s0 = s - d;
  }
}

double Estimator::quantStep(int b) const {
  const std::vector<double>& st = quant_[b].steps;
  if (quant_cache_n_[b] == st.size() && st.size() > 0) return quant_cache_q_[b];
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double lo = p_.quant_step_lo_kmh, hi = p_.quant_step_hi_kmh;
  double q = nan;
  if (static_cast<double>(st.size()) >= p_.quant_min_steps && hi > lo) {
    // modal 2e-6 km/h bin, then the median of the steps within the tolerance of it
    constexpr double kBin = 2e-6;
    constexpr int kMaxBins = 2048;
    const int nb = std::min(kMaxBins, static_cast<int>((hi - lo) / kBin) + 1);
    int hist[kMaxBins] = {0};
    for (double v : st) {
      const int bi = static_cast<int>((v - lo) / kBin);
      if (bi >= 0 && bi < nb) ++hist[bi];
    }
    int best = 0;
    for (int bi = 1; bi < nb; ++bi)
      if (hist[bi] > hist[best]) best = bi;
    const double mode = lo + (best + 0.5) * kBin;
    quant_scratch_.clear();
    for (double v : st)
      if (std::abs(v - mode) < p_.quant_step_tol_kmh) quant_scratch_.push_back(v);
    const double n = static_cast<double>(quant_scratch_.size());
    if (n >= p_.quant_min_steps && n >= p_.quant_min_share * static_cast<double>(st.size())) {
      const auto mid = quant_scratch_.begin() + quant_scratch_.size() / 2;
      std::nth_element(quant_scratch_.begin(), mid, quant_scratch_.end());
      q = *mid;
    }
  }
  quant_cache_n_[b] = st.size();
  quant_cache_q_[b] = q;
  return q;
}

void Estimator::quantFuse(FilterState& f) const {
  if (f.kq_done || p_.quant_k_enable < 0.5 || epochs_.empty() || !init_.have_first) return;
  const double qf = quantStep(0), qr = quantStep(1);
  if (!std::isfinite(qf) || !std::isfinite(qr)) return;
  const int date = dateMsk(init_.t_first);
  double k = 0.0, pkk = 0.0;
  for (int j = 0; j < kNumModes; ++j) k += f.mu[j] * f.x[j](kK, 0);
  for (int j = 0; j < kNumModes; ++j) {
    const double dk = f.x[j](kK, 0) - k;
    pkk += f.mu[j] * (f.P[j](kK, kK) + dk * dk);
  }
  const double r = p_.quant_k_sigma * p_.quant_k_sigma;
  const double sd = std::sqrt(std::max(pkk, 0.0) + r);
  // candidates: per vehicle the latest wheel epoch not later than the run date
  double z1 = std::numeric_limits<double>::infinity(), z2 = z1, k1 = 0.0;
  for (const WheelEpoch& e : epochs_) {
    if (e.date > date) continue;
    bool latest = true;
    for (const WheelEpoch& o : epochs_)
      if (o.vehicle == e.vehicle && o.date <= date && o.date > e.date) latest = false;
    if (!latest) continue;
    const double kq = 0.5 * (qf / e.c_front + qr / e.c_rear) - 1.0;
    if (!(std::abs(kq) <= p_.quant_k_max)) continue;  // e.g. wheels turned since: an unknown epoch
    const double z = std::abs(kq - k) / sd;
    if (z < z1) {
      z2 = z1;
      z1 = z;
      k1 = kq;
    } else if (z < z2) {
      z2 = z;
    }
  }
  // only the candidate consistent with the filter's k, and no other one near it
  if (!(z1 < p_.quant_k_gate_sd) || z2 < z1 + p_.quant_k_ambig_sd) return;
  static const bool dbg = std::getenv("TBO_DEBUG_QK") != nullptr;
  if (dbg)
    std::fprintf(stderr, "QK t=%.1f date=%d q=%.7f/%.7f k=%.5f sd=%.5f -> k_q=%.5f (z %.2f, next %.2f)\n",
                 toSec(f.t), date, qf, qr, k, sd, k1, z1, z2);
  for (int j = 0; j < kNumModes; ++j) {  // one scalar measurement of k on every mode
    StateVec& x = f.x[j];
    StateCov& P = f.P[j];
    const double S = P(kK, kK) + r;
    if (!(S > 0.0)) continue;
    StateVec K;
    for (int i = 0; i < kNx; ++i) K(i, 0) = P(i, kK) / S;
    x += K * (k1 - x(kK, 0));
    Mat<1, kNx> H;
    H(0, kK) = 1.0;
    const StateCov IKH = StateCov::identity() - K * H;
    Mat<1, 1> R;
    R(0, 0) = r;
    P = IKH * P * transpose(IKH) + K * R * transpose(K);
    symmetrize(P);
    x(kK, 0) = std::clamp(x(kK, 0), -0.05, 0.05);
    if (!(x(kV, 0) >= 0.0)) x(kV, 0) = 0.0;
  }
  f.kq_done = true;
  f.kq = k1;
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

double Estimator::trackAccel(const FilterState& f, double ds) const {
  double s = ds, v = 0.0;
  for (int j = 0; j < kNumModes; ++j) {
    s += f.mu[j] * f.x[j](kS, 0);
    v += f.mu[j] * f.x[j](kV, 0);
  }
  // grade and curvature averaged over the car body (the mass is spread along ~16.5 m)
  constexpr int kSamples = 5;
  double grade = 0.0, curv = 0.0;
  int n = 0;
  for (int q = 0; q < kSamples; ++q) {
    const double off = p_.body_rear_m + (p_.body_front_m - p_.body_rear_m) * q / (kSamples - 1);
    const TrackMap* m = nullptr;
    double sm = 0.0;
    if (!routeAt(s + off, m, sm) || !m->hasProfile()) continue;
    grade += m->gradeAt(sm);
    curv += std::abs(m->curvatureAt(sm));
    ++n;
  }
  if (n == 0) return 0.0;
  grade /= n;
  curv /= n;
  const double kg = f.notch < 0 ? p_.kg_brake : (f.notch > 0 ? p_.kg_traction : p_.kg_coast);
  double a = -kg * p_.map_grade_gain * grade;
  if (p_.curve_resist_coef > 0.0 && v > 0.1) a -= p_.curve_resist_coef * curv;
  if (!dfield_.empty() && p_.dfield_gain != 0.0 && v > 0.5) {  // learned field: main cycle, moving
    const TrackMap* m = nullptr;
    double sm = 0.0;
    if (routeAt(s, m, sm) && m == map_) a += p_.dfield_gain * dfield_.at(sm);
  }
  return a;
}

void Estimator::predictMode(StateVec& x, StateCov& P, double a, double a_ext, double h,
                            bool standstill, double sigma_accel, double q_d, double da_ds) const {
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
  F(kV, kS) = da_ds * h;             // grade seen through the dynamics (0 unless enabled)
  F(kS, kS) += 0.5 * da_ds * h * h;
  P = F * P * transpose(F);
  const double qa = sigma_accel * sigma_accel;
  P(kS, kS) += qa * h * h * h / 3.0;
  P(kS, kS) += p_.q_along * std::abs(ds);  // odometry error that varies from place to place
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
  bool newly_stuck[2] = {false, false};
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
          std::abs(vc - w.v_at_same) >= p_.stuck_min_change) {
        if (!w.stuck) newly_stuck[i] = true;
        w.stuck = true;
      }
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
    const double along[2] = {p_.front_bogie_along_m, p_.rear_bogie_along_m};
    for (int i = 0; i < 2; ++i) {
      const TrackMap* m = nullptr;
      double sm = 0.0;
      if (!routeAt(s + along[i], m, sm) || !m->hasProfile()) continue;
      const double k = m->curvatureAt(sm);
      z[i] *= 1.0 + std::min(p_.wheel_curv_abs * std::abs(k), p_.wheel_curv_sat) + p_.wheel_curv_signed * k;
    }
  }

  // stuck_reset: a bogie was just proven frozen. While it froze the filter may have followed it and
  // learnt a disturbance explaining its missing acceleration, and the IMM may be rejecting the healthy
  // bogie. Re-anchor to the healthy one (if plausible) and drop the learnt excess disturbance.
  if (p_.stuck_reset > 0.5)
    for (int i = 0; i < 2; ++i) {
      const int o = 1 - i;
      if (!newly_stuck[i] || !avail[o] || implausible[o]) continue;
      for (int j = 0; j < kNumModes; ++j) {
        pinVelocity(f.x[j], f.P[j], z[o] / (1.0 + f.x[j](kK, 0)), p_.sigma_wheel * p_.sigma_wheel);
        f.x[j](kD, 0) = std::clamp(f.x[j](kD, 0), -p_.disturbance_max, p_.disturbance_max);
      }
      for (int j = 0; j < kNumModes; ++j) f.mu[j] = 0.01;
      f.mu[kModeNominal] = 1.0 - 0.01 * (kNumModes - 1);
      f.recovered_t = e.t;
      f.bad_since = f.agree_since = -1;
    }

  // ---- zero-velocity (standstill) detection ----
  const double thr = p_.standstill_kmh * p_.wheel_kmh_to_ms;
  bool all_low = true;
  for (int i = 0; i < 2; ++i)
    if (avail[i] && z[i] > thr) all_low = false;
  if (p_.agree_tau_s > 0.0)  // low-passed plausible bogie speeds for the recovery agreement test
    for (int i = 0; i < 2; ++i)
      if (avail[i] && !implausible[i]) {
        const double gap = f.t_ema[i] >= 0 ? toSec(e.t - f.t_ema[i]) : 1e9;
        if (gap > 1.0) f.ema_z[i] = z[i];
        else f.ema_z[i] += (1.0 - std::exp(-gap / p_.agree_tau_s)) * (z[i] - f.ema_z[i]);
        f.t_ema[i] = e.t;
      }
  bool zero_long = false;  // lockup_guard_s: bogies at ~0 far longer than any wheel lock lasts
  if (p_.lockup_guard_s > 0.0) {
    if (all_low) {
      if (f.zero_since < 0) f.zero_since = e.t;
      zero_long = toSec(e.t - f.zero_since) >= p_.lockup_guard_s;
    } else {
      f.zero_since = -1;
    }
  }
  if (all_low && (vc < p_.standstill_max_v || zero_long)) {
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
      if (!stubCheck(f) && p_.landmark_enable > 0.5 && placeUpdate(f, e.t, landmarks_, p_.landmark_p_random, 0.0))
        f.lm_t = e.t;
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
  double boost = effort ? p_.slip_context_boost : 1.0;
  if (!fault_prior_.empty()) {  // known low-adhesion places: faults are a priori more likely there
    double s = 0.0;
    for (int j = 0; j < kNumModes; ++j) s += f.mu[j] * f.x[j](kS, 0);
    const TrackMap* m = nullptr;
    double sm = 0.0;
    if (routeAt(s, m, sm) && m == map_) boost *= std::clamp(fault_prior_.at(sm), 1.0, p_.fault_prior_max);
  }
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
  if (p_.d_clamp_model_only > 0.5 && f.mu[kModeBothBad] > 0.5)  // wheels distrusted: no wheel-learnt
    xbar(kD, 0) = std::clamp(xbar(kD, 0), -p_.disturbance_max, p_.disturbance_max);  // excess accel
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
  bool agree = avail[0] && avail[1] && std::abs(z[0] - z[1]) < p_.recover_agree &&
               !implausible[0] && !implausible[1];
  if (p_.agree_tau_s > 0.0)  // noisy bogies: compare low-passed speeds instead of single samples
    agree = avail[0] && avail[1] && f.t_ema[0] >= 0 && f.t_ema[1] >= 0 && toSec(e.t - f.t_ema[0]) < 0.5 &&
            toSec(e.t - f.t_ema[1]) < 0.5 && std::abs(f.ema_z[0] - f.ema_z[1]) < p_.recover_agree;
  if (agree) {
    if (f.agree_since < 0) f.agree_since = e.t;
  } else {
    f.agree_since = -1;
  }
  const bool zero_ok = p_.lockup_guard_s > 0.0 && f.zero_since >= 0 &&
                       toSec(e.t - f.zero_since) >= p_.lockup_guard_s;
  if (f.bad_since >= 0 && f.agree_since >= 0 && toSec(e.t - f.bad_since) >= p_.recover_min_bad_s &&
      toSec(e.t - f.agree_since) >= p_.recover_time_s) {
    const double zm = 0.5 * (z[0] + z[1]);
    const double vnow = combinedV(f);
    // never re-anchor onto wheels locked near zero while the model says we still move
    // (lockup_guard_s: unless they have read ~0 for longer than any lock lasts)
    if (zm > 2.0 * thr || vnow < p_.standstill_max_v || zero_ok) {
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
  // single_bogie_recover: the other bogie is out (dropout / stuck) and the only one left has been
  // rejected for recover_min_bad_s + recover_time_s while reading smoothly -> re-anchor to it
  if (p_.single_bogie_recover > 0.5) {
    const int n_av = (avail[0] ? 1 : 0) + (avail[1] ? 1 : 0);
    bool armed = false;
    if (n_av == 1) {
      const int i = avail[0] ? 0 : 1;
      const WheelTrack& wo = f.wheel[1 - i];
      const bool other_out = !wo.have || toSec(e.t - wo.t) > p_.wheel_timeout_s || wo.stuck;
      const double bad_i = f.mu[i == 0 ? kModeFrontBad : kModeRearBad] + f.mu[kModeBothBad];
      if (other_out && bad_i > 0.5 && !implausible[i]) {
        armed = true;
        if (f.single_since < 0) f.single_since = e.t;
        const double vnow = combinedV(f);
        if (toSec(e.t - f.single_since) >= p_.recover_min_bad_s + p_.recover_time_s &&
            (z[i] > 2.0 * thr || vnow < p_.standstill_max_v || zero_ok)) {
          for (int j = 0; j < kNumModes; ++j)
            pinVelocity(f.x[j], f.P[j], z[i] / (1.0 + f.x[j](kK, 0)), p_.sigma_wheel * p_.sigma_wheel);
          for (int j = 0; j < kNumModes; ++j) f.mu[j] = 0.01;
          f.mu[kModeNominal] = 1.0 - 0.01 * (kNumModes - 1);
          f.recovered_t = e.t;
          f.single_since = -1;
          armed = false;
        }
      }
    }
    if (!armed) f.single_since = -1;
  }
}

bool Estimator::jointMonitor(FilterState& f, const Event& e, const bool* avail, const double* z) const {
  StateVec xm;
  for (int j = 0; j < kNumModes; ++j) xm += f.mu[j] * f.x[j];
  const double vc = std::max(0.0, xm(kV, 0));
  const double k = xm(kK, 0);
  // Reference: the controller model with a grade-sized disturbance only. A large negative d
  // learned during an unmodelled brake must not make normal driving look like a slip.
  double d_src = xm(kD, 0);
  if (p_.joint_d_tau_s > 0.0) {
    // the filter's d absorbs a slowly rising joint slip within ~0.5 s (it follows the wheels), which
    // hides ramps from the CUSUM: reference a slow copy of d, frozen while any monitor is active
    const bool quiet = !f.latch && f.mu[kModeNominal] > 0.9 && f.wheel[0].cusum_pos <= 0.0 &&
                       f.wheel[0].cusum_neg <= 0.0 && f.wheel[1].cusum_pos <= 0.0 && f.wheel[1].cusum_neg <= 0.0;
    if (!f.d_slow_init) {
      f.d_slow = d_src;
      f.d_slow_init = true;
    } else if (quiet) {
      const double dt = std::clamp(toSec(e.t - f.t_dslow), 0.0, 1.0);
      f.d_slow += (1.0 - std::exp(-dt / p_.joint_d_tau_s)) * (d_src - f.d_slow);
    }
    f.t_dslow = e.t;
    d_src = f.d_slow;
  }
  const double d_ref = std::clamp(d_src, -p_.disturbance_max, p_.disturbance_max);
  // same terms as the filter dynamics: drive + disturbance + map (grade, learned field); without the
  // grade a 3 % slope alone used up to half of the CUSUM allowance
  const double a_map = trackAccel(f);
  const double a_model = xm(kG, 0) * f.a_drive + d_ref + a_map;
  // Slide reference: the brake table error changes by ~1 m/s^2 within one braking (d from +0.7 to
  // -0.3 as the brake builds up), so the slow copy lags and fakes a joint slide; braking therefore
  // uses the filter's own disturbance within its physical bounds.
  double d_brake = xm(kD, 0);
  if (p_.slide_ref_tau_s > 0.0) {  // smooth heavy wheel noise; freeze while a slide builds up
    const bool slide_quiet = !f.latch && f.wheel[0].cusum_neg <= 0.0 && f.wheel[1].cusum_neg <= 0.0;
    if (!f.d_med_init) {
      f.d_med = d_brake;
      f.d_med_init = true;
    } else if (slide_quiet) {
      const double dt = f.t_dmed >= 0 ? std::clamp(toSec(e.t - f.t_dmed), 0.0, 1.0) : 0.0;
      f.d_med += (1.0 - std::exp(-dt / p_.slide_ref_tau_s)) * (d_brake - f.d_med);
    }
    f.t_dmed = e.t;
    d_brake = f.d_med;
  }
  if (avail[0] && avail[1]) {  // bogie noise level from the front-rear difference (common motion cancels)
    const double dz = z[0] - z[1];
    const double dt = f.t_noise >= 0 ? std::clamp(toSec(e.t - f.t_noise), 0.0, 1.0) : 1.0;
    f.noise_var += (1.0 - std::exp(-dt / 5.0)) * (dz * dz - f.noise_var);
    f.t_noise = e.t;
  }
  const bool fast_ref = p_.slide_ref_fast > 0.5 &&
                        f.noise_var < p_.slide_fast_max_noise * p_.slide_fast_max_noise;
  const double d_fast =
      fast_ref ? std::clamp(d_brake, -p_.disturbance_max_decel, p_.disturbance_max_free) : d_ref;
  const double a_model_slide = xm(kG, 0) * f.a_drive + d_fast + a_map;
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
      f.latch_end_t = e.t;
      setNominal();
      resetMonitor();
      f.snap_t = -1;
      return false;  // this sample updates the filter normally
    }
    const double latch_max = f.latch_sign < 0 ? p_.slide_latch_max_s : p_.latch_max_s;
    if (toSec(e.t - f.latch_t) > latch_max) {  // model drift now dominates: re-anchor
      const bool both = avail[0] && avail[1];
      if ((both && std::abs(z[0] - z[1]) < p_.recover_agree) || (avail[0] != avail[1])) {
        const double zm = both ? 0.5 * (z[0] + z[1]) : (avail[0] ? z[0] : z[1]);
        for (int j = 0; j < kNumModes; ++j)
          pinVelocity(f.x[j], f.P[j], zm / (1.0 + f.x[j](kK, 0)), p_.sigma_wheel * p_.sigma_wheel);
        f.latch = false;
        f.latch_end_t = e.t;
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
    w.hams[slot] = a_model_slide;
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
      const double a_wheel = (z[i] - w.hz[ref]) / age;
      const double excess = a_wheel - 0.5 * (a_model + w.ham[ref]);
      const double excess_slide = a_wheel - 0.5 * (a_model_slide + w.hams[ref]);
      excess_sum += excess;
      ++n_excess;
      w.cusum_pos = traction ? std::max(0.0, w.cusum_pos + (excess - p_.cusum_slip_accel) * dtc) : 0.0;
      w.cusum_neg = braking ? std::max(0.0, w.cusum_neg + (-excess_slide - p_.cusum_slide_accel) * dtc) : 0.0;
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
    if (p_.cmd_check_absolute > 0.5) {
      // (1) under a braking notch only an actual speed-up is impossible (a weaker-than-tabulated
      //     brake near a stop is not); (2) no evidence while a bogie is distrusted or right after a
      //     joint latch: a wheel spinning back up after a slide is not the controller's fault
      const double bad = f.mu[kModeFrontBad] + f.mu[kModeRearBad] + f.mu[kModeBothBad];
      const bool pause = f.latch || bad > 0.5 || (f.latch_end_t >= 0 && toSec(e.t - f.latch_end_t) < 3.0);
      if (!pause) {
        if (f.notch < 0) impossible = ex + a_model;        // absolute bogie acceleration
        else if (f.notch == 0) impossible = ex;
        else if (f.a_target > 0.1) impossible = -ex - 1.0;
      }
    } else if (f.notch <= 0) {
      impossible = ex;                                   // speeding up with no traction
    } else if (f.a_target > 0.1) {
      impossible = -ex - 1.0;                            // strong braking under traction
    }
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
      f.snap_d = braking ? d_fast : d_ref;
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
    const double a = f.onset_g * f.a_drive + f.onset_d + a_map;
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
  bool joint = alarms == n_av;
  // near a stop a joint slide is harmless (< 1 m) but a false one latches the model through the stop
  if (joint && sign < 0 && vc < p_.slide_latch_min_v) joint = false;
  if (joint && n_av == 1 && p_.joint_need_both > 0.5) {
    // A lone alarm is "joint" only if the other bogie is really out. Otherwise its sample of the
    // same stamp pair has merely not arrived yet (query() between the W0 and W1 messages), and
    // latching would publish a rolled-back model speed exactly at a wheel stamp.
    const WheelTrack& wo = f.wheel[avail[0] ? 1 : 0];
    const bool other_out = !wo.have || toSec(e.t - wo.t) > p_.wheel_timeout_s || wo.stuck;
    if (!other_out) joint = false;
  }
  if (joint) {
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

bool Estimator::placeUpdate(FilterState& f, Stamp t, const std::vector<Landmark>& places,
                            double p_random, double lead) const {
  if (places.empty() || !map_) return false;
  StateVec xm;
  for (int j = 0; j < kNumModes; ++j) xm += f.mu[j] * f.x[j];
  StateCov Pm;
  for (int j = 0; j < kNumModes; ++j) {
    const StateVec dx = f.x[j] - xm;
    Pm += f.mu[j] * (f.P[j] + dx * transpose(dx));
  }
  const TrackMap* m = nullptr;
  double sm = 0.0;
  if (!routeAt(xm(kS, 0), m, sm) || m != map_) return false;  // places exist on the main cycle only
  sm = map_->wrap(sm + std::max(0.0, xm(kV, 0)) * lead);
  const double L = map_->length();
  const double extra2 = p_.landmark_sigma_extra * p_.landmark_sigma_extra;
  // Association only: odometry drifts from place to place by up to ~0.5 % between fixes (5-95 % of
  // train segments: -0.41..+0.48 %), more than the filter's own along-track variance says. A right
  // place 1.8 m away after 320 m was rejected (d927f360). The update itself keeps the filter variance.
  const double var_s = std::max(Pm(kS, kS), 0.0) + p_.landmark_assoc_q * std::max(f.odo - f.lm_odo, 0.0);
  const double g = p_.landmark_gate_sigma;
  // candidates inside the gate: probabilistic data association (PDA) over all of them plus the
  // "not a known place" hypothesis, so two close places (e.g. 6 m apart) blend instead of
  // snapping to one and collapsing the along-track variance onto a possibly wrong place
  constexpr int kMaxCand = 8;
  double c_delta[kMaxCand], c_r[kMaxCand], c_w[kMaxCand];
  int nc = 0;
  double sum_w = 0.0;
  for (const Landmark& l : places) {
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
    if (nc < kMaxCand) {
      c_delta[nc] = delta;
      c_r[nc] = r;
      c_w[nc] = w;
      ++nc;
      sum_w += w;
    }
  }
  const double width = 2.0 * g * std::sqrt(var_s + extra2 + 0.25);
  const double w_random = p_random / std::max(width, 1.0);
  const double p_known = nc > 0 ? sum_w / (sum_w + w_random) : 0.0;
  static const bool dbg = std::getenv("TBO_DEBUG_LM") != nullptr;
  if (dbg)
    std::fprintf(stderr, "LM t=%.1f s_map=%.1f sd=%.2f n=%d d0=%.2f p_known=%.2f k=%.4f\n", toSec(t), sm,
                 std::sqrt(var_s), nc, nc > 0 ? c_delta[0] : 0.0, p_known, xm(kK, 0));
  if (nc == 0 || p_known < p_.landmark_min_prob) return false;
  // association probabilities (beta_0: a stop away from any known place)
  double beta[kMaxCand], r_bar = 0.0;
  for (int i = 0; i < nc; ++i) {
    beta[i] = c_w[i] / (sum_w + w_random);
    r_bar += beta[i] * c_r[i];
  }
  const double beta0 = w_random / (sum_w + w_random);
  r_bar /= (1.0 - beta0);
  for (int j = 0; j < kNumModes; ++j) {
    StateVec& x = f.x[j];
    StateCov& P = f.P[j];
    const double S = P(kS, kS) + r_bar;
    if (!(S > 0.0)) continue;
    StateVec K;
    // a landmark fixes position and (through the s-k correlation) the wheel scale only
    K(kS, 0) = P(kS, kS) / S;
    K(kK, 0) = P(kK, kS) / S;
    double nu = 0.0, nu2 = 0.0;  // combined innovation and its spread over the hypotheses
    for (int i = 0; i < nc; ++i) {
      const double ni = xm(kS, 0) + c_delta[i] - x(kS, 0);
      nu += beta[i] * ni;
      nu2 += beta[i] * ni * ni;
    }
    const double dk = K(kK, 0) * nu;
    if (std::abs(dk) > p_.landmark_max_dk && std::abs(K(kK, 0)) > 0.0) {
      // inflate the scale-position coupling so one fix cannot rewrite the wheel calibration
      const double shrink = p_.landmark_max_dk / std::abs(dk);
      K(kK, 0) *= shrink;
    }
    x += K * nu;
    Mat<1, kNx> H;
    H(0, kS) = 1.0;
    const StateCov IKH = StateCov::identity() - K * H;
    Mat<1, 1> R;
    R(0, 0) = r_bar;
    const StateCov Pc = IKH * P * transpose(IKH) + K * R * transpose(K);  // correct association
    // PDA: keep the prior with beta0, add the spread of the candidate innovations
    P = beta0 * P + (1.0 - beta0) * Pc + (nu2 - nu * nu) * (K * transpose(K));
    symmetrize(P);
    x(kK, 0) = std::clamp(x(kK, 0), -0.05, 0.05);
  }
  f.lm_odo = f.odo;
  return true;
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
  // a fix stamped far from the input time base is a glitch (GNSS headers jump by +-1 s at most)
  if (started_ && std::llabs(stamp - latest_) > fromSec(p_.max_future_s)) return;
  if (!init_.have_first) {
    init_.have_first = true;
    init_.t_first = stamp;
  }
  // The window counts from the first fix, not from the first message: the start-up burst
  // delivers wheel/controller history up to ~6 s old, and GNSS may start after the wheels.
  if (!init_.have_gnss) {
    init_.have_gnss = true;
    init_.t_first_gnss = stamp;
  }
  if (stamp > init_.t_first_gnss + fromSec(p_.gnss_init_window_s) || gnssWindowClosed()) {
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
  if (cfg_.output_frame == "mgrs") {  // Autoware MGRS map frame, one fixed 100 km square
    const geo::Geodetic g = map_lc_.reverse({mx, my, mz});
    const geo::Utm u = geo::geodeticToUtm(g.lat_deg, g.lon_deg, static_cast<int>(p_.mgrs_zone));
    ox = u.easting - p_.mgrs_origin_e;
    oy = u.northing - p_.mgrs_origin_n;
    oz = g.h - p_.base_link_height_m;
    return;
  }
  if (cfg_.output_frame == "utm") {
    const geo::Geodetic g = map_lc_.reverse({mx, my, mz});
    const geo::Utm o = geo::geodeticToUtm(init_.origin.lat_deg, init_.origin.lon_deg);
    const geo::Utm u = geo::geodeticToUtm(g.lat_deg, g.lon_deg, o.zone);
    ox = u.easting - o.easting;
    oy = u.northing - o.northing;
    oz = g.h - init_.origin.h - p_.base_link_height_m;
    return;
  }
  ox = rot_[0][0] * mx + rot_[0][1] * my + rot_[0][2] * mz + trans_[0];
  oy = rot_[1][0] * mx + rot_[1][1] * my + rot_[1][2] * mz + trans_[1];
  oz = rot_[2][0] * mx + rot_[2][1] * my + rot_[2][2] * mz + trans_[2] - p_.base_link_height_m;
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
  std::vector<Stamp> ts;
  for (const Fix& fx : init_.fixes) ts.push_back(fx.t);
  std::nth_element(ts.begin(), ts.begin() + static_cast<long>(ts.size() / 2), ts.end());
  const Stamp t_med = ts[ts.size() / 2];

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

  // distance travelled at the median fix time (the tram may already move inside the window)
  const double s_rel = distanceAt(t_med);
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
  if (cfg_.output_frame == "mgrs") {
    const geo::Utm u = geo::geodeticToUtm(med.lat_deg, med.lon_deg, static_cast<int>(p_.mgrs_zone));
    init_.dr_x = u.easting - p_.mgrs_origin_e;
    init_.dr_y = u.northing - p_.mgrs_origin_n;
    init_.dr_z = med.h - p_.base_link_height_m;
  } else if (cfg_.output_frame == "utm") {
    const geo::Utm oz = geo::geodeticToUtm(init_.origin.lat_deg, init_.origin.lon_deg);
    const geo::Utm u = geo::geodeticToUtm(med.lat_deg, med.lon_deg, oz.zone);
    init_.dr_x = u.easting - oz.easting;
    init_.dr_y = u.northing - oz.northing;
    init_.dr_z = med.h - init_.origin.h - p_.base_link_height_m;
  } else {
    const geo::Enu q = out_lc_.forward(med);
    init_.dr_x = q.e;
    init_.dr_y = q.n;
    init_.dr_z = q.u - p_.base_link_height_m;
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
  o.v_var = std::max(p_.speed_var_scale * Pm(kV, kV), p_.speed_var_floor);
  o.s = xm(kS, 0);
  o.s_var = Pm(kS, kS);
  o.disturbance = xm(kD, 0);
  o.scale = xm(kK, 0);
  o.gain = xm(kG, 0);
  o.a_model = f.a_target;
  o.a_ext = f.started ? trackAccel(f) : 0.0;
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
  // antenna 1 on the map (antenna) path, aligned with the fix timing and then with the judge reference
  const double s_ant = o.s + o.v * (p_.position_lead_s - p_.position_output_delay_s);
  const double s_pub = s_ant + p_.base_link_along_m;
  {
    const TrackMap* ra = nullptr;
    double sa = 0.0;
    if (routeAt(o.s, ra, sa) && ra == map_) o.s_map = sa;
  }
  const bool on_stub = init_.anchored && has_stub_ && f.stub;
  if (on_stub || (init_.anchored && routeAt(s_pub, rm, sm))) {
    // base_link = antenna 1 + base_link_along_m along the car body axis, the line through both
    // antennas (organisers' TF: master x = -9.873, rover x = +2.563 in base_link). The map is the
    // antenna path, which swings ~0.7 m outside the rails in the 16 m loops, so the path point
    // base_link_along_m further on is not the bogie pivot; the body-axis construction is the rigid TF.
    MapPose p;
    double bx = 0.0, by = 0.0, hx = 1.0, hy = 0.0;
    if (on_stub) {  // the same construction on the dead-end stub polyline
      const double da = s_ant - f.stub_s0 + p_.stub_arc_offset_m;
      p = stub_.at(da + p_.base_link_along_m);
      const MapPose A = stub_.at(da), Rv = stub_.at(da + p_.antenna_baseline_m);
      hx = std::cos(A.heading);
      hy = std::sin(A.heading);
      const double dx = Rv.x - A.x, dy = Rv.y - A.y, len = std::hypot(dx, dy);
      if (len > 1.0) {
        hx = dx / len;
        hy = dy / len;
      }
      bx = A.x + p_.base_link_along_m * hx;
      by = A.y + p_.base_link_along_m * hy;
      o.s_map = -1.0;
    } else {
      p = rm->at(sm);
      bx = p.x;
      by = p.y;
      hx = std::cos(p.heading);
      hy = std::sin(p.heading);
      const TrackMap *ra = nullptr, *rr = nullptr;
      double sa = 0.0, sr = 0.0;
      if (p_.antenna_baseline_m > 1.0 && routeAt(s_ant, ra, sa) && routeAt(s_ant + p_.antenna_baseline_m, rr, sr)) {
        const MapPose A = ra->at(sa), Rv = rr->at(sr);
        const double dx = Rv.x - A.x, dy = Rv.y - A.y, len = std::hypot(dx, dy);
        if (len > 1.0) {
          hx = dx / len;
          hy = dy / len;
          bx = A.x + p_.base_link_along_m * hx;
          by = A.y + p_.base_link_along_m * hy;
        }
      }
    }
    double ox, oy, oz, ox2, oy2, oz2;
    mapToOutput(bx, by, p.z, ox, oy, oz);
    mapToOutput(bx + hx, by + hy, p.z, ox2, oy2, oz2);  // body heading in the output frame
    o.x = ox;
    o.y = oy;
    o.z = oz;
    o.yaw = std::atan2(oy2 - oy, ox2 - ox);
    o.map_matched = true;
    const double sc2 = p_.map_sigma_cross * p_.map_sigma_cross;
    const double c = std::cos(o.yaw), s = std::sin(o.yaw);
    o.cov_xx = sig_s2 * c * c + sc2 * s * s;
    o.cov_yy = sig_s2 * s * s + sc2 * c * c;
    o.cov_xy = (sig_s2 - sc2) * s * c;
    o.cov_zz = p_.map_sigma_z * p_.map_sigma_z;
  } else if (init_.anchored) {
    const double yaw = init_.have_yaw ? init_.yaw0 : 0.0;
    const double along = o.s + p_.base_link_along_m;
    o.x = init_.dr_x + along * std::cos(yaw);
    o.y = init_.dr_y + along * std::sin(yaw);
    o.z = init_.dr_z;
    o.yaw = yaw;
    const double cross = 1.0 + 0.05 * std::abs(o.s);  // heading unknown along curves
    o.cov_xx = sig_s2 + cross * cross;
    o.cov_yy = sig_s2 + cross * cross;
    o.cov_zz = 4.0 + (0.01 * o.s) * (0.01 * o.s);
    fl |= kFlagNoMap;
  } else {  // no GNSS at start: relative odometry from the start point (x forward along the track)
    o.x = o.s + p_.base_link_along_m;
    o.y = 0.0;
    o.z = 0.0;
    o.cov_xx = sig_s2;
    o.cov_yy = 1e4;
    o.cov_zz = 1e4;
    fl |= kFlagNotInitialized | kFlagNoMap;
  }
  o.protection_level = p_.protection_k * std::sqrt(std::max(sig_s2, 0.0));
  if (p_.speed_output_delay_s != 0.0 && !f.standstill)  // speed of (stamp - delay): v - a * delay
    o.v = std::max(0.0, o.v - (o.accel + o.a_ext) * p_.speed_output_delay_s);
  o.flags = fl;
  // Without any GNSS: relative odometry only if asked for (in the MGRS frame it is meaningless);
  // otherwise the position waits for the GNSS-free map fix.
  o.pos_valid = init_.anchored || (init_.have_first && t - init_.t_first >= fromSec(p_.gnss_wait_s) &&
                                   (!globalAvailable() || p_.nognss_relative > 0.5));
  return o;
}

}  // namespace tbo
