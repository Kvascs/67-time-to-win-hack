#include "tbo/global_localizer.hpp"

#include <algorithm>
#include <cmath>

#include "tbo/estimator.hpp"  // Landmark

namespace tbo {

namespace {
constexpr double kSqrt2Pi = 2.5066282746310002;

// Adds sum_i w_i N(u; c_i, sd_i^2) to a fine periodic table (spacing df, period n * df).
void addBumps(std::vector<double>& tab, double df, const std::vector<double>& centre,
              const std::vector<double>& sd, const std::vector<double>& w) {
  const long n = static_cast<long>(tab.size());
  for (size_t m = 0; m < centre.size(); ++m) {
    const long c = static_cast<long>(std::llround(centre[m] / df));
    const long half = static_cast<long>(std::ceil(5.0 * sd[m] / df)) + 1;
    for (long k = -half; k <= half; ++k) {
      const double d = (c + k) * df - centre[m];
      long idx = (c + k) % n;
      if (idx < 0) idx += n;
      tab[static_cast<size_t>(idx)] += w[m] * std::exp(-0.5 * d * d / (sd[m] * sd[m])) / (kSqrt2Pi * sd[m]);
    }
  }
}

// Centred moving average over `size` samples on a periodic table, then division by lam.
std::vector<float> boxAverage(const std::vector<double>& tab, int size, double lam) {
  const long n = static_cast<long>(tab.size());
  std::vector<float> out(tab.size());
  const long lo = -(size / 2), hi = lo + size - 1;
  double acc = 0.0;
  for (long k = lo; k <= hi; ++k) acc += tab[static_cast<size_t>(((k % n) + n) % n)];
  for (long i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = static_cast<float>(acc / size / lam);
    acc += tab[static_cast<size_t>(((i + hi + 1) % n + n) % n)] - tab[static_cast<size_t>(((i + lo) % n + n) % n)];
  }
  return out;
}
}  // namespace

bool GlobalLocalizer::init(const TrackMap& main, const std::vector<Landmark>& stops,
                           const std::vector<Landmark>& cutoffs, const TrackField& vmax, const Params& p) {
  p_ = p;
  nx_ = 0;
  if (!main.cyclic() || main.empty() || stops.empty()) return false;
  L_ = main.length();
  // ---- wheel-distance coordinate: the uncorrected odometer runs at ds / (1 + rho(s)) ----
  constexpr double kDs = 0.5;
  const int ns = static_cast<int>(L_ / kDs);
  sg_.resize(ns + 1);
  ug_.resize(ns + 1);
  auto corr = [&](double s) {
    const double c = main.curvatureAt(main.wrap(s));
    return std::min(p.curv_abs * std::abs(c), p.curv_sat) + p.curv_signed * c;
  };
  double u = 0.0;
  for (int i = 0; i <= ns; ++i) {
    const double s = i < ns ? i * kDs : L_;
    sg_[i] = s;
    ug_[i] = u;
    if (i < ns) {
      const double ds = std::min(kDs, L_ - s);
      const double rho = 0.5 * (corr(s + p.front_along) + corr(s + p.rear_along));
      u += ds / (1.0 + rho);
    }
  }
  LU_ = u;
  nx_ = static_cast<int>(std::llround(LU_ / p.du));
  if (nx_ < 100) {
    nx_ = 0;
    return false;
  }
  du_ = LU_ / nx_;
  // ---- rows: wheel scale error ----
  nk_ = static_cast<int>(std::llround((p.k_max - p.k_min) / p.k_step)) + 1;
  kap_.resize(nk_);
  c_.resize(nk_);
  std::vector<double> prior(nk_);
  double psum = 0.0;
  for (int j = 0; j < nk_; ++j) {
    kap_[j] = p.k_min + j * p.k_step;
    c_[j] = 1.0 / (1.0 + kap_[j]);
    prior[j] = std::exp(-0.5 * (kap_[j] / p.k_prior_sd) * (kap_[j] / p.k_prior_sd));
    psum += prior[j];
  }
  G_.assign(static_cast<size_t>(nk_) * nx_, 0.0f);
  for (int j = 0; j < nk_; ++j)
    std::fill(G_.begin() + static_cast<long>(j) * nx_, G_.begin() + static_cast<long>(j + 1) * nx_,
              static_cast<float>(prior[j] / psum / nx_));
  // ---- stop and cut-off likelihood tables on a fine periodic grid ----
  const double df = du_ / kSub;
  std::vector<double> lm_u, lm_sd, lm_p;
  for (const Landmark& l : stops) {
    lm_u.push_back(uOfS(l.s));
    lm_sd.push_back(std::hypot(l.sigma, p.lm_sd_extra));
    lm_p.push_back(std::min(l.p_stop, 1.0));
  }
  std::vector<double> tab(static_cast<size_t>(nx_) * kSub, p.stop_rate);
  addBumps(tab, df, lm_u, lm_sd, lm_p);
  stop_tab_ = boxAverage(tab, kSub, p.stop_rate);
  std::vector<double> c_sd;
  cut_u_.clear();
  cut_q_.clear();
  for (const Landmark& l : cutoffs) {
    cut_u_.push_back(uOfS(l.s));
    c_sd.push_back(std::max(l.sigma, p.cut_sd_min));
    cut_q_.push_back(std::min(l.p_stop, 1.0));
  }
  std::fill(tab.begin(), tab.end(), p.cut_rate);
  addBumps(tab, df, cut_u_, c_sd, cut_q_);
  cut_tab_ = boxAverage(tab, kSub, p.cut_rate);
  for (double& q : cut_q_) q = std::min(q, p.p_max);
  // ---- known stop places passed without stopping: cumulative log(1 - p) over three laps ----
  std::vector<double> lp(3 * static_cast<size_t>(nx_), 0.0);
  for (size_t m = 0; m < lm_u.size(); ++m) {
    const long k = ((std::llround(lm_u[m] / du_) % nx_) + nx_) % nx_;
    const double v = std::log1p(-std::min(lm_p[m], p.p_max));
    for (int lap = 0; lap < 3; ++lap) lp[static_cast<size_t>(k + static_cast<long>(lap) * nx_)] += v;
  }
  pass_cum_.resize(lp.size());
  double acc = 0.0;
  for (size_t i = 0; i < lp.size(); ++i) pass_cum_[i] = (acc += lp[i]);
  pass_lap_ = pass_cum_[static_cast<size_t>(nx_) - 1];
  // ---- speed envelope over the last checkpoint behind each place ----
  v_tab_.clear();
  if (!vmax.empty()) {
    std::vector<double> vm(nx_);
    for (int i = 0; i < nx_; ++i) vm[i] = vmax.at(sOfU(i * du_));
    const int n = std::max(1, static_cast<int>(1.2 * p.checkpoint / du_));
    v_tab_.resize(nx_);
    for (int i = 0; i < nx_; ++i) {
      double m = -1e9;
      for (int k = 0; k < n; ++k) m = std::max(m, vm[((i - k) % nx_ + nx_) % nx_]);
      v_tab_[i] = static_cast<float>(m);
    }
  }
  // ---- bookkeeping ----
  r_last_ = r_blur_ = var_acc_ = eps_acc_ = 0.0;
  r_pass_ = off_pass_ = r_cp_ = vmax_cp_ = 0.0;
  moved_ = still_ = stop_done_ = false;
  still_t_acc_ = 0.0;
  prev_notch_ = 0;
  notch_hist_.clear();
  cut_events_.clear();
  cues_ = 0;
  fixed_ = false;
  t_last_ = -1.0;
  return true;
}

double GlobalLocalizer::uOfS(double s) const {
  s = std::fmod(s, L_);
  if (s < 0.0) s += L_;
  const size_t i = std::min(static_cast<size_t>(s / (sg_[1] - sg_[0])), sg_.size() - 2);
  const double w = (s - sg_[i]) / std::max(1e-12, sg_[i + 1] - sg_[i]);
  return ug_[i] + w * (ug_[i + 1] - ug_[i]);
}

double GlobalLocalizer::sOfU(double u) const {
  u = std::fmod(u, LU_);
  if (u < 0.0) u += LU_;
  const size_t i = static_cast<size_t>(std::upper_bound(ug_.begin(), ug_.end(), u) - ug_.begin());
  const size_t k = std::clamp<size_t>(i, 1, ug_.size() - 1);
  const double w = (u - ug_[k - 1]) / std::max(1e-12, ug_[k] - ug_[k - 1]);
  return sg_[k - 1] + w * (sg_[k] - sg_[k - 1]);
}

void GlobalLocalizer::normalise() {
  double sum = 0.0;
  for (float g : G_) sum += g;
  if (!(sum > 0.0)) {  // numerical collapse: restart from a flat posterior
    std::fill(G_.begin(), G_.end(), 1.0f / static_cast<float>(G_.size()));
    return;
  }
  const float inv = static_cast<float>(1.0 / sum);
  for (float& g : G_) g *= inv;
}

void GlobalLocalizer::blur(double r, bool force_jump) {
  const double dist = std::max(0.0, r - r_blur_);
  r_blur_ = r;
  var_acc_ += p_.q_x * dist;
  eps_acc_ += p_.jump_rate * dist;
  // rows are copied into a buffer padded with the wrapped ends, so the inner loops need no modulo
  auto padded = [&](const float* g, int margin, std::vector<float>& pad) {
    pad.resize(static_cast<size_t>(nx_) + 2 * margin);
    for (int k = 0; k < margin; ++k) {
      pad[k] = g[((k - margin) % nx_ + nx_) % nx_];
      pad[static_cast<size_t>(margin) + nx_ + k] = g[k % nx_];
    }
    std::copy(g, g + nx_, pad.begin() + margin);
  };
  std::vector<float> pad, tmp(nx_);
  if (var_acc_ >= du_ * du_) {  // odometry random walk: Gaussian blur along u0
    const double sig = std::sqrt(var_acc_) / du_;
    const int rad = std::min(nx_ / 2 - 1, std::max(1, static_cast<int>(std::ceil(3.0 * sig))));
    std::vector<float> w(2 * rad + 1);
    double ws = 0.0;
    for (int k = -rad; k <= rad; ++k) ws += std::exp(-0.5 * k * k / (sig * sig));
    for (int k = -rad; k <= rad; ++k) w[k + rad] = static_cast<float>(std::exp(-0.5 * k * k / (sig * sig)) / ws);
    for (int j = 0; j < nk_; ++j) {
      float* g = &G_[static_cast<size_t>(j) * nx_];
      padded(g, rad, pad);
      for (int i = 0; i < nx_; ++i) {
        const float* q = &pad[i];
        float a = 0.0f;
        for (int k = 0; k <= 2 * rad; ++k) a += w[k] * q[k];
        tmp[i] = a;
      }
      std::copy(tmp.begin(), tmp.end(), g);
    }
    var_acc_ = 0.0;
  }
  if (eps_acc_ > 0.0 && (force_jump || eps_acc_ > 0.01)) {  // rare odometry jumps: box mixing
    const double eps = std::min(0.5, eps_acc_);
    const int half = std::min(nx_ / 2 - 1, static_cast<int>(p_.jump_hw / du_));
    const int width = 2 * half + 1;
    for (int j = 0; j < nk_; ++j) {
      float* g = &G_[static_cast<size_t>(j) * nx_];
      padded(g, half + 1, pad);
      double acc = 0.0;
      for (int k = 1; k <= width; ++k) acc += pad[k];  // cells i - half .. i + half for i = 0
      for (int i = 0; i < nx_; ++i) {
        tmp[i] = static_cast<float>((1.0 - eps) * pad[i + half + 1] + eps * acc / width);
        acc += pad[static_cast<size_t>(i) + width + 1] - pad[static_cast<size_t>(i) + 1];
      }
      std::copy(tmp.begin(), tmp.end(), g);
    }
    eps_acc_ = 0.0;
  }
  if (p_.floor > 0.0)  // re-seeding: a wrong mode can always be left
    for (int j = 0; j < nk_; ++j) {
      float* g = &G_[static_cast<size_t>(j) * nx_];
      double rs = 0.0;
      for (int i = 0; i < nx_; ++i) rs += g[i];
      const float add = static_cast<float>(p_.floor * rs / nx_);
      for (int i = 0; i < nx_; ++i) g[i] += add;
    }
}

void GlobalLocalizer::passThrough(double r_to, double off_to) {
  for (int j = 0; j < nk_; ++j) {
    const long a = std::llround(shiftCells(r_pass_, off_pass_, j));
    const long b = std::llround(shiftCells(r_to, off_to, j));
    if (b <= a) continue;
    const long laps = (b - a) / nx_;
    const long bj = b - laps * nx_;
    const long a0 = ((a % nx_) + nx_) % nx_;
    const long b0 = a0 + (bj - a);
    float* g = &G_[static_cast<size_t>(j) * nx_];
    for (int i = 0; i < nx_; ++i) {
      const double dl = pass_cum_[static_cast<size_t>(b0 + i)] - pass_cum_[static_cast<size_t>(a0 + i)] +
                        static_cast<double>(laps) * pass_lap_;
      if (dl != 0.0) g[i] *= static_cast<float>(std::exp(dl));
    }
  }
}

int GlobalLocalizer::notchAt(double r) const {
  // history is appended in odometer order: last change at or before r
  const auto it = std::upper_bound(notch_hist_.begin(), notch_hist_.end(), r,
                                   [](double v, const std::pair<double, int>& h) { return v < h.first; });
  return it == notch_hist_.begin() ? 0 : std::prev(it)->second;
}

void GlobalLocalizer::cutPass(double r0, double r1) {
  for (size_t c = 0; c < cut_u_.size(); ++c) {
    const double uci = cut_u_[c] / du_;
    for (int j = 0; j < nk_; ++j) {
      const long a = static_cast<long>(std::floor(shiftCells(r0, 0.0, j)));
      const long b = static_cast<long>(std::floor(shiftCells(r1, 0.0, j)));
      float* g = &G_[static_cast<size_t>(j) * nx_];
      for (long k = a + 1; k <= b; ++k) {
        const double r_at = k * du_ / c_[j];
        if (notchAt(r_at - 8.0) < p_.cut_notch) continue;
        bool near_event = false;
        for (double ev : cut_events_)
          if (std::abs(ev - r_at) <= 15.0) near_event = true;
        if (near_event) continue;
        long cell = static_cast<long>(std::floor(uci - static_cast<double>(k))) % nx_;
        if (cell < 0) cell += nx_;
        g[cell] *= static_cast<float>(1.0 - cut_q_[c]);
      }
    }
  }
}

void GlobalLocalizer::multiplyFine(const std::vector<float>& tab, double r) {
  const long nf = static_cast<long>(nx_) * kSub;
  for (int j = 0; j < nk_; ++j) {
    long a = std::llround(shiftCells(r, 0.0, j) * kSub) % nf;
    if (a < 0) a += nf;
    float* g = &G_[static_cast<size_t>(j) * nx_];
    for (int i = 0; i < nx_; ++i) {
      long idx = a + static_cast<long>(i) * kSub;
      if (idx >= nf) idx -= nf;
      g[i] *= tab[static_cast<size_t>(idx)];
    }
  }
}

void GlobalLocalizer::onStop(double r) {
  blur(r, true);
  passThrough(r, -p_.guard);
  multiplyFine(stop_tab_, r);
  ++cues_;
  r_pass_ = r;
  off_pass_ = p_.guard;
  normalise();
  afterUpdate(r);
}

void GlobalLocalizer::onCutoff(double r) {
  cut_events_.push_back(r);
  blur(r, false);
  multiplyFine(cut_tab_, r);
  ++cues_;
  normalise();
  afterUpdate(r);
}

void GlobalLocalizer::onCheckpoint(double r) {
  blur(r, false);
  passThrough(r, -p_.guard);
  r_pass_ = r;
  off_pass_ = -p_.guard;
  cutPass(r_cp_, r);
  if (!v_tab_.empty())
    for (int j = 0; j < nk_; ++j) {
      long a = std::llround(shiftCells(r, 0.0, j)) % nx_;
      if (a < 0) a += nx_;
      float* g = &G_[static_cast<size_t>(j) * nx_];
      for (int i = 0; i < nx_; ++i) {
        long idx = a + i;
        if (idx >= nx_) idx -= nx_;
        if (vmax_cp_ > v_tab_[static_cast<size_t>(idx)] + p_.v_margin) g[i] *= static_cast<float>(p_.v_eps);
      }
    }
  r_cp_ = r;
  vmax_cp_ = 0.0;
  normalise();
  afterUpdate(r);
}

void GlobalLocalizer::afterUpdate(double r) {
  const Estimate e = estimate(r);
  if (e.conf >= p_.p_fix && cues_ >= p_.min_cues) {
    fixed_ = true;
    fix_ = e;
  }
}

GlobalLocalizer::Estimate GlobalLocalizer::estimate(double r) const {
  Estimate e;
  if (nx_ == 0) return e;
  std::vector<double> H(nx_, 0.0);
  std::vector<long> a(nk_);
  for (int j = 0; j < nk_; ++j) {
    a[j] = ((std::llround(shiftCells(r, 0.0, j)) % nx_) + nx_) % nx_;
    const float* g = &G_[static_cast<size_t>(j) * nx_];
    for (int i = 0; i < nx_; ++i) {
      long idx = a[j] + i;
      if (idx >= nx_) idx -= nx_;
      H[idx] += g[i];
    }
  }
  double tot = 0.0;
  for (double h : H) tot += h;
  if (!(tot > 0.0)) return e;
  const int w = static_cast<int>(std::llround(p_.win / du_));
  const int W = 2 * w + 1;
  double acc = 0.0;
  for (int k = 0; k < W; ++k) acc += H[k % nx_];
  double best = acc;
  int i0 = 0;
  for (int i = 1; i < nx_; ++i) {
    acc += H[(i + W - 1) % nx_] - H[i - 1];
    if (acc > best) {
      best = acc;
      i0 = i;
    }
  }
  double m0 = 0.0, m1 = 0.0, m2 = 0.0;
  for (int k = 0; k < W; ++k) {
    const double h = H[(i0 + k) % nx_];
    m0 += h;
    m1 += h * k;
    m2 += h * k * k;
  }
  const double kbar = m1 / std::max(m0, 1e-300);
  e.conf = best / tot;
  e.s_var = std::max(0.0, m2 / std::max(m0, 1e-300) - kbar * kbar) * du_ * du_;
  e.s = sOfU((i0 + kbar) * du_);
  double kw_sum = 0.0, kw_k = 0.0, kw_k2 = 0.0;
  for (int j = 0; j < nk_; ++j) {
    const float* g = &G_[static_cast<size_t>(j) * nx_];
    long lo = (i0 - a[j]) % nx_;
    if (lo < 0) lo += nx_;
    double kw = 0.0;
    for (int k = 0; k < W; ++k) kw += g[(lo + k) % nx_];
    kw_sum += kw;
    kw_k += kw * kap_[j];
    kw_k2 += kw * kap_[j] * kap_[j];
  }
  e.kappa = kw_k / std::max(kw_sum, 1e-300);
  e.kappa_sd = std::sqrt(std::max(0.0, kw_k2 / std::max(kw_sum, 1e-300) - e.kappa * e.kappa));
  e.cues = cues_;
  return e;
}

void GlobalLocalizer::step(double t, double r, double v, bool standstill, int notch) {
  if (nx_ == 0 || fixed_) return;
  const double dt = t_last_ >= 0.0 ? std::clamp(t - t_last_, 0.0, 1.0) : 0.0;
  t_last_ = t;
  r_last_ = r;
  if (!moved_ && r > 20.0) moved_ = true;  // the standstill at the start of a run is no cue
  vmax_cp_ = std::max(vmax_cp_, v);
  if (standstill) {
    if (!still_) {
      still_ = true;
      still_t_acc_ = 0.0;
      stop_done_ = false;
    } else {
      still_t_acc_ += dt;
    }
    if (moved_ && !stop_done_ && still_t_acc_ >= p_.stop_dwell) {
      stop_done_ = true;
      onStop(r);
    }
  } else {
    still_ = false;
  }
  if (notch != prev_notch_) {
    notch_hist_.emplace_back(r, notch);
    if (moved_ && prev_notch_ >= p_.cut_notch && notch == 0 && v > p_.cut_vmin) onCutoff(r + v * p_.cut_lead);
    prev_notch_ = notch;
  }
  if (!fixed_ && r - r_cp_ >= p_.checkpoint) onCheckpoint(r);
}

}  // namespace tbo
