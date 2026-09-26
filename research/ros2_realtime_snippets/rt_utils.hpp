// rt_utils.hpp -- ROS-free, allocation-free helpers for the tram odometry node.
// C++17, header-only. Compiles with g++ >= 9 (Humble/Jammy ships g++ 11).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tram_rt {

// ---------------------------------------------------------------------------
// StampSanitizer: per-topic header.stamp validation.
//   accept  iff  last < s <= last + rate*(recv - last_recv) + tol
//   K mutually consistent rejected stamps in a row (no accept in between) -> resync.
// Validated offline on all 122 bags: rejects ~0.005 % of messages (only the
// +-1 s "ghost" duplicates); accepted stamps are strictly monotonic.
// ---------------------------------------------------------------------------
class StampSanitizer {
public:
  enum class Verdict : uint8_t { kAccepted, kRejected, kResynced };
  struct Config { double tol_s = 0.35; double rate = 1.0; int resync_k = 8; };

  StampSanitizer() = default;
  explicit StampSanitizer(const Config & c) : cfg_(c) {}

  // stamp_s: header stamp [s]; recv_s: local steady receive time [s]
  Verdict check(double stamp_s, double recv_s) {
    if (!(stamp_s > 0.0) || !std::isfinite(stamp_s)) { ++rejected_; return Verdict::kRejected; }
    if (!has_last_) { accept(stamp_s, recv_s); return Verdict::kAccepted; }
    if (consistent(last_s_, last_r_, stamp_s, recv_s)) {
      accept(stamp_s, recv_s);
      return Verdict::kAccepted;
    }
    // candidate chain for resync (a real clock step / bag loop)
    if (pend_n_ > 0 && consistent(pend_s_, pend_r_, stamp_s, recv_s)) { ++pend_n_; } else { pend_n_ = 1; }
    pend_s_ = stamp_s; pend_r_ = recv_s;
    if (pend_n_ >= cfg_.resync_k) { accept(stamp_s, recv_s); ++resyncs_; return Verdict::kResynced; }
    ++rejected_;
    return Verdict::kRejected;
  }
  double last_stamp() const { return last_s_; }
  uint64_t rejected() const { return rejected_; }
  uint64_t resyncs() const { return resyncs_; }
  void reset() { has_last_ = false; pend_n_ = 0; }

private:
  bool consistent(double s0, double r0, double s, double r) const {
    const double dr = std::max(0.0, r - r0);
    return (s > s0) && (s - s0 <= cfg_.rate * dr + cfg_.tol_s);
  }
  void accept(double s, double r) { last_s_ = s; last_r_ = r; has_last_ = true; pend_n_ = 0; }
  Config cfg_{};
  bool has_last_ = false;
  double last_s_ = 0.0, last_r_ = 0.0, pend_s_ = 0.0, pend_r_ = 0.0;
  int pend_n_ = 0;
  uint64_t rejected_ = 0, resyncs_ = 0;
};

// ---------------------------------------------------------------------------
// Fixed-capacity ring buffer (no heap). Used for state history (OOSM re-filtering).
// ---------------------------------------------------------------------------
template <typename T, std::size_t N>
class RingBuffer {
public:
  void push(const T & v) { buf_[head_] = v; head_ = (head_ + 1) % N; if (size_ < N) ++size_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  // i = 0 -> oldest, i = size()-1 -> newest
  const T & at(std::size_t i) const { return buf_[(head_ + N - size_ + i) % N]; }
  T & at(std::size_t i) { return buf_[(head_ + N - size_ + i) % N]; }
  const T & back() const { return at(size_ - 1); }
  void clear() { size_ = 0; head_ = 0; }
  // keep elements 0..i, drop everything newer
  void truncate_after(std::size_t i) {
    const std::size_t drop = size_ - (i + 1);
    head_ = (head_ + N - drop) % N; size_ = i + 1;
  }
private:
  std::array<T, N> buf_{};
  std::size_t head_ = 0, size_ = 0;
};

// ---------------------------------------------------------------------------
// LatencyHistogram: O(1)-memory streaming percentiles (0.05 ms bins up to 500 ms).
// ---------------------------------------------------------------------------
class LatencyHistogram {
public:
  static constexpr int kBins = 10000;  // 10000 * 0.05 ms = 500 ms
  static constexpr double kBinMs = 0.05;
  void add(double ms) {
    if (!std::isfinite(ms) || ms < 0) ms = 0;
    int b = static_cast<int>(ms / kBinMs);
    if (b >= kBins) b = kBins - 1;
    ++h_[b]; ++n_; max_ = std::max(max_, ms); sum_ += ms;
    if (ms > 100.0) ++over100_;
    if (ms > 250.0) ++over250_;
  }
  double percentile(double p) const {
    if (n_ == 0) return 0.0;
    const uint64_t target = static_cast<uint64_t>(std::ceil(p / 100.0 * static_cast<double>(n_)));
    uint64_t c = 0;
    for (int b = 0; b < kBins; ++b) { c += h_[b]; if (c >= target) return (b + 1) * kBinMs; }
    return max_;
  }
  double max() const { return max_; }
  double mean() const { return n_ ? sum_ / static_cast<double>(n_) : 0.0; }
  uint64_t count() const { return n_; }
  uint64_t over100() const { return over100_; }
  uint64_t over250() const { return over250_; }
  void reset() { h_.fill(0); n_ = over100_ = over250_ = 0; max_ = sum_ = 0.0; }
private:
  std::array<uint32_t, kBins> h_{};
  uint64_t n_ = 0, over100_ = 0, over250_ = 0;
  double max_ = 0.0, sum_ = 0.0;
};

// ---------------------------------------------------------------------------
// Along/cross-track covariance -> ENU covariance, row-major 6x6 for nav_msgs/Odometry
//   Sigma_xy = R(psi) diag(var_along, var_cross) R(psi)^T
// ---------------------------------------------------------------------------
inline void fill_pose_covariance(std::array<double, 36> & c, double yaw, double var_along,
                                 double var_cross, double var_z, double var_rp, double var_yaw) {
  c.fill(0.0);
  const double cs = std::cos(yaw), sn = std::sin(yaw);
  c[0] = cs * cs * var_along + sn * sn * var_cross;  // xx
  c[1] = cs * sn * (var_along - var_cross);          // xy
  c[6] = c[1];                                       // yx
  c[7] = sn * sn * var_along + cs * cs * var_cross;  // yy
  c[14] = var_z;                                     // zz
  c[21] = var_rp;                                    // roll
  c[28] = var_rp;                                    // pitch
  c[35] = var_yaw;                                   // yaw
}

}  // namespace tram_rt
