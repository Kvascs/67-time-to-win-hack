// Shared plain types of the estimator core.
#pragma once

#include <cmath>
#include <cstdint>

namespace tbo {

// All times are header stamps in integer nanoseconds (bag time domain), which keeps
// replays bit-reproducible and avoids float rounding in stamp comparisons.
using Stamp = std::int64_t;
constexpr Stamp kNsPerSec = 1000000000LL;
inline double toSec(Stamp t) { return static_cast<double>(t) * 1e-9; }
inline Stamp fromSec(double s) { return static_cast<Stamp>(std::llround(s * 1e9)); }

enum class Sensor : std::uint8_t { Front = 0, Rear = 1 };
enum class GnssSource : std::uint8_t { Master = 0, Rover = 1 };

// Interacting-multiple-model hypotheses about the wheel sensors.
enum Mode : int {
  kModeNominal = 0,   // both bogie speeds trustworthy
  kModeFrontBad = 1,  // front bogie slipping / sliding / faulty
  kModeRearBad = 2,   // rear bogie slipping / sliding / faulty
  kModeBothBad = 3,   // both untrustworthy -> model-only dead reckoning
  kNumModes = 4
};

// Diagnostic bit flags published with every output.
enum HealthFlag : std::uint32_t {
  kFlagFrontSlip = 1u << 0,       // front wheel faster than vehicle (traction slip)
  kFlagRearSlip = 1u << 1,
  kFlagFrontSlide = 1u << 2,      // front wheel slower than vehicle (braking slide / lock)
  kFlagRearSlide = 1u << 3,
  kFlagFrontDropout = 1u << 4,    // no valid front message within timeout
  kFlagRearDropout = 1u << 5,
  kFlagCmdDropout = 1u << 6,      // no controller message within timeout
  kFlagFrontStuck = 1u << 7,      // frozen value while vehicle speed changes
  kFlagRearStuck = 1u << 8,
  kFlagFrontInvalid = 1u << 9,    // NaN / negative / absurd sample rejected recently
  kFlagRearInvalid = 1u << 10,
  kFlagModelOnly = 1u << 11,      // speed propagated by the dynamics model only
  kFlagStandstill = 1u << 12,     // zero-velocity update active
  kFlagNotInitialized = 1u << 13, // absolute position not anchored yet (no GNSS init)
  kFlagNoMap = 1u << 14,          // position is dead reckoning, not map-constrained
  kFlagRecovered = 1u << 15,      // speed re-anchored to wheels after a long anomaly
  kFlagLateData = 1u << 16,       // a message older than the fixed-lag window was dropped
};

struct Output {
  Stamp stamp{0};
  bool valid{false};
  double v{0.0}, v_var{0.0};          // longitudinal speed, m/s
  double accel{0.0};                  // estimated longitudinal acceleration, m/s^2
  double a_model{0.0};                // drive acceleration predicted from the notch, m/s^2
  double s{0.0}, s_var{0.0};          // distance along the track map (or odometer), m
  double x{0.0}, y{0.0}, z{0.0};      // position in the output frame, m
  double yaw{0.0};                    // heading in the output frame, rad (ENU: 0 = east, CCW)
  double cov_xx{0.0}, cov_xy{0.0}, cov_yy{0.0}, cov_zz{0.0};
  double mode_prob[kNumModes]{1.0, 0.0, 0.0, 0.0};
  double slip_front{0.0}, slip_rear{0.0};  // longitudinal slip ratio (wheel - vehicle) / vehicle
  double disturbance{0.0};            // grade + resistance mismatch, m/s^2
  double scale{0.0};                  // wheel scale error estimate (reading / true - 1)
  double gain{1.0};                   // traction gain (1 / relative mass)
  double wheel_trust{1.0};            // 1 - P(both bad)
  std::uint32_t flags{0};
  bool map_matched{false};
};

}  // namespace tbo
