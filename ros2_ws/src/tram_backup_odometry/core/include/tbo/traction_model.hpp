// Nonlinear traction / braking model: target longitudinal acceleration as a function
// of the driver notch and the vehicle speed, a*(notch, v), stored as a lookup table
// calibrated offline on training bags. The table already includes running resistance
// on level track (notch 0 row = coasting deceleration). Realised drive acceleration
// follows a* with a first-order lag (tau) and optional dead time (see Params).
#pragma once

#include <string>
#include <vector>

namespace tbo {

class TractionModel {
 public:
  static constexpr int kNotchMin = -15;
  static constexpr int kNotchMax = 15;
  static constexpr int kRows = kNotchMax - kNotchMin + 1;

  TractionModel();  // built-in fallback table

  // CSV: header "notch,<v0>,<v1>,..." (speeds in m/s, ascending), then one row per notch
  // -15..15. Lines starting with '#' are comments.
  bool loadCsv(const std::string& path, std::string* err);

  // Target acceleration, m/s^2 (linear in speed between grid points, clamped outside).
  double target(int notch, double v) const;
  // d target / d v, used for the EKF Jacobian.
  double dTargetDv(int notch, double v) const;

  bool isBuiltin() const { return builtin_; }
  const std::vector<double>& speedGrid() const { return vgrid_; }

 private:
  void locate(double v, int& i0, double& w) const;
  std::vector<double> vgrid_;
  std::vector<double> tab_;  // kRows x vgrid_.size(), row-major
  bool builtin_ = true;
};

}  // namespace tbo
