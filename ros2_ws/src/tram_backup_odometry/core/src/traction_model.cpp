#include "tbo/traction_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace tbo {

namespace {
// Placeholder from the first identification pass on training bags (median transient
// acceleration per notch, m/s^2); replaced by the calibrated CSV in deployment.
const double kNotchAccel[TractionModel::kRows] = {
    -1.669, -1.550, -1.438, -1.369, -1.295, -1.208, -1.028, -0.846,  // -15..-8
    -0.774, -0.720, -0.688, -0.658, -0.614, -0.515, -0.182,          // -7..-1
    -0.031,                                                          //  0
    0.071,  0.119,  0.133,  0.091,  0.228,  0.687,  0.816,  0.945,   //  1..8
    0.943,  0.928,  0.895,  0.857,  0.823,  0.801,  0.718};          //  9..15
constexpr double kPowerPerMass = 7.0;  // m^2/s^3, traction limit a <= P/(m v)
}  // namespace

TractionModel::TractionModel() {
  for (int i = 0; i <= 20; ++i) vgrid_.push_back(static_cast<double>(i));
  tab_.assign(static_cast<size_t>(kRows) * vgrid_.size(), 0.0);
  for (int r = 0; r < kRows; ++r)
    for (size_t j = 0; j < vgrid_.size(); ++j) {
      double a = kNotchAccel[r];
      const int notch = r + kNotchMin;
      if (notch > 0 && vgrid_[j] > 0.5) a = std::min(a, kPowerPerMass / vgrid_[j]);
      tab_[r * vgrid_.size() + j] = a;
    }
  builtin_ = true;
}

bool TractionModel::loadCsv(const std::string& path, std::string* err) {
  std::ifstream in(path);
  if (!in) {
    if (err) *err = "cannot open traction table " + path;
    return false;
  }
  std::vector<double> grid;
  std::vector<double> tab(static_cast<size_t>(kRows), std::nan(""));
  std::vector<bool> seen(kRows, false);
  std::string line;
  bool header = false;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    for (char& c : line)
      if (c == ',' || c == ';' || c == '\t') c = ' ';
    std::istringstream ss(line);
    if (!header) {
      std::string first;
      ss >> first;  // "notch"
      double v;
      while (ss >> v) grid.push_back(v);
      if (grid.size() < 2 || !std::is_sorted(grid.begin(), grid.end())) {
        if (err) *err = "bad speed grid header in " + path;
        return false;
      }
      tab.assign(static_cast<size_t>(kRows) * grid.size(), std::nan(""));
      header = true;
      continue;
    }
    double notch_d;
    if (!(ss >> notch_d)) continue;
    const int notch = static_cast<int>(std::lround(notch_d));
    if (notch < kNotchMin || notch > kNotchMax) continue;
    const int r = notch - kNotchMin;
    for (size_t j = 0; j < grid.size(); ++j) {
      double a;
      if (!(ss >> a) || !std::isfinite(a)) {
        if (err) *err = "row for notch " + std::to_string(notch) + " is incomplete";
        return false;
      }
      tab[r * grid.size() + j] = a;
    }
    seen[r] = true;
  }
  for (int r = 0; r < kRows; ++r)
    if (!seen[r]) {
      if (err) *err = "traction table misses notch " + std::to_string(r + kNotchMin);
      return false;
    }
  vgrid_ = std::move(grid);
  tab_ = std::move(tab);
  builtin_ = false;
  return true;
}

void TractionModel::locate(double v, int& i0, double& w) const {
  const int n = static_cast<int>(vgrid_.size());
  if (!(v > vgrid_.front())) {  // also NaN
    i0 = 0;
    w = 0.0;
    return;
  }
  if (v >= vgrid_.back()) {
    i0 = n - 2;
    w = 1.0;
    return;
  }
  const auto it = std::upper_bound(vgrid_.begin(), vgrid_.end(), v);
  i0 = static_cast<int>(it - vgrid_.begin()) - 1;
  w = (v - vgrid_[i0]) / (vgrid_[i0 + 1] - vgrid_[i0]);
}

double TractionModel::target(int notch, double v) const {
  notch = std::clamp(notch, kNotchMin, kNotchMax);
  int i0;
  double w;
  locate(v, i0, w);
  const double* row = &tab_[static_cast<size_t>(notch - kNotchMin) * vgrid_.size()];
  return row[i0] * (1.0 - w) + row[i0 + 1] * w;
}

double TractionModel::dTargetDv(int notch, double v) const {
  notch = std::clamp(notch, kNotchMin, kNotchMax);
  int i0;
  double w;
  locate(v, i0, w);
  if (!(v > vgrid_.front()) || v >= vgrid_.back()) return 0.0;
  const double* row = &tab_[static_cast<size_t>(notch - kNotchMin) * vgrid_.size()];
  return (row[i0 + 1] - row[i0]) / (vgrid_[i0 + 1] - vgrid_[i0]);
}

}  // namespace tbo
