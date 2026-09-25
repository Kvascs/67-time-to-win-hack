// A scalar field along the main map cycle, learned offline from many runs: here the mean
// unexplained longitudinal acceleration at each place (grade-model and resistance errors,
// curve resistance, typical driver behaviour at that spot). Linear interpolation, cyclic.
#pragma once

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace tbo {

class TrackField {
 public:
  // CSV "s,value" (comment lines start with '#'), s ascending on the main cycle.
  bool loadCsv(const std::string& path, double period, std::string* err) {
    std::ifstream in(path);
    if (!in) {
      if (err) *err = "cannot open field " + path;
      return false;
    }
    s_.clear();
    v_.clear();
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#' || line[0] == 's') continue;
      for (char& c : line)
        if (c == ',') c = ' ';
      std::istringstream ss(line);
      double s = 0.0, v = 0.0;
      if (ss >> s >> v) {
        s_.push_back(s);
        v_.push_back(v);
      }
    }
    period_ = period;
    if (s_.size() < 2) {
      if (err) *err = "field " + path + " has fewer than 2 rows";
      return false;
    }
    return true;
  }
  bool empty() const { return s_.size() < 2; }

  double at(double s) const {
    if (empty()) return 0.0;
    if (period_ > 0.0) {
      s = std::fmod(s, period_);
      if (s < 0.0) s += period_;
    }
    if (s <= s_.front() || s >= s_.back()) {  // wrap segment (or clamp on an open track)
      if (period_ <= 0.0) return s <= s_.front() ? v_.front() : v_.back();
      const double gap = s_.front() + period_ - s_.back();
      const double u = s >= s_.back() ? s - s_.back() : s + period_ - s_.back();
      return gap > 0.0 ? v_.back() + (v_.front() - v_.back()) * u / gap : v_.back();
    }
    size_t lo = 0, hi = s_.size() - 1;
    while (hi - lo > 1) {
      const size_t mid = (lo + hi) / 2;
      (s_[mid] <= s ? lo : hi) = mid;
    }
    const double w = (s - s_[lo]) / (s_[hi] - s_[lo]);
    return v_[lo] + w * (v_[hi] - v_[lo]);
  }

 private:
  std::vector<double> s_, v_;
  double period_ = 0.0;
};

}  // namespace tbo
