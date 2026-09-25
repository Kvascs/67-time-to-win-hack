#include "tbo/track_map.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace tbo {

namespace {
bool parseMeta(const std::string& line, const char* key, double& out) {
  const auto p = line.find(key);
  if (p == std::string::npos) return false;
  const auto eq = line.find('=', p);
  if (eq == std::string::npos) return false;
  out = std::strtod(line.c_str() + eq + 1, nullptr);
  return true;
}
}  // namespace

bool TrackMap::loadCsv(const std::string& path, std::string* err) {
  std::ifstream in(path);
  if (!in) {
    if (err) *err = "cannot open map " + path;
    return false;
  }
  std::vector<MapPose> pts;
  std::vector<double> grade, curv;
  double lat = 0, lon = 0, h = 0, cyc = 0, join = 0;
  bool have_origin = false, have_join = false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      have_origin |= parseMeta(line, "origin_lat", lat);
      parseMeta(line, "origin_lon", lon);
      parseMeta(line, "origin_h", h);
      parseMeta(line, "cyclic", cyc);
      have_join |= parseMeta(line, "join_s", join);
      continue;
    }
    if (line[0] == 's' || std::isalpha(static_cast<unsigned char>(line[0]))) continue;  // header
    for (char& c : line)
      if (c == ',' || c == ';') c = ' ';
    std::istringstream ss(line);
    double s, x, y, z;
    if (ss >> s >> x >> y >> z) {
      pts.push_back({x, y, z, 0.0});
      double g = 0.0, k = 0.0;
      if (ss >> g) {
        grade.push_back(g);
        if (ss >> k) curv.push_back(k);
      }
    }
  }
  if (pts.size() < 2 || !have_origin) {
    if (err) *err = "map " + path + " has < 2 points or no origin metadata";
    return false;
  }
  setPoints(pts, cyc > 0.5, {lat, lon, h});
  has_join_ = have_join;
  join_s_ = join;
  // keep the profile only if it is complete and aligned with the kept vertices
  if (grade.size() == pts.size() && s_.size() >= pts.size()) {
    grade_.assign(grade.begin(), grade.end());
    if (curv.size() == pts.size()) curv_.assign(curv.begin(), curv.end());
    while (grade_.size() < s_.size()) grade_.push_back(grade_.front());  // closing vertex
    while (!curv_.empty() && curv_.size() < s_.size()) curv_.push_back(curv_.front());
  }
  return true;
}

void TrackMap::setPoints(const std::vector<MapPose>& pts, bool cyclic, const geo::Geodetic& origin) {
  grade_.clear();
  curv_.clear();
  s_.clear();
  x_.clear();
  y_.clear();
  z_.clear();
  origin_ = origin;
  cyclic_ = cyclic;
  double s = 0.0;
  for (size_t i = 0; i < pts.size(); ++i) {
    if (i > 0) {
      const double ds = std::hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
      if (ds < 1e-6) continue;  // drop duplicate vertices
      s += ds;
    }
    s_.push_back(s);
    x_.push_back(pts[i].x);
    y_.push_back(pts[i].y);
    z_.push_back(pts[i].z);
  }
  if (cyclic_ && s_.size() >= 2) {  // close the loop with a final segment back to the start
    const double ds = std::hypot(x_.front() - x_.back(), y_.front() - y_.back());
    if (ds > 1e-6) {
      s += ds;
      s_.push_back(s);
      x_.push_back(x_.front());
      y_.push_back(y_.front());
      z_.push_back(z_.front());
    }
  }
  length_ = s_.empty() ? 0.0 : s_.back();
}

double TrackMap::wrap(double s) const {
  if (empty()) return s;
  if (cyclic_) {
    double w = std::fmod(s, length_);
    if (w < 0) w += length_;
    return w;
  }
  return std::clamp(s, 0.0, length_);
}

size_t TrackMap::segmentAt(double s) const {
  const auto it = std::upper_bound(s_.begin(), s_.end(), s);
  if (it == s_.begin()) return 0;
  size_t i = static_cast<size_t>(it - s_.begin()) - 1;
  if (i >= s_.size() - 1) i = s_.size() - 2;
  return i;
}

MapPose TrackMap::at(double s) const {
  MapPose p;
  if (empty()) return p;
  double ss = wrap(s);
  const size_t i = segmentAt(ss);
  const double seg = s_[i + 1] - s_[i];
  double w = seg > 0 ? (ss - s_[i]) / seg : 0.0;
  if (!cyclic_) {  // extrapolate linearly beyond the ends of an open map
    if (s < 0.0 && i == 0) w = s / seg;
    if (s > length_ && i == s_.size() - 2) w = 1.0 + (s - length_) / seg;
  }
  p.x = x_[i] + w * (x_[i + 1] - x_[i]);
  p.y = y_[i] + w * (y_[i + 1] - y_[i]);
  p.z = z_[i] + std::clamp(w, 0.0, 1.0) * (z_[i + 1] - z_[i]);
  p.heading = std::atan2(y_[i + 1] - y_[i], x_[i + 1] - x_[i]);
  return p;
}

double TrackMap::interpProfile(const std::vector<double>& v, double s) const {
  if (v.size() != s_.size() || empty()) return 0.0;
  const double ss = wrap(s);
  const size_t i = segmentAt(ss);
  const double seg = s_[i + 1] - s_[i];
  const double w = seg > 0 ? std::clamp((ss - s_[i]) / seg, 0.0, 1.0) : 0.0;
  return v[i] + w * (v[i + 1] - v[i]);
}

double TrackMap::gradeAt(double s) const { return interpProfile(grade_, s); }
double TrackMap::curvatureAt(double s) const { return interpProfile(curv_, s); }

MapProjection TrackMap::projectSegment(size_t i, double x, double y) const {
  MapProjection r;
  const double dx = x_[i + 1] - x_[i], dy = y_[i + 1] - y_[i];
  const double len2 = dx * dx + dy * dy;
  double t = len2 > 0 ? ((x - x_[i]) * dx + (y - y_[i]) * dy) / len2 : 0.0;
  t = std::clamp(t, 0.0, 1.0);
  const double px = x_[i] + t * dx, py = y_[i] + t * dy;
  const double len = std::sqrt(len2);
  r.ok = true;
  r.s = s_[i] + t * len;
  r.dist = std::hypot(x - px, y - py);
  r.lateral = len > 0 ? (dx * (y - y_[i]) - dy * (x - x_[i])) / len : 0.0;
  r.heading = std::atan2(dy, dx);
  return r;
}

MapProjection TrackMap::project(double x, double y) const {
  MapProjection best;
  if (empty()) return best;
  best.dist = 1e300;
  for (size_t i = 0; i + 1 < s_.size(); ++i) {
    const MapProjection r = projectSegment(i, x, y);
    if (r.dist < best.dist) best = r;
  }
  return best;
}

int TrackMap::projectAll(double x, double y, double gate, MapProjection* out, int max_out) const {
  if (empty() || max_out <= 0) return 0;
  // Candidates are local minima of the point-to-path distance along the path (each
  // pass of the path near the point, e.g. the two tracks of a double line), within gate.
  const size_t nseg = s_.size() - 1;
  std::vector<MapProjection> seg(nseg);
  for (size_t i = 0; i < nseg; ++i) seg[i] = projectSegment(i, x, y);
  int n = 0;
  for (size_t i = 0; i < nseg && n < max_out; ++i) {
    if (seg[i].dist > gate) continue;
    const bool has_prev = i > 0 || cyclic_;
    const bool has_next = i + 1 < nseg || cyclic_;
    const MapProjection& prev = seg[i > 0 ? i - 1 : nseg - 1];
    const MapProjection& next = seg[i + 1 < nseg ? i + 1 : 0];
    if (has_prev && prev.dist < seg[i].dist) continue;
    if (has_next && next.dist < seg[i].dist) continue;
    bool dup = false;  // plateaus / vertex minima shared by two segments
    for (int k = 0; k < n; ++k)
      if (std::abs(out[k].s - seg[i].s) < 1.0 ||
          (cyclic_ && std::abs(std::abs(out[k].s - seg[i].s) - length_) < 1.0))
        dup = true;
    if (!dup) out[n++] = seg[i];
  }
  return n;
}

}  // namespace tbo
