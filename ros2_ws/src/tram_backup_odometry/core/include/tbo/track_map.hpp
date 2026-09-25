// Track map ("pathgraph"): the tram's path as a polyline parameterised by arc length s,
// in a local ENU frame anchored at a geodetic origin. Built offline from GNSS traces
// of training runs (both directions + terminal loops form one closed cycle).
// Converts the 1-D along-track estimate into x/y/z and heading.
#pragma once

#include <string>
#include <vector>

#include "tbo/geo.hpp"

namespace tbo {

struct MapPose {
  double x{0.0}, y{0.0}, z{0.0};
  double heading{0.0};  // rad, ENU (0 = east, CCW positive), direction of increasing s
};

struct MapProjection {
  bool ok{false};
  double s{0.0};
  double lateral{0.0};   // signed distance to the left of the path, m
  double dist{0.0};      // |lateral| (or distance to an end point)
  double heading{0.0};
};

class TrackMap {
 public:
  // CSV with header "s,x,y,z" (metres, map ENU frame). Metadata in comment lines:
  //   # origin_lat=<deg> origin_lon=<deg> origin_h=<m> cyclic=<0|1>
  bool loadCsv(const std::string& path, std::string* err);
  // Programmatic construction (tests, tools). s is recomputed from x/y.
  void setPoints(const std::vector<MapPose>& pts, bool cyclic, const geo::Geodetic& origin);

  bool empty() const { return s_.size() < 2; }
  // Branches merging into the main cycle carry the main-cycle arc length of the merge point.
  bool hasJoin() const { return has_join_; }
  double joinS() const { return join_s_; }
  bool cyclic() const { return cyclic_; }
  double length() const { return length_; }
  const geo::Geodetic& origin() const { return origin_; }
  size_t size() const { return s_.size(); }

  // Wraps s for cyclic maps, clamps otherwise.
  double wrap(double s) const;
  MapPose at(double s) const;

  // Track grade dz/ds (positive = uphill along increasing s) and curvature at s; 0 if absent.
  double gradeAt(double s) const;
  double curvatureAt(double s) const;
  bool hasProfile() const { return !grade_.empty(); }

  // Nearest point on the whole map.
  MapProjection project(double x, double y) const;
  // All local nearest points (distinct branches) within `gate` metres; returns count.
  int projectAll(double x, double y, double gate, MapProjection* out, int max_out) const;

 private:
  MapProjection projectSegment(size_t i, double x, double y) const;
  size_t segmentAt(double s) const;
  double interpProfile(const std::vector<double>& v, double s) const;
  std::vector<double> s_, x_, y_, z_;
  std::vector<double> grade_, curv_;  // optional per-vertex profile
  bool cyclic_ = false;
  bool has_join_ = false;
  double join_s_ = 0.0;
  double length_ = 0.0;
  geo::Geodetic origin_{};
};

}  // namespace tbo
