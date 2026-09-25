// Geodesy helpers: WGS84 geodetic <-> ECEF <-> local ENU tangent plane, and UTM.
// Pure C++17, no dependencies, no heap allocation. Used to map the initial GNSS fix
// onto the track map and to express output positions in the frame the jury expects.
#pragma once

namespace tbo::geo {

struct Geodetic {
  double lat_deg{0.0};
  double lon_deg{0.0};
  double h{0.0};  // ellipsoidal height, m
};

struct Ecef {
  double x{0.0}, y{0.0}, z{0.0};
};

struct Enu {
  double e{0.0}, n{0.0}, u{0.0};
};

struct Utm {
  int zone{0};
  bool north{true};
  double easting{0.0};
  double northing{0.0};
};

Ecef geodeticToEcef(const Geodetic& g);
Geodetic ecefToGeodetic(const Ecef& p);

// East-North-Up tangent plane anchored at an origin (same convention as
// GeographicLib::LocalCartesian and PROJ "topocentric").
class LocalCartesian {
 public:
  LocalCartesian() = default;
  explicit LocalCartesian(const Geodetic& origin) { reset(origin); }
  void reset(const Geodetic& origin);

  Enu forward(const Geodetic& g) const { return fromEcef(geodeticToEcef(g)); }
  Geodetic reverse(const Enu& p) const { return ecefToGeodetic(toEcef(p)); }
  Enu fromEcef(const Ecef& p) const;
  Ecef toEcef(const Enu& p) const;
  const Geodetic& origin() const { return origin_; }

 private:
  Geodetic origin_{};
  Ecef o_{};
  double r_[3][3]{};  // rows: unit vectors e, n, u expressed in ECEF
};

int utmZoneFor(double lon_deg);
// force_zone <= 0 selects the standard zone for the longitude.
Utm geodeticToUtm(double lat_deg, double lon_deg, int force_zone = 0);
Geodetic utmToGeodetic(const Utm& u);  // returned h = 0

}  // namespace tbo::geo
