#include "tbo/geo.hpp"

#include <cmath>

namespace tbo::geo {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr double kA = 6378137.0;                 // WGS84 semi-major axis
constexpr double kF = 1.0 / 298.257223563;       // WGS84 flattening
constexpr double kE2 = kF * (2.0 - kF);          // first eccentricity squared
constexpr double kK0 = 0.9996;                   // UTM scale on central meridian

// Krueger series coefficients (3rd order in n), accurate to ~1 mm inside a zone.
struct KruegerCoeffs {
  double n, A, alpha[3], beta[3], delta[3];
};

KruegerCoeffs makeCoeffs() {
  KruegerCoeffs c{};
  const double n = kF / (2.0 - kF);
  const double n2 = n * n, n3 = n2 * n;
  c.n = n;
  c.A = kA / (1.0 + n) * (1.0 + n2 / 4.0 + n2 * n2 / 64.0);
  c.alpha[0] = n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0;
  c.alpha[1] = 13.0 * n2 / 48.0 - 3.0 * n3 / 5.0;
  c.alpha[2] = 61.0 * n3 / 240.0;
  c.beta[0] = n / 2.0 - 2.0 * n2 / 3.0 + 37.0 * n3 / 96.0;
  c.beta[1] = n2 / 48.0 + n3 / 15.0;
  c.beta[2] = 17.0 * n3 / 480.0;
  c.delta[0] = 2.0 * n - 2.0 * n2 / 3.0 - 2.0 * n3;
  c.delta[1] = 7.0 * n2 / 3.0 - 8.0 * n3 / 5.0;
  c.delta[2] = 56.0 * n3 / 15.0;
  return c;
}

const KruegerCoeffs& coeffs() {
  static const KruegerCoeffs c = makeCoeffs();
  return c;
}

}  // namespace

Ecef geodeticToEcef(const Geodetic& g) {
  const double lat = g.lat_deg * kDeg, lon = g.lon_deg * kDeg;
  const double s = std::sin(lat), c = std::cos(lat);
  const double N = kA / std::sqrt(1.0 - kE2 * s * s);
  return {(N + g.h) * c * std::cos(lon), (N + g.h) * c * std::sin(lon),
          (N * (1.0 - kE2) + g.h) * s};
}

Geodetic ecefToGeodetic(const Ecef& p) {
  const double rho = std::hypot(p.x, p.y);
  const double lon = std::atan2(p.y, p.x);
  double lat = std::atan2(p.z, rho * (1.0 - kE2));
  double h = 0.0;
  for (int i = 0; i < 6; ++i) {  // Bowring-style fixed-point, converges to < 1e-9 rad
    const double s = std::sin(lat), c = std::cos(lat);
    const double N = kA / std::sqrt(1.0 - kE2 * s * s);
    h = (std::abs(c) > 1e-9) ? rho / c - N : std::abs(p.z) / std::abs(s) - N * (1.0 - kE2);
    lat = std::atan2(p.z, rho * (1.0 - kE2 * N / (N + h)));
  }
  return {lat / kDeg, lon / kDeg, h};
}

void LocalCartesian::reset(const Geodetic& origin) {
  origin_ = origin;
  o_ = geodeticToEcef(origin);
  const double lat = origin.lat_deg * kDeg, lon = origin.lon_deg * kDeg;
  const double sl = std::sin(lat), cl = std::cos(lat);
  const double so = std::sin(lon), co = std::cos(lon);
  const double r[3][3] = {{-so, co, 0.0}, {-sl * co, -sl * so, cl}, {cl * co, cl * so, sl}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r_[i][j] = r[i][j];
}

Enu LocalCartesian::fromEcef(const Ecef& p) const {
  const double d[3] = {p.x - o_.x, p.y - o_.y, p.z - o_.z};
  double out[3];
  for (int i = 0; i < 3; ++i) out[i] = r_[i][0] * d[0] + r_[i][1] * d[1] + r_[i][2] * d[2];
  return {out[0], out[1], out[2]};
}

Ecef LocalCartesian::toEcef(const Enu& p) const {
  const double v[3] = {p.e, p.n, p.u};
  double d[3];
  for (int j = 0; j < 3; ++j) d[j] = r_[0][j] * v[0] + r_[1][j] * v[1] + r_[2][j] * v[2];
  return {o_.x + d[0], o_.y + d[1], o_.z + d[2]};
}

int utmZoneFor(double lon_deg) {
  int zone = static_cast<int>(std::floor((lon_deg + 180.0) / 6.0)) + 1;
  if (zone < 1) zone = 1;
  if (zone > 60) zone = 60;
  return zone;
}

Utm geodeticToUtm(double lat_deg, double lon_deg, int force_zone) {
  const KruegerCoeffs& c = coeffs();
  const int zone = force_zone > 0 ? force_zone : utmZoneFor(lon_deg);
  const double lon0 = (zone * 6.0 - 183.0) * kDeg;
  const double lat = lat_deg * kDeg, dlon = lon_deg * kDeg - lon0;
  const double k = 2.0 * std::sqrt(c.n) / (1.0 + c.n);
  const double t = std::sinh(std::atanh(std::sin(lat)) - k * std::atanh(k * std::sin(lat)));
  const double xi_p = std::atan2(t, std::cos(dlon));
  const double eta_p = std::atanh(std::sin(dlon) / std::sqrt(1.0 + t * t));
  double xi = xi_p, eta = eta_p;
  for (int j = 1; j <= 3; ++j) {
    xi += c.alpha[j - 1] * std::sin(2.0 * j * xi_p) * std::cosh(2.0 * j * eta_p);
    eta += c.alpha[j - 1] * std::cos(2.0 * j * xi_p) * std::sinh(2.0 * j * eta_p);
  }
  Utm u;
  u.zone = zone;
  u.north = lat_deg >= 0.0;
  u.easting = 500000.0 + kK0 * c.A * eta;
  u.northing = (u.north ? 0.0 : 10000000.0) + kK0 * c.A * xi;
  return u;
}

Geodetic utmToGeodetic(const Utm& u) {
  const KruegerCoeffs& c = coeffs();
  const double lon0 = (u.zone * 6.0 - 183.0) * kDeg;
  const double xi = (u.northing - (u.north ? 0.0 : 10000000.0)) / (kK0 * c.A);
  const double eta = (u.easting - 500000.0) / (kK0 * c.A);
  double xi_p = xi, eta_p = eta;
  for (int j = 1; j <= 3; ++j) {
    xi_p -= c.beta[j - 1] * std::sin(2.0 * j * xi) * std::cosh(2.0 * j * eta);
    eta_p -= c.beta[j - 1] * std::cos(2.0 * j * xi) * std::sinh(2.0 * j * eta);
  }
  const double chi = std::asin(std::sin(xi_p) / std::cosh(eta_p));
  double lat = chi;
  for (int j = 1; j <= 3; ++j) lat += c.delta[j - 1] * std::sin(2.0 * j * chi);
  const double lon = lon0 + std::atan2(std::sinh(eta_p), std::cos(xi_p));
  return {lat / kDeg, lon / kDeg, 0.0};
}

}  // namespace tbo::geo
