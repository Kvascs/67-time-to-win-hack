// Fixed-size dense matrices for the tiny estimator state (N <= 6).
// Header-only, value semantics, no heap allocation: safe for the real-time path.
#pragma once

#include <cmath>
#include <cstddef>

namespace tbo {

template <int R, int C>
struct Mat {
  double a[R][C]{};

  static Mat zero() { return Mat{}; }
  static Mat identity() {
    static_assert(R == C, "identity needs a square matrix");
    Mat m;
    for (int i = 0; i < R; ++i) m.a[i][i] = 1.0;
    return m;
  }

  double& operator()(int r, int c) { return a[r][c]; }
  double operator()(int r, int c) const { return a[r][c]; }

  Mat& operator+=(const Mat& o) {
    for (int i = 0; i < R; ++i)
      for (int j = 0; j < C; ++j) a[i][j] += o.a[i][j];
    return *this;
  }
  Mat& operator-=(const Mat& o) {
    for (int i = 0; i < R; ++i)
      for (int j = 0; j < C; ++j) a[i][j] -= o.a[i][j];
    return *this;
  }
  Mat& operator*=(double s) {
    for (int i = 0; i < R; ++i)
      for (int j = 0; j < C; ++j) a[i][j] *= s;
    return *this;
  }
};

template <int N>
using Vec = Mat<N, 1>;

template <int R, int C>
Mat<R, C> operator+(Mat<R, C> x, const Mat<R, C>& y) { return x += y; }
template <int R, int C>
Mat<R, C> operator-(Mat<R, C> x, const Mat<R, C>& y) { return x -= y; }
template <int R, int C>
Mat<R, C> operator*(Mat<R, C> x, double s) { return x *= s; }
template <int R, int C>
Mat<R, C> operator*(double s, Mat<R, C> x) { return x *= s; }

template <int R, int K, int C>
Mat<R, C> operator*(const Mat<R, K>& x, const Mat<K, C>& y) {
  Mat<R, C> out;
  for (int i = 0; i < R; ++i)
    for (int k = 0; k < K; ++k) {
      const double xik = x.a[i][k];
      if (xik == 0.0) continue;
      for (int j = 0; j < C; ++j) out.a[i][j] += xik * y.a[k][j];
    }
  return out;
}

template <int R, int C>
Mat<C, R> transpose(const Mat<R, C>& x) {
  Mat<C, R> out;
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j) out.a[j][i] = x.a[i][j];
  return out;
}

template <int N>
void symmetrize(Mat<N, N>& m) {
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      const double v = 0.5 * (m.a[i][j] + m.a[j][i]);
      m.a[i][j] = v;
      m.a[j][i] = v;
    }
}

// Clamp diagonal to a floor and keep the matrix symmetric; guards against
// covariance collapse after many updates.
template <int N>
void conditionCovariance(Mat<N, N>& m, double floor_var) {
  symmetrize(m);
  for (int i = 0; i < N; ++i)
    if (!(m.a[i][i] >= floor_var)) m.a[i][i] = floor_var;  // also catches NaN
}

template <int N>
bool allFinite(const Mat<N, 1>& v) {
  for (int i = 0; i < N; ++i)
    if (!std::isfinite(v.a[i][0])) return false;
  return true;
}

template <int R, int C>
bool allFiniteM(const Mat<R, C>& m) {
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j)
      if (!std::isfinite(m.a[i][j])) return false;
  return true;
}

// Inverse and determinant for the innovation covariance (dimension 1 or 2).
inline bool invert(const Mat<1, 1>& s, Mat<1, 1>& inv, double& det) {
  det = s.a[0][0];
  if (!(det > 0.0)) return false;
  inv.a[0][0] = 1.0 / det;
  return true;
}

inline bool invert(const Mat<2, 2>& s, Mat<2, 2>& inv, double& det) {
  det = s.a[0][0] * s.a[1][1] - s.a[0][1] * s.a[1][0];
  if (!(det > 0.0) || !(s.a[0][0] > 0.0)) return false;
  const double id = 1.0 / det;
  inv.a[0][0] = s.a[1][1] * id;
  inv.a[1][1] = s.a[0][0] * id;
  inv.a[0][1] = -s.a[0][1] * id;
  inv.a[1][0] = -s.a[1][0] * id;
  return true;
}

}  // namespace tbo
