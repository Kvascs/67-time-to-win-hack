// Minimal self-registering test framework: lets the core be tested with any
// compiler (MinGW on Windows, gcc in ros:humble) without gtest being installed.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace mini_test {

struct Case {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

inline int& failures() {
  static int f = 0;
  return f;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

inline int runAll() {
  int failed_cases = 0;
  for (const auto& c : registry()) {
    const int before = failures();
    c.fn();
    const bool ok = failures() == before;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", c.name);
    if (!ok) ++failed_cases;
  }
  std::printf("%zu cases, %d failed\n", registry().size(), failed_cases);
  // A run that collected zero tests is a failure, never a silent pass.
  return (registry().empty() || failed_cases > 0) ? 1 : 0;
}

}  // namespace mini_test

#define MT_CONCAT2(a, b) a##b
#define MT_CONCAT(a, b) MT_CONCAT2(a, b)
#define TEST_CASE(name)                                                             \
  static void MT_CONCAT(test_fn_, __LINE__)();                                      \
  static mini_test::Registrar MT_CONCAT(test_reg_, __LINE__)(name, &MT_CONCAT(test_fn_, __LINE__)); \
  static void MT_CONCAT(test_fn_, __LINE__)()

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::printf("  %s:%d CHECK failed: %s\n", __FILE__, __LINE__, #cond);         \
      ++mini_test::failures();                                                      \
    }                                                                               \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                       \
  do {                                                                              \
    const double mt_a = (a), mt_b = (b);                                            \
    if (!(std::fabs(mt_a - mt_b) <= (tol))) {                                       \
      std::printf("  %s:%d CHECK_NEAR failed: %s=%.9g %s=%.9g tol=%g\n", __FILE__,   \
                  __LINE__, #a, mt_a, #b, mt_b, (double)(tol));                     \
      ++mini_test::failures();                                                      \
    }                                                                               \
  } while (0)
