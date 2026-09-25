#include <cstdio>

#include "mini_test.hpp"

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep output even if a test crashes
  return mini_test::runAll();
}
