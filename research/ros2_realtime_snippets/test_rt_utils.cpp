#include "rt_utils.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char ** argv) {
  using namespace tram_rt;
  using V = StampSanitizer::Verdict;
  {  // synthetic cases
    StampSanitizer s;
    assert(s.check(100.0, 0.0) == V::kAccepted);
    assert(s.check(100.1, 0.1) == V::kAccepted);
    assert(s.check(101.15, 0.15) == V::kRejected);  // +1 s ghost duplicate
    assert(s.check(100.2, 0.2) == V::kAccepted);
    assert(s.check(99.25, 0.25) == V::kRejected);   // -1 s ghost duplicate
    assert(s.check(173.0, 73.2) == V::kAccepted);   // 73 s dropout, consistent with elapsed time
    // bag loop: time jumps back 1000 s and stays there -> resync after K msgs
    int resynced = 0;
    for (int i = 0; i < 10; ++i) if (s.check(10.0 + 0.1 * i, 80.0 + 0.1 * i) == V::kResynced) ++resynced;
    assert(resynced == 1);
  }
  {
    RingBuffer<int, 4> rb;
    for (int i = 0; i < 6; ++i) rb.push(i);
    assert(rb.size() == 4 && rb.at(0) == 2 && rb.back() == 5);
    rb.truncate_after(1);
    assert(rb.size() == 2 && rb.back() == 3);
  }
  {
    LatencyHistogram h;
    for (int i = 1; i <= 100; ++i) h.add(i * 0.1);
    assert(std::abs(h.percentile(50) - 5.0) < 0.06);
  }
  {
    std::array<double, 36> c;
    fill_pose_covariance(c, 1.5707963267948966, 4.0, 1.0, 1, 1, 1);
    assert(std::abs(c[0] - 1.0) < 1e-12 && std::abs(c[7] - 4.0) < 1e-12);
  }
  std::puts("unit checks OK");

  // replay CSV files: recv,stamp (one topic per file), measure per-call cost
  for (int a = 1; a < argc; ++a) {
    std::ifstream f(argv[a]);
    std::string line;
    StampSanitizer s;
    double last = -1;
    long n = 0, nonmono = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (std::getline(f, line)) {
      double r, st; char comma;
      std::istringstream is(line);
      is >> r >> comma >> st;
      ++n;
      if (s.check(st, r) != V::kRejected) { if (st <= last) ++nonmono; last = st; }
    }
    auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: n=%ld rejected=%llu resyncs=%llu nonmono=%ld (%.2f us/msg incl. CSV parse)\n", argv[a], n,
                static_cast<unsigned long long>(s.rejected()), static_cast<unsigned long long>(s.resyncs()), nonmono,
                us / n);
  }
  return 0;
}
