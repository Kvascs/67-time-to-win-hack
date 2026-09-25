// Decides which output stamps to publish after each input, shared by the ROS node and
// the offline replay so both emit exactly the same stream:
//  * at every input stamp (controller and/or bogie messages) -> "input time" outputs;
//  * on a fixed stamp grid (multiples of publish_grid_s), as soon as input time has
//    passed the grid point -> outputs aligned with 10 Hz GNSS reference stamps.
// Published stamps are strictly increasing.
#pragma once

#include "tbo/params.hpp"
#include "tbo/types.hpp"

namespace tbo {

class OutputScheduler {
 public:
  explicit OutputScheduler(const Params& p) : p_(p) {}
  void reset() {
    last_pub_ = -1;
    next_grid_ = -1;
  }

  // Returns the number of stamps written to `out` (ascending), at most `max_out`.
  int onInput(bool is_cmd, Stamp input_stamp, Stamp latest_stamp, Stamp* out, int max_out) {
    int n = 0;
    const Stamp grid = p_.publish_grid_s > 0.0 ? fromSec(p_.publish_grid_s) : 0;
    const bool want_input = is_cmd ? p_.publish_on_cmd > 0.5 : p_.publish_on_wheel > 0.5;
    if (grid > 0) {
      if (next_grid_ < 0) next_grid_ = (latest_stamp / grid) * grid;  // first grid point at/before start
      while (next_grid_ <= latest_stamp && n < max_out) {
        if (want_input && input_stamp < next_grid_ && input_stamp > last_pub_ && n < max_out) {
          out[n++] = input_stamp;
          last_pub_ = input_stamp;
        }
        if (next_grid_ > last_pub_) {
          out[n++] = next_grid_;
          last_pub_ = next_grid_;
        }
        next_grid_ += grid;
      }
    }
    if (want_input && input_stamp > last_pub_ && n < max_out) {
      out[n++] = input_stamp;
      last_pub_ = input_stamp;
    }
    return n;
  }

 private:
  const Params& p_;
  Stamp last_pub_ = -1;
  Stamp next_grid_ = -1;
};

}  // namespace tbo
