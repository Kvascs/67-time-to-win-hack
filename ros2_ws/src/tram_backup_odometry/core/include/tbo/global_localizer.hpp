// GNSS-free global localisation on the closed main cycle (port of research/global_loc/glocal.py).
//
// Without a GNSS anchor the filter still integrates the travelled distance r, but on the raw wheel
// scale (no map: no curve correction) and with an unknown wheel scale error kappa. A recursive grid
// (histogram) filter over the start place u0 (in the wheel-distance coordinate u of the cycle, step du)
// and kappa (rows) finds the place from cues that happen at known places:
//   stop     standstill >= dwell: likelihood 1 + sum_m p_m N(u; u_m, sd_m^2) / lambda_stop
//   pass     known stop places passed without a stop: prod (1 - p_m)
//   cutoff   traction notch >= 4 -> 0 at speed: 1 + sum_c q_c N(u; u_c, sd_c^2) / lambda_cut
//   cutpass  a cut-off place passed at notch >= 4 without a cut-off: (1 - q_c)
//   speed    max speed over the last 50 m above the envelope of that place + margin: v_eps
// Cells never move: the current place of cell i in row j after odometer r is u0_i + r / (1 + kappa_j),
// so every map lookup of a row is a contiguous slice of a periodic table. The odometry noise is a
// blur along u0. Fix: posterior mass within +-win of the mode >= p_fix after >= min_cues positive cues.
// Validation of the prototype: fix after median 176 s / 355 m, 0 false fixes on 64 runs with a reference.
#pragma once

#include <vector>

#include "tbo/track_field.hpp"
#include "tbo/track_map.hpp"

namespace tbo {

struct Landmark;

class GlobalLocalizer {
 public:
  struct Params {
    double du = 0.5;              // grid step of u0, m
    double k_min = -0.025, k_max = 0.025, k_step = 0.001, k_prior_sd = 0.012;
    double q_x = 0.004;           // odometry random walk, m^2 per m travelled
    double jump_rate = 1e-4;      // odometry jumps per m travelled
    double jump_hw = 20.0;        // half-width of a jump, m
    double floor = 1e-9;          // uniform re-seeding per update
    double checkpoint = 50.0;     // m of travel between moving updates
    double stop_dwell = 1.5;      // s of standstill that makes a stop
    double lm_sd_extra = 0.5;     // m, added in quadrature to the stop spread
    double stop_rate = 7.49e-4 * 1.5;  // random stops per m (train rate, conservative x1.5)
    double guard = 3.0;           // m around a stop that explains a landmark (no pass penalty)
    double p_max = 0.9;           // cap for negative information
    double cut_sd_min = 1.5, cut_rate = 5e-5;
    int cut_notch = 4;
    double cut_vmin = 2.0, cut_lead = 0.045;
    double v_margin = 1.5, v_eps = 0.05;
    double win = 10.0, p_fix = 0.99;
    int min_cues = 2;
    // wheel curve correction of the core and the bogie places (the odometer runs uncorrected)
    double curv_abs = 0.415, curv_signed = -0.054, curv_sat = 0.009;
    double front_along = 9.873, rear_along = 2.323;
  };
  struct Estimate {
    double conf = 0.0;   // posterior mass within +-win of the mode
    double s = 0.0;      // main-cycle arc length of the current place (antenna 1)
    double s_var = 0.0;  // variance of the place inside the window, m^2
    double kappa = 0.0;  // wheel scale error
    double kappa_sd = 1.0;  // its posterior spread within the window
    int cues = 0;
  };

  // Returns false (and stays inactive) when the map is not a closed cycle or there are no stops.
  bool init(const TrackMap& main, const std::vector<Landmark>& stops, const std::vector<Landmark>& cutoffs,
            const TrackField& vmax, const Params& p);
  bool active() const { return nx_ > 0; }
  // One committed filter step at time t (s): odometer r (m since the localiser started), speed,
  // standstill flag and controller notch.
  void step(double t, double r, double v, bool standstill, int notch);
  bool fixed() const { return fixed_; }
  const Estimate& fix() const { return fix_; }
  Estimate estimate(double r) const;
  double lastOdometer() const { return r_last_; }

 private:
  // map geometry in the wheel-distance coordinate
  double uOfS(double s) const;
  double sOfU(double u) const;
  double shiftCells(double r, double off, int j) const { return (r * c_[j] + off) / du_; }
  void normalise();
  void blur(double r, bool force_jump);
  void passThrough(double r_to, double off_to);
  void cutPass(double r0, double r1);
  void multiplyFine(const std::vector<float>& tab, double r);
  int notchAt(double r) const;
  void onStop(double r);
  void onCutoff(double r);
  void onCheckpoint(double r);
  void afterUpdate(double r);

  Params p_;
  double L_ = 0.0, LU_ = 0.0, du_ = 0.0;
  int nx_ = 0, nk_ = 0;
  static constexpr int kSub = 10;
  std::vector<double> sg_, ug_;          // s grid and u at those s
  std::vector<double> kap_, c_;          // kappa per row and 1 / (1 + kappa)
  std::vector<float> G_;                 // nk x nx posterior over (kappa, u0)
  std::vector<float> stop_tab_, cut_tab_;  // fine periodic likelihood tables (nx * kSub)
  std::vector<double> pass_cum_;         // cumulative log(1 - p) over 3 laps
  double pass_lap_ = 0.0;
  std::vector<float> v_tab_;             // speed envelope over the last checkpoint, 3 laps (empty: off)
  std::vector<double> cut_u_, cut_q_;
  // odometry bookkeeping
  double r_last_ = 0.0, r_blur_ = 0.0, var_acc_ = 0.0, eps_acc_ = 0.0;
  double r_pass_ = 0.0, off_pass_ = 0.0, r_cp_ = 0.0, vmax_cp_ = 0.0;
  bool moved_ = false, still_ = false, stop_done_ = false;
  double still_r_ = 0.0, still_t_acc_ = 0.0;
  int prev_notch_ = 0;
  std::vector<std::pair<double, int>> notch_hist_;  // (odometer, notch) at changes
  std::vector<double> cut_events_;
  int cues_ = 0;
  bool fixed_ = false;
  Estimate fix_;
  double t_last_ = -1.0;
};

}  // namespace tbo
