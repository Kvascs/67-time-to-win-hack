// All tunable numbers of the estimator in one place, with a registry that the ROS
// node uses to declare parameters and the offline replay uses to read the same YAML.
#pragma once

#include <string>

namespace tbo {

struct Params {
  // ---- sensor units / calibration ----
  double wheel_kmh_to_ms = 1.0 / 3.5965;  // nominal: bogie topics carry km/h, calibrated on train bags
  double wheel_max_kmh = 110.0;           // readings above are physically impossible -> invalid
  double wheel_delay_s = 0.0;             // measurement latency: value at stamp t describes t - delay
  double cmd_delay_s = 0.0;               // dead time from notch change to drive response

  // ---- timing ----
  double lag_window_s = 0.30;       // fixed-lag window to absorb out-of-order messages
  double max_step_s = 0.05;         // max integration step
  double wheel_timeout_s = 0.6;     // no valid sample for longer -> dropout
  double cmd_timeout_s = 0.6;       // controller silent for longer -> assume neutral + flag
  double max_future_s = 2.0;        // stamps further ahead of the latest are rejected
  double max_backjump_s = 30.0;     // stamps jumping back more than this reset the time base

  // ---- process model noise ----
  double sigma_accel = 0.12;        // white acceleration noise of the model, m/s^2
  double q_disturbance = 0.004;     // random walk of disturbance d, (m/s^2)^2 / s
  double q_scale = 1e-9;            // random walk of wheel scale error k, 1 / s
  double q_gain = 2e-5;             // random walk of traction gain g, 1 / s
  double init_sigma_v = 0.3;
  double init_sigma_d = 0.10;
  double init_sigma_scale = 0.004;
  double init_sigma_gain = 0.08;
  double gain_min = 0.6, gain_max = 1.6;
  double disturbance_max = 1.0;     // |d| clamp, m/s^2

  // ---- measurement model ----
  double sigma_wheel = 0.05;        // per-bogie speed noise, m/s
  double sigma_wheel_rel = 0.004;   // + relative part (quantisation, creep), fraction of speed
  double outlier_range = 30.0;      // support of the uniform "bad sensor" density, m/s

  // ---- IMM mode switching ----
  double rate_to_bad = 0.05;        // 1/s: nominal -> one sensor bad
  double rate_to_both_bad = 0.01;   // 1/s: nominal -> both bad
  double rate_recover = 0.7;        // 1/s: bad -> nominal
  double mode_prob_floor = 1e-6;
  double slip_context_boost = 4.0;  // x rate_to_bad under high traction / braking effort

  // ---- plausibility gates ----
  double max_wheel_accel = 2.5;     // |dv/dt| of a bogie beyond this is not vehicle motion, m/s^2
  double stuck_time_s = 1.2;        // identical readings for this long ...
  double stuck_min_change = 0.4;    // ... while the vehicle speed changed by more than this (m/s)

  // ---- standstill (zero-velocity update) ----
  double standstill_kmh = 0.15;     // both bogies below -> candidate standstill
  double standstill_time_s = 0.25;  // held for this long
  double standstill_max_v = 1.2;    // only if the filter speed is already this low (lock-up guard)

  // ---- recovery after long anomalies ----
  double recover_time_s = 3.0;      // wheels mutually consistent this long while filter rejects them
  double recover_agree = 0.35;      // |front - rear| agreement, m/s
  double recover_min_bad_s = 5.0;   // only after the filter has been in bad modes this long

  // ---- traction / drive model ----
  double drive_tau_s = 0.30;        // first-order lag of realised drive acceleration

  // ---- initialisation / map ----
  double gnss_init_window_s = 5.0;  // GNSS used only this long after the first message
  double gnss_min_fixes = 3.0;
  double map_gate_m = 25.0;         // max distance of the init fix from the map
  double map_heading_gate_deg = 60.0;
  double map_sigma_cross = 0.30;    // map cross-track accuracy for the pose covariance, m
  double map_sigma_z = 0.50;
  double init_sigma_s = 1.0;        // along-track uncertainty after GNSS init, m
  double use_baseline_heading = 1.0;

  // ---- output ----
  double publish_grid_s = 0.05;     // also publish on a fixed stamp grid (0 disables)
  double publish_on_cmd = 1.0;      // publish at every controller stamp
  double publish_on_wheel = 1.0;    // publish at every bogie stamp
};

struct ParamInfo {
  const char* name;
  double Params::*member;
  const char* unit;
  const char* doc;
};

// Registry of every numeric parameter (single source of truth for names/defaults).
const ParamInfo* paramRegistry(int* count);
bool setParam(Params& p, const std::string& name, double value);

// String-valued configuration kept separate from the numeric registry.
struct Config {
  Params p;
  std::string map_file;                  // track map CSV (empty -> dead reckoning)
  std::string traction_file;             // traction LUT CSV (empty -> built-in table)
  std::string output_frame = "enu";      // enu | utm | map
  std::string init_source = "master";    // master | rover
  std::string frame_id = "map";
  std::string child_frame_id = "base_link";
};

// Reads "key: value" pairs from a (possibly nested) YAML file, ignoring structure.
// Unknown keys are reported through `unknown` so tests can assert none exist.
bool loadFlatYaml(const std::string& path, Config& cfg, std::string* err, std::string* unknown = nullptr);

}  // namespace tbo
