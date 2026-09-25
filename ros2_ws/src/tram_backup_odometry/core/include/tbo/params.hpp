// All tunable numbers of the estimator in one place, with a registry that the ROS
// node uses to declare parameters and the offline replay uses to read the same YAML.
#pragma once

#include <string>

namespace tbo {

struct Params {
  // ---- sensor units / calibration ----
  double wheel_kmh_to_ms = 1.00037 / 3.6;  // bogie topics carry km/h; straight-track scale from train bags
  double wheel_curv_abs = 0.415;    // wheels under-read in curves: v = v_wheel (1 + a|k| + b k), a
  double wheel_curv_signed = -0.054;  // b (per 1/m of track curvature k from the map)
  double wheel_curv_sat = 0.009;    // the curve under-reading saturates at ~0.9 % (R < 60 m)
  double front_bogie_along_m = 9.9; // bogie positions ahead of antenna 1 along the track (curvature
  double rear_bogie_along_m = 2.35; //   for each bogie's wheel correction is taken at its own place)
  double body_rear_m = -2.1;        // car body extent relative to antenna 1 (grade averaged over it)
  double body_front_m = 14.4;
  double wheel_max_kmh = 110.0;           // readings above are physically impossible -> invalid
  double wheel_delay_s = 0.0;             // measurement latency: value at stamp t describes t - delay
  double cmd_delay_s = 0.0;               // dead time from notch change to drive response

  // ---- timing ----
  double lag_window_s = 0.30;       // fixed-lag window to absorb out-of-order messages
  double max_step_s = 0.05;         // max integration step
  double wheel_timeout_s = 0.6;     // no valid sample for longer -> dropout
  double cmd_timeout_s = 0.6;       // controller silent for longer -> assume neutral + flag
  // Time-base jumps. Header stamps are continuous (gaps <= 0.23 s in all bags), but arrival order
  // interleaves topics: up to 2.6 s ahead / 2.7 s behind in the start-up burst, ~1 s mid-run.
  // A stamp outside [latest - max_backjump_s, latest + max_future_s] is a glitch unless a second
  // message within jump_confirm_s of it confirms that the time base itself jumped.
  double max_future_s = 5.0;        // further ahead: needs confirmation (then: inputs were silent)
  double max_backjump_s = 10.0;     // further back: needs confirmation (then: bag replayed, new run)
  double jump_confirm_s = 1.0;      // two messages this close confirm a time jump
  double new_run_gap_s = 60.0;      // a confirmed forward jump this long is a new run (another bag)

  // ---- process model noise ----
  double sigma_accel = 0.12;        // white acceleration noise of the model, m/s^2
  double q_disturbance = 0.004;     // random walk of disturbance d, (m/s^2)^2 / s
  double q_scale = 1e-9;            // random walk of wheel scale error k, 1 / s
  double q_gain = 2e-5;             // random walk of traction gain g, 1 / s
  double init_sigma_v = 0.3;
  double init_sigma_d = 0.10;
  double init_sigma_scale = 0.015;  // fleet k spans +-1.6 % by vehicle/date: a wide prior lets stops calibrate it
  double init_sigma_gain = 0.08;
  double gain_min = 0.6, gain_max = 1.6;
  double disturbance_max = 0.6;     // max unexplained acceleration (grade), m/s^2: more is slip
  double disturbance_max_decel = 4.0;  // max unexplained deceleration (grade, emergency brake), m/s^2

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
  double rate_to_maneuver = 0.05;   // 1/s: nominal -> unmodelled acceleration (emergency brake...)
  double rate_maneuver_end = 0.5;   // 1/s: maneuver -> nominal
  double sigma_accel_maneuver = 0.5;  // acceleration noise in the maneuver mode, m/s^2
  double q_disturbance_maneuver = 8.0;  // fast disturbance adaptation in maneuver mode, (m/s^2)^2/s
  double wrong_sign_penalty = 3.0;  // log-penalty for a "bad sensor" residual of the impossible sign

  // ---- plausibility gates ----
  double max_wheel_accel = 6.0;     // |dv/dt| of a bogie beyond this is not vehicle motion, m/s^2
  double stuck_time_s = 1.2;        // identical readings for this long ...
  double stuck_min_change = 0.15;   // ... while the vehicle speed changed by more than this (m/s)
  double zero_stuck_other = 0.5;    // a bogie at 0 while the other reads above this (m/s) is dead
  double single_bogie_latch_mult = 2.0;  // CUSUM threshold multiplier with only one live bogie

  // ---- joint slip/slide monitor: Page CUSUM on (bogie acceleration - model acceleration) ----
  // Only physically possible anomalies latch: slip under traction (wheels fast), slide under
  // braking (wheels slow). The opposite signs mean the controller signal is wrong instead.
  double cusum_slip_accel = 0.60;   // tolerated excess acceleration under traction, m/s^2
  double cusum_slide_accel = 0.80;  // tolerated extra deceleration under braking notches, m/s^2
  double cusum_h = 0.30;            // alarm threshold, m/s (accumulated excess speed)
  double cmd_fault_accel = 0.50;    // excess of the impossible sign that indicts the controller, m/s^2
  double cmd_fault_h = 0.40;        // accumulated impossible-sign excess to raise the fault, m/s
  double cmd_fault_hold_s = 8.0;    // controller distrusted this long after the last evidence
  double disturbance_max_free = 2.0;  // max unexplained acceleration when not under traction, m/s^2
  double latch_release_abs = 0.30;  // wheels back on the model trajectory within this, m/s ...
  double latch_release_rel = 0.04;  // ... or this fraction of speed
  double latch_release_n = 2.0;     // consecutive consistent samples to release
  double latch_max_s = 10.0;        // longest joint anomaly bridged by the model before re-anchoring

  // ---- standstill (zero-velocity update) ----
  double standstill_kmh = 0.15;     // both bogies below -> candidate standstill
  double standstill_time_s = 0.25;  // held for this long
  double standstill_max_v = 1.2;    // only if the filter speed is already this low (lock-up guard)

  // ---- recovery after long anomalies ----
  double recover_time_s = 3.0;      // wheels mutually consistent this long while filter rejects them
  double recover_agree = 0.35;      // |front - rear| agreement, m/s
  double recover_min_bad_s = 5.0;   // only after the filter has been in bad modes this long

  // ---- traction / drive model ----
  double drive_tau_s = 0.452;       // first-order lag of realised drive force (output-error fit)
  double map_grade_gain = 1.0;      // multiplier of the grade term below (0 disables)
  double kg_brake = 8.22;           // grade -> acceleration, m/s^2 per unit grade: g/(1+rho),
  double kg_coast = 7.98;           //   rho = rotating-mass factor; identified per regime
  double kg_traction = 7.36;
  double curve_resist_coef = 0.0;   // curve resistance a = -coef * |curvature|, m^2/s^2
  double dfield_gain = 1.0;         // weight of the learned disturbance field d(s) (when a file is given)
  double grade_s_coupling = 0.0;    // 1: EKF Jacobian includes d(grade accel)/ds (position seen via grade)

  // ---- initialisation / map ----
  double gnss_init_window_s = 5.0;  // GNSS used only this long after the first fix
  double gnss_wait_s = 4.0;         // hold position output this long for the first fix, then go relative (start-up burst ~2.5 s)
  double gnss_min_fixes = 3.0;
  double map_gate_m = 25.0;         // max distance of the init fix from the map
  double map_heading_gate_deg = 60.0;
  double map_sigma_cross = 0.30;    // map cross-track accuracy for the pose covariance, m
  double map_sigma_z = 0.50;
  double init_sigma_s = 1.0;        // along-track uncertainty after GNSS init, m
  double init_sigma_s_per_m = 0.7;  // + this x distance of the init fix from the map (unmapped tracks)
  double use_baseline_heading = 1.0;

  // ---- stop landmarks ("virtual balises"): known standstill positions on the map ----
  double landmark_enable = 1.0;
  double landmark_dwell_s = 1.5;    // standstill this long before using the stop as a fix
  double landmark_gate_sigma = 3.0; // association gate in standard deviations
  double landmark_min_prob = 0.6;   // posterior probability required to apply the fix
  double landmark_p_random = 0.15;  // prior share of stops not at any landmark (traffic)
  double landmark_sigma_extra = 0.3;  // added to the landmark spread, m
  double landmark_max_dk = 0.004;   // max wheel-scale change applied by one landmark fix
  double cutoff_enable = 1.0;       // traction cut-off landmarks (notch >= cutoff_notch -> 0)
  double cutoff_notch = 4.0;
  double cutoff_min_v = 2.0;        // only while moving faster than this, m/s
  double cutoff_p_random = 0.05;    // share of abrupt cut-offs away from the known places

  // ---- output ----
  double position_lead_s = 0.045;   // GNSS fixes lead wheel/vel stamps: publish s(t + lead)
  // Reference frame of the jury: Autoware map frame = MGRS 100 km square (Moscow: 37U DB).
  double mgrs_zone = 37.0;
  double mgrs_origin_e = 400000.0;  // UTM easting of the square's west edge
  double mgrs_origin_n = 6100000.0; // UTM northing of the square's south edge
  // Published point = base_link (centre of the front bogie at rail-top level, REP-103).
  double base_link_along_m = 9.9;   // front bogie ahead of antenna 1 (fitted: body heading vs chord)
  double base_link_height_m = 3.5;  // antenna 1 above rail top (assumption until the TF is given)
  double bogie_base_m = 7.55;       // distance between bogie centres (organisers)
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
  std::string branch_files;              // comma-separated branch CSVs merging into the main cycle
  std::string landmark_file;             // stop landmarks CSV (main cycle)
  std::string cutoff_file;               // traction cut-off landmarks CSV (main cycle)
  std::string dfield_file;               // learned disturbance field d(s) CSV (main cycle), empty = off
  std::string output_frame = "mgrs";     // mgrs (jury) | enu (first fix) | utm | map
  std::string init_source = "master";    // master | rover
  std::string frame_id = "map";
  std::string child_frame_id = "base_link";
};

// Reads "key: value" pairs from a (possibly nested) YAML file, ignoring structure.
// Unknown keys are reported through `unknown` so tests can assert none exist.
bool loadFlatYaml(const std::string& path, Config& cfg, std::string* err, std::string* unknown = nullptr);

}  // namespace tbo
