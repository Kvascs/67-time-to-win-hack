#include "tbo/params.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace tbo {

#define TBO_P(field, unit, doc) {#field, &Params::field, unit, doc}

static const ParamInfo kRegistry[] = {
    TBO_P(wheel_kmh_to_ms, "m/s per km/h", "nominal bogie speed calibration"),
    TBO_P(wheel_curv_abs, "m", "wheel under-reading in curves, |curvature| coefficient"),
    TBO_P(wheel_curv_signed, "m", "wheel under-reading in curves, signed curvature coefficient"),
    TBO_P(wheel_curv_sat, "-", "saturation of the curve correction"),
    TBO_P(front_bogie_along_m, "m", "front bogie ahead of antenna 1"),
    TBO_P(rear_bogie_along_m, "m", "rear bogie ahead of antenna 1"),
    TBO_P(body_rear_m, "m", "car body rear end relative to antenna 1"),
    TBO_P(body_front_m, "m", "car body front end relative to antenna 1"),
    TBO_P(wheel_max_kmh, "km/h", "readings above are rejected as invalid"),
    TBO_P(wheel_delay_s, "s", "bogie speed latency compensation"),
    TBO_P(cmd_delay_s, "s", "notch -> drive dead time"),
    TBO_P(lag_window_s, "s", "fixed-lag window for out-of-order messages"),
    TBO_P(max_step_s, "s", "max integration step"),
    TBO_P(wheel_timeout_s, "s", "bogie dropout timeout"),
    TBO_P(cmd_timeout_s, "s", "controller dropout timeout"),
    TBO_P(max_future_s, "s", "stamps further ahead need a confirming message"),
    TBO_P(max_backjump_s, "s", "stamps further back need a confirming message (then new run)"),
    TBO_P(jump_confirm_s, "s", "two messages this close confirm a time-base jump"),
    TBO_P(new_run_gap_s, "s", "confirmed forward jump that starts a new run"),
    TBO_P(sigma_accel, "m/s^2", "model acceleration white noise"),
    TBO_P(q_disturbance, "(m/s^2)^2/s", "disturbance random walk"),
    TBO_P(q_scale, "1/s", "wheel scale random walk"),
    TBO_P(q_gain, "1/s", "traction gain random walk"),
    TBO_P(init_sigma_v, "m/s", "initial speed sigma"),
    TBO_P(init_sigma_d, "m/s^2", "initial disturbance sigma"),
    TBO_P(init_sigma_scale, "-", "initial wheel scale sigma"),
    TBO_P(init_sigma_gain, "-", "initial traction gain sigma"),
    TBO_P(gain_min, "-", "traction gain lower clamp"),
    TBO_P(gain_max, "-", "traction gain upper clamp"),
    TBO_P(disturbance_max, "m/s^2", "max unexplained acceleration (beyond = slip)"),
    TBO_P(disturbance_max_decel, "m/s^2", "max unexplained deceleration (emergency brake)"),
    TBO_P(sigma_wheel, "m/s", "bogie speed noise"),
    TBO_P(sigma_wheel_rel, "-", "relative bogie speed noise"),
    TBO_P(outlier_range, "m/s", "uniform outlier density support"),
    TBO_P(rate_to_bad, "1/s", "IMM nominal -> one bad"),
    TBO_P(rate_to_both_bad, "1/s", "IMM nominal -> both bad"),
    TBO_P(rate_recover, "1/s", "IMM bad -> nominal"),
    TBO_P(mode_prob_floor, "-", "IMM probability floor"),
    TBO_P(slip_context_boost, "-", "rate boost under high effort"),
    TBO_P(rate_to_maneuver, "1/s", "IMM nominal -> maneuver (unmodelled acceleration)"),
    TBO_P(rate_maneuver_end, "1/s", "IMM maneuver -> nominal"),
    TBO_P(sigma_accel_maneuver, "m/s^2", "acceleration noise in maneuver mode"),
    TBO_P(q_disturbance_maneuver, "(m/s^2)^2/s", "disturbance random walk in maneuver mode"),
    TBO_P(wrong_sign_penalty, "-", "log-penalty for impossible-sign fault residual"),
    TBO_P(max_wheel_accel, "m/s^2", "implausible bogie acceleration"),
    TBO_P(stuck_time_s, "s", "frozen reading duration"),
    TBO_P(stuck_min_change, "m/s", "speed change proving a frozen reading"),
    TBO_P(zero_stuck_other, "m/s", "other bogie speed proving a zero reading dead"),
    TBO_P(single_bogie_latch_mult, "-", "latch threshold multiplier with one live bogie"),
    TBO_P(cusum_slip_accel, "m/s^2", "CUSUM: tolerated excess accel under traction"),
    TBO_P(cusum_slide_accel, "m/s^2", "CUSUM: tolerated extra decel under braking"),
    TBO_P(cusum_h, "m/s", "CUSUM alarm threshold"),
    TBO_P(cmd_fault_accel, "m/s^2", "impossible-sign excess indicting the controller"),
    TBO_P(cmd_fault_h, "m/s", "accumulated excess raising controller fault"),
    TBO_P(cmd_fault_hold_s, "s", "controller distrust hold time"),
    TBO_P(disturbance_max_free, "m/s^2", "max unexplained accel when not under traction"),
    TBO_P(latch_release_abs, "m/s", "latch release gate (absolute)"),
    TBO_P(latch_release_rel, "-", "latch release gate (relative)"),
    TBO_P(latch_release_n, "-", "consistent samples to release latch"),
    TBO_P(latch_max_s, "s", "max joint anomaly bridged by the model"),
    TBO_P(standstill_kmh, "km/h", "standstill threshold"),
    TBO_P(standstill_time_s, "s", "standstill confirmation time"),
    TBO_P(standstill_max_v, "m/s", "lock-up guard for standstill"),
    TBO_P(recover_time_s, "s", "consistency time before re-anchoring"),
    TBO_P(recover_agree, "m/s", "front/rear agreement for re-anchoring"),
    TBO_P(recover_min_bad_s, "s", "min anomaly duration before re-anchoring"),
    TBO_P(drive_tau_s, "s", "drive acceleration lag"),
    TBO_P(map_grade_gain, "-", "gain of the map grade term in the dynamics"),
    TBO_P(kg_brake, "m/s^2", "grade coefficient under braking"),
    TBO_P(kg_coast, "m/s^2", "grade coefficient when coasting"),
    TBO_P(kg_traction, "m/s^2", "grade coefficient under traction"),
    TBO_P(curve_resist_coef, "m^2/s^2", "curve resistance coefficient (0 = off)"),
    TBO_P(dfield_gain, "-", "weight of the learned disturbance field d(s)"),
    TBO_P(grade_s_coupling, "bool", "EKF Jacobian includes d(grade accel)/ds"),
    TBO_P(gnss_init_window_s, "s", "GNSS accepted only this long after the first fix"),
    TBO_P(gnss_wait_s, "s", "position withheld this long waiting for the first fix"),
    TBO_P(gnss_min_fixes, "-", "fixes needed for initialisation"),
    TBO_P(map_gate_m, "m", "init fix to map distance gate"),
    TBO_P(map_heading_gate_deg, "deg", "init heading consistency gate"),
    TBO_P(map_sigma_cross, "m", "map cross-track sigma"),
    TBO_P(map_sigma_z, "m", "map altitude sigma"),
    TBO_P(init_sigma_s, "m", "along-track sigma after init"),
    TBO_P(init_sigma_s_per_m, "-", "extra along-track sigma per metre of init fix-to-map distance"),
    TBO_P(use_baseline_heading, "bool", "use master-rover baseline to pick direction"),
    TBO_P(landmark_enable, "bool", "use stop landmarks as along-track fixes"),
    TBO_P(landmark_dwell_s, "s", "standstill before a landmark fix"),
    TBO_P(landmark_gate_sigma, "sigma", "landmark association gate"),
    TBO_P(landmark_min_prob, "-", "landmark association posterior threshold"),
    TBO_P(landmark_p_random, "-", "share of stops not at a landmark"),
    TBO_P(landmark_sigma_extra, "m", "extra landmark position sigma"),
    TBO_P(landmark_max_dk, "-", "max wheel-scale change per landmark fix"),
    TBO_P(cutoff_enable, "bool", "use traction cut-off landmarks"),
    TBO_P(cutoff_notch, "-", "min notch before an abrupt cut to 0"),
    TBO_P(cutoff_min_v, "m/s", "min speed for a cut-off fix"),
    TBO_P(cutoff_p_random, "-", "share of cut-offs away from known places"),
    TBO_P(position_lead_s, "s", "position published for stamp + lead (reference fix timing)"),
    TBO_P(mgrs_zone, "-", "UTM zone of the MGRS output frame"),
    TBO_P(mgrs_origin_e, "m", "UTM easting of the MGRS square origin"),
    TBO_P(mgrs_origin_n, "m", "UTM northing of the MGRS square origin"),
    TBO_P(base_link_along_m, "m", "published point ahead of antenna 1 along the track"),
    TBO_P(base_link_height_m, "m", "antenna 1 height above rail top"),
    TBO_P(bogie_base_m, "m", "distance between bogie centres (heading chord)"),
    TBO_P(publish_grid_s, "s", "fixed stamp grid period (0 = off)"),
    TBO_P(publish_on_cmd, "bool", "publish at controller stamps"),
    TBO_P(publish_on_wheel, "bool", "publish at bogie stamps"),
};

#undef TBO_P

const ParamInfo* paramRegistry(int* count) {
  if (count) *count = static_cast<int>(sizeof(kRegistry) / sizeof(kRegistry[0]));
  return kRegistry;
}

bool setParam(Params& p, const std::string& name, double value) {
  int n = 0;
  const ParamInfo* reg = paramRegistry(&n);
  for (int i = 0; i < n; ++i)
    if (name == reg[i].name) {
      p.*(reg[i].member) = value;
      return true;
    }
  return false;
}

namespace {
std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\"'");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\"'");
  return s.substr(b, e - b + 1);
}
}  // namespace

bool loadFlatYaml(const std::string& path, Config& cfg, std::string* err, std::string* unknown) {
  std::ifstream in(path);
  if (!in) {
    if (err) *err = "cannot open " + path;
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = trim(line.substr(0, colon));
    const std::string val = trim(line.substr(colon + 1));
    if (key.empty() || val.empty()) continue;  // section headers like "ros__parameters:"
    if (key == "map_file") { cfg.map_file = val; continue; }
    if (key == "traction_file") { cfg.traction_file = val; continue; }
    if (key == "branch_files") { cfg.branch_files = val; continue; }
    if (key == "landmark_file") { cfg.landmark_file = val; continue; }
    if (key == "cutoff_file") { cfg.cutoff_file = val; continue; }
    if (key == "dfield_file") { cfg.dfield_file = val; continue; }
    if (key == "output_frame") { cfg.output_frame = val; continue; }
    if (key == "init_source") { cfg.init_source = val; continue; }
    if (key == "frame_id") { cfg.frame_id = val; continue; }
    if (key == "child_frame_id") { cfg.child_frame_id = val; continue; }
    char* end = nullptr;
    double v = 0.0;
    if (val == "true") v = 1.0;
    else if (val == "false") v = 0.0;
    else {
      v = std::strtod(val.c_str(), &end);
      if (end == val.c_str()) {  // non-numeric value for an unknown key
        if (unknown) *unknown += key + " ";
        continue;
      }
    }
    if (!setParam(cfg.p, key, v) && unknown) *unknown += key + " ";
  }
  return true;
}

}  // namespace tbo
