#include "tbo/params.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace tbo {

#define TBO_P(field, unit, doc) {#field, &Params::field, unit, doc}

static const ParamInfo kRegistry[] = {
    TBO_P(wheel_kmh_to_ms, "m/s per km/h", "nominal bogie speed calibration"),
    TBO_P(wheel_max_kmh, "km/h", "readings above are rejected as invalid"),
    TBO_P(wheel_delay_s, "s", "bogie speed latency compensation"),
    TBO_P(cmd_delay_s, "s", "notch -> drive dead time"),
    TBO_P(lag_window_s, "s", "fixed-lag window for out-of-order messages"),
    TBO_P(max_step_s, "s", "max integration step"),
    TBO_P(wheel_timeout_s, "s", "bogie dropout timeout"),
    TBO_P(cmd_timeout_s, "s", "controller dropout timeout"),
    TBO_P(max_future_s, "s", "reject stamps further ahead than this"),
    TBO_P(max_backjump_s, "s", "reset time base if stamps jump back further"),
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
    TBO_P(disturbance_max, "m/s^2", "disturbance clamp"),
    TBO_P(sigma_wheel, "m/s", "bogie speed noise"),
    TBO_P(sigma_wheel_rel, "-", "relative bogie speed noise"),
    TBO_P(outlier_range, "m/s", "uniform outlier density support"),
    TBO_P(rate_to_bad, "1/s", "IMM nominal -> one bad"),
    TBO_P(rate_to_both_bad, "1/s", "IMM nominal -> both bad"),
    TBO_P(rate_recover, "1/s", "IMM bad -> nominal"),
    TBO_P(mode_prob_floor, "-", "IMM probability floor"),
    TBO_P(slip_context_boost, "-", "rate boost under high effort"),
    TBO_P(max_wheel_accel, "m/s^2", "implausible bogie acceleration"),
    TBO_P(stuck_time_s, "s", "frozen reading duration"),
    TBO_P(stuck_min_change, "m/s", "speed change proving a frozen reading"),
    TBO_P(standstill_kmh, "km/h", "standstill threshold"),
    TBO_P(standstill_time_s, "s", "standstill confirmation time"),
    TBO_P(standstill_max_v, "m/s", "lock-up guard for standstill"),
    TBO_P(recover_time_s, "s", "consistency time before re-anchoring"),
    TBO_P(recover_agree, "m/s", "front/rear agreement for re-anchoring"),
    TBO_P(recover_min_bad_s, "s", "min anomaly duration before re-anchoring"),
    TBO_P(drive_tau_s, "s", "drive acceleration lag"),
    TBO_P(gnss_init_window_s, "s", "GNSS accepted only this long after start"),
    TBO_P(gnss_min_fixes, "-", "fixes needed for initialisation"),
    TBO_P(map_gate_m, "m", "init fix to map distance gate"),
    TBO_P(map_heading_gate_deg, "deg", "init heading consistency gate"),
    TBO_P(map_sigma_cross, "m", "map cross-track sigma"),
    TBO_P(map_sigma_z, "m", "map altitude sigma"),
    TBO_P(init_sigma_s, "m", "along-track sigma after init"),
    TBO_P(use_baseline_heading, "bool", "use master-rover baseline to pick direction"),
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
