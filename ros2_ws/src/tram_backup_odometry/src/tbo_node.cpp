// ROS 2 node: thin wrapper around the ROS-independent estimator core (tbo::Estimator).
// Subscribes to bogie speeds and the driver controller, uses GNSS only inside the
// initialisation window (then unsubscribes), publishes /result/velocity,
// /result/position (nav_msgs/Odometry), /result/status and /diagnostics.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <tbo_msgs/msg/estimator_status.hpp>
#include <tram_vehicle_msgs/msg/driver_controller_command.hpp>
#include <tram_vehicle_msgs/msg/velocity_sensor.hpp>

#include "tbo/estimator.hpp"
#include "tbo/scheduler.hpp"

namespace {

using SteadyClock = std::chrono::steady_clock;
using VelocitySensor = tram_vehicle_msgs::msg::VelocitySensor;
using DriverCmd = tram_vehicle_msgs::msg::DriverControllerCommand;
using NavSatFix = sensor_msgs::msg::NavSatFix;
using Odometry = nav_msgs::msg::Odometry;
using Status = tbo_msgs::msg::EstimatorStatus;
using DiagArray = diagnostic_msgs::msg::DiagnosticArray;
using DiagStatus = diagnostic_msgs::msg::DiagnosticStatus;
using KeyValue = diagnostic_msgs::msg::KeyValue;

tbo::Stamp toStamp(const builtin_interfaces::msg::Time& t) {
  return static_cast<tbo::Stamp>(t.sec) * tbo::kNsPerSec + static_cast<tbo::Stamp>(t.nanosec);
}

builtin_interfaces::msg::Time toMsg(tbo::Stamp s) {
  builtin_interfaces::msg::Time t;
  t.sec = static_cast<int32_t>(s / tbo::kNsPerSec);
  t.nanosec = static_cast<uint32_t>(s % tbo::kNsPerSec);
  return t;
}

double asDouble(const rclcpp::Parameter& p, double fallback) {
  switch (p.get_type()) {
    case rclcpp::ParameterType::PARAMETER_DOUBLE: return p.as_double();
    case rclcpp::ParameterType::PARAMETER_INTEGER: return static_cast<double>(p.as_int());
    case rclcpp::ParameterType::PARAMETER_BOOL: return p.as_bool() ? 1.0 : 0.0;
    default: return fallback;
  }
}

KeyValue kv(const std::string& k, const std::string& v) {
  KeyValue x;
  x.key = k;
  x.value = v;
  return x;
}

}  // namespace

class TboNode : public rclcpp::Node {
 public:
  TboNode() : Node("tram_backup_odometry") {
    declareParameters();
    loadModelAndMap();
    est_ = std::make_unique<tbo::Estimator>(cfg_, model_, map_.empty() ? nullptr : &map_, branch_ptrs_);
    if (!cfg_.landmark_file.empty()) {
      std::vector<tbo::Landmark> lms;
      std::string err;
      if (tbo::loadLandmarks(resolve(cfg_.landmark_file), lms, &err)) est_->setLandmarks(std::move(lms));
      else RCLCPP_WARN(get_logger(), "landmarks not loaded (%s)", err.c_str());
    }
    if (!cfg_.cutoff_file.empty()) {
      std::vector<tbo::Landmark> cut;
      std::string err;
      if (tbo::loadLandmarks(resolve(cfg_.cutoff_file), cut, &err)) est_->setCutoffs(std::move(cut));
      else RCLCPP_WARN(get_logger(), "cut-off landmarks not loaded (%s)", err.c_str());
    }
    if (!cfg_.dfield_file.empty()) {
      tbo::TrackField fld;
      std::string err;
      if (fld.loadCsv(resolve(cfg_.dfield_file), map_.cyclic() ? map_.length() : 0.0, &err))
        est_->setDisturbanceField(std::move(fld));
      else RCLCPP_WARN(get_logger(), "disturbance field not loaded (%s)", err.c_str());
    }
    sched_ = std::make_unique<tbo::OutputScheduler>(est_->config().p);

    // Best-effort subscribers are compatible with both reliable and best-effort publishers.
    const auto in_qos = rclcpp::QoS(rclcpp::KeepLast(100)).best_effort();
    sub_front_ = create_subscription<VelocitySensor>(
        topic_front_, in_qos, [this](VelocitySensor::ConstSharedPtr m) { onWheel(tbo::Sensor::Front, *m); });
    sub_rear_ = create_subscription<VelocitySensor>(
        topic_rear_, in_qos, [this](VelocitySensor::ConstSharedPtr m) { onWheel(tbo::Sensor::Rear, *m); });
    sub_cmd_ = create_subscription<DriverCmd>(topic_cmd_, in_qos,
                                              [this](DriverCmd::ConstSharedPtr m) { onCmd(*m); });
    subscribeGnss();

    const auto out_qos = rclcpp::QoS(rclcpp::KeepLast(50));
    pub_vel_ = create_publisher<VelocitySensor>("/result/velocity", out_qos);
    pub_pos_ = create_publisher<Odometry>("/result/position", out_qos);
    pub_status_ = create_publisher<Status>("/result/status", out_qos);
    pub_diag_ = create_publisher<DiagArray>("/diagnostics", rclcpp::QoS(10));
    diag_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { onDiagTimer(); });

    RCLCPP_INFO(get_logger(),
                "tram_backup_odometry ready: map=%s (%.0f m, cyclic=%d), traction=%s, frame=%s, "
                "GNSS init window %.1f s",
                map_.empty() ? "none" : cfg_.map_file.c_str(), map_.length(), map_.cyclic() ? 1 : 0,
                model_.isBuiltin() ? "built-in" : cfg_.traction_file.c_str(), cfg_.output_frame.c_str(),
                cfg_.p.gnss_init_window_s);
  }

 private:
  void declareParameters() {
    int n = 0;
    const tbo::ParamInfo* reg = tbo::paramRegistry(&n);
    for (int i = 0; i < n; ++i) {
      rcl_interfaces::msg::ParameterDescriptor d;
      d.description = std::string(reg[i].doc) + " [" + reg[i].unit + "]";
      d.dynamic_typing = true;  // accept 1, 1.0 or true in YAML
      const double def = cfg_.p.*(reg[i].member);
      declare_parameter(reg[i].name, rclcpp::ParameterValue(def), d);
      cfg_.p.*(reg[i].member) = asDouble(get_parameter(reg[i].name), def);
    }
    cfg_.map_file = declare_parameter<std::string>("map_file", "maps/track_map.csv");
    cfg_.traction_file = declare_parameter<std::string>("traction_file", "config/traction_lut.csv");
    cfg_.branch_files = declare_parameter<std::string>(
        "branch_files", "maps/branch_fan_F2.csv,maps/branch_fan_F3.csv,maps/branch_wb_detour.csv");
    cfg_.landmark_file = declare_parameter<std::string>("landmark_file", "maps/landmarks.csv");
    cfg_.cutoff_file = declare_parameter<std::string>("cutoff_file", "maps/cutoffs.csv");
    cfg_.dfield_file = declare_parameter<std::string>("dfield_file", "");
    cfg_.output_frame = declare_parameter<std::string>("output_frame", "mgrs");
    cfg_.init_source = declare_parameter<std::string>("init_source", "master");
    cfg_.frame_id = declare_parameter<std::string>("frame_id", "map");
    cfg_.child_frame_id = declare_parameter<std::string>("child_frame_id", "base_link");
    topic_front_ = declare_parameter<std::string>("topic_front", "/vehicle/front_bogie_velocity");
    topic_rear_ = declare_parameter<std::string>("topic_rear", "/vehicle/rear_bogie_velocity");
    topic_cmd_ = declare_parameter<std::string>("topic_cmd", "/vehicle/driver_position_cmd");
    topic_fix_master_ = declare_parameter<std::string>("topic_fix_master", "/sensing/gnss/master/fix");
    topic_fix_rover_ = declare_parameter<std::string>("topic_fix_rover", "/sensing/gnss/rover/fix");
  }

  std::string resolve(const std::string& path) const {
    if (path.empty() || path[0] == '/') return path;
    try {
      return ament_index_cpp::get_package_share_directory("tram_backup_odometry") + "/" + path;
    } catch (const std::exception&) {
      return path;
    }
  }

  void loadModelAndMap() {
    std::string err;
    cfg_.traction_file = resolve(cfg_.traction_file);
    cfg_.map_file = resolve(cfg_.map_file);
    if (!cfg_.traction_file.empty() && !model_.loadCsv(cfg_.traction_file, &err))
      RCLCPP_WARN(get_logger(), "traction table not loaded (%s); using built-in table", err.c_str());
    if (!cfg_.map_file.empty() && !map_.loadCsv(cfg_.map_file, &err))
      RCLCPP_WARN(get_logger(), "track map not loaded (%s); position = dead reckoning", err.c_str());
    size_t pos = 0;
    const std::string list = cfg_.branch_files;
    while (!list.empty() && pos != std::string::npos) {
      const size_t comma = list.find(',', pos);
      const std::string path =
          resolve(list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
      pos = comma == std::string::npos ? comma : comma + 1;
      if (path.empty()) continue;
      tbo::TrackMap b;
      if (b.loadCsv(path, &err)) branches_.push_back(std::move(b));
      else RCLCPP_WARN(get_logger(), "branch map not loaded (%s)", err.c_str());
    }
    for (const auto& b : branches_) branch_ptrs_.push_back(&b);
  }

  void subscribeGnss() {
    if (cfg_.p.gnss_init_window_s <= 0.0) return;
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(50)).best_effort();
    sub_fix_master_ = create_subscription<NavSatFix>(
        topic_fix_master_, qos, [this](NavSatFix::ConstSharedPtr m) { onFix(tbo::GnssSource::Master, *m); });
    sub_fix_rover_ = create_subscription<NavSatFix>(
        topic_fix_rover_, qos, [this](NavSatFix::ConstSharedPtr m) { onFix(tbo::GnssSource::Rover, *m); });
    gnss_subscribed_ = true;
  }

  void onWheel(tbo::Sensor s, const VelocitySensor& m) {
    const auto t0 = SteadyClock::now();
    try {
      const tbo::Stamp st = toStamp(m.header.stamp);
      if (est_->onWheel(s, st, m.velocity)) afterInput(false, st, t0);
    } catch (const std::exception& e) {
      ++callback_errors_;
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "wheel callback error: %s", e.what());
    }
  }

  void onCmd(const DriverCmd& m) {
    const auto t0 = SteadyClock::now();
    try {
      const tbo::Stamp st = toStamp(m.header.stamp);
      if (est_->onCmd(st, static_cast<int>(m.position))) afterInput(true, st, t0);
    } catch (const std::exception& e) {
      ++callback_errors_;
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "cmd callback error: %s", e.what());
    }
  }

  void onFix(tbo::GnssSource src, const NavSatFix& m) {
    try {
      const tbo::Stamp st = toStamp(m.header.stamp);
      est_->onGnssFix(src, st, m.latitude, m.longitude, m.altitude, static_cast<int>(m.status.status));
    } catch (const std::exception&) {
      ++callback_errors_;
    }
  }

  void afterInput(bool is_cmd, tbo::Stamp st, SteadyClock::time_point t0) {
    if (!est_->started()) return;
    const auto resets = est_->diagnostics().resets;
    if (resets != last_resets_) {  // bag restarted or another bag: new run, new init window
      last_resets_ = resets;
      sched_->reset();
      if (!gnss_subscribed_) subscribeGnss();
    }
    tbo::Stamp stamps[64];
    const int n = sched_->onInput(is_cmd, st, est_->latestStamp(), stamps, 64);
    for (int k = 0; k < n; ++k) publish(est_->query(stamps[k]), t0);
    // GNSS is used only for initialisation: drop the subscriptions once the window closed.
    if (gnss_subscribed_ && est_->gnssWindowClosed()) {
      sub_fix_master_.reset();
      sub_fix_rover_.reset();
      gnss_subscribed_ = false;
      RCLCPP_INFO(get_logger(), "GNSS init window closed (map_matched=%d): GNSS unsubscribed",
                  est_->mapMatched() ? 1 : 0);
    }
  }

  void publish(const tbo::Output& o, SteadyClock::time_point t0) {
    const auto stamp = toMsg(o.stamp);
    VelocitySensor v;
    v.header.stamp = stamp;
    v.header.frame_id = cfg_.child_frame_id;
    v.velocity = o.v;
    pub_vel_->publish(v);
    if (!o.pos_valid) return;  // no anchor yet: wait (<= gnss_wait_s) for the first GNSS fix

    Odometry od;
    od.header.stamp = stamp;
    od.header.frame_id = cfg_.frame_id;
    od.child_frame_id = cfg_.child_frame_id;
    od.pose.pose.position.x = o.x;
    od.pose.pose.position.y = o.y;
    od.pose.pose.position.z = o.z;
    od.pose.pose.orientation.z = std::sin(0.5 * o.yaw);
    od.pose.pose.orientation.w = std::cos(0.5 * o.yaw);
    auto& pc = od.pose.covariance;  // row-major 6x6: x y z roll pitch yaw
    pc[0] = o.cov_xx;
    pc[1] = o.cov_xy;
    pc[6] = o.cov_xy;
    pc[7] = o.cov_yy;
    pc[14] = o.cov_zz;
    pc[21] = 0.0025;  // roll: rail vehicle close to level
    pc[28] = 0.0025;  // pitch: track grades are a few percent
    pc[35] = o.map_matched ? 3e-4 : 0.03;
    od.twist.twist.linear.x = o.v;
    auto& tc = od.twist.covariance;
    tc[0] = std::max(o.v_var, 1e-6);
    tc[7] = 1e-4;   // no lateral velocity on rails
    tc[14] = 1e-4;  // no vertical velocity
    tc[21] = 1e-2;
    tc[28] = 1e-2;
    tc[35] = 1e-2;
    pub_pos_->publish(od);

    const double proc_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - t0).count();
    Status s;
    s.header = od.header;
    s.speed = o.v;
    s.speed_std = std::sqrt(std::max(0.0, o.v_var));
    s.accel = o.accel;
    s.model_accel = o.a_model;
    s.distance = o.s;
    s.distance_std = std::sqrt(std::max(0.0, o.s_var));
    for (int j = 0; j < tbo::kNumModes; ++j) s.mode_prob[j] = o.mode_prob[j];
    s.slip_ratio_front = o.slip_front;
    s.slip_ratio_rear = o.slip_rear;
    s.adhesion_used = std::abs(o.accel) / 9.81;
    s.disturbance = o.disturbance;
    s.wheel_scale = o.scale;
    s.traction_gain = o.gain;
    s.wheel_trust = o.wheel_trust;
    s.flags = o.flags;
    s.map_matched = o.map_matched;
    s.initialized = !(o.flags & tbo::kFlagNotInitialized);
    s.processing_ms = proc_ms;
    pub_status_->publish(s);

    last_flags_ = o.flags;
    last_out_ = o;
    ++published_;
    proc_max_ms_ = std::max(proc_max_ms_, proc_ms);
    proc_sum_ms_ += proc_ms;
  }

  void onDiagTimer() {
    DiagArray arr;
    arr.header.stamp = now();
    DiagStatus st;
    st.name = "tram_backup_odometry: estimator";
    st.hardware_id = "tram";
    const auto& d = est_->diagnostics();
    const std::uint32_t f = last_flags_;
    st.level = DiagStatus::OK;
    std::string msg = "nominal";
    if (f & (tbo::kFlagFrontSlip | tbo::kFlagRearSlip | tbo::kFlagFrontSlide | tbo::kFlagRearSlide)) {
      st.level = DiagStatus::WARN;
      msg = "wheel slip/slide detected: odometry de-weighted";
    }
    if (f & tbo::kFlagUnmodeledAccel) {
      st.level = DiagStatus::WARN;
      msg = "acceleration not explained by the controller (emergency/track brake?): wheels trusted";
    }
    if (f & tbo::kFlagModelOnly) {
      st.level = DiagStatus::WARN;
      msg = "model-only dead reckoning (wheel data untrusted or missing)";
    }
    if (f & tbo::kFlagNotInitialized) {
      st.level = DiagStatus::WARN;
      msg = "not anchored: no GNSS fix inside the init window";
    }
    if (!est_->started()) {
      st.level = DiagStatus::STALE;
      msg = "waiting for input";
    }
    st.message = msg;
    const double rate = static_cast<double>(published_ - published_last_);
    st.values.push_back(kv("output_rate_hz", std::to_string(rate)));
    st.values.push_back(kv("processing_ms_max", std::to_string(proc_max_ms_)));
    st.values.push_back(kv("processing_ms_mean",
                           std::to_string(published_ > published_last_ ? proc_sum_ms_ / rate : 0.0)));
    st.values.push_back(kv("flags", std::to_string(f)));
    st.values.push_back(kv("speed", std::to_string(last_out_.v)));
    st.values.push_back(kv("p_nominal", std::to_string(last_out_.mode_prob[0])));
    st.values.push_back(kv("p_model_only", std::to_string(last_out_.mode_prob[3])));
    st.values.push_back(kv("p_unmodeled_accel", std::to_string(last_out_.mode_prob[4])));
    st.values.push_back(kv("map_matched", est_->mapMatched() ? "true" : "false"));
    st.values.push_back(kv("gnss_subscribed", gnss_subscribed_ ? "true" : "false"));
    st.values.push_back(kv("wheel_msgs", std::to_string(d.wheel_msgs)));
    st.values.push_back(kv("cmd_msgs", std::to_string(d.cmd_msgs)));
    st.values.push_back(kv("invalid_wheel", std::to_string(d.invalid_wheel)));
    st.values.push_back(kv("rejected_stamps", std::to_string(d.rejected_stamps)));
    st.values.push_back(kv("late_dropped", std::to_string(d.late_dropped)));
    st.values.push_back(kv("landmark_flag", (f & tbo::kFlagLandmark) ? "true" : "false"));
    st.values.push_back(kv("callback_errors", std::to_string(callback_errors_)));
    arr.status.push_back(st);
    pub_diag_->publish(arr);
    published_last_ = published_;
    proc_max_ms_ = 0.0;
    proc_sum_ms_ = 0.0;
  }

  tbo::Config cfg_;
  tbo::TractionModel model_;
  tbo::TrackMap map_;
  std::vector<tbo::TrackMap> branches_;
  std::vector<const tbo::TrackMap*> branch_ptrs_;
  std::unique_ptr<tbo::Estimator> est_;
  std::unique_ptr<tbo::OutputScheduler> sched_;

  std::string topic_front_, topic_rear_, topic_cmd_, topic_fix_master_, topic_fix_rover_;
  rclcpp::Subscription<VelocitySensor>::SharedPtr sub_front_, sub_rear_;
  rclcpp::Subscription<DriverCmd>::SharedPtr sub_cmd_;
  rclcpp::Subscription<NavSatFix>::SharedPtr sub_fix_master_, sub_fix_rover_;
  rclcpp::Publisher<VelocitySensor>::SharedPtr pub_vel_;
  rclcpp::Publisher<Odometry>::SharedPtr pub_pos_;
  rclcpp::Publisher<Status>::SharedPtr pub_status_;
  rclcpp::Publisher<DiagArray>::SharedPtr pub_diag_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  bool gnss_subscribed_ = false;
  std::uint64_t last_resets_ = 0;
  std::uint64_t callback_errors_ = 0;
  std::uint64_t published_ = 0, published_last_ = 0;
  double proc_max_ms_ = 0.0, proc_sum_ms_ = 0.0;
  std::uint32_t last_flags_ = 0;
  tbo::Output last_out_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TboNode>());
  rclcpp::shutdown();
  return 0;
}
