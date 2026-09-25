// Backup odometry estimator: IMM (interacting multiple model) extended Kalman filter
// over along-track state [s, v, d, k, g] driven by a nonlinear traction model of the
// driver notch and corrected by front/rear bogie speeds; map-constrained position.
//
//   s  distance travelled along the track since start, m
//   v  longitudinal speed, m/s (>= 0)
//   d  disturbance acceleration: grade + resistance mismatch, m/s^2 (random walk)
//   k  wheel scale error: bogie reading / true speed - 1 (random walk)
//   g  traction gain: realised / tabulated drive acceleration (mass variation)
//
// Dynamics: ds/dt = v, dv/dt = g * a_drive + d, a_drive follows a*(notch, v) with lag.
// Modes: nominal / front bad / rear bad / both bad (model-only dead reckoning).
// Messages may arrive out of order: a fixed-lag buffer re-sorts them by stamp and
// outputs are computed on demand for any stamp at or after the committed state.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "tbo/geo.hpp"
#include "tbo/params.hpp"
#include "tbo/small_matrix.hpp"
#include "tbo/traction_model.hpp"
#include "tbo/track_map.hpp"
#include "tbo/types.hpp"

namespace tbo {

constexpr int kNx = 5;
enum StateIndex : int { kS = 0, kV = 1, kD = 2, kK = 3, kG = 4 };
using StateVec = Vec<kNx>;
using StateCov = Mat<kNx, kNx>;

struct Landmark {
  double s = 0.0;       // main-cycle arc length of the standstill position, m
  double sigma = 0.3;   // spread of observed stops, m
  double p_stop = 0.5;  // probability that a pass stops here
};
bool loadLandmarks(const std::string& path, std::vector<Landmark>& out, std::string* err);

struct Diagnostics {
  std::uint64_t wheel_msgs = 0, cmd_msgs = 0, gnss_msgs = 0;
  std::uint64_t invalid_wheel = 0, invalid_cmd = 0, rejected_stamps = 0;
  std::uint64_t late_dropped = 0, merged_pairs = 0, recoveries = 0, resets = 0;
  std::uint64_t implausible_wheel = 0, gnss_ignored_after_window = 0, landmark_fixes = 0;
  int max_buffer = 0;
};

class Estimator {
 public:
  Estimator(const Config& cfg, const TractionModel& model, const TrackMap* map,
            std::vector<const TrackMap*> branches = {});

  // Inputs. `stamp` is the message header stamp (bag time). Never throws; bad data is
  // counted, flagged and ignored.
  void onWheel(Sensor sensor, Stamp stamp, double speed_kmh);
  void onCmd(Stamp stamp, int notch);
  void onGnssFix(GnssSource src, Stamp stamp, double lat, double lon, double alt, int status);
  void onGnssVel(GnssSource src, Stamp stamp, double vx, double vy, double vz);

  // Best estimate at time t given everything received so far (pure, const).
  Output query(Stamp t) const;

  bool started() const { return started_; }
  Stamp latestStamp() const { return latest_; }
  bool initialized() const { return init_.anchored; }
  bool mapMatched() const { return init_.map_matched; }
  const Diagnostics& diagnostics() const { return diag_; }
  const Config& config() const { return cfg_; }
  void reset();
  void setLandmarks(std::vector<Landmark> lms) { landmarks_ = std::move(lms); }
  void setCutoffs(std::vector<Landmark> lms) { cutoffs_ = std::move(lms); }

  // ---- internals exposed for tests ----
  struct WheelTrack {
    bool have = false;
    double z = 0.0;          // last valid calibrated speed, m/s
    Stamp t = 0;
    bool have_prev = false;  // previous valid sample for plausibility checks
    double z_prev = 0.0;
    Stamp t_prev = 0;
    Stamp same_since = -1;   // start of a run of identical non-zero readings
    double v_at_same = 0.0;
    bool stuck = false;
    bool implausible = false;
    Stamp invalid_t = -1;    // last invalid sample
    Stamp zero_stuck_t = -1; // last time the bogie read 0 while the other bogie showed motion
    // short history for the acceleration-excess (CUSUM) monitor
    static constexpr int kHist = 8;
    Stamp ht[kHist]{};
    double hz[kHist]{};
    double ham[kHist]{};
    int hn = 0;
    int hhead = 0;
    double cusum_pos = 0.0, cusum_neg = 0.0;
    Stamp t_cusum = -1;
    bool alarm = false;
  };

  struct FilterState {
    bool started = false;
    Stamp t = 0;
    StateVec x[kNumModes];
    StateCov P[kNumModes];
    double mu[kNumModes] = {1.0, 0.0, 0.0, 0.0, 0.0};
    Stamp t_mix = 0;
    int notch = 0;
    bool have_cmd = false;
    Stamp last_cmd = 0;
    double a_drive = 0.0;
    double a_target = 0.0;
    WheelTrack wheel[2];
    bool standstill = false;
    Stamp still_since = -1;
    Stamp bad_since = -1;
    Stamp agree_since = -1;
    Stamp recovered_t = -1;
    Stamp late_t = -1;
    // joint slip/slide handling: model-only latch with roll-back to the anomaly onset
    bool onset = false;
    Stamp onset_t = -1, mod_t = -1;
    double mod_v = 0.0, mod_s = 0.0, onset_d = 0.0, onset_g = 1.0;
    Stamp snap_t = -1;       // last state before any CUSUM became active
    double snap_v = 0.0, snap_s = 0.0, snap_d = 0.0, snap_g = 1.0;
    bool latch = false;
    Stamp latch_t = -1;
    int release_count = 0;
    int latch_sign = 0;      // +1 slip (wheels fast), -1 slide (wheels slow)
    bool lm_done = false;    // landmark fix already attempted during the current stop
    Stamp lm_t = -1;
    double cmd_cusum = 0.0;  // evidence that the controller signal is wrong
    Stamp cmd_fault_t = -1;  // last time that evidence crossed the threshold
    Stamp t_cmd_cusum = -1;
  };

 private:
  struct Event {
    Stamp t = 0;             // effective time (stamp minus configured latency)
    std::uint64_t seq = 0;   // arrival order, tie-break
    bool is_cmd = false;
    int notch = 0;
    bool has[2] = {false, false};
    double z[2] = {0.0, 0.0};  // calibrated m/s, NaN marks an invalid sample
  };

  void insert(const Event& e);
  void commitOlderThan(Stamp t);
  void applyEvent(FilterState& f, const Event& e) const;
  void advance(FilterState& f, Stamp t) const;
  void predictMode(StateVec& x, StateCov& P, double a, double a_ext, double h, bool standstill,
                   double sigma_accel, double q_d) const;
  double trackAccel(const FilterState& f) const;  // grade + curve terms from the map
  void wheelUpdate(FilterState& f, const Event& e) const;
  bool jointMonitor(FilterState& f, const Event& e, const bool* avail, const double* z) const;
  // Along-track fix from a list of known places; `lead` shifts the place by v*lead (GNSS timing).
  bool placeUpdate(FilterState& f, Stamp t, const std::vector<Landmark>& places, double p_random,
                   double lead) const;
  void startFilter(FilterState& f, Stamp t) const;
  bool acceptStamp(Stamp stamp);
  Output makeOutput(const FilterState& f, Stamp t) const;
  double combinedV(const FilterState& f) const;
  void updateAnchor();

  Config cfg_;
  const Params& p_;
  const TractionModel& model_;
  const TrackMap* map_;
  std::vector<const TrackMap*> branches_;  // alternative start tracks merging into main
  std::vector<Landmark> landmarks_;
  std::vector<Landmark> cutoffs_;
  // Maps a relative distance to (edge, arc length) along the anchored route.
  bool routeAt(double s_rel, const TrackMap*& m, double& s) const;

  FilterState committed_;
  std::vector<Event> buf_;  // sorted by (t, seq); capacity reserved up front
  bool started_ = false;
  Stamp latest_ = 0;
  std::uint64_t seq_ = 0;
  Diagnostics diag_;

  // ---- GNSS initialisation (first seconds only) ----
  struct Fix {
    Stamp t;
    double lat, lon, alt;
  };
  struct InitState {
    bool have_first = false;
    Stamp t_first = 0;
    bool origin_set = false;
    geo::Geodetic origin{};      // output frame origin (first fix of init_source)
    std::vector<Fix> fixes;      // init_source fixes inside the window
    std::vector<Fix> other;      // the other antenna (baseline heading)
    bool anchored = false;
    bool map_matched = false;
    double s_offset = 0.0;       // s_map = wrap(s_offset + s_rel)
    const TrackMap* prefix = nullptr;  // start branch (if the run starts off the main cycle)
    double prefix_s0 = 0.0, prefix_len = 0.0;
    double dr_x = 0.0, dr_y = 0.0, dr_z = 0.0;  // dead-reckoning start (output frame)
    double yaw0 = 0.0;           // forward yaw (map ENU)
    bool have_yaw = false;
    double match_dist = 0.0;
    bool var_applied = false;    // anchor uncertainty injected into the filter once
  } init_;
  geo::LocalCartesian map_lc_;   // map origin
  geo::LocalCartesian out_lc_;   // output origin (enu mode)
  double rot_[3][3]{};           // map ENU -> output ENU rotation
  double trans_[3]{};            // map ENU -> output ENU translation
  bool frame_ready_ = false;
  void configureFrame();
  void mapToOutput(double mx, double my, double mz, double& ox, double& oy, double& oz) const;
};

}  // namespace tbo
