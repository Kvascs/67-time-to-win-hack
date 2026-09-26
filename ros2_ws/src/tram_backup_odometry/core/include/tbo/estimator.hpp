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
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "tbo/geo.hpp"
#include "tbo/global_localizer.hpp"
#include "tbo/params.hpp"
#include "tbo/small_matrix.hpp"
#include "tbo/traction_model.hpp"
#include "tbo/track_field.hpp"
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
  std::uint64_t late_dropped = 0, merged_pairs = 0, recoveries = 0, resets = 0, time_gaps = 0;
  std::uint64_t implausible_wheel = 0, gnss_ignored_after_window = 0, landmark_fixes = 0, global_fixes = 0;
  int max_buffer = 0;
};

class Estimator {
 public:
  Estimator(const Config& cfg, const TractionModel& model, const TrackMap* map,
            std::vector<const TrackMap*> branches = {});

  // Inputs. `stamp` is the message header stamp (bag time). Never throws; bad data is
  // counted, flagged and ignored. Returns false if the message was dropped (bad stamp or
  // notch): callers must not publish an output at that stamp.
  bool onWheel(Sensor sensor, Stamp stamp, double speed_kmh);
  bool onCmd(Stamp stamp, int notch);
  void onGnssFix(GnssSource src, Stamp stamp, double lat, double lon, double alt, int status);
  void onGnssVel(GnssSource src, Stamp stamp, double vx, double vy, double vz);

  // Best estimate at time t given everything received so far (pure, const).
  Output query(Stamp t) const;

  bool started() const { return started_; }
  Stamp latestStamp() const { return latest_; }
  bool initialized() const { return init_.anchored; }
  bool mapMatched() const { return init_.map_matched; }
  // GNSS init window of the current run is over (the node then drops its GNSS subscriptions).
  bool gnssWindowClosed() const {
    return init_.have_gnss && latest_ > init_.t_first_gnss + fromSec(p_.gnss_init_window_s + 1.0);
  }
  const Diagnostics& diagnostics() const { return diag_; }
  const Config& config() const { return cfg_; }
  void reset();
  void setLandmarks(std::vector<Landmark> lms) { landmarks_ = std::move(lms); }
  void setCutoffs(std::vector<Landmark> lms) { cutoffs_ = std::move(lms); }
  void setDisturbanceField(TrackField f) { dfield_ = std::move(f); }
  void setFaultPrior(TrackField f) { fault_prior_ = std::move(f); }
  // Dead-end stub leaving the main cycle (its join_s = main arc of its first point); some runs end on it.
  void setStub(TrackMap stub) {
    stub_ = std::move(stub);
    has_stub_ = !stub_.empty() && stub_.hasJoin();
  }
  // Cues for the GNSS-free global localisation (used only when no GNSS fix arrives at all).
  void setGlobalLocalisation(std::vector<Landmark> stops, std::vector<Landmark> cutoffs, TrackField vmax) {
    gl_stops_ = std::move(stops);
    gl_cutoffs_ = std::move(cutoffs);
    vmax_env_ = std::move(vmax);
  }
  bool globalLocalisationRunning() const { return gl_ != nullptr; }

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
    double ham[kHist]{};     // model acceleration, slip reference (slow disturbance)
    double hams[kHist]{};    // model acceleration, slide reference (filter disturbance)
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
    double odo = 0.0;        // distance travelled in this run (integral of the combined speed), m
    double lm_odo = 0.0;     // odo at the last accepted place fix (landmark or cut-off)
    bool stub = false;       // on the dead-end stub (decided by a stop at its far part)
    double stub_s0 = 0.0;    // filter arc where the stub starts
    double stub_rs = 0.0;    // sum of log(front/rear)^2 in the switch window
    int stub_rn = 0;
    bool stub_rdone = false; // roughness decision taken for this pass of the switch
    double stub_noise0 = 0.0;  // slow bogie noise level when entering the window, (m/s)^2
    double noise_slow = -1.0;  // slow EMA (60 s) of (front - rear)^2 while moving, (m/s)^2
    Stamp t_noise_slow = -1;
    double cmd_cusum = 0.0;  // evidence that the controller signal is wrong
    Stamp cmd_fault_t = -1;  // last time that evidence crossed the threshold
    Stamp t_cmd_cusum = -1;
    // ---- anomaly-suite fix prototypes ----
    double d_slow = 0.0;     // slow disturbance reference for the joint CUSUM
    double d_med = 0.0;      // medium disturbance reference for the slide CUSUM
    double noise_var = 0.0;  // EMA of (front - rear)^2: bogie sensor noise level
    Stamp t_noise = -1;
    bool d_med_init = false;
    Stamp t_dmed = -1;
    bool d_slow_init = false;
    Stamp t_dslow = -1;
    double ema_z[2] = {0.0, 0.0};  // low-passed bogie speeds (recovery agreement test)
    Stamp t_ema[2] = {-1, -1};
    Stamp zero_since = -1;   // both available bogies ~0 since
    Stamp single_since = -1; // only one bogie available and distrusted since
    Stamp latch_end_t = -1;  // last joint-latch release
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
                   double sigma_accel, double q_d, double da_ds = 0.0) const;
  double trackAccel(const FilterState& f, double ds = 0.0) const;  // map terms at s + ds
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
  void noteCommitted();          // record (t, s) of the committed state
  void feedGlobal();             // GNSS-free localisation: one committed step, handover on a fix
  void handoverGlobal();
  bool globalAvailable() const;
  double distanceAt(Stamp t) const;  // travelled distance at any recent time (anchor at fix time)

  Config cfg_;
  const Params& p_;
  const TractionModel& model_;
  const TrackMap* map_;
  std::vector<const TrackMap*> branches_;  // alternative start tracks merging into main
  std::vector<Landmark> landmarks_;
  std::vector<Landmark> cutoffs_;
  TrackField dfield_;
  TrackField fault_prior_;  // slip/slide rate multiplier by place (learned adhesion map)
  std::vector<Landmark> gl_stops_, gl_cutoffs_;
  TrackField vmax_env_;
  TrackMap stub_;          // dead-end stub (antenna path from its start on the main line)
  bool has_stub_ = false;
  std::unique_ptr<GlobalLocalizer> gl_;
  double gl_s0_ = 0.0;      // committed distance when the localiser started (its odometer origin)
  bool gl_tried_ = false;
  // Maps a relative distance to (edge, arc length) along the anchored route.
  bool routeAt(double s_rel, const TrackMap*& m, double& s) const;
  // Stop check for the dead-end stub: true while the tram is taken to be on it (no landmarks there).
  bool stubCheck(FilterState& f) const;
  // Roughness of the front/rear speed ratio at the switch: the other (earlier) cue for the stub.
  void stubRoughness(FilterState& f, const Event& e) const;

  FilterState committed_;
  std::vector<Event> buf_;  // sorted by (t, seq); capacity reserved up front
  bool started_ = false;
  Stamp latest_ = 0;
  Stamp pending_jump_ = 0;  // stamp of a lone message far from the current time (glitch or jump?)
  std::uint64_t seq_ = 0;
  Diagnostics diag_;
  // (t, s) of the committed state over the last seconds: the GNSS anchor needs the distance at
  // the fix time, which may lie before the fixed-lag window when fixes arrive late or the tram moves.
  std::deque<std::pair<Stamp, double>> s_hist_;

  // ---- GNSS initialisation (first seconds only) ----
  struct Fix {
    Stamp t;
    double lat, lon, alt;
  };
  struct InitState {
    bool have_first = false;
    Stamp t_first = 0;             // first input of the run (wheel, controller or GNSS)
    bool have_gnss = false;
    Stamp t_first_gnss = 0;        // the init window counts from the first valid fix
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
