# Diagnostic status of the backup odometry estimator, published with every estimate.

# flags bit mask
uint32 FLAG_FRONT_SLIP = 1          # front bogie faster than vehicle (traction slip)
uint32 FLAG_REAR_SLIP = 2
uint32 FLAG_FRONT_SLIDE = 4         # front bogie slower than vehicle (braking slide / lock)
uint32 FLAG_REAR_SLIDE = 8
uint32 FLAG_FRONT_DROPOUT = 16      # no valid front message within timeout
uint32 FLAG_REAR_DROPOUT = 32
uint32 FLAG_CMD_DROPOUT = 64        # no controller message within timeout (neutral assumed)
uint32 FLAG_FRONT_STUCK = 128       # frozen reading while vehicle speed changes
uint32 FLAG_REAR_STUCK = 256
uint32 FLAG_FRONT_INVALID = 512     # NaN / negative / absurd sample rejected recently
uint32 FLAG_REAR_INVALID = 1024
uint32 FLAG_MODEL_ONLY = 2048       # speed from the dynamics model only
uint32 FLAG_STANDSTILL = 4096       # zero-velocity update active
uint32 FLAG_NOT_INITIALIZED = 8192  # absolute position not anchored (no GNSS in init window)
uint32 FLAG_NO_MAP = 16384          # position is dead reckoning, not map-constrained
uint32 FLAG_RECOVERED = 32768       # speed re-anchored to wheels after a long anomaly
uint32 FLAG_LATE_DATA = 65536       # message older than the fixed-lag window dropped

std_msgs/Header header
float64 speed                 # m/s
float64 speed_std             # m/s
float64 accel                 # estimated longitudinal acceleration, m/s^2
float64 model_accel           # drive acceleration predicted from the notch, m/s^2
float64 distance              # along-track distance since start, m
float64 distance_std          # m
float64[4] mode_prob          # IMM: nominal, front bad, rear bad, both bad (model only)
float64 slip_ratio_front      # (wheel - vehicle) / vehicle
float64 slip_ratio_rear
float64 adhesion_used         # |accel| / g: adhesion utilisation (upper bound during slip)
float64 disturbance           # grade + resistance mismatch, m/s^2
float64 wheel_scale           # wheel scale error estimate (reading / true - 1)
float64 traction_gain         # realised / tabulated drive acceleration
float64 wheel_trust           # 1 - P(both bogies untrustworthy)
uint32 flags
bool map_matched
bool initialized
float64 processing_ms         # callback entry -> publish, wall clock
