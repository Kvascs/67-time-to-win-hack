"""Topic keys, message types, column layouts and anomaly label flags.

The npz files produced by ``tools/extract_bags.py`` store one float64 array per topic.
The key is the topic name without the leading slash and with ``/`` replaced by ``__``.
Column 0 is the bag receive time [s], column 1 the header stamp [s], then payload columns.
"""
from __future__ import annotations

import enum
import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parents[1]                      # C:\MosTransHack
DATA_ROOT = Path(os.environ.get('MOSTRANS_DATA', PROJECT_ROOT / 'data'))
NPZ_DIR = DATA_ROOT / 'npz'
BAG_DIR = DATA_ROOT / 'bags'
MSG_DIR = DATA_ROOT / 'tram_vehicle_msgs' / 'msg'
SPLITS_JSON = DATA_ROOT / 'splits.json'
DEFAULT_SUITE = PKG_DIR / 'scenarios' / 'suite.yaml'
DEFAULT_OUT = PKG_DIR / 'out'

# --------------------------------------------------------------------------------------
# Topics
# --------------------------------------------------------------------------------------
FRONT = 'vehicle__front_bogie_velocity'
REAR = 'vehicle__rear_bogie_velocity'
CMD = 'vehicle__driver_position_cmd'
GNSS_MASTER_FIX = 'sensing__gnss__master__fix'
GNSS_ROVER_FIX = 'sensing__gnss__rover__fix'
GNSS_MASTER_VEL = 'sensing__gnss__master__vel'
GNSS_ROVER_VEL = 'sensing__gnss__rover__vel'

WHEELS = (FRONT, REAR)
VEHICLE = (FRONT, REAR, CMD)
GNSS = (GNSS_MASTER_FIX, GNSS_ROVER_FIX, GNSS_MASTER_VEL, GNSS_ROVER_VEL)
ALL_TOPICS = VEHICLE + GNSS

#: Short aliases usable in scenario files (``topics: [front, cmd]``).
TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    'front': (FRONT,),
    'rear': (REAR,),
    'cmd': (CMD,),
    'notch': (CMD,),
    'controller': (CMD,),
    'wheels': WHEELS,
    'both': WHEELS,
    'vehicle': VEHICLE,
    'all_vehicle': VEHICLE,
    'gnss': GNSS,
    'master_fix': (GNSS_MASTER_FIX,),
    'rover_fix': (GNSS_ROVER_FIX,),
    'master_vel': (GNSS_MASTER_VEL,),
    'rover_vel': (GNSS_ROVER_VEL,),
}
for _k in ALL_TOPICS:
    TOPIC_ALIASES[_k] = (_k,)


def topic_name(key: str) -> str:
    """npz key -> ROS topic name (``vehicle__front_bogie_velocity`` -> ``/vehicle/front_bogie_velocity``)."""
    return '/' + key.replace('__', '/')


def topic_key(name: str) -> str:
    """ROS topic name -> npz key."""
    return name.strip('/').replace('/', '__')


def resolve_topics(spec) -> tuple[str, ...]:
    """Expand a topic spec (alias, key, ROS name or list thereof) to a tuple of npz keys."""
    if spec is None:
        return ()
    if isinstance(spec, str):
        spec = [spec]
    out: list[str] = []
    for item in spec:
        key = item if item in TOPIC_ALIASES else topic_key(item)
        if key not in TOPIC_ALIASES:
            raise ValueError(f'unknown topic/alias {item!r}; known: {sorted(TOPIC_ALIASES)}')
        for k in TOPIC_ALIASES[key]:
            if k not in out:
                out.append(k)
    return tuple(out)


MSGTYPES = {
    FRONT: 'tram_vehicle_msgs/msg/VelocitySensor',
    REAR: 'tram_vehicle_msgs/msg/VelocitySensor',
    CMD: 'tram_vehicle_msgs/msg/DriverControllerCommand',
    GNSS_MASTER_FIX: 'sensor_msgs/msg/NavSatFix',
    GNSS_ROVER_FIX: 'sensor_msgs/msg/NavSatFix',
    GNSS_MASTER_VEL: 'geometry_msgs/msg/TwistStamped',
    GNSS_ROVER_VEL: 'geometry_msgs/msg/TwistStamped',
}

#: Payload column names (after ``t_bag``, ``t_hdr``) per topic, as written by extract_bags.py.
COLUMNS = {
    FRONT: ('velocity',),
    REAR: ('velocity',),
    CMD: ('position',),
    GNSS_MASTER_FIX: ('lat', 'lon', 'alt', 'status', 'cov_xx', 'cov_yy', 'cov_zz'),
    GNSS_ROVER_FIX: ('lat', 'lon', 'alt', 'status', 'cov_xx', 'cov_yy', 'cov_zz'),
    GNSS_MASTER_VEL: ('vx', 'vy', 'vz', 'wz'),
    GNSS_ROVER_VEL: ('vx', 'vy', 'vz', 'wz'),
}

#: Default header.frame_id seen in the recordings (used when a message has to be synthesised).
FRAME_IDS = {FRONT: 'base_link', REAR: 'base_link', CMD: ''}

#: Nominal publishing periods [s] (measured: wheels 10 Hz with +-12 ms stamp jitter, cmd 20 Hz).
NOMINAL_PERIOD = {FRONT: 0.1, REAR: 0.1, CMD: 0.05, GNSS_MASTER_FIX: 0.1, GNSS_ROVER_FIX: 0.1,
                  GNSS_MASTER_VEL: 0.1, GNSS_ROVER_VEL: 0.1}

#: Wheel sensors publish km/h (not m/s as the dataset README claims); ratio to GNSS m/s is ~3.60-3.67.
KMH_PER_MS = 3.6
#: The wheel sensors output exactly 0 below ~0.15 km/h (smallest non-zero value seen in the data).
WHEEL_DEADBAND_KMH = 0.15
#: int8 range of DriverControllerCommand.position.
INT8_MIN, INT8_MAX = -128, 127


# --------------------------------------------------------------------------------------
# Anomaly labels (bit flags stored per output row)
# --------------------------------------------------------------------------------------
class Label(enum.IntFlag):
    """Per-message anomaly flags. Stored as uint32 in ``_label__<topic>`` arrays."""

    NONE = 0
    SLIP = 1 << 0            # traction wheel slip: wheel over-reads
    SLIDE = 1 << 1           # braking slide / skid: wheel under-reads
    LOCK = 1 << 2            # wheel locked (reads ~0 while vehicle moves)
    RESUME = 1 << 3          # first message after an injected dropout
    STALL = 1 << 4           # delivered late (stall + burst drain)
    SPIKE = 1 << 5           # additive / multiplicative outlier
    NAN = 1 << 6
    INF = 1 << 7
    NEGATIVE = 1 << 8        # sign flipped / negative value
    ABSURD = 1 << 9          # physically impossible magnitude
    ZERO = 1 << 10           # spurious 0 while moving
    FROZEN = 1 << 11         # stuck value
    NOISE = 1 << 12          # increased measurement noise
    STAMP_JITTER = 1 << 13   # header stamp perturbed
    ARRIVAL_JITTER = 1 << 14  # bag receive time perturbed
    DUPLICATE = 1 << 15      # duplicated message (the extra copy)
    OUT_OF_ORDER = 1 << 16   # delivered after a newer message
    ZERO_STAMP = 1 << 17     # header.stamp == 0
    STAMP_GLITCH = 1 << 18   # header stamp off by ~+-1 s (sec/nsec roll-over bug seen in real bags)
    CLOCK_OFFSET = 1 << 19   # header clock offset / drift segment
    SCALE_DRIFT = 1 << 20    # wheel-diameter scale error
    NOTCH_FAULT = 1 << 21    # controller value corrupted (stuck / jump / invalid)

    @classmethod
    def names(cls, mask: int) -> list[str]:
        return [f.name for f in cls if f.value and (int(mask) & f.value)]


#: Labels that mean "the value is not a valid measurement of the true state".
VALUE_FAULTS = (Label.SLIP | Label.SLIDE | Label.LOCK | Label.SPIKE | Label.NAN | Label.INF | Label.NEGATIVE
                | Label.ABSURD | Label.ZERO | Label.FROZEN | Label.NOTCH_FAULT)
#: Labels that mean "the timing information of this message is corrupted".
TIMING_FAULTS = (Label.STAMP_JITTER | Label.ARRIVAL_JITTER | Label.DUPLICATE | Label.OUT_OF_ORDER
                 | Label.ZERO_STAMP | Label.STAMP_GLITCH | Label.CLOCK_OFFSET | Label.STALL)
