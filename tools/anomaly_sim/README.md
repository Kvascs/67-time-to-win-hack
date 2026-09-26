# anomaly_sim: anomaly injection for tram wheel-odometry robustness testing

`anomaly_sim` takes a clean recording (the `.npz` produced by `tools/extract_bags.py`) and produces
corrupted copies. Each copy has exact per-message labels and an event log. It can also be written back
to a **rosbag2 (sqlite3)** bag, so the ROS 2 node can be exercised with `ros2 bag play`. That is how the
jury will test criterion 3 ("robustness to slip / slide / dropouts / outliers, GNSS only in the first
seconds").

The simulator is built around the anomalies actually present in the dataset (statistics in `out/natural/`):
- **Traction slip and braking slide** come from a wheel-rail adhesion model (Polach-type creep curve), a
  wheelset inertia model and an anti-slip / wheel-slide-protection controller. It is driven by the recorded
  notch and speed, so the shapes (humps, saw-tooth, creep plateau, lock) arise from the physics.
- **Timing faults** reproduce the patterns observed in the real bags: +-1 s stamp glitches, a bus stall
  followed by a slow queue drain, header-clock offset and drift, and long single-sensor silences.

```
tools/anomaly_sim/
  constants.py     topic keys, units, Label bit flags
  run.py           Run/Stream data model, npz load/save (extractor layout + side arrays), load_pair
  context.py       clean "truth" signals (speed, accel, notch, distance) + event placement
  physics.py       adhesion / wheelset / anti-slip (cutoff, creep) / WSP / lock simulation
  injectors.py     15 injector types (slip ... gnss_cut), stage-ordered
  scenario.py      suites (YAML/JSON), deterministic seeding, batch generation
  bagio.py         rosbag2 writer (Humble-clone layout or rosbags Writer v8) + read-back verification
  characterize.py  statistics / catalogue of natural anomalies in the dataset
  evaluate.py      causal replay harness, reference estimators, robustness metrics
  plotting.py, report_figures.py
  scenarios/suite.yaml   21 named, seeded scenarios
  tests/                 pytest suite (69 tests)
  out/                   generated data (npz, bags, eval, natural statistics)
```

## Quick start

Run everything from `C:\MosTransHack\tools`. It needs numpy, scipy, pyyaml, matplotlib and
`rosbags>=0.10` (for bag I/O).

```bash
python -m anomaly_sim list -v                          # the 21 scenarios
python -m anomaly_sim list --injectors                 # injector types + default parameters

# corrupted npz for all scenarios x validation bags  ->  anomaly_sim/out/npz/<scenario>/<bag>.npz (+ .events.json)
python -m anomaly_sim run --bags val --workers 6

# only some scenarios / bags, and also write rosbag2 dirs -> anomaly_sim/out/bags/<scenario>/<bag>/
python -m anomaly_sim run --scenarios S01 S05 S18 --bags 30618_e3d94878 30639_3b3d9eb8 --write-bags

# convert an existing corrupted npz to a bag
python -m anomaly_sim tobag anomaly_sim/out/npz/S05_slide_wheel_lock/30618_e3d94878.npz --out anomaly_sim/out/bags --overwrite

python -m anomaly_sim plot anomaly_sim/out/npz/S02_slip_both_bogies/30639_3b3d9eb8.npz   # PNG next to the npz
python -m anomaly_sim validate anomaly_sim/out/npz --bags anomaly_sim/out/bags --npz-root anomaly_sim/out/npz     # schema + bag read-back
python -m anomaly_sim characterize                     # natural anomaly statistics -> anomaly_sim/out/natural
python -m anomaly_sim evaluate                         # reference estimators on anomaly_sim/out/npz -> anomaly_sim/out/eval
python -m pytest anomaly_sim/tests -n 6                # tests (~2 min)
```

Python API:

```python
from anomaly_sim import load_run, load_suite, get_scenario, apply_scenario, save_run, write_bag
clean = load_run('30618_e3d94878')                       # data/npz/<bag>.npz
bad = apply_scenario(clean, get_scenario(load_suite(), 'S05'))
save_run(bad, 'out/npz/S05/30618_e3d94878.npz')
write_bag(bad, 'out/bags/S05/30618_e3d94878')             # verified by reading it back

# ad-hoc scenario
from anomaly_sim import Scenario
sc = Scenario('my_test', seed=1, injectors=[{'type': 'slip', 'bogie': 'both', 'count': 4, 'peak_rel': [0.1, 0.2]},
                                            {'type': 'dropout', 'topics': 'wheels', 'at': [120.0], 'duration': 8.0}])
```

## Testing the ROS 2 node with the corrupted bags

```bash
# build tram_vehicle_msgs in the workspace first (see dataset README)
ros2 bag play anomaly_sim/out/bags/S05_slide_wheel_lock/30618_e3d94878 --clock
ros2 topic echo /result/velocity
```

* The default bag format `humble` is a byte-level clone of the Humble recorder layout used by the
  dataset: the same sqlite DDL, `schema` = (3, humble), `metadata.yaml` version 5, and the original QoS
  strings, including the two QoS profiles of `/vehicle/driver_position_cmd`. The test suite checks that
  the schema is identical to the original bag and that the QoS strings parse with integer fields, the
  way Humble's `Rosbag2QoS` decoder reads them.
* Copy-on-write: unmodified messages keep their **original CDR bytes and int64 timestamps**. An
  uncorrupted crop is byte-identical to the source. Only corrupted or synthetic messages are
  re-serialised, using the rosbags typestore with `VelocitySensor` / `DriverControllerCommand` registered
  from `data/tram_vehicle_msgs/msg`.
* `--fmt rosbags` uses the rosbags `Writer`. rosbags >= 0.10 can only write metadata v8/v9 (Jazzy
  era); v8 is chosen because its QoS encoding is the integer form Humble expects. Humble ignores
  unknown YAML keys, so this should play, but it has not been tested on a real Humble install here.
* Each bag directory also contains `anomaly_events.json` (scenario, seed and the injected events).
  `ros2 bag` ignores it.
* The bag contains GNSS only for the first `gnss_keep_s` seconds, exactly as the jury will provide it.
  Use the original bag or npz for reference metrics.

## Output format (npz)

`out/npz/<scenario>/<bag>.npz` has exactly the keys and columns of the clean extractor output, so any
existing loader works unchanged. Every topic array holds `[t_bag, t_header, payload...]`, sorted by
`t_bag` (the `ros2 bag play` order). The wheel speeds are in km/h, as in the recordings.

The following side arrays are added (with a leading underscore, so they are easy to skip):

| key | meaning |
|---|---|
| `_label__<topic>` | uint32 bit mask per delivered message, see `constants.Label` (SLIP, SLIDE, LOCK, RESUME, STALL, SPIKE, NAN, INF, NEGATIVE, ABSURD, ZERO, FROZEN, NOISE, STAMP_JITTER, ARRIVAL_JITTER, DUPLICATE, OUT_OF_ORDER, ZERO_STAMP, STAMP_GLITCH, CLOCK_OFFSET, SCALE_DRIFT, NOTCH_FAULT) |
| `_clean__<topic>` | payload the message would have had without corruption (vehicle topics) |
| `_hdr0__<topic>` | uncorrupted header stamp |
| `_src__<topic>` | row index of the source message in the clean npz (-1 = synthetic) |
| `_meta_json` | scenario, seed, injector specs, placement, source npz, **events** |

Invariant, enforced by the tests: `label == 0` means the value and stamp are identical to the clean
recording. `<bag>.events.json` holds the same metadata plus the event list in readable form. Each
event has `type`, `topics`, `t0`/`t1` (absolute measurement time), `t0_rel`/`t1_rel` (seconds from the
start of the run), the sampled `params`, and `stats` (for example achieved `peak_rel`, `active_s`,
`locked_s`, controller type, `mu_low`).

`anomaly_sim.load_pair(path)` returns `(corrupted_run, clean_run)`. The clean run carries the complete
GNSS, for reference metrics.

## Scenario suite (`scenarios/suite.yaml`)

Every scenario also removes GNSS after 5 s (3 s in S18), and no event starts in the first 10 s.

| # | scenario | what is injected | tests (jury criterion) |
|---|---|---|---|
| S00 | baseline_gnss_cut | nothing | reference accuracy with GNSS only at start (1, 2) |
| S01 | slip_single_bogie | 3-6 physical traction slips on one bogie, 5-30 %, 1-10 s | slip detection via front/rear mismatch (3) |
| S02 | slip_both_bogies | slips on both bogies, simultaneous or spatially lagged | model-based slip detection (3) |
| S03 | slip_creep_plateau | slip held smoothly at 5-15 % for 4-10 s by a creep controller | hardest slip case, slow bias (3) |
| S04 | slide_braking_wsp | braking slides 10-50 % with WSP cycling | slide handling (3) |
| S05 | slide_wheel_lock | wheel reads 0 for 0.5-3 s while moving (40 % both bogies) | lock handling, no false stop (3) |
| S06 | dropout_short | 8-15 gaps of 0.2-2 s (one / both wheels, notch, all) | short dropouts (3) |
| S07 | dropout_long | both wheels 3-10 s; one sensor silent 20-75 s from a departure | model dead reckoning (3) |
| S08 | stall_burst | 0.5-2 s stall, then late delivery draining at 0.9x period | latency and stamp handling (3, 4) |
| S09 | outliers_spikes | 1 % spikes of 5-60 km/h, zeros, sign flips | outlier rejection (3) |
| S10 | invalid_values | NaN, +-inf, 1e300, -3.4e38, 6553.5, negative; notch codes 127/-128/100/+-16 | no crash, no NaN output (3) |
| S11 | frozen_sensor | stuck last value (70 %) or 0 (30 %) for 2-20 s | stuck-sensor detection (3) |
| S12 | noise_increase | +1-3 km/h white, +3 % proportional, AR(1) coloured, minutes long | adaptive noise (1, 3) |
| S13 | timing_jitter_dup_reorder | 20 ms stamp jitter with tails, 3 % duplicates, 1 % out of order | message handling (3, 4) |
| S14 | stamp_faults | zero stamps, +-1 s glitches, +-1 s clock offset with drift | output stamp sanity (2, 3) |
| S15 | wheel_wear_scale | front scale drifting to -1..-2 %, rear constant +0.5..+2 % | scale adaptation, drift % (2) |
| S16 | notch_faults | stuck, jumping, invalid and offset notch; notch dropouts | model inputs not trusted blindly (3) |
| S17 | combined_moderate | a realistic bad day mixing the above | overall (1-3) |
| S18 | combined_severe | everything at high intensity, GNSS 3 s | no crash, no divergence, recovery (3, 4) |
| S19 | jury_style_simple | rectangular x1.2 for 5 s, -30 % for 2 s, zeros 3 s, 5 s gaps | naive synthetic tests (3) |
| S20 | natural_magnitude_slip_slide | natural-size slips (achieved p50 +39 %, p95 +64 %) at 1-8 m/s, slides -30..-70 %, 30 % locks | slip handling at real magnitudes (3) |

Parameters: a scalar is fixed, `[lo, hi]` is uniform, `{a: w, b: w}` is a weighted choice. Placement
keys: `count`, `duration`, `when` (any, motion, standstill, traction, braking, coast, accel, decel,
departure, arrival), `min_speed`/`max_speed`, `min_notch`/`max_notch`, `min_hold`, `min_gap`,
`t_min`/`t_max`, `lead`, and `at` (explicit start times). Randomness is deterministic: the generator is
seeded with `(scenario seed, crc32(bag name), injector index)`.

## Injector reference (key parameters)

| type | stage | key parameters (defaults) |
|---|---|---|
| `scale_drift` | 10 | `topics` wheels, `start` 0, `end` [-0.02, 0.02], `profile` linear / constant / step |
| `slip` | 20 | `bogie` {front .4, rear .4, both .2}, `peak_rel` [.05, .30], `duration` [1, 10], `controller` {cutoff, creep}, `model` physical / step / trapezoid, `both_mode` simultaneous / spatial, `G` [40, 100], `rho` [.55, .85], `min_notch` 5 |
| `slide` | 20 | `depth_rel` [.1, .5], `duration` [.5, 3], `lock_prob` .25, `lock_duration` [.5, 3], `controller` {cutoff .6, creep .4}, `max_notch` -3 |
| `noise` | 30 | `sigma_kmh` [.5, 2], `rel_sigma`, `ar_sigma_kmh`, `ar_tau`, `count` 0 = whole run |
| `frozen` | 40 | `topics` {front, rear, cmd}, `mode` {hold .7, zero .3}, `duration` [2, 20], `stamp_frozen` |
| `notch_fault` | 40 | `kinds` {stuck, jump, invalid, offset}, `duration` [.5, 5] |
| `outliers` | 50 | `rate` .005, `burst` [1, 3], `kinds` {spike, zero, negative, nan, inf, absurd}, `spike_kmh` [5, 60] |
| `dropout` | 60 | `topics` {front, rear, wheels, cmd, vehicle}, `duration` [.2, 10], `mode` drop / stall, `drain_factor` .9 |
| `duplicates` | 70 | `rate` .02, `delay_ms` [0, 5] |
| `reorder` | 70 | `rate` .01, `delay` [.15, .6] s |
| `stamp_jitter` | 80 | `sigma_ms` 15, `tail_prob` .01, `tail_ms` [50, 300], `arrival_sigma_ms` |
| `stamp_glitch` | 80 | `offset_s` +-1, `pattern` single / alternating (all vehicle topics together) |
| `clock_offset` | 80 | `offset_s` [-1, 1], `drift` [-.05, .05] s/s, `duration` [10, 60] |
| `zero_stamps` | 85 | `rate` .005 and/or `count` windows |
| `gnss_cut` | 90 | `keep_s` 5 |

Stages run in ascending order. Physics (on clean timing) runs first, then sensor faults, then delivery
faults, then timing faults, and GNSS removal runs last.

## Physics of slip and slide (`physics.py`)

Per bogie, with forces normalised by the axle load:
- Slip velocity: `dw/dt = G (u f_dem(notch) - f_a(w, v)) - a_vehicle`.
- Adhesion: `f_a = mu (A + (1-A) e^{-B|w|}) (2/pi) atan(s / s_c)`, where `s = w / v`.
- `f_dem` comes from the measured notch-to-acceleration table.
- A low-adhesion patch (`mu_low = rho * f_dem`) starts a run-away that the controller stops:
  - `cutoff` (classic re-adhesion or WSP) cuts torque above `w_on` and re-applies it with a ramp, which
    gives humps and saw-tooth patterns;
  - `creep` (slip-regulating PI) holds `s_target`, which gives a plateau;
  - `none` in braking gives a **lock**.

Only the measured wheel speed is corrupted; the true motion (GNSS) is unchanged. The equation is
integrated with a linearly implicit Euler scheme (stiff creep region, dt = 4 ms, ~0.1 s CPU per event).
Setting `ANOMALY_SIM_NUMBA=1` enables a numba JIT of the loop, which is about 100x faster but costs a
6-10 s import.

## Evaluation harness (`evaluate.py`)

`evaluate_run(npz, estimators=(...))` replays the vehicle messages in arrival order. Any object with
`reset()` and `on_message(key, t_bag, t_hdr, value) -> (stamp, v) | None` can be plugged in. The
harness scores the estimator against the robust GNSS reference of the clean run and reports:
- `rmse_rt`, `p99_err`, `max_err`, `rmse_anom` (inside events), `recovery_s`;
- `drift_pct` (integrated distance vs reference);
- `match_rate` (output stamps within 0.5 s of real time);
- `bad_out` (NaN / inf / absurd outputs).

Two yardsticks are included: `naive` (average of the latest wheels) and `kf_baseline` (a 2-state KF
with a notch model, bias, chi-squared gating, plausibility checks and re-sync). Results are in
`out/eval/eval_summary.md` (regenerate with `python -m anomaly_sim evaluate`).

## Limitations

* The true vehicle motion is not altered by injected slip. In reality, adhesion loss also reduces the
  tractive force, so the notch model would over-predict acceleration slightly during a slip.
* Placement uses a clean-wheel speed estimate as truth. The rare natural anomalies stay in the
  recordings (they are listed in `out/natural/natural_events.json` and are useful as a real test set).
* Bags are written for single-file sqlite3 recordings (all bags of this dataset).
