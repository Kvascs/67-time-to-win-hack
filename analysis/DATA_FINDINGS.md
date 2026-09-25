## KEY POINTS
1. **Units and time base hold up.** On 69 GNSS bags the wheel topics are in km/h; the median ratio wheel/GNSS is k = 3.5956. Wheel header stamps line up with GNSS vel header stamps (lag −0.4 ms, IQR −2.6…+1.5). GNSS fix positions lead wheel and vel by +43–44 ms. Header stamps fit better than bag times (residual 0.028 vs 0.034 m/s).
2. **The harness computes its metrics correctly.** A trivial estimator gave the same numbers as my independent hand computation on 3 bags, to 4 decimals. The only sensitivity is the end-drift denominator on bad-GNSS runs: for `defd0170` it is 5.2 km in the harness and 13.5 km with a naive path.
3. **Wheel scale k is the dominant drift term.** It is constant within a run (median change −0.008%, p95 +0.12%) but changes by vehicle and day, up to ±1.6%. It needs a prior from calibration plus online estimation from landmarks. The GNSS window cannot calibrate it: at most 1–2 m of travel.
4. **The map is accurate.** Validation RTK fixes sit 0.87 cm from the train-only map at the median (p90 23 cm, 96.9% within 1 m).
5. **Curve correction: the linear model is wrong in the loops.** Wheels read low in curves, but the effect levels off at about 0.9% for curve radius R < 60 m. map_build's linear 0.41|k| over-predicts by about 0.6% at R ≈ 26 m. Use a saturating lookup table.
6. **The traction model is confirmed but optimistic.** It is about 6× better than holding the last speed, and grade halves the error. Its published error at 1/3/5/10 s is 1.4–2.2× too low. My protocol gives 0.135/0.30/0.42/0.60 m/s on windows without a parked handle and 0.22/0.50/0.69/0.92 on all windows.
7. **New mid-route landmarks are confirmed.** Abrupt traction cut-offs (notch ≥ +4 → 0 in one step) all fall at 4 fixed places on the track, with spread 0.2–2.5 m and 0 outliers in 68 RTK events.
8. **"Stationary during GNSS init" is refuted for a 5 s window.** 27 of 86 long bags start moving within 5 s of bag start, and 4 of 17 validation bags reach 0.5–0.94 m/s inside that window.
9. **Output stamps must come from the cmd messages.** Wheel stamps give fewer than 10 distinct stamps per second. Publishing on both wheel and cmd messages gives 10–31% backward stamps. The node cannot see bag time without `/clock`.
10. **The biggest open risks are on the judge's side:** the output frame (ENU vs UTM, up to 110 m), the reference time base, ±1 s GNSS header-stamp episodes, and how latency is measured given the start-up burst.

---

# DATA_FINDINGS — consolidated and cross-checked (2026-09-25)

## 0. Scope and artifacts
- **Sources reviewed:** the six agents' structured outputs; their code and CSVs; `data/README.md`; the case PDF; `research/SYNTHESIS.md`.
- **New code in `C:\MosTransHack\analysis\cross_check\`:**
  - `xc_common.py` — shared helpers
  - `check_scale_lag.py` → `scale_lag_per_bag.csv`
  - `check_harness.py`
  - `check_curve.py` → `curve_windows.npy`
  - `check_traction.py`
  - `check_cutoffs.py`
  - `check_misc.py` — start of run, clock episodes, stamps, within-run k, detour, grade agreement
  - `make_plots.py` → `fig_scale_and_curve.png` (k per run by time of day; curve excess vs |k| against both models)
- **Possible answers to the open questions:** `research/qa_session/` holds an organiser Q&A recording (Zoom, about 55 min). Its whisper transcription was about 4% done when I checked. Read it before asking the organisers anything in §5.

## 1. Scoring rules that drive the design (PDF and README)
- **Points:** speed 30, position 35, robustness 20, real-time 15.
- **Real-time limits:**
  - latency "input → published result" ≤ 100 ms, peaks ≤ 250 ms
  - publishing rate ≥ 10 Hz (20–50 Hz recommended)
  - ≤ 2 CPU cores, ≤ 0.5 GB RAM
  - standard `colcon build` with no internet
- **Deadline:** 27 Sep, 23:59 MSK.
- **Stamps:** `header.stamp` = "time of the corresponding input (time from the bag)". The judge pairs by nearest stamp within about 0.05 s.
- **Position:** in a "local metric frame consistent with the reference". Compared per component (x/y/z) and as 3-D distance. Relative odometry from the start point is allowed if no absolute anchor is given. Cross-track error is measured against the organisers' "pathgraph" map "if needed".
- **Odometry message:** correct filling counts (stamp, frame_id, position, longitudinal speed, covariances).

## 2. Verified facts (independent checks)

| # | Claim (agents) | Independent result | Verdict |
|---|---|---|---|
| 1 | Wheel topics are km/h, k ≈ 3.596 (all) | 69 bags, clean moving samples: fleet median 3.5956, range 3.5435–3.6554. Straight-track values are about 0.06% higher. | Confirmed |
| 2 | k depends on vehicle and date (all) | See k table below | Confirmed |
| 3 | k is constant within a run (wheel_anomalies) | Last-quarter k / first-quarter k − 1: median −0.008%, p5–p95 −0.09…+0.12%, max 0.33% (on the drifting days) | Confirmed; allow slow drift |
| 4 | Wheel header stamp = measurement time, lag 0 vs GNSS vel (timing_reference, traction_id, harness) | Header vs header −0.4 ms (IQR −2.6…+1.5). Bag vs bag −38.8 ms (IQR −46.7…−31.6). RMS residual 0.028 (header) vs 0.034 m/s (bag) | Confirmed |
| 5 | Fix positions lead wheel/vel by 45–48 ms (timing_reference, harness) | Wheel vs \|dp/dt\| +42.6 ms (IQR +30…+52, n = 60). Vel vs \|dp/dt\| +44.0 ms (IQR 37–50) | Confirmed; use +45 ms, per-bag spread ±15 ms |
| 6 | cmd-stamped outputs match about 100% of reference epochs | Trivial estimator in the harness: 20.05 Hz, match rate 1.000 / 0.9997 / 1.000, stamp age median 0.7–1.4 ms | Confirmed |
| 7 | Wheel stamps unsuitable as the only trigger | 8.99–9.80 distinct stamps/s. Front and rear share stamps in 99.4–100% of messages. Newest wheel sample at a cmd stamp is 87 ms old (p95 156, p99 190). Wheel+cmd union in arrival order has 10–31% backward stamps | Confirmed |
| 8 | Start-up burst: about 2.5 s of history arrives at bag start | First rows are 1.3–2.8 s old; cmd-stamped outputs are up to 2.66–3.06 s old at publish time | Confirmed; latency risk (Q7) |
| 9 | Harness metrics are correct | Hand computation identical for speed RMSE/MAE/bias/match, 3-D RMSE/max, z RMSE and final error | Confirmed; drift % denominator depends on the judge (5.2 vs 13.5 km on defd0170) |
| 10 | Map is cm-accurate (map_build) | Validation status-2 fixes vs train-only map, 28.9k points: p50 0.87 cm, p68 1.46 cm, p90 23 cm, p95 75 cm; 86.8% < 10 cm | Confirmed |
| 11 | Curve under-read: min(0.5\|k\|, 0.95%) (wheel_anomalies) vs 0.41\|k\| (map_build) | 9124 windows of 30 m: excess 0% (\|k\| < 0.003), +0.11% (0.004), +0.27% (0.0074), +0.39% (0.013), +0.88% (0.017), +0.92% (0.023), +0.91% (0.038, R ≈ 26 m). A least-squares linear fit gives c = 0.417 but hides the saturation. Tightest radii: 15.9 m (west loop), 17.9 m (east loop) | Saturating model confirmed |
| 12 | Traction model and grade (traction_id) | 17 val bags, 6277 windows. Model: 0.135/0.30/0.42/0.60 m/s at 1/3/5/10 s on handle-not-parked windows (90%), 0.22/0.50/0.69/0.92 on all. Without grade 0.17/0.42/0.64/1.10. Hold 0.55/1.52/2.32/3.76, which reproduces their hold numbers | Confirmed qualitatively; their headline 0.10/0.21/0.29/0.41 is 1.4–2.2× optimistic |
| 13 | Grade profiles of the two maps agree | Difference RMS 0.14% grade, correlation 0.996 | map_build grade can be used with traction_id's grade coefficients |
| 14 | Traction cut-offs at fixed spots (wheel_anomalies) | 68 RTK events on main, all in 4 clusters: s = 349 m (n 21, robust σ 2.5 m), 3707 (27, 0.28), 5948 (17, 0.22), 8723 (3, 0.01) | Confirmed; usable landmarks at speed |
| 15 | Westbound detour on 27–28 Jul (map_build) | All 13 RTK westbound runs on 07-27 are 4.1–5.7 m right of main at s 1200–1700 m; other dates within 6 cm | Confirmed |
| 16 | Tram stationary in the init window in all runs (timing_reference "0/97", wheel_anomalies, SYNTHESIS §0.8) | First motion (> 1 km/h) after bag start: min 3.5 s, median 7.9 s. 27/86 long bags move within 5 s. Val bags defd0170, 2b4a6347, 9c362687, d927f360 reach 0.5–0.94 m/s inside 5 s | **Refuted** for a 5 s bag-time window; scale still not calibratable |
| 17 | GNSS header stamps jump ±1 s for minutes | 3b3d9eb8: all GNSS topics shifted +1.0 s against bag time from 579 to 822 s. 40ffd323: −1 s at 85–145 s and +1 s at 446–506 s. Wheel/cmd only show 0.3–3 s blips at recorder clock steps | Confirmed; this is an error on the judge's side |
| 18 | Vehicle identifiable from the antennas | RTK baseline 12.4426 ± 0.0020 m (30618, n 56) vs 12.4283 ± 0.0036 (30639, n 12), ranges do not overlap. Lateral offset −4 vs +22 cm. Baseline yaw minus course −0.18° vs +0.98° | Vehicle known within the init window |
| 19 | z = 0 costs about 12 m; ENU vs UTM-local differ by about 110 m | Trivial estimator z RMSE 9.3–12.6 m. Analytic: grid convergence ≈ −1.3° gives about 114 m at 5 km | Confirmed |

**k by vehicle and local date** (median [min–max], n runs; curves included):

| Vehicle | Date | k |
|---|---|---|
| 30618 | 07-27 | 3.5974 [3.5924–3.6006], 26 |
| 30618 | 08-10 | 3.5947 [3.5877–3.5959], 10 |
| 30618 | 08-26 | 3.5955 [3.5918–3.5974], 12 |
| 30618 | 09-03 | 3.5600 [3.5435 morning … 3.5772 evening], 4 |
| 30639 | 05-05 | 3.6289 [3.6158–3.6554], 9; falls steadily through the day |
| 30639 | 08-26 | 3.5860 [3.5840–3.5902], 8 |

The 17 no-GNSS bags are 30618 on 08-10 (16) and 30639 on 05-05 (1). They have no GNSS at all, so they cannot be scored on position.

## 3. Contradictions resolved
- **C1 Output trigger and stamp.** Five different recommendations; resolved as publish on every cmd message with the cmd header stamp.
  - A cmd stamp is within about 1 ms of the cmd bag time, so it satisfies both readings of the README's "time of the input".
  - Match rate 1.000 at 20 Hz, stamps monotonic.
  - Wheel-only publishing gives fewer than 10 distinct stamps/s and about 10% of reference epochs unmatched.
  - The wheel+cmd union produces backward stamps.
  - Bag time (map_build's proposal) cannot be seen by the node without `/clock`.
- **C2 Curve model.** Saturating lookup table (#11), not the linear model.
- **C3 Drive dynamics.** Follow traction_id (lag τ = 0.45 s, no dead time; bias adaptation hurt at ≥ 10 s) over anomaly_sim's "0.6–0.8 s lag plus bias state".
- **C4 Notch −1 acceleration** reported as −0.06 / −0.14 / −0.35 m/s² by different agents. The difference is raw vs grade-compensated; drivers use notch −1 mostly downhill. Use the grade-compensated table.
- **C5 Scale from the first GNSS seconds** (traction_id) is impossible. Use a prior plus online landmark estimation.
- **C6 Raw-odometry along-track baselines** of 11.6 m (timing_reference, 8 runs), 4.44 m median (harness B1) and 2.60 m median (map_build drift_sim) differ by protocol: mean vs median, all fixes vs RTK-only, zero-order-hold lag, initialisation on main. Always quote the protocol with the number.
- **C7 Handle-parked (automation) share.** 2.1% of pooled moving time vs 0.17% median per bag: the distribution is skewed (max 3.5%). Both hold.
- **C8 Map conventions.** traction_id uses a centreline with s from the west terminus (5497 m, UTM); map_build uses a closed loop, 11053.66 m, s = 0 at the east platform, local ENU. Use map_build as the single runtime map; grade is the same in both.

## 4. Errors in agent outputs
- **E1 wheel_anomalies:** "publish on every wheel and cmd message … with monotonic stamps" cannot both hold (10–31% backward stamps).
- **E2 map_build:** "stamp = bag time of the triggering wheel message" cannot be observed by the node and gives under 10 Hz.
- **E3 map_build and SYNTHESIS:** 0.41|k| extrapolated to 1.6% in the loops; the data saturate at about 0.9%.
- **E4 timing_reference, wheel_anomalies and SYNTHESIS §0.8:** "stationary in the init window in all runs" (see #16).
- **E5 traction_id:** headline bridging errors are 1.4–2.2× optimistic, because the protocol excludes unclean and automation windows.
- **E6 anomaly_sim:** "traction saturates at +0.88 m/s² for notch ≥ 9" and "brake −0.14 m/s² at notch −1" are raw statistics without grade compensation. Do not use them for the model.
- **E7 harness:** the drift % denominator depends on its own outlier cleaning; the judge's could be 2.6× larger on float-GNSS runs.

## 5. Questions for the organisers (priority order; our default in brackets)
1. **Reference frame:** ENU tangent plane at the first fix, or UTM/MGRS minus the origin? Which origin: first master fix, first status-2 fix? [ENU at the first master fix by header stamp.] Wrong guess costs up to 110 m.
2. **Reference antenna:** master, rover or midpoint? [Master.] Wrong guess costs 12.4 m.
3. **z:** is it compared? ENU-up or alt − alt0? [ENU-up from the map, relative to the exact origin fix.] z = 0 costs 12 m RMS; alt − alt0 vs ENU-up differs by 1.8 m.
4. **Reference time base:** GNSS `header.stamp` or bag receive time? [Header.] Wrong guess costs about 0.02 m/s RMSE, ±0.05 m/s bias on transients and about 0.3 m.
5. **±1 s GNSS header-stamp episodes:** are they repaired in the reference?
6. **Reference speed:** vel topic horizontal |v|, 3-D, or derived from positions? [Horizontal |v| of master vel.]
7. **Latency measurement:** `/clock` minus stamp, or input arrival to output arrival? The start-up burst makes any output that copies input stamps look about 2.6 s old.
8. **GNSS window in test bags:** how many seconds, measured in bag time? Both antennas? Is vel included?
9. **Coverage and rate:** are unmatched reference epochs penalised? Is the ≥ 10 Hz rate counted by messages or by distinct stamps?
10. **Test bags:** same two vehicles and same days as the dataset? Which injected faults?
11. **End-drift distance:** measured along the reference path or from integrated speed?
12. **Pathgraph map:** will it be provided?

## 6. Prioritized design decisions

**P0 — must have**
1. **Frame, antenna and z.**
   - Frame is a parameter; default exact WGS84 ENU at the first master fix by header stamp.
   - Publish the master antenna point.
   - z from the map, relative to the same origin fix (including that fix's own bias).
   - Keep internal positions absolute.
2. **Stamps and trigger.**
   - Publish on every cmd message using its header stamp.
   - Watchdog publishes if cmd is silent for more than 75 ms.
   - Stamps strictly increasing.
   - Hedge for the start-up burst: suppress outputs while inputs arrive faster than real time (wall-clock gap < 0.3 × stamp gap). This costs about 25 stationary reference epochs.
3. **Values at the stamp.**
   - Filter in header time.
   - Speed: predict with the model acceleration to the stamp (newest wheel sample is 87 ms old; zero-order hold would add about 0.03 m/s of bias on transients); no speed lead.
   - Position: +45 ms lead.
4. **Input hygiene.**
   - Mark a bogie stale after 0.3 s silence; handle the 30639 "stuck at zero" start (one bogie reads 0 then goes silent for up to 73 s).
   - Reject NaN/inf and out-of-range values; drop stamps ≤ the last accepted stamp per topic.
   - Accept burst messages into the state.
   - Example: naive averaging with a stale rear bogie gives 1.03 m/s RMSE on val bag 3b3d9eb8.
5. **One-dimensional state along the map** (map/ for deployment, map_train/ for honest validation).
   - Yaw, z and grade come from the map.
   - Date-aware westbound detour for 07-27/28 (otherwise 4.2 m cross-track over 625 m, about 1.4 m of 3-D RMSE on those runs).
6. **Wheel scale.**
   - Prior from a (vehicle, date) table: vehicle from the antenna fingerprint, date from the stamps. Fall back to 3.596 ± 1%.
   - Online estimation from landmark-to-landmark distances, with a slow random walk (drift ≤ 0.33% per run).
7. **Stop landmarks.**
   - Require both bogies at 0 for ≥ 3 s and no stuck-zero state.
   - Gate scales with the along-track uncertainty.
   - Evidence: drift_sim median along-track RMSE 2.6 → 0.74–0.78 m, end drift 0.106% → 0.008%.

**P1 — high value**
8. **Curve correction table** (saturating at about 0.9%), applied to distance and to published speed (the reference is the antenna speed).
9. **Traction model as process model.** OE lookup table plus grade plus online traction/brake gains; innovation monitor for a parked handle and for emergency braking outside the notch (down to −5 m/s²); use it to bridge slip, slide and dropouts.
10. **Cut-off landmarks** at s = 349 / 3707 / 5948 / 8723 m with a gate.
11. **Odometry fields and diagnostics.** Realistic covariances; `twist.linear.x` = speed; `/diagnostics` with slip flags (criterion-3 bonus).

**P2 — optional**
12. Snap stamps to a 0.05 s grid, only if a header-time reference is confirmed (removes pairing error of 0.009 m/s and 0.11 m).
13. Multiple track hypotheses for non-RTK starts in the west yard.

## 7. Expected error budget
- **Speed:** floor 0.028 m/s (wheel vs vel with per-run k) plus 0.009 pairing cost. A 1% k error costs 0.1 m/s at 10 m/s, so online k also matters for speed. Transient bias about ±0.005–0.01 m/s under a header-time judge.
- **Position:** along-track median RMSE about 0.8 m with landmarks and online k (mean about 1.8 m); end drift about 0.01%; cross-track at map accuracy (cm) except on detour dates. Bad-reference runs (14 are grade C) put a floor under the achievable score for every team.

## 8. Gaps not yet analysed
1. No end-to-end evaluation of the integrated estimator yet: honest map_train map, no oracle, full harness judge metrics.
2. Landmark failure modes: queues and secondary berths 5–18 m away, stuck-zero standstills, gate behaviour when k is off by more than 1%.
3. GNSS windows of 1/3/5/10 s and starts where the tram is already moving.
4. Handling of the start-up burst and its latency.
5. The anomaly_sim S00–S20 scenarios and the harness fault suites run on the integrated node.
6. Transient front/rear speed differences of up to 0.25 m/s in the loops are unexplained.
7. The cause of the day-to-day k changes (probably on the sensor side).