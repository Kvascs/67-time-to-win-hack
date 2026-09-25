#!/usr/bin/env bash
# Live check of the node: play a dataset bag in real time, measure latency / rate / CPU / RAM,
# record the outputs and compute judge-like accuracy against the bag's GNSS (antenna 1).
#   check_run.sh <bag name under /bags> [seconds of playback, 0 = whole bag] [rate]
# Accuracy is evaluated in antenna mode (ENU at the first fix, no base_link shift) because the
# GNSS antenna is the only reference inside the dataset bags.
set -eo pipefail
BAG="${1:?bag name}"
DUR="${2:-120}"
RATE="${3:-1.0}"
OUT="/out/${BAG}"
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
mkdir -p "${OUT}"
rm -rf "${OUT}/run_out"

ros2 run tram_backup_odometry tbo_node --ros-args \
    --params-file "$(ros2 pkg prefix tram_backup_odometry)/share/tram_backup_odometry/config/params.yaml" \
    -p output_frame:=enu -p base_link_along_m:=0.0 -p base_link_height_m:=0.0 \
    > "${OUT}/node.log" 2>&1 &
NODE=$!
sleep 2
ros2 bag record -o "${OUT}/run_out" /result/velocity /result/position /result/status /diagnostics \
    > "${OUT}/record.log" 2>&1 &
REC=$!
ros2 run tram_backup_odometry_tools latency_probe --csv "${OUT}/latency.csv" > "${OUT}/probe.log" 2>&1 &
PROBE=$!
sleep 2

if [ "${DUR}" = "0" ]; then
  ros2 bag play "/bags/${BAG}" --rate "${RATE}" > "${OUT}/play.log" 2>&1
else
  timeout --signal=INT "${DUR}" ros2 bag play "/bags/${BAG}" --rate "${RATE}" > "${OUT}/play.log" 2>&1 || true
fi
sleep 2
kill -INT ${PROBE} ${REC} 2>/dev/null || true
sleep 3
kill -INT ${NODE} 2>/dev/null || true
wait ${NODE} 2>/dev/null || true

echo "=== latency / rate / resources (last reports) ==="
grep -E "outputs=" "${OUT}/probe.log" | tail -n 3
echo "=== node log (tail) ==="
tail -n 5 "${OUT}/node.log"
echo "=== accuracy vs GNSS antenna 1 (played part) ==="
ros2 run tram_backup_odometry_tools evaluate_run --input-bag "/bags/${BAG}" --output-bag "${OUT}/run_out" || true
