#!/usr/bin/env bash
# Live check of the node exactly as the jury runs it (packaged parameters: MGRS 37U CB, base_link):
# play a dataset bag in real time, measure latency / rate / CPU / RAM, record the outputs and score
# them against a base_link reference built from both GNSS antennas with the organisers' TF.
#   check_run.sh <bag name under /bags> [seconds of playback, 0 = whole bag] [rate]
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
echo "=== accuracy vs base_link reference from both antennas (played part) ==="
ros2 run tram_backup_odometry_tools evaluate_run --input-bag "/bags/${BAG}" --output-bag "${OUT}/run_out" --frame mgrs || true
