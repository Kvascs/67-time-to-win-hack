#!/usr/bin/env bash
# Live check of the node exactly as the jury runs it (packaged parameters: MGRS 37U CB, base_link):
# play a dataset bag in real time, measure latency / rate / CPU / RAM, record the outputs and score
# them against a base_link reference built from both GNSS antennas with the organisers' TF.
#   check_run.sh <bag name under /bags> [seconds of playback, 0 = whole bag] [rate]
# Executables are started directly (not through `ros2 run`, whose wrapper does not pass SIGINT on),
# so the CPU/RAM figures are those of the node itself and every process stops cleanly.
set -eo pipefail
BAG="${1:?bag name}"
DUR="${2:-120}"
RATE="${3:-1.0}"
OUT="/out/${BAG}"
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
mkdir -p "${OUT}"
rm -rf "${OUT}/run_out"
PFX="$(ros2 pkg prefix tram_backup_odometry)"
TPFX="$(ros2 pkg prefix tram_backup_odometry_tools)"

stop() {  # SIGINT, then SIGKILL after 10 s
  kill -INT "$1" 2>/dev/null || return 0
  for _ in $(seq 1 20); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.5; done
  kill -KILL "$1" 2>/dev/null || true
}

"${PFX}/lib/tram_backup_odometry/tbo_node" --ros-args \
    --params-file "${PFX}/share/tram_backup_odometry/config/params.yaml" > "${OUT}/node.log" 2>&1 &
NODE=$!
sleep 2
ros2 bag record -o "${OUT}/run_out" /result/velocity /result/position /result/status /diagnostics \
    > "${OUT}/record.log" 2>&1 &
REC=$!
"${TPFX}/lib/tram_backup_odometry_tools/latency_probe" --csv "${OUT}/latency.csv" > "${OUT}/probe.log" 2>&1 &
PROBE=$!
sleep 2

if [ "${DUR}" = "0" ]; then
  ros2 bag play "/bags/${BAG}" --rate "${RATE}" > "${OUT}/play.log" 2>&1
else
  timeout --signal=INT "${DUR}" ros2 bag play "/bags/${BAG}" --rate "${RATE}" > "${OUT}/play.log" 2>&1 || true
fi
sleep 2
stop ${PROBE}
stop ${REC}
stop ${NODE}

echo "=== latency / rate / resources (last reports) ==="
grep -E "outputs=" "${OUT}/probe.log" | tail -n 3
echo "=== node log (tail) ==="
tail -n 5 "${OUT}/node.log"
echo "=== accuracy vs base_link reference from both antennas (played part) ==="
"${TPFX}/lib/tram_backup_odometry_tools/evaluate_run" --input-bag "/bags/${BAG}" --output-bag "${OUT}/run_out" --frame mgrs || true
