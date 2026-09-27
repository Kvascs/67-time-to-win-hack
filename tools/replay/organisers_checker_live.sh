#!/usr/bin/env bash
# Live check with the organisers' own scoring code: their `metrics.py` (hackathon_solution_checker from
# check-code) runs next to our node in the same container while the bag plays in real time.
#   docker run --rm --cpus=2 --memory=512m -v <check-code>:/check:ro -v <bags>:/bags:ro -v $(pwd)/out:/out \
#       -v $(pwd)/tools/replay:/chk:ro tram_backup_odometry bash /chk/organisers_checker_live.sh 30618_88aea4d9
# /check = the organisers' check-code folder (read-only), /bags = bags, /out = logs (out/chk/*.log).
set -o pipefail
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
mkdir -p /chk_ws/src /out/chk
cp -r /check/src/checker_ros /chk_ws/src/
cd /chk_ws && colcon build --packages-select hackathon_solution_checker > /out/chk/build.log 2>&1 || { tail -20 /out/chk/build.log; exit 1; }
source /chk_ws/install/setup.bash

stop() {  # SIGINT, then SIGKILL after 10 s
  kill -INT "$1" 2>/dev/null || return 0
  for _ in $(seq 1 20); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.5; done
  kill -KILL "$1" 2>/dev/null || true
}

PFX="$(ros2 pkg prefix tram_backup_odometry)"
"${PFX}/lib/tram_backup_odometry/tbo_node" --ros-args --params-file "${PFX}/share/tram_backup_odometry/config/params.yaml" > /out/chk/node.log 2>&1 &
NODE=$!
sleep 2
"$(ros2 pkg prefix hackathon_solution_checker)/lib/hackathon_solution_checker/metrics" > /out/chk/metrics.log 2>&1 &
MET=$!
sleep 3
ros2 bag play "/bags/${1:-30618_88aea4d9}" > /out/chk/play.log 2>&1
sleep 3
stop ${MET}
stop ${NODE}
echo "=== organisers' checker, final report ==="
grep -E "metrics \[" /out/chk/metrics.log | tail -2
