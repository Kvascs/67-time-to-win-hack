# Reproducible build + run environment for the jury (ROS 2 Humble, Ubuntu 22.04).
#   docker build -t tram_backup_odometry .
#   docker run --rm -it --cpus=2 --memory=512m -v <dataset bags>:/bags:ro -v <out>:/out tram_backup_odometry \
#          ros2 run tram_backup_odometry check_run.sh 30618_2f104a1d 120
FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy python3-psutil \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY ros2_ws/src ./src
RUN . /opt/ros/humble/setup.sh && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON && \
    ./build/tram_backup_odometry/core/tbo_core_tests

# Every `docker run <command>` sees the workspace, not only interactive shells: the entrypoint of the
# base image sources ROS only, so `ros2 run tram_backup_odometry ...` would not find the packages.
RUN printf '#!/bin/bash\nset -e\nsource /opt/ros/humble/setup.bash --\nsource /ws/install/setup.bash --\nexec "$@"\n' \
        > /ros_entrypoint.sh && chmod +x /ros_entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash' >> /root/.bashrc
CMD ["bash"]
