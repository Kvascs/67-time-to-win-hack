cmake_minimum_required(VERSION 3.8)
project(tram_backup_odometry)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(tram_vehicle_msgs REQUIRED)
find_package(tbo_msgs REQUIRED)

# ROS-independent estimator core (same code as the offline replay and unit tests).
set(TBO_BUILD_TESTS ${BUILD_TESTING} CACHE BOOL "" FORCE)
add_subdirectory(core)

add_executable(tbo_node src/tbo_node.cpp)
target_link_libraries(tbo_node tbo_core)
ament_target_dependencies(tbo_node
  ament_index_cpp rclcpp nav_msgs sensor_msgs diagnostic_msgs tram_vehicle_msgs tbo_msgs)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tbo_node PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS tbo_node tbo_replay DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config maps DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  install(TARGETS tbo_core_tests DESTINATION lib/${PROJECT_NAME})
endif()

ament_package()
