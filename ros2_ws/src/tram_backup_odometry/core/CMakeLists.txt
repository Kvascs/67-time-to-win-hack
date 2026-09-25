# ROS-independent estimator core. Builds standalone (any CMake + C++17 compiler)
# for offline replay and unit tests, and is also added as a subdirectory by the
# ROS 2 package so the node and the offline tools run exactly the same code.
cmake_minimum_required(VERSION 3.14)
project(tbo_core CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(tbo_core STATIC
  src/geo.cpp
)
target_include_directories(tbo_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(tbo_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(TBO_BUILD_TESTS "Build core unit tests" ON)
if(TBO_BUILD_TESTS)
  add_executable(tbo_core_tests
    tests/test_main.cpp
    tests/test_geo.cpp
  )
  target_link_libraries(tbo_core_tests PRIVATE tbo_core)
  enable_testing()
  add_test(NAME tbo_core_tests COMMAND tbo_core_tests)
endif()
