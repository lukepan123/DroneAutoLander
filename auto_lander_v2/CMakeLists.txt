cmake_minimum_required(VERSION 3.8)
project(auto_lander_v2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# ---- Add your C++ node ----
add_executable(controller_state_orchestrator
  src/controller_state_orchestrator.cpp
)

# ---- Link dependencies ----
ament_target_dependencies(controller_state_orchestrator
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  mavros_msgs
  tf2_ros
)

# ---- Install the executable so ros2 run / launch can find it ----
install(TARGETS
  controller_state_orchestrator
  DESTINATION lib/${PROJECT_NAME}
)

# ---- Optional testing / linting ----
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ---- Finish package ----
ament_package()