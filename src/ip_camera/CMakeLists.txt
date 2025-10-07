cmake_minimum_required(VERSION 3.0.2)
project(ip_camera)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/ip_camera_streamer.py
  scripts/low_latency_rtsp_streamer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
