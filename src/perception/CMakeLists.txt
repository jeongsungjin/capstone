cmake_minimum_required(VERSION 3.0.2)
project(perception)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  capstone_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs capstone_msgs
)

catkin_install_python(PROGRAMS
  scripts/imu_uplink_receiver.py
  scripts/inference_receiver.py
  scripts/heading_fuser.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

