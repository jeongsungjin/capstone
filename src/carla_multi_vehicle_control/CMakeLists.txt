cmake_minimum_required(VERSION 3.0.2)
project(carla_multi_vehicle_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  ackermann_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/simple_multi_vehicle_controller.py
  scripts/simple_udp_ackermann_sender.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


