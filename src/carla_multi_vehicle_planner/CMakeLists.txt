cmake_minimum_required(VERSION 3.0.2)
project(carla_multi_vehicle_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  ackermann_msgs
  capstone_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/multi_vehicle_spawner.py
  scripts/simple_multi_vehicle_controller.py
  scripts/simple_multi_agent_planner.py
  scripts/simple_udp_ackermann_sender.py
  scripts/simple_bev_teleporter.py
  scripts/key_selector.py
  scripts/rviz_goal_mux.py
  scripts/path_relay.py
  scripts/rviz_click_teleporter.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
