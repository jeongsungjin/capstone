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
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/multi_vehicle_spawner.py
  scripts/networkx_path_planner.py
  scripts/multi_vehicle_controller.py
  scripts/bev_visualizer.py
  scripts/occupancy_grid_publisher.py
  scripts/key_selector.py
  scripts/rviz_goal_mux.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
