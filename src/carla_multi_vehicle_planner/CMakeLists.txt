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
  scripts/multi_vehicle_controller.py
  scripts/bev_visualizer.py
  scripts/occupancy_grid_publisher.py
  scripts/inference_recv.py
  scripts/key_selector.py
  scripts/rviz_goal_mux.py
  scripts/rviz_person_spawner.py
  scripts/udp_ackermann_sender.py
  scripts/path_relay.py
  scripts/platoon_manager.py
  scripts/bev_id_teleporter.py
  scripts/rviz_click_teleporter.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
