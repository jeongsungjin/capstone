cmake_minimum_required(VERSION 3.0.2)
project(carla_camera_stream)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  geometry_msgs
  nav_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/spawn_cameras.py
  scripts/spawn_traffic_vehicles.py
  scripts/click_coordinate.py
  scripts/spawn_cameras_n.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(FILES
  config/positions.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

