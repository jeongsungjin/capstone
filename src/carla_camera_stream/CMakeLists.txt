cmake_minimum_required(VERSION 3.0.2)
project(carla_camera_stream)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/spawn_cameras.py
  scripts/spawn_cameras_n.py
  scripts/spawn_single_camera.py
  scripts/spawn_traffic_vehicles.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

