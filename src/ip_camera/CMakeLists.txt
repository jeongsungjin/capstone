cmake_minimum_required(VERSION 3.0.2)
project(ip_camera)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  nodelet
  pluginlib
)

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs cv_bridge nodelet pluginlib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/ip_camera_streamer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(ip_camera_streamer_node
  src/ip_camera_node.cpp
  src/ip_camera_streamer.cpp
)

target_link_libraries(ip_camera_streamer_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_dependencies(ip_camera_streamer_node ${catkin_EXPORTED_TARGETS})

# Nodelet library
add_library(ip_camera_streamer_nodelet SHARED
  src/ip_camera_streamer_nodelet.cpp
  src/ip_camera_streamer.cpp
)

target_link_libraries(ip_camera_streamer_nodelet
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_dependencies(ip_camera_streamer_nodelet ${catkin_EXPORTED_TARGETS})

# Install nodelet plugin description
install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
