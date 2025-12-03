cmake_minimum_required(VERSION 3.0.2)
project(capstone_visualization)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/bev_visualizer.py
  scripts/occupancy_grid_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


