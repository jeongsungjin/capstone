cmake_minimum_required(VERSION 3.0.2)
project(perception)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Cuda
link_directories(/usr/local/cuda/lib64)
include_directories(/usr/local/cuda/include)

# TensorRT
link_directories(/usr/lib/x86_64-linux-gnu)
include_directories(/usr/lib/x86_64-linux-gnu)

# xtensor
link_directories(/usr/local/include)
include_directories(/usr/local/include)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES perception
#  CATKIN_DEPENDS cv_bridge roscpp sensor_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(perception
  src/perception.cpp
  src/perception_node.cpp
  src/TAFv25.cpp
)

target_link_libraries(perception 
  nvinfer
  nvinfer_plugin
  cudart
  xtensor
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_dependencies(perception ${catkin_EXPORTED_TARGETS})

