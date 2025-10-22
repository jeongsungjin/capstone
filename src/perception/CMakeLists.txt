cmake_minimum_required(VERSION 3.0.2)
project(perception)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  roslib
  sensor_msgs
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Cuda
link_directories(/usr/local/cuda/lib64)
include_directories(/usr/local/cuda/include)

# TensorRT
link_directories(/usr/lib/x86_64-linux-gnu)
include_directories(/usr/include/x86_64-linux-gnu)

# TensorRT (환경변수 경로 우선)
if(DEFINED ENV{TENSORRT_ROOT})
  message(STATUS "Using TENSORRT_ROOT=$ENV{TENSORRT_ROOT}")
  include_directories($ENV{TENSORRT_ROOT}/include)
  link_directories($ENV{TENSORRT_ROOT}/lib)
endif()

# xtensor
link_directories(/home/ctrl/miniconda3/envs/perception/include)
include_directories(/home/ctrl/miniconda3/envs/perception/include)

find_package(BLAS REQUIRED)

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
  src/Model.cpp
)

target_link_libraries(perception 
  nvinfer
  nvinfer_plugin
  cudart
  ${BLAS_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_dependencies(perception ${catkin_EXPORTED_TARGETS})

