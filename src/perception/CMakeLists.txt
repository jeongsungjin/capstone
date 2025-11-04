cmake_minimum_required(VERSION 3.10)

# Avoid enabling CUDA as a CMake language to bypass the built-in CUDA compiler probe.
project(perception LANGUAGES CXX)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  roslib
  sensor_msgs
  message_filters
)

find_package(OpenCV 4.10 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "Using OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")

# CUDA (via deprecated FindCUDA module, still available in CMake 3.16)
find_package(CUDA REQUIRED)

# TensorRT
link_directories(/usr/lib/x86_64-linux-gnu)
include_directories(/usr/include/x86_64-linux-gnu)

# xtensor
link_directories(/home/ctrl/miniconda3/envs/xtensor/include)
include_directories(/home/ctrl/miniconda3/envs/xtensor/include)

find_package(BLAS REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES perception
  CATKIN_DEPENDS cv_bridge roscpp sensor_msgs message_filters
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# Build CUDA source as a separate static library using nvcc via FindCUDA
cuda_add_library(preprocess_cuda STATIC
  src/preprocess.cu
)

add_executable(perception
  src/perception.cpp
  src/perception_node.cpp
  src/timer.cpp
  src/timer_logger.cpp
  src/Model.cpp
)

target_link_libraries(perception 
  preprocess_cuda
  nvinfer
  nvinfer_plugin
  cudart
  ${BLAS_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
)

add_dependencies(perception ${catkin_EXPORTED_TARGETS})
