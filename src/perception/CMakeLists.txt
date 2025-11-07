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
  nodelet
  pluginlib
  capstone_msgs
)

find_package(OpenCV 4.10 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "Using OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")

# CUDA (via deprecated FindCUDA module, still available in CMake 3.16)
find_package(CUDA REQUIRED)

# Ensure CUDA objects are compiled with PIC so they can be linked into shared libs (nodelets)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-fPIC")

# TensorRT
link_directories(/usr/lib/x86_64-linux-gnu)
include_directories(/usr/include/x86_64-linux-gnu)

# xtensor
link_directories(/home/ctrl/miniconda3/envs/xtensor/include)
include_directories(/home/ctrl/miniconda3/envs/xtensor/include)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES perception
  CATKIN_DEPENDS cv_bridge roscpp sensor_msgs message_filters nodelet pluginlib
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
# Also mark target as position independent (belt-and-suspenders for various CMake/FindCUDA versions)
set_target_properties(preprocess_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(perception
  src/perception.cpp
  src/perception_node.cpp
  src/timer.cpp
  src/timer_logger.cpp
  src/Model.cpp
  include/nvidia_helper/logger.cpp
)

target_link_libraries(perception 
  preprocess_cuda
  nvinfer
  nvinfer_plugin
  cudart
  # ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  -lblas
  -lcblas
)

add_dependencies(perception ${catkin_EXPORTED_TARGETS})

# Nodelet library (reuse sources except main)
add_library(perception_nodelet SHARED
  src/perception_nodelet.cpp
  src/perception_node.cpp
  src/timer.cpp
  src/timer_logger.cpp
  src/Model.cpp
  include/nvidia_helper/logger.cpp
)

target_link_libraries(perception_nodelet 
  preprocess_cuda
  nvinfer
  nvinfer_plugin
  cudart
  # ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  -lblas
  -lcblas
)

add_dependencies(perception_nodelet ${catkin_EXPORTED_TARGETS})

install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
