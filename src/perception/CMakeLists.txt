cmake_minimum_required(VERSION 3.8)
project(perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(message_filters REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV 4.10 REQUIRED)

find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-fPIC")

cuda_add_library(preprocess_cuda STATIC
  src/preprocess.cu
)
target_include_directories(preprocess_cuda PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
set_target_properties(preprocess_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

# TensorRT
link_directories(/usr/lib/x86_64-linux-gnu)
include_directories(/usr/include/x86_64-linux-gnu)

# xtensor
link_directories(/home/airkon2/anaconda3/envs/xtensor/include)
include_directories(/home/airkon2/anaconda3/envs/xtensor/include)

find_package(ament_index_cpp REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_library(perception_component SHARED
  src/perception_node.cpp
  src/model.cpp
  src/timer.cpp
  src/timer_logger.cpp
  include/nvidia_helper/logger.cpp
)

target_include_directories(perception_component 
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(perception_component
  rclcpp
  rclcpp_components
  sensor_msgs
  message_filters
  cv_bridge
  ament_index_cpp
)

target_link_libraries(perception_component
  preprocess_cuda
  nvinfer
  nvinfer_plugin
  cudart
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
)

rclcpp_components_register_nodes(perception_component
  PLUGIN "PerceptionNode"
)

install(TARGETS 
  perception_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY
  launch
  engine
  DESTINATION share/${PROJECT_NAME} 
)

ament_package()
