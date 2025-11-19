cmake_minimum_required(VERSION 3.8)
project(ip_camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

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

add_executable(ipcamera_node
  src/ip_camera_node.cpp
  src/ip_camera_streamer.cpp
)

target_include_directories(ipcamera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(ipcamera_node
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
)

# IP Camera Node
target_link_libraries(ipcamera_node 
  ${OpenCV_LIBRARIES}
)

install(TARGETS ipcamera_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY 
  include
  include/ip_camera
  DESTINATION include
)
###############################################################


# IP Camera Component
add_library(ipcamera_component SHARED
  src/ip_camera_streamer.cpp
)

ament_target_dependencies(ipcamera_component
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
)

# include dirs for component
target_include_directories(ipcamera_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# link OpenCV for component
target_link_libraries(ipcamera_component 
  ${OpenCV_LIBRARIES}
)

rclcpp_components_register_nodes(ipcamera_component
  PLUGIN "IPCameraStreamer"
)

install(TARGETS
  ipcamera_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  include/ip_camera
  DESTINATION include
)

# 컴포넌트가 launch 파일 등에서 사용될 수 있도록 export (패키지 의존성)
ament_export_dependencies(rclcpp rclcpp_components sensor_msgs cv_bridge OpenCV)
###############################################################

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME} 
)

ament_export_include_directories("include")

ament_package()