cmake_minimum_required(VERSION 3.10)
project(TensorRT_Plan_Load)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# TensorRT
include_directories(/usr/include/x86_64-linux-gnu)
link_directories(/usr/lib/x86_64-linux-gnu)

# xtensor
find_package(xtensor REQUIRED)

# 소스 파일
set(SOURCES
    main.cpp
)

# Executable
add_executable(test
    test.cpp
)

# Link libraries
target_link_libraries(test
    nvinfer
    nvinfer_plugin
    cudart
    xtensor
    ${OpenCV_LIBS}
)

add_definitions(-O3 -pthread)
