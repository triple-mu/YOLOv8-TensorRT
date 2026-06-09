cmake_minimum_required(VERSION 3.18)
project(yolov8_trt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# All executables land in build/bin.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(Function)
include(TrtDefs)

# Default to the TensorRT that matches the Python runtime (override with -DTensorRT_ROOT=...).
if(NOT TensorRT_ROOT AND NOT DEFINED ENV{TensorRT_ROOT} AND EXISTS /data/TensorRT-10.16.1.11)
    set(TensorRT_ROOT /data/TensorRT-10.16.1.11)
endif()

find_package(CUDAToolkit REQUIRED)  # provides CUDA::cudart (no CUDA language needed)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
print_var(TensorRT_VERSION_STRING)
print_var(OpenCV_VERSION)

# Shared library used by every task.
add_subdirectory(csrc/core)

# Task executables (added as each is migrated to the core library).
add_subdirectory(csrc/detect/normal)
add_subdirectory(csrc/detect/end2end)
add_subdirectory(csrc/segment/normal)
add_subdirectory(csrc/segment/simple)
add_subdirectory(csrc/pose/normal)
add_subdirectory(csrc/obb/normal)
add_subdirectory(csrc/cls/normal)

# The task executables above are platform-agnostic: to deploy on Jetson, build
# them natively on the device (or cross-compile) with -DTensorRT_ROOT pointing at
# the aarch64 TensorRT. No separate Jetson sources are needed.

# DeepStream parser plugin — needs the DeepStream SDK, so it is off by default.
option(BUILD_DEEPSTREAM "Build the DeepStream parser plugin" OFF)
if(BUILD_DEEPSTREAM)
    add_subdirectory(csrc/deepstream)
endif()
