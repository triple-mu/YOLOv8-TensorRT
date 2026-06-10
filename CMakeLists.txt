cmake_minimum_required(VERSION 3.18)
project(yolov8_trt LANGUAGES CXX)

# Default to C++17 (std::filesystem). Override with -DCMAKE_CXX_STANDARD=14 to
# build against the vendored ghc::filesystem fallback for older toolchains.
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
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

# Optional CUDA toolchain for the GPU preprocessing kernel (opt-in --gpu-preprocess).
# If nvcc is available we enable the CUDA language and compile preprocess.cu into the
# core lib; otherwise the GPU path compiles to a stub that throws (CPU path unaffected).
include(CheckLanguage)
if(NOT CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # Set before enable_language(CUDA), which would otherwise default it to "52".
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)  # Turing..Hopper; override with -D for others
    endif()
    enable_language(CUDA)
    set(YOLOV8_GPU_PREPROCESS ON)
    message(STATUS "GPU preprocessing: enabled (CUDA ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "GPU preprocessing: disabled (no CUDA compiler; --gpu-preprocess will be unavailable)")
endif()
print_var(TensorRT_VERSION_STRING)
print_var(OpenCV_VERSION)

# TensorRT < 10 plugins (libnvinfer_plugin) link against cuDNN 8; TensorRT 10+
# dropped that hard dependency. Locate cuDNN so older TensorRT links cleanly.
# Hint via -DCUDNN_ROOT=<dir> (e.g. a conda env that provides libcudnn.so.8).
if(TensorRT_VERSION_MAJOR LESS 10)
    find_library(CUDNN_LIBRARY NAMES cudnn
        HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib lib64)
    if(CUDNN_LIBRARY)
        get_filename_component(_cudnn_dir ${CUDNN_LIBRARY} DIRECTORY)
        message(STATUS "cuDNN for TensorRT ${TensorRT_VERSION_STRING}: ${CUDNN_LIBRARY}")
        link_directories(${_cudnn_dir})
    else()
        message(WARNING "TensorRT ${TensorRT_VERSION_STRING} needs cuDNN 8 but it was not found; "
                        "linking relies on --allow-shlib-undefined and runtime needs libcudnn.so.8 "
                        "on LD_LIBRARY_PATH (pass -DCUDNN_ROOT=<dir> to silence this).")
    endif()
endif()

# Be resilient to transitive shared-lib dependencies of the prebuilt TensorRT
# libraries that are not on the dev link path (e.g. cuDNN for TensorRT 8). They
# are resolved at runtime via LD_LIBRARY_PATH.
add_link_options(-Wl,--allow-shlib-undefined)

# Shared library used by every task.
add_subdirectory(csrc/core)

# Thin per-task executables (one source file each in csrc/apps).
add_subdirectory(csrc/apps)

# Core unit tests (self-contained, no gtest dependency): -DBUILD_TESTS=ON + ctest.
option(BUILD_TESTS "Build C++ core unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(csrc/tests)
endif()

# The task executables above are platform-agnostic: to deploy on Jetson, build
# them natively on the device (or cross-compile) with -DTensorRT_ROOT pointing at
# the aarch64 TensorRT. No separate Jetson sources are needed.

# Custom TensorRT postprocess plugins (libyolov8_plugins.so) — opt-in, needs CUDA.
option(BUILD_PLUGINS "Build the custom TensorRT postprocess plugins" OFF)
if(BUILD_PLUGINS)
    if(NOT YOLOV8_GPU_PREPROCESS)
        message(FATAL_ERROR "BUILD_PLUGINS requires a CUDA compiler (none found at configure time)")
    endif()
    add_subdirectory(csrc/plugins)
endif()

# DeepStream parser plugin — needs the DeepStream SDK, so it is off by default.
option(BUILD_DEEPSTREAM "Build the DeepStream parser plugin" OFF)
if(BUILD_DEEPSTREAM)
    add_subdirectory(csrc/deepstream)
endif()
