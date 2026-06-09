cmake_minimum_required(VERSION 3.1)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(nvdsinfer_custom_bbox_yoloV8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wall -Werror -shared -fPIC")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")


# TensorRT — override with -DTensorRT_INCLUDE_DIRS=... -DTensorRT_LIBRARIES=...
set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu CACHE PATH "TensorRT headers")
set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu CACHE PATH "TensorRT libraries")

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")


# DeepStream SDK root — override with -DDEEPSTREAM=/opt/nvidia/deepstream/deepstream-X.Y
set(DEEPSTREAM /opt/nvidia/deepstream/deepstream CACHE PATH "DeepStream SDK root")
set(DS_LIBRARIES ${DEEPSTREAM}/lib)
set(DS_INCLUDE_DIRS ${DEEPSTREAM}/sources/includes)
message(STATUS "DS Libs: \n${DS_LIBRARIES}\n")
message(STATUS "DS Headers: \n${DS_INCLUDE_DIRS}\n")



list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${DS_INCLUDE_DIRS}
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_LIB_DIR}
        ${TensorRT_LIBRARIES}
        ${DS_LIBRARIES}
        )

include_directories(${INCLUDE_DIRS})

add_library(
        ${PROJECT_NAME}
        SHARED
        custom_bbox_parser/nvdsparsebbox_yoloV8.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin)
