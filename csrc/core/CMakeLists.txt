add_library(yolov8_core STATIC
    src/engine.cpp
    src/logger.cpp
    src/preprocess.cpp
    src/labels.cpp
    src/config.cpp
    src/runner.cpp
    src/draw.cpp
)

target_include_directories(yolov8_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolov8_core PUBLIC
    CUDA::cudart
    ${OpenCV_LIBS}
    TensorRT::TensorRT
)

target_compile_features(yolov8_core PUBLIC cxx_std_17)
target_compile_options(yolov8_core PRIVATE -Wall -Wextra)

# TRT_10 / BATCHED_NMS are PUBLIC so task executables compile the same code paths.
yolov8_apply_compile_defs(yolov8_core)
