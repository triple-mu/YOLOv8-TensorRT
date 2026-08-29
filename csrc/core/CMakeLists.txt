add_library(yolov8_core STATIC
    src/engine.cpp
    src/logger.cpp
    src/preprocess.cpp
    src/labels.cpp
    src/config.cpp
    src/runner.cpp
    src/draw.cpp
    src/profiler.cpp
)

target_include_directories(yolov8_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Vendored ghc::filesystem (C++14 fallback) as a SYSTEM include so -Wall stays quiet on it.
target_include_directories(yolov8_core SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8/3rdparty
)

target_link_libraries(yolov8_core PUBLIC
    CUDA::cudart
    ${OpenCV_LIBS}
    TensorRT::TensorRT
    ${CMAKE_DL_LIBS}  # dlopen for custom plugin .so
)

# GPU preprocessing kernel (opt-in). Compiled into the core lib only when the root
# build found a CUDA compiler; otherwise the engine's --gpu-preprocess branch throws.
if(YOLOV8_GPU_PREPROCESS)
    target_sources(yolov8_core PRIVATE src/preprocess.cu)
    target_compile_definitions(yolov8_core PUBLIC YOLOV8_GPU_PREPROCESS)
    set_target_properties(yolov8_core PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Require at least C++14 (the ghc::filesystem floor); the project default is C++17
# (std::filesystem). Build with -DCMAKE_CXX_STANDARD=14 to exercise the ghc fallback.
target_compile_features(yolov8_core PUBLIC cxx_std_14)
# Warning flags only for C++ TUs (nvcc rejects -Wall/-Wextra passed directly).
target_compile_options(yolov8_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)

# TRT_10 / BATCHED_NMS are PUBLIC so task executables compile the same code paths.
yolov8_apply_compile_defs(yolov8_core)
