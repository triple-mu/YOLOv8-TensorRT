# Custom TensorRT postprocess plugins (libyolov8_plugins.so). Like the engine, the
# .so is tied to the TensorRT version it is built against — build it with the same
# -DTensorRT_ROOT you use for the engine. Off by default (-DBUILD_PLUGINS=ON); needs
# the CUDA language (enabled by the root build when nvcc is found).
add_library(yolov8_plugins SHARED
    src/det_plugin.cu
    src/seg_plugin.cu
    src/pose_plugin.cu
)

set_target_properties(yolov8_plugins PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_include_directories(yolov8_plugins PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(yolov8_plugins PRIVATE CUDA::cudart TensorRT::TensorRT)

# IPluginV2DynamicExt is deprecated on TRT 10/11 but still the single-source API that
# also works on TRT 8 — silence the deprecation noise for this target only.
target_compile_options(yolov8_plugins PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
    # 997: dynamic-API methods intentionally hide the legacy IPluginV2 overloads.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations;--diag-suppress=997>
)
