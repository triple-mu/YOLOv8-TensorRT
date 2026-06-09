# Centralised compile definitions for the version-dependent code paths.
# Apply once to the core library; task executables inherit them transitively.
#
#   TRT_10       -> TensorRT major version >= 10 (tensor-name API instead of binding-index API)
#   BATCHED_NMS  -> OpenCV >= 4.7.0 (cv::dnn::NMSBoxesBatched is available)

function(yolov8_apply_compile_defs target)
    if(TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
        message(STATUS "[${target}] TensorRT ${TensorRT_VERSION_STRING}: enabling -DTRT_10")
        target_compile_definitions(${target} PUBLIC TRT_10)
    endif()
    if(OpenCV_VERSION VERSION_GREATER_EQUAL 4.7.0)
        message(STATUS "[${target}] OpenCV ${OpenCV_VERSION}: enabling -DBATCHED_NMS")
        target_compile_definitions(${target} PUBLIC BATCHED_NMS)
    endif()
endfunction()
