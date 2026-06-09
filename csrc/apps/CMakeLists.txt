# One thin executable per task, all linking the shared core library.
# Each <source>:<target> pair maps a file in this directory to its binary name.
set(YOLOV8_APPS
    detect:yolov8_detect
    detect_e2e:yolov8_detect_e2e
    segment:yolov8_seg
    segment_simple:yolov8_seg_simple
    pose:yolov8_pose
    obb:yolov8_obb
    cls:yolov8_cls
)

foreach(app ${YOLOV8_APPS})
    string(REPLACE ":" ";" pair ${app})
    list(GET pair 0 src)
    list(GET pair 1 tgt)
    add_executable(${tgt} ${src}.cpp)
    target_link_libraries(${tgt} PRIVATE yolov8_core)
endforeach()
