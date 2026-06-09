add_executable(yolov8_core_tests test_core.cpp)
target_link_libraries(yolov8_core_tests PRIVATE yolov8_core)
add_test(
    NAME core_tests
    COMMAND yolov8_core_tests ${CMAKE_SOURCE_DIR}/data/labels/coco.txt
)
