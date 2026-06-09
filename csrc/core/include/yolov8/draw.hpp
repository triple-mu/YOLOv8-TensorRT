#pragma once
#include "opencv2/opencv.hpp"
#include "yolov8/types.hpp"
#include <string>
#include <vector>

namespace yolov8 {

// Draws labelled boxes onto a copy of `image`. Shared by the detection-style tasks.
void draw_detections(const cv::Mat&                  image,
                     cv::Mat&                        res,
                     const std::vector<Object>&      objs,
                     const std::vector<std::string>& names);

// Draws labelled boxes plus translucent mask overlays. Shared by the segmentation tasks.
void draw_segments(const cv::Mat&                  image,
                   cv::Mat&                        res,
                   const std::vector<Object>&      objs,
                   const std::vector<std::string>& names);

}  // namespace yolov8
