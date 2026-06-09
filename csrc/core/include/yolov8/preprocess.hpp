#pragma once
#include "opencv2/opencv.hpp"
#include "yolov8/types.hpp"

namespace yolov8 {

// Aspect-preserving resize with gray padding into an NCHW float blob (RGB, /255).
// Records the scale/pad in `pparam` so detections can be mapped back.
void letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size, PreParam& pparam);

// Plain resize (no padding) into an NCHW float blob (RGB, /255). Used for classification.
void resize_blob(const cv::Mat& image, cv::Mat& out, const cv::Size& size, PreParam& pparam);

}  // namespace yolov8
