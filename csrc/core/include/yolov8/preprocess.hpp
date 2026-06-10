#pragma once
#include "opencv2/opencv.hpp"
#include "yolov8/types.hpp"

namespace yolov8 {

// Aspect-preserving resize with gray padding into an NCHW float blob (RGB, /255).
// Records the scale/pad in `pparam` so detections can be mapped back.
void letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size, PreParam& pparam);

// Plain resize (no padding) into an NCHW float blob (RGB, /255). Used for classification.
void resize_blob(const cv::Mat& image, cv::Mat& out, const cv::Size& size, PreParam& pparam);

// GPU counterparts (defined in preprocess.cu, compiled only when CUDA is available).
// `src` is a device-side uint8 HWC BGR image; `dst` is the device NCHW float input
// buffer (RGB, /255) written directly. `stream` is a cudaStream_t passed as void* to
// keep this header free of CUDA headers. Records scale/pad in `pparam` exactly like
// the CPU versions above.
void letterbox_cuda(
    const unsigned char* src, int src_w, int src_h, float* dst, int dst_w, int dst_h, PreParam& pparam, void* stream);
void resize_cuda(
    const unsigned char* src, int src_w, int src_h, float* dst, int dst_w, int dst_h, PreParam& pparam, void* stream);

}  // namespace yolov8
