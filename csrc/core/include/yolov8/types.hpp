#pragma once
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <vector>

namespace yolov8 {

struct Binding {
    size_t         size  = 1;  // number of elements
    size_t         dsize = 1;  // bytes per element
    nvinfer1::Dims dims{};
    std::string    name;
};

// Letterbox parameters, used to map detections back to the original image.
struct PreParam {
    float ratio  = 1.0f;
    float dw     = 0.0f;
    float dh     = 0.0f;
    float height = 0.0f;
    float width  = 0.0f;
};

// One detection. Task-specific fields are left empty/default when unused, which
// keeps a single value type usable across det/seg/pose/obb/cls without polymorphism.
struct Object {
    cv::Rect_<float>   rect;
    int                label = 0;
    float              prob  = 0.0f;
    std::vector<float> kps;      // pose keypoints (x, y, score) triplets
    cv::Mat            boxMask;  // segmentation mask cropped to rect
    cv::RotatedRect    rrect;    // oriented box (valid when has_rrect)
    bool               has_rrect = false;
};

inline size_t volume(const nvinfer1::Dims& dims)
{
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

inline size_t dtype_size(nvinfer1::DataType dtype)
{
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

inline float clamp(float val, float lo, float hi)
{
    return val < lo ? lo : (val > hi ? hi : val);
}

}  // namespace yolov8
