// Self-contained unit tests for the core pure helpers (no gtest dependency).
// Returns non-zero on the first failure; wired into ctest by csrc/tests/CMakeLists.txt.
#include "opencv2/opencv.hpp"
#include "yolov8/labels.hpp"
#include "yolov8/preprocess.hpp"
#include "yolov8/types.hpp"
#include <cstdio>
#include <string>

static int failures = 0;

#define CHECK_TRUE(cond)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                                                \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)

int main(int argc, char** argv)
{
    using namespace yolov8;

    // clamp
    CHECK_TRUE(clamp(5.f, 0.f, 10.f) == 5.f);
    CHECK_TRUE(clamp(-1.f, 0.f, 10.f) == 0.f);
    CHECK_TRUE(clamp(20.f, 0.f, 10.f) == 10.f);

    // volume / dtype_size
    nvinfer1::Dims dims{};
    dims.nbDims = 4;
    dims.d[0] = 1, dims.d[1] = 3, dims.d[2] = 8, dims.d[3] = 8;
    CHECK_TRUE(volume(dims) == 192);
    CHECK_TRUE(dtype_size(nvinfer1::DataType::kFLOAT) == 4);
    CHECK_TRUE(dtype_size(nvinfer1::DataType::kHALF) == 2);

    // letterbox: 480x640 BGR -> NCHW blob 1x3x640x640, normalized, with vertical padding
    cv::Mat  image(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat  blob;
    PreParam pp;
    letterbox(image, blob, cv::Size(640, 640), pp);
    CHECK_TRUE(blob.size[0] == 1 && blob.size[1] == 3 && blob.size[2] == 640 && blob.size[3] == 640);
    CHECK_TRUE(std::abs(pp.ratio - 1.0f) < 1e-6f);
    CHECK_TRUE(std::abs(pp.dh - 80.0f) < 1.0f && std::abs(pp.dw) < 1e-6f);

    // load_labels: COCO file passed as argv[1]
    if (argc > 1) {
        auto names = load_labels(argv[1]);
        CHECK_TRUE(names.size() == 80);
        CHECK_TRUE(names.front() == "person");
    }

    if (failures == 0) {
        std::printf("all core tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
