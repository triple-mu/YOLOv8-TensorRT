#include "yolov8/preprocess.hpp"
#include <algorithm>
#include <cmath>

namespace yolov8 {

// Fills an NCHW float blob from a (already sized) HWC BGR image: swap to RGB and scale by 1/255.
static void to_blob(const cv::Mat& src, cv::Mat& out)
{
    const int h = src.rows;
    const int w = src.cols;
    out.create({1, 3, h, w}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    cv::Mat r(h, w, CV_32F, out.ptr<float>(0));
    cv::Mat g(h, w, CV_32F, out.ptr<float>(0) + static_cast<size_t>(h) * w);
    cv::Mat b(h, w, CV_32F, out.ptr<float>(0) + static_cast<size_t>(h) * w * 2);

    channels[0].convertTo(b, CV_32F, 1 / 255.f);  // BGR -> plane order RGB
    channels[1].convertTo(g, CV_32F, 1 / 255.f);
    channels[2].convertTo(r, CV_32F, 1 / 255.f);
}

void letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size, PreParam& pparam)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    const float height = image.rows;
    const float width  = image.cols;

    const float r    = std::min(inp_h / height, inp_w / width);
    const int   padw = std::round(width * r);
    const int   padh = std::round(height * r);

    cv::Mat tmp;
    if (static_cast<int>(width) != padw || static_cast<int>(height) != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = (inp_w - padw) / 2.0f;
    float dh = (inp_h - padh) / 2.0f;

    const int top    = static_cast<int>(std::round(dh - 0.1f));
    const int bottom = static_cast<int>(std::round(dh + 0.1f));
    const int left   = static_cast<int>(std::round(dw - 0.1f));
    const int right  = static_cast<int>(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});
    to_blob(tmp, out);

    pparam.ratio  = 1 / r;
    pparam.dw     = dw;
    pparam.dh     = dh;
    pparam.height = height;
    pparam.width  = width;
}

void resize_blob(const cv::Mat& image, cv::Mat& out, const cv::Size& size, PreParam& pparam)
{
    cv::Mat tmp;
    cv::resize(image, tmp, size);
    to_blob(tmp, out);

    pparam.ratio  = 1.0f;
    pparam.dw     = 0.0f;
    pparam.dh     = 0.0f;
    pparam.height = image.rows;
    pparam.width  = image.cols;
}

}  // namespace yolov8
