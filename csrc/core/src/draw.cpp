#include "yolov8/draw.hpp"
#include "yolov8/labels.hpp"
#include <cstdio>
#include <string>

namespace yolov8 {

void draw_detections(const cv::Mat&                  image,
                     cv::Mat&                        res,
                     const std::vector<Object>&      objs,
                     const std::vector<std::string>& names)
{
    const Palette& colors = palette();
    res                   = image.clone();
    for (const auto& obj : objs) {
        const Color& c = colors[obj.label % colors.size()];
        cv::rectangle(res, obj.rect, cv::Scalar(c[0], c[1], c[2]), 2);

        const std::string name =
            obj.label < static_cast<int>(names.size()) ? names[obj.label] : std::to_string(obj.label);
        char text[256];
        std::snprintf(text, sizeof(text), "%s %.1f%%", name.c_str(), obj.prob * 100);

        int      base_line  = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &base_line);

        const int x = static_cast<int>(obj.rect.x);
        int       y = static_cast<int>(obj.rect.y) + 1;
        if (y > res.rows) {
            y = res.rows;
        }
        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + base_line), {0, 0, 255}, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}

void draw_segments(const cv::Mat&                  image,
                   cv::Mat&                        res,
                   const std::vector<Object>&      objs,
                   const std::vector<std::string>& names)
{
    const Palette& colors      = palette();
    const Palette& mask_colors = mask_palette();
    res                        = image.clone();
    cv::Mat mask               = image.clone();
    for (const auto& obj : objs) {
        const Color& c  = colors[obj.label % colors.size()];
        const Color& mc = mask_colors[obj.label % mask_colors.size()];
        cv::rectangle(res, obj.rect, cv::Scalar(c[0], c[1], c[2]), 2);
        if (!obj.boxMask.empty()) {
            mask(obj.rect).setTo(cv::Scalar(mc[0], mc[1], mc[2]), obj.boxMask);
        }

        const std::string name =
            obj.label < static_cast<int>(names.size()) ? names[obj.label] : std::to_string(obj.label);
        char text[256];
        std::snprintf(text, sizeof(text), "%s %.1f%%", name.c_str(), obj.prob * 100);

        int       base_line  = 0;
        cv::Size  label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &base_line);
        const int x          = static_cast<int>(obj.rect.x);
        int       y          = static_cast<int>(obj.rect.y) + 1;
        if (y > res.rows) {
            y = res.rows;
        }
        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + base_line), {0, 0, 255}, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
    cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
}

}  // namespace yolov8
