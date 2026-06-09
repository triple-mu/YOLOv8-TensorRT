#include "opencv2/opencv.hpp"
#include "yolov8/engine.hpp"
#include "yolov8/labels.hpp"
#include "yolov8/runner.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace yolov8;

// YOLOv8 pose: single class with 17 keypoints. Output [1, 4+1+51, anchors].
class PoseEngine: public Engine {
public:
    using Engine::Engine;

    void postprocess(std::vector<Object>& objs) override
    {
        objs.clear();
        const int   num_anchors = output_bindings_[0].dims.d[2];
        const float dw = pparam_.dw, dh = pparam_.dh;
        const float width = pparam_.width, height = pparam_.height, ratio = pparam_.ratio;

        std::vector<cv::Rect>           bboxes;
        std::vector<float>              scores;
        std::vector<int>                labels;
        std::vector<int>                indices;
        std::vector<std::vector<float>> kpss;

        cv::Mat output(output_bindings_[0].dims.d[1], num_anchors, CV_32F, host_ptrs_[0]);
        output = output.t();
        for (int i = 0; i < num_anchors; ++i) {
            auto        row_ptr   = output.row(i).ptr<float>();
            auto        bbox_ptr  = row_ptr;
            auto        score_ptr = row_ptr + 4;
            auto        kps_ptr   = row_ptr + 5;
            const float score     = *score_ptr;
            if (score > config_.score_thres) {
                const float x  = *bbox_ptr++ - dw;
                const float y  = *bbox_ptr++ - dh;
                const float w  = *bbox_ptr++;
                const float h  = *bbox_ptr;
                const float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
                const float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
                const float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
                const float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

                std::vector<float> kps;
                for (int k = 0; k < 17; ++k) {
                    float kx = clamp((*(kps_ptr + 3 * k) - dw) * ratio, 0.f, width);
                    float ky = clamp((*(kps_ptr + 3 * k + 1) - dh) * ratio, 0.f, height);
                    kps.push_back(kx);
                    kps.push_back(ky);
                    kps.push_back(*(kps_ptr + 3 * k + 2));
                }

                bboxes.emplace_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
                labels.push_back(0);
                scores.push_back(score);
                kpss.push_back(std::move(kps));
            }
        }

#ifdef BATCHED_NMS
        cv::dnn::NMSBoxesBatched(bboxes, scores, labels, config_.score_thres, config_.iou_thres, indices);
#else
        cv::dnn::NMSBoxes(bboxes, scores, config_.score_thres, config_.iou_thres, indices);
#endif

        int cnt = 0;
        for (int idx : indices) {
            if (cnt >= config_.topk) {
                break;
            }
            Object obj;
            obj.rect  = bboxes[idx];
            obj.prob  = scores[idx];
            obj.label = labels[idx];
            obj.kps   = kpss[idx];
            objs.push_back(obj);
            ++cnt;
        }
    }

    void draw(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) const override
    {
        const Palette& kps_colors  = pose::kps_colors();
        const Palette& limb_colors = pose::limb_colors();
        const auto&    skeleton    = pose::skeleton();
        const int      num_point   = 17;
        res                        = image.clone();

        for (const auto& obj : objs) {
            cv::rectangle(res, obj.rect, {0, 0, 255}, 2);
            char text[256];
            std::snprintf(text, sizeof(text), "person %.1f%%", obj.prob * 100);
            int       base_line  = 0;
            cv::Size  label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &base_line);
            const int x          = static_cast<int>(obj.rect.x);
            int       y          = static_cast<int>(obj.rect.y) + 1;
            if (y > res.rows) {
                y = res.rows;
            }
            cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + base_line), {0, 0, 255}, -1);
            cv::putText(
                res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);

            const auto& kps = obj.kps;
            for (int k = 0; k < num_point + 2; ++k) {
                if (k < num_point) {
                    const int   kx = std::round(kps[k * 3]);
                    const int   ky = std::round(kps[k * 3 + 1]);
                    const float ks = kps[k * 3 + 2];
                    if (ks > 0.5f) {
                        const Color& c = kps_colors[k];
                        cv::circle(res, {kx, ky}, 5, cv::Scalar(c[0], c[1], c[2]), -1);
                    }
                }
                const auto& ske = skeleton[k];
                const int   p1x = std::round(kps[(ske[0] - 1) * 3]);
                const int   p1y = std::round(kps[(ske[0] - 1) * 3 + 1]);
                const int   p2x = std::round(kps[(ske[1] - 1) * 3]);
                const int   p2y = std::round(kps[(ske[1] - 1) * 3 + 1]);
                const float s1  = kps[(ske[0] - 1) * 3 + 2];
                const float s2  = kps[(ske[1] - 1) * 3 + 2];
                if (s1 > 0.5f && s2 > 0.5f) {
                    const Color& c = limb_colors[k];
                    cv::line(res, {p1x, p1y}, {p2x, p2y}, cv::Scalar(c[0], c[1], c[2]), 2);
                }
            }
        }
    }
};

int main(int argc, char** argv)
{
    try {
        const CliArgs args = parse_args(argc, argv);
        PoseEngine    engine(args.engine, args.config);
        engine.make_pipe();
        return run(engine, args);
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
