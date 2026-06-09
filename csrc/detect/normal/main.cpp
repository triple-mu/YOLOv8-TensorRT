#include "yolov8/engine.hpp"
#include "yolov8/labels.hpp"
#include "yolov8/runner.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace yolov8;

// YOLOv8 detection: a single output [1, 4+num_classes, num_anchors].
class DetectEngine : public Engine {
public:
    DetectEngine(const std::string& engine_path, const InferConfig& config) : Engine(engine_path, config)
    {
        class_names_ = config.labels_path.empty() ? coco_labels() : load_labels(config.labels_path);
    }

    void postprocess(std::vector<Object>& objs) override
    {
        objs.clear();
        const int   num_channels = output_bindings_[0].dims.d[1];
        const int   num_anchors  = output_bindings_[0].dims.d[2];
        const int   num_labels   = num_channels - 4;
        const float dw = pparam_.dw, dh = pparam_.dh;
        const float width = pparam_.width, height = pparam_.height, ratio = pparam_.ratio;

        std::vector<cv::Rect> bboxes;
        std::vector<float>    scores;
        std::vector<int>      labels;
        std::vector<int>      indices;

        cv::Mat output(num_channels, num_anchors, CV_32F, host_ptrs_[0]);
        output = output.t();
        for (int i = 0; i < num_anchors; ++i) {
            auto        row_ptr   = output.row(i).ptr<float>();
            auto        bbox_ptr  = row_ptr;
            auto        score_ptr = row_ptr + 4;
            auto        max_ptr   = std::max_element(score_ptr, score_ptr + num_labels);
            const float score     = *max_ptr;
            if (score > config_.score_thres) {
                const float x = *bbox_ptr++ - dw;
                const float y = *bbox_ptr++ - dh;
                const float w = *bbox_ptr++;
                const float h = *bbox_ptr;

                const float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
                const float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
                const float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
                const float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

                bboxes.emplace_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
                labels.push_back(static_cast<int>(max_ptr - score_ptr));
                scores.push_back(score);
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
            objs.push_back(obj);
            ++cnt;
        }
    }

    void draw(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) const override
    {
        const Palette& colors = palette();
        res                   = image.clone();
        for (const auto& obj : objs) {
            const Color& c = colors[obj.label % colors.size()];
            cv::Scalar   color(c[0], c[1], c[2]);
            cv::rectangle(res, obj.rect, color, 2);

            const std::string name = obj.label < static_cast<int>(class_names_.size())
                                         ? class_names_[obj.label]
                                         : std::to_string(obj.label);
            char text[256];
            std::snprintf(text, sizeof(text), "%s %.1f%%", name.c_str(), obj.prob * 100);

            int      base_line = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &base_line);

            const int x = static_cast<int>(obj.rect.x);
            int       y = static_cast<int>(obj.rect.y) + 1;
            if (y > res.rows) {
                y = res.rows;
            }
            cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + base_line), {0, 0, 255}, -1);
            cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        {255, 255, 255}, 1);
        }
    }

private:
    std::vector<std::string> class_names_;
};

int main(int argc, char** argv)
{
    try {
        const CliArgs args = parse_args(argc, argv);
        DetectEngine  engine(args.engine, args.config);
        engine.make_pipe();
        return run(engine, args);
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
