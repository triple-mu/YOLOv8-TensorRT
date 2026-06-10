#include "opencv2/opencv.hpp"
#include "yolov8/engine.hpp"
#include "yolov8/labels.hpp"
#include "yolov8/runner.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace yolov8;

// YOLOv8 oriented bounding boxes. Output [1, 4+nc+1, anchors] (the last channel is the angle).
class ObbEngine: public Engine {
public:
    ObbEngine(const std::string& engine_path, const InferConfig& config): Engine(engine_path, config)
    {
        if (!config.labels_path.empty()) {
            class_names_ = load_labels(config.labels_path);
        }
    }

    void postprocess(std::vector<Object>& objs) override
    {
        objs.clear();
        if (binding_index("num_dets") >= 0) {
            postprocess_plugin(objs);
        }
        else {
            postprocess_raw(objs);
        }
    }

private:
    int binding_index(const std::string& name) const
    {
        for (size_t i = 0; i < output_bindings_.size(); ++i) {
            if (output_bindings_[i].name == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    // YoloObbPostprocess plugin: rotated NMS in-engine; only rescale + build rrect.
    void postprocess_plugin(std::vector<Object>& objs)
    {
        const int*   num_dets = static_cast<const int*>(host_ptrs_[binding_index("num_dets")]);
        const float* boxes    = static_cast<const float*>(host_ptrs_[binding_index("bboxes")]);
        const float* scores   = static_cast<const float*>(host_ptrs_[binding_index("scores")]);
        const int*   labels   = static_cast<const int*>(host_ptrs_[binding_index("labels")]);
        const float* angles   = static_cast<const float*>(host_ptrs_[binding_index("angles")]);
        const float  dw = pparam_.dw, dh = pparam_.dh;
        const float  width = pparam_.width, height = pparam_.height, ratio = pparam_.ratio;

        for (int i = 0; i < num_dets[0]; ++i) {
            const float* bp = boxes + i * 4;
            const float  x  = clamp((bp[0] - dw) * ratio, 0.f, width);
            const float  y  = clamp((bp[1] - dh) * ratio, 0.f, height);
            const float  w  = clamp(bp[2] * ratio, 0.f, width);
            const float  h  = clamp(bp[3] * ratio, 0.f, height);
            if (w < 1.f || h < 1.f) {
                continue;
            }
            Object obj;
            obj.rrect     = cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(w, h), angles[i] / CV_PI * 180.f);
            obj.has_rrect = true;
            obj.prob      = scores[i];
            obj.label     = labels[i];
            objs.push_back(obj);
        }
    }

    void postprocess_raw(std::vector<Object>& objs)
    {
        const int   num_channels = output_bindings_[0].dims.d[1];
        const int   num_anchors  = output_bindings_[0].dims.d[2];
        const int   num_labels   = num_channels - 5;  // 4 box coords + nc + 1 angle
        const float dw = pparam_.dw, dh = pparam_.dh;
        const float width = pparam_.width, height = pparam_.height, ratio = pparam_.ratio;

        std::vector<cv::RotatedRect> bboxes;
        std::vector<float>           scores;
        std::vector<int>             labels;
        std::vector<int>             indices;

        cv::Mat output(num_channels, num_anchors, CV_32F, host_ptrs_[0]);
        output = output.t();
        for (int i = 0; i < num_anchors; ++i) {
            auto        row_ptr   = output.row(i).ptr<float>();
            auto        bbox_ptr  = row_ptr;
            auto        score_ptr = row_ptr + 4;
            auto        max_ptr   = std::max_element(score_ptr, score_ptr + num_labels);
            auto        angle_ptr = row_ptr + 4 + num_labels;
            const float score     = *max_ptr;
            if (score > config_.score_thres) {
                float x = (*bbox_ptr++ - dw) * ratio;
                float y = (*bbox_ptr++ - dh) * ratio;
                float w = (*bbox_ptr++) * ratio;
                float h = (*bbox_ptr) * ratio;
                if (w < 1.f || h < 1.f) {
                    continue;
                }
                x = clamp(x, 0.f, width);
                y = clamp(y, 0.f, height);
                w = clamp(w, 0.f, width);
                h = clamp(h, 0.f, height);

                const float     angle = *angle_ptr / CV_PI * 180.f;
                cv::RotatedRect bbox(cv::Point2f(x, y), cv::Size2f(w, h), angle);
                bboxes.push_back(bbox);
                labels.push_back(static_cast<int>(std::distance(score_ptr, max_ptr)));
                scores.push_back(score);
            }
        }

        cv::dnn::NMSBoxes(bboxes, scores, config_.score_thres, config_.iou_thres, indices);

        int cnt = 0;
        for (int idx : indices) {
            if (cnt >= config_.topk) {
                break;
            }
            Object obj;
            obj.rrect     = bboxes[idx];
            obj.has_rrect = true;
            obj.prob      = scores[idx];
            obj.label     = labels[idx];
            objs.push_back(obj);
            ++cnt;
        }
    }

public:
    void draw(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) const override
    {
        const Palette& colors = palette();
        res                   = image.clone();
        for (const auto& obj : objs) {
            cv::Mat points;
            cv::boxPoints(obj.rrect, points);
            points.convertTo(points, CV_32S);
            const Color& c = colors[obj.label % colors.size()];
            cv::polylines(res, points, true, cv::Scalar(c[0], c[1], c[2]), 2);

            const std::string name =
                obj.label < static_cast<int>(class_names_.size()) ? class_names_[obj.label] : std::to_string(obj.label);
            char text[256];
            std::snprintf(text, sizeof(text), "%s %.1f%%", name.c_str(), obj.prob * 100);
            int       base_line  = 0;
            cv::Size  label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &base_line);
            const int x          = static_cast<int>(obj.rrect.center.x);
            int       y          = static_cast<int>(obj.rrect.center.y) + 1;
            if (y > res.rows) {
                y = res.rows;
            }
            cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + base_line), {0, 0, 255}, -1);
            cv::putText(
                res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
        }
    }

private:
    std::vector<std::string> class_names_;
};

int main(int argc, char** argv)
{
    try {
        const CliArgs args = parse_args(argc, argv);
        ObbEngine     engine(args.engine, args.config);
        engine.make_pipe();
        return run(engine, args);
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
