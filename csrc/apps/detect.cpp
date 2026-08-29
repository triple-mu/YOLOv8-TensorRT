#include "opencv2/opencv.hpp"
#include "yolov8/draw.hpp"
#include "yolov8/engine.hpp"
#include "yolov8/labels.hpp"
#include "yolov8/runner.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace yolov8;

// YOLOv8 detection. Auto-detects the engine kind from its outputs: a single
// [1, 4+num_classes, num_anchors] tensor → raw decode + cv::dnn NMS; four tensors
// (num_dets, bboxes, scores, labels) → NMS already baked in (EfficientNMS_TRT or the
// YoloDetPostprocess plugin), so we only rescale. One binary handles all three.
class DetectEngine: public Engine {
public:
    DetectEngine(const std::string& engine_path, const InferConfig& config): Engine(engine_path, config)
    {
        class_names_ = config.labels_path.empty() ? coco_labels() : load_labels(config.labels_path);
    }

    void postprocess(std::vector<Object>& objs) override
    {
        objs.clear();
        if (output_bindings_.size() >= 4) {
            postprocess_end2end(objs);
        }
        else {
            postprocess_raw(objs);
        }
    }

    void draw(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) const override
    {
        draw_detections(image, res, objs, class_names_);
    }

private:
    // NMS baked into the engine: outputs are num_dets, bboxes, scores, labels.
    void postprocess_end2end(std::vector<Object>& objs)
    {
        const int*   num_dets = static_cast<const int*>(host_ptrs_[0]);
        const float* boxes    = static_cast<const float*>(host_ptrs_[1]);
        const float* scores   = static_cast<const float*>(host_ptrs_[2]);
        const int*   labels   = static_cast<const int*>(host_ptrs_[3]);
        const float  dw = pparam_.dw, dh = pparam_.dh;
        const float  width = pparam_.width, height = pparam_.height, ratio = pparam_.ratio;

        for (int i = 0; i < num_dets[0]; ++i) {
            const float* ptr = boxes + i * 4;
            const float  x0  = clamp((ptr[0] - dw) * ratio, 0.f, width);
            const float  y0  = clamp((ptr[1] - dh) * ratio, 0.f, height);
            const float  x1  = clamp((ptr[2] - dw) * ratio, 0.f, width);
            const float  y1  = clamp((ptr[3] - dh) * ratio, 0.f, height);

            Object obj;
            obj.rect  = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
            obj.prob  = scores[i];
            obj.label = labels[i];
            objs.push_back(obj);
        }
    }

    // Raw head output [1, 4+num_classes, num_anchors]: decode + NMS on the host.
    void postprocess_raw(std::vector<Object>& objs)
    {
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
