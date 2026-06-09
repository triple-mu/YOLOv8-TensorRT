#include "opencv2/opencv.hpp"
#include "yolov8/draw.hpp"
#include "yolov8/engine.hpp"
#include "yolov8/labels.hpp"
#include "yolov8/runner.hpp"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace yolov8;

// Segmentation for the "seg-sim" ONNX layout: the detection output is already
// transposed to [1, anchors, 4+1+1+seg_channels] (x0 y0 x1 y1, score, label, mask).
class SegSimpleEngine: public Engine {
public:
    SegSimpleEngine(const std::string& engine_path, const InferConfig& config): Engine(engine_path, config)
    {
        class_names_ = config.labels_path.empty() ? coco_labels() : load_labels(config.labels_path);
    }

    void postprocess(std::vector<Object>& objs) override
    {
        objs.clear();
        const int seg_channels = config_.seg_channels, seg_h = config_.seg_h, seg_w = config_.seg_w;
        const int input_h      = input_bindings_[0].dims.d[2];
        const int input_w      = input_bindings_[0].dims.d[3];
        const int num_anchors  = output_bindings_[0].dims.d[1];
        const int num_channels = output_bindings_[0].dims.d[2];

        const float dw = pparam_.dw, dh = pparam_.dh;
        const float width = pparam_.width, height = pparam_.height, ratio = pparam_.ratio;

        float*  output = static_cast<float*>(host_ptrs_[0]);
        cv::Mat protos(seg_channels, seg_h * seg_w, CV_32F, host_ptrs_[1]);

        std::vector<int>      labels;
        std::vector<float>    scores;
        std::vector<cv::Rect> bboxes;
        std::vector<cv::Mat>  mask_confs;
        std::vector<int>      indices;

        for (int i = 0; i < num_anchors; ++i) {
            float*      ptr   = output + i * num_channels;
            const float score = *(ptr + 4);
            if (score > config_.score_thres) {
                float x0 = clamp((*ptr++ - dw) * ratio, 0.f, width);
                float y0 = clamp((*ptr++ - dh) * ratio, 0.f, height);
                float x1 = clamp((*ptr++ - dw) * ratio, 0.f, width);
                float y1 = clamp((*ptr++ - dh) * ratio, 0.f, height);

                const int label     = static_cast<int>(*(++ptr));
                cv::Mat   mask_conf = cv::Mat(1, seg_channels, CV_32F, ++ptr);
                bboxes.emplace_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
                labels.push_back(label);
                scores.push_back(score);
                mask_confs.push_back(mask_conf);
            }
        }

#ifdef BATCHED_NMS
        cv::dnn::NMSBoxesBatched(bboxes, scores, labels, config_.score_thres, config_.iou_thres, indices);
#else
        cv::dnn::NMSBoxes(bboxes, scores, config_.score_thres, config_.iou_thres, indices);
#endif

        cv::Mat masks;
        int     cnt = 0;
        for (int idx : indices) {
            if (cnt >= config_.topk) {
                break;
            }
            Object obj;
            obj.rect  = bboxes[idx];
            obj.prob  = scores[idx];
            obj.label = labels[idx];
            masks.push_back(mask_confs[idx]);
            objs.push_back(obj);
            ++cnt;
        }

        if (masks.empty()) {
            return;
        }
        cv::Mat matmul   = (masks * protos).t();
        cv::Mat mask_mat = matmul.reshape(static_cast<int>(objs.size()), {seg_h, seg_w});

        std::vector<cv::Mat> mask_channels;
        cv::split(mask_mat, mask_channels);
        const int      scale_dw = dw / input_w * seg_w;
        const int      scale_dh = dh / input_h * seg_h;
        const cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

        for (size_t i = 0; i < objs.size(); ++i) {
            cv::Mat dest, mask;
            cv::exp(-mask_channels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size(static_cast<int>(width), static_cast<int>(height)), cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > 0.5f;
        }
    }

    void draw(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) const override
    {
        draw_segments(image, res, objs, class_names_);
    }

private:
    std::vector<std::string> class_names_;
};

int main(int argc, char** argv)
{
    try {
        const CliArgs   args = parse_args(argc, argv);
        SegSimpleEngine engine(args.engine, args.config);
        engine.make_pipe();
        return run(engine, args);
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
