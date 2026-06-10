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

// YOLOv8 instance segmentation. Auto-detects the engine kind: raw outputs
// (detections [1, 4+nc+seg_channels, anchors] + prototypes) → decode + NMS on the host;
// YoloSegPostprocess plugin outputs (num_dets/bboxes/scores/labels/mask_coeffs + proto)
// → NMS already done, gather coeffs only. Mask assembly is shared by both.
class SegEngine: public Engine {
public:
    SegEngine(const std::string& engine_path, const InferConfig& config): Engine(engine_path, config)
    {
        class_names_ = config.labels_path.empty() ? coco_labels() : load_labels(config.labels_path);
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

    void draw(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) const override
    {
        draw_segments(image, res, objs, class_names_);
    }

private:
    // Index of an output binding by name, or -1.
    int binding_index(const std::string& name) const
    {
        for (size_t i = 0; i < output_bindings_.size(); ++i) {
            if (output_bindings_[i].name == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    // matmul kept coefficients with the prototypes, sigmoid, crop the padded region,
    // resize to the original image and threshold. Shared by both postprocess paths.
    void assemble_masks(std::vector<Object>& objs, const cv::Mat& masks, const cv::Mat& protos)
    {
        if (masks.empty()) {
            return;
        }
        const int   seg_h = config_.seg_h, seg_w = config_.seg_w;
        const int   input_h = input_bindings_[0].dims.d[2];
        const int   input_w = input_bindings_[0].dims.d[3];
        const float dw = pparam_.dw, dh = pparam_.dh;
        const float width = pparam_.width, height = pparam_.height;

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

    // Plugin engine: NMS + coeff gather done in-engine; only rescale + assemble masks.
    void postprocess_plugin(std::vector<Object>& objs)
    {
        const int    seg_channels = config_.seg_channels;
        const int*   num_dets     = static_cast<const int*>(host_ptrs_[binding_index("num_dets")]);
        const float* boxes        = static_cast<const float*>(host_ptrs_[binding_index("bboxes")]);
        const float* scores       = static_cast<const float*>(host_ptrs_[binding_index("scores")]);
        const int*   labels       = static_cast<const int*>(host_ptrs_[binding_index("labels")]);
        const float* coeffs       = static_cast<const float*>(host_ptrs_[binding_index("mask_coeffs")]);
        cv::Mat      protos(seg_channels, config_.seg_h * config_.seg_w, CV_32F, host_ptrs_[binding_index("proto")]);
        const float  dw = pparam_.dw, dh = pparam_.dh;
        const float  width = pparam_.width, height = pparam_.height, ratio = pparam_.ratio;

        cv::Mat masks;
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
            masks.push_back(cv::Mat(1, seg_channels, CV_32F, const_cast<float*>(coeffs + i * seg_channels)));
        }
        assemble_masks(objs, masks, protos);
    }

    // Raw engine: detections [1, 4+nc+seg_channels, anchors] + prototypes; decode + NMS.
    void postprocess_raw(std::vector<Object>& objs)
    {
        const int seg_channels = config_.seg_channels, seg_h = config_.seg_h, seg_w = config_.seg_w;

        // Find the 3-D detection output; the other output holds the prototypes.
        int det_idx      = -1;
        int num_channels = 0, num_anchors = 0;
        for (size_t i = 0; i < output_bindings_.size(); ++i) {
            if (output_bindings_[i].dims.nbDims == 3) {
                num_channels = output_bindings_[i].dims.d[1];
                num_anchors  = output_bindings_[i].dims.d[2];
                det_idx      = static_cast<int>(i);
            }
        }
        TRT_CHECK(det_idx >= 0);
        const int num_classes = num_channels - seg_channels - 4;

        const float dw = pparam_.dw, dh = pparam_.dh;
        const float width = pparam_.width, height = pparam_.height, ratio = pparam_.ratio;

        cv::Mat output(num_channels, num_anchors, CV_32F, host_ptrs_[det_idx]);
        output = output.t();
        cv::Mat protos(seg_channels, seg_h * seg_w, CV_32F, host_ptrs_[1 - det_idx]);

        std::vector<int>      labels;
        std::vector<float>    scores;
        std::vector<cv::Rect> bboxes;
        std::vector<cv::Mat>  mask_confs;
        std::vector<int>      indices;

        for (int i = 0; i < num_anchors; ++i) {
            auto        row_ptr   = output.row(i).ptr<float>();
            auto        bbox_ptr  = row_ptr;
            auto        score_ptr = row_ptr + 4;
            auto        mask_ptr  = row_ptr + 4 + num_classes;
            auto        max_ptr   = std::max_element(score_ptr, score_ptr + num_classes);
            const float score     = *max_ptr;
            if (score > config_.score_thres) {
                const float x  = *bbox_ptr++ - dw;
                const float y  = *bbox_ptr++ - dh;
                const float w  = *bbox_ptr++;
                const float h  = *bbox_ptr;
                const float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
                const float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
                const float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
                const float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

                bboxes.emplace_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
                labels.push_back(static_cast<int>(max_ptr - score_ptr));
                scores.push_back(score);
                mask_confs.push_back(cv::Mat(1, seg_channels, CV_32F, mask_ptr));
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
        assemble_masks(objs, masks, protos);
    }

    std::vector<std::string> class_names_;
};

int main(int argc, char** argv)
{
    try {
        const CliArgs args = parse_args(argc, argv);
        SegEngine     engine(args.engine, args.config);
        engine.make_pipe();
        return run(engine, args);
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
