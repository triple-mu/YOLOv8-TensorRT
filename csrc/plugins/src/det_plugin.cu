// YoloDetPostprocess — a TensorRT plugin that fuses YOLOv8 detection
// score-threshold + NMS + top-k on the GPU. It consumes the already-decoded head
// outputs (corner boxes + per-class scores, exactly what models/common.py PostDetect
// feeds EfficientNMS_TRT) and emits the same four tensors as EfficientNMS_TRT, so it
// is a drop-in alternative. Built into libyolov8_plugins.so and registered via
// REGISTER_TENSORRT_PLUGIN. Implemented against IPluginV2DynamicExt, whose API is
// stable across TensorRT 8 / 10 / 11 (deprecated on 10+ but functional).

#include "NvInferPlugin.h"
#include "nms_common.cuh"
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace {

using namespace yolov8_plugin;  // iou_xyxy, argmax_kernel, iota_kernel, nms_* helpers

const char* kPluginName    = "YoloDetPostprocess";
const char* kPluginVersion = "1";

struct DetParams {
    float score_threshold  = 0.25f;
    float iou_threshold    = 0.65f;
    int   max_output       = 100;  // top-k
    int   background       = -1;   // unused (kept for EfficientNMS attribute parity)
    int   box_coding       = 0;    // 0 = corner [x1,y1,x2,y2] (the only mode we emit)
    int   score_activation = 0;    // scores already sigmoided upstream
};

// Parallel NMS suppression is shared (suppress_xyxy_kernel in nms_common.cuh).
// Part 2 (single thread, no IoU): gather survivors in score order up to
// top-k. Cheap O(candidates) — the expensive O(M^2) overlap work is done in parallel above.
__global__ void compact_kernel(const float* boxes,
                               const float* sorted_score,
                               const int*   sorted_idx,
                               const int*   cls,
                               const int*   flag,
                               int          num_anchors,
                               float        score_thr,
                               int          topk,
                               int*         num_dets,
                               float*       out_boxes,
                               float*       out_scores,
                               int*         out_classes)
{
    int keep = 0;
    for (int r = 0; r < num_anchors && keep < topk; ++r) {
        if (sorted_score[r] <= score_thr) {
            break;
        }
        if (!flag[r]) {
            continue;
        }
        const int    a          = sorted_idx[r];
        const float* bx         = boxes + static_cast<size_t>(a) * 4;
        out_boxes[keep * 4 + 0] = bx[0];
        out_boxes[keep * 4 + 1] = bx[1];
        out_boxes[keep * 4 + 2] = bx[2];
        out_boxes[keep * 4 + 3] = bx[3];
        out_scores[keep]        = sorted_score[r];
        out_classes[keep]       = cls[a];
        ++keep;
    }
    *num_dets = keep;
}

// ---- plugin ----------------------------------------------------------------

class YoloDetPostprocess: public nvinfer1::IPluginV2DynamicExt {
public:
    explicit YoloDetPostprocess(DetParams p): params_(p) {}
    YoloDetPostprocess(const void* data, size_t length)
    {
        const char* p = static_cast<const char*>(data);
        std::memcpy(&params_, p, sizeof(DetParams));
        (void)length;
    }
    YoloDetPostprocess(const YoloDetPostprocess&) = default;

    // IPluginV2DynamicExt
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        auto* q = new YoloDetPostprocess(*this);
        q->setPluginNamespace(namespace_.c_str());
        return q;
    }

    nvinfer1::DimsExprs getOutputDimensions(int32_t                    outputIndex,
                                            const nvinfer1::DimsExprs* inputs,
                                            int32_t                    nbInputs,
                                            nvinfer1::IExprBuilder&    expr) noexcept override
    {
        (void)nbInputs;
        nvinfer1::DimsExprs out;
        const auto*         batch = inputs[0].d[0];
        const auto*         topk  = expr.constant(params_.max_output);
        if (outputIndex == 0) {  // num_dets [B, 1]
            out.nbDims = 2;
            out.d[0]   = batch;
            out.d[1]   = expr.constant(1);
        }
        else if (outputIndex == 1) {  // boxes [B, topk, 4]
            out.nbDims = 3;
            out.d[0]   = batch;
            out.d[1]   = topk;
            out.d[2]   = expr.constant(4);
        }
        else {  // scores [B, topk] / classes [B, topk]
            out.nbDims = 2;
            out.d[0]   = batch;
            out.d[1]   = topk;
        }
        return out;
    }

    bool supportsFormatCombination(int32_t                           pos,
                                   const nvinfer1::PluginTensorDesc* inOut,
                                   int32_t                           nbInputs,
                                   int32_t                           nbOutputs) noexcept override
    {
        (void)nbInputs;
        (void)nbOutputs;
        const auto& d = inOut[pos];
        if (d.format != nvinfer1::TensorFormat::kLINEAR) {
            return false;
        }
        // inputs 0,1 float; outputs: 0 int32 (num_dets), 1 float (boxes), 2 float (scores), 3 int32 (classes)
        if (pos == 2 || pos == 5) {
            return d.type == nvinfer1::DataType::kINT32;
        }
        return d.type == nvinfer1::DataType::kFLOAT;
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,
                         int32_t,
                         const nvinfer1::DynamicPluginTensorDesc*,
                         int32_t) noexcept override
    {
    }

    // Workspace: 5 int/float arrays of length A (keys in/out, idx in/out, class) plus
    // cub's radix-sort temp storage. cub with pre-allocated temp is CUDA-graph/stream-
    // capture safe (no host sync, no implicit allocation) — thrust's sort is not.
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int32_t                           nbInputs,
                            const nvinfer1::PluginTensorDesc*,
                            int32_t) const noexcept override
    {
        (void)nbInputs;
        return nms_workspace_bytes(inputs[1].dims.d[1]);
    }

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc*,
                    const void* const* inputs,
                    void* const*       outputs,
                    void*              workspace,
                    cudaStream_t       stream) noexcept override
    {
        const int B  = inputDesc[0].dims.d[0];
        const int A  = inputDesc[1].dims.d[1];
        const int nc = inputDesc[1].dims.d[2];

        const float* boxes       = static_cast<const float*>(inputs[0]);
        const float* scores      = static_cast<const float*>(inputs[1]);
        int*         num_dets    = static_cast<int*>(outputs[0]);
        float*       out_boxes   = static_cast<float*>(outputs[1]);
        float*       out_scores  = static_cast<float*>(outputs[2]);
        int*         out_classes = static_cast<int*>(outputs[3]);

        char*  ws       = static_cast<char*>(workspace);
        float* keys_in  = reinterpret_cast<float*>(ws);
        float* keys_out = keys_in + A;
        int*   idx_in   = reinterpret_cast<int*>(keys_out + A);
        int*   idx_out  = idx_in + A;
        int*   cls_buf  = idx_out + A;
        int*   flag     = cls_buf + A;

        const int t     = params_.max_output;
        const int block = 256;
        const int grid  = (A + block - 1) / block;
        for (int b = 0; b < B; ++b) {
            const float* bx = boxes + static_cast<size_t>(b) * A * 4;
            argmax_kernel<<<grid, block, 0, stream>>>(
                scores + static_cast<size_t>(b) * A * nc, A, nc, keys_in, cls_buf);
            iota_kernel<<<grid, block, 0, stream>>>(idx_in, A);
            sort_scores_desc(workspace, A, keys_in, keys_out, idx_in, idx_out, stream);
            suppress_xyxy_kernel<<<grid, block, 0, stream>>>(
                bx, keys_out, idx_out, cls_buf, A, params_.score_threshold, params_.iou_threshold, flag);
            compact_kernel<<<1, 1, 0, stream>>>(bx,
                                                keys_out,
                                                idx_out,
                                                cls_buf,
                                                flag,
                                                A,
                                                params_.score_threshold,
                                                t,
                                                num_dets + b,
                                                out_boxes + static_cast<size_t>(b) * t * 4,
                                                out_scores + static_cast<size_t>(b) * t,
                                                out_classes + static_cast<size_t>(b) * t);
        }
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType*, int32_t) const noexcept override
    {
        return (index == 0 || index == 3) ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kFLOAT;
    }

    // IPluginV2
    const char* getPluginType() const noexcept override
    {
        return kPluginName;
    }
    const char* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }
    int32_t getNbOutputs() const noexcept override
    {
        return 4;
    }
    int32_t initialize() noexcept override
    {
        return 0;
    }
    void   terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override
    {
        return sizeof(DetParams);
    }
    void serialize(void* buffer) const noexcept override
    {
        std::memcpy(buffer, &params_, sizeof(DetParams));
    }
    void destroy() noexcept override
    {
        delete this;
    }
    void setPluginNamespace(const char* ns) noexcept override
    {
        namespace_ = ns ? ns : "";
    }
    const char* getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }

private:
    DetParams   params_;
    std::string namespace_;
};

class YoloDetPostprocessCreator: public nvinfer1::IPluginCreator {
public:
    YoloDetPostprocessCreator()
    {
        fields_ = {
            {"score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
            {"iou_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
            {"max_output_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"background_class", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"box_coding", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"score_activation", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
        };
        fc_.nbFields = static_cast<int>(fields_.size());
        fc_.fields   = fields_.data();
    }

    const char* getPluginName() const noexcept override
    {
        return kPluginName;
    }
    const char* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override
    {
        return &fc_;
    }

    nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection* fc) noexcept override
    {
        DetParams p;
        for (int i = 0; i < fc->nbFields; ++i) {
            const auto&       f    = fc->fields[i];
            const std::string name = f.name;
            if (name == "score_threshold") {
                p.score_threshold = *static_cast<const float*>(f.data);
            }
            else if (name == "iou_threshold") {
                p.iou_threshold = *static_cast<const float*>(f.data);
            }
            else if (name == "max_output_boxes") {
                p.max_output = *static_cast<const int*>(f.data);
            }
            else if (name == "background_class") {
                p.background = *static_cast<const int*>(f.data);
            }
            else if (name == "box_coding") {
                p.box_coding = *static_cast<const int*>(f.data);
            }
            else if (name == "score_activation") {
                p.score_activation = *static_cast<const int*>(f.data);
            }
        }
        auto* plugin = new YoloDetPostprocess(p);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(const char*, const void* data, size_t length) noexcept override
    {
        auto* plugin = new YoloDetPostprocess(data, length);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    void setPluginNamespace(const char* ns) noexcept override
    {
        namespace_ = ns ? ns : "";
    }
    const char* getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }

private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection    fc_{};
    std::string                        namespace_;
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(YoloDetPostprocessCreator);
