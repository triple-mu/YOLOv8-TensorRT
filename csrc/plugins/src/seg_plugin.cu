// YoloSegPostprocess — segmentation postprocess plugin. Same decode+NMS+top-k as
// YoloDetPostprocess, but it takes a third input (per-anchor mask coefficients) and
// emits a fifth output: the coefficients of the kept boxes. Mask assembly (proto
// matmul + sigmoid + crop + resize) stays outside the engine. Registered in
// libyolov8_plugins.so via REGISTER_TENSORRT_PLUGIN.

#include "NvInferPlugin.h"
#include "nms_common.cuh"
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace {

using namespace yolov8_plugin;  // iou_xyxy, argmax_kernel, iota_kernel, nms_* helpers

const char* kPluginName    = "YoloSegPostprocess";
const char* kPluginVersion = "1";

struct SegParams {
    float score_threshold  = 0.25f;
    float iou_threshold    = 0.65f;
    int   max_output       = 100;  // top-k
    int   background       = -1;
    int   box_coding       = 0;
    int   score_activation = 0;
};

// Greedy NMS over score-sorted anchors (class-aware), gathering the nm mask
// coefficients of each kept box.
__global__ void nms_seg_kernel(const float* boxes,
                               const float* coeffs,
                               int          nm,
                               const float* sorted_score,
                               const int*   sorted_idx,
                               const int*   cls,
                               int          num_anchors,
                               float        score_thr,
                               float        iou_thr,
                               int          topk,
                               int*         num_dets,
                               float*       out_boxes,
                               float*       out_scores,
                               int*         out_classes,
                               float*       out_coeffs)
{
    int keep = 0;
    for (int r = 0; r < num_anchors && keep < topk; ++r) {
        const float sc = sorted_score[r];
        if (sc <= score_thr) {
            break;
        }
        const int    a  = sorted_idx[r];
        const float* bx = boxes + static_cast<size_t>(a) * 4;
        const int    cl = cls[a];

        bool suppressed = false;
        for (int k = 0; k < keep; ++k) {
            if (out_classes[k] == cl && iou_xyxy(bx, out_boxes + k * 4) > iou_thr) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            out_boxes[keep * 4 + 0] = bx[0];
            out_boxes[keep * 4 + 1] = bx[1];
            out_boxes[keep * 4 + 2] = bx[2];
            out_boxes[keep * 4 + 3] = bx[3];
            out_scores[keep]        = sc;
            out_classes[keep]       = cl;
            const float* cf         = coeffs + static_cast<size_t>(a) * nm;
            float*       dst        = out_coeffs + static_cast<size_t>(keep) * nm;
            for (int j = 0; j < nm; ++j) {
                dst[j] = cf[j];
            }
            ++keep;
        }
    }
    *num_dets = keep;
}

class YoloSegPostprocess: public nvinfer1::IPluginV2DynamicExt {
public:
    explicit YoloSegPostprocess(SegParams p): params_(p) {}
    YoloSegPostprocess(const void* data, size_t length)
    {
        std::memcpy(&params_, data, sizeof(SegParams));
        (void)length;
    }
    YoloSegPostprocess(const YoloSegPostprocess&) = default;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        auto* q = new YoloSegPostprocess(*this);
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
        else if (outputIndex == 4) {  // mask_coeffs [B, topk, nm]
            out.nbDims = 3;
            out.d[0]   = batch;
            out.d[1]   = topk;
            out.d[2]   = inputs[2].d[2];
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
        // 3 float inputs; outputs: 3 num_dets(int32), 4 boxes, 5 scores, 6 classes(int32), 7 coeffs.
        if (pos == 3 || pos == 6) {
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
        const int nm = inputDesc[2].dims.d[2];

        const float* boxes       = static_cast<const float*>(inputs[0]);
        const float* scores      = static_cast<const float*>(inputs[1]);
        const float* coeffs      = static_cast<const float*>(inputs[2]);
        int*         num_dets    = static_cast<int*>(outputs[0]);
        float*       out_boxes   = static_cast<float*>(outputs[1]);
        float*       out_scores  = static_cast<float*>(outputs[2]);
        int*         out_classes = static_cast<int*>(outputs[3]);
        float*       out_coeffs  = static_cast<float*>(outputs[4]);

        char*  ws       = static_cast<char*>(workspace);
        float* keys_in  = reinterpret_cast<float*>(ws);
        float* keys_out = keys_in + A;
        int*   idx_in   = reinterpret_cast<int*>(keys_out + A);
        int*   idx_out  = idx_in + A;
        int*   cls_buf  = idx_out + A;

        const int t  = params_.max_output;
        const int bk = 256;
        const int gd = (A + bk - 1) / bk;
        for (int b = 0; b < B; ++b) {
            argmax_kernel<<<gd, bk, 0, stream>>>(scores + static_cast<size_t>(b) * A * nc, A, nc, keys_in, cls_buf);
            iota_kernel<<<gd, bk, 0, stream>>>(idx_in, A);
            sort_scores_desc(workspace, A, keys_in, keys_out, idx_in, idx_out, stream);
            nms_seg_kernel<<<1, 1, 0, stream>>>(boxes + static_cast<size_t>(b) * A * 4,
                                                coeffs + static_cast<size_t>(b) * A * nm,
                                                nm,
                                                keys_out,
                                                idx_out,
                                                cls_buf,
                                                A,
                                                params_.score_threshold,
                                                params_.iou_threshold,
                                                t,
                                                num_dets + b,
                                                out_boxes + static_cast<size_t>(b) * t * 4,
                                                out_scores + static_cast<size_t>(b) * t,
                                                out_classes + static_cast<size_t>(b) * t,
                                                out_coeffs + static_cast<size_t>(b) * t * nm);
        }
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType*, int32_t) const noexcept override
    {
        return (index == 0 || index == 3) ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kFLOAT;
    }

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
        return 5;
    }
    int32_t initialize() noexcept override
    {
        return 0;
    }
    void   terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override
    {
        return sizeof(SegParams);
    }
    void serialize(void* buffer) const noexcept override
    {
        std::memcpy(buffer, &params_, sizeof(SegParams));
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
    SegParams   params_;
    std::string namespace_;
};

class YoloSegPostprocessCreator: public nvinfer1::IPluginCreator {
public:
    YoloSegPostprocessCreator()
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
        SegParams p;
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
        auto* plugin = new YoloSegPostprocess(p);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(const char*, const void* data, size_t length) noexcept override
    {
        auto* plugin = new YoloSegPostprocess(data, length);
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

REGISTER_TENSORRT_PLUGIN(YoloSegPostprocessCreator);
