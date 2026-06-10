// YoloObbPostprocess — oriented-bounding-box postprocess plugin. Like pose it takes the
// single transposed raw head tensor [B, A, C] (C = 4 + nc + 1: xc,yc,w,h | nc scores |
// angle in radians) and does argmax + NMS + top-k in-engine, but NMS uses ProbIoU (the
// closed-form Gaussian overlap ultralytics itself uses for OBB), which is cheaper than
// polygon intersection. Outputs center boxes + angle so the consumer can rebuild the rrect.

#include "NvInferPlugin.h"
#include "nms_common.cuh"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace {

using namespace yolov8_plugin;  // iota_kernel, cub sort helpers

const char* kPluginName    = "YoloObbPostprocess";
const char* kPluginVersion = "1";

struct ObbParams {
    float score_threshold  = 0.25f;
    float iou_threshold    = 0.65f;
    int   max_output       = 100;
    int   background       = -1;
    int   box_coding       = 0;
    int   score_activation = 0;
};

// ---- rotated IoU (ProbIoU) -------------------------------------------------
// ultralytics computes OBB NMS overlap with ProbIoU (a closed-form probabilistic
// IoU from the boxes' Gaussian covariances), not polygon intersection. It is both
// cheaper (no clipping) and matches ultralytics' own postprocess. Verified to agree
// with ultralytics.utils.metrics.batch_probiou to 5 decimals.

__device__ inline void cov_matrix(float w, float h, float r, float& a, float& b, float& c)
{
    const float av = w * w / 12.0f, bv = h * h / 12.0f;
    const float cr = cosf(r), sr = sinf(r);
    a = av * cr * cr + bv * sr * sr;
    b = av * sr * sr + bv * cr * cr;
    c = (av - bv) * sr * cr;
}

// boxA/boxB = (cx,cy,w,h); angles in radians. Returns ProbIoU in [0,1].
__device__ inline float iou_probiou(const float* boxA, float rA, const float* boxB, float rB)
{
    const float eps = 1e-7f;
    float       a1, b1, c1, a2, b2, c2;
    cov_matrix(boxA[2], boxA[3], rA, a1, b1, c1);
    cov_matrix(boxB[2], boxB[3], rB, a2, b2, c2);
    const float cx1 = boxA[0], cy1 = boxA[1], cx2 = boxB[0], cy2 = boxB[1];
    const float denom = (a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps;

    const float t1 = ((a1 + a2) * powf(cy1 - cy2, 2) + (b1 + b2) * powf(cx1 - cx2, 2)) / denom;
    const float t2 = ((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / denom;
    const float t3 =
        logf(((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2))
                 / (4.0f * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps)
             + eps);
    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    bd       = fmaxf(fminf(bd, 100.0f), eps);
    return 1.0f - sqrtf(1.0f - expf(-bd) + eps);
}

// ---- kernels ---------------------------------------------------------------

// Per-anchor argmax over the nc class channels of a row-major [A, C] tensor (offset 4).
__global__ void obb_argmax_kernel(const float* data, int A, int C, int nc, float* best_score, int* best_cls)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= A) {
        return;
    }
    const float* s    = data + static_cast<size_t>(a) * C + 4;
    float        best = s[0];
    int          bi   = 0;
    for (int c = 1; c < nc; ++c) {
        if (s[c] > best) {
            best = s[c];
            bi   = c;
        }
    }
    best_score[a] = best;
    best_cls[a]   = bi;
}

// Greedy rotated NMS (class-aware). Keeps center box (cx,cy,w,h) + angle for each survivor.
__global__ void obb_nms_kernel(const float* data,
                               int          A,
                               int          C,
                               int          nc,
                               const float* sorted_score,
                               const int*   sorted_idx,
                               const int*   cls,
                               float        score_thr,
                               float        iou_thr,
                               int          topk,
                               int*         num_dets,
                               float*       out_boxes,
                               float*       out_scores,
                               int*         out_classes,
                               float*       out_angles)
{
    int keep = 0;
    for (int r = 0; r < A && keep < topk; ++r) {
        const float sc = sorted_score[r];
        if (sc <= score_thr) {
            break;
        }
        const int    a      = sorted_idx[r];
        const float* row    = data + static_cast<size_t>(a) * C;
        const float  box[4] = {row[0], row[1], row[2], row[3]};
        const float  ang    = row[4 + nc];
        const int    cl     = cls[a];

        bool suppressed = false;
        for (int k = 0; k < keep; ++k) {
            if (out_classes[k] == cl && iou_probiou(box, ang, out_boxes + k * 4, out_angles[k]) > iou_thr) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            out_boxes[keep * 4 + 0] = box[0];
            out_boxes[keep * 4 + 1] = box[1];
            out_boxes[keep * 4 + 2] = box[2];
            out_boxes[keep * 4 + 3] = box[3];
            out_scores[keep]        = sc;
            out_classes[keep]       = cl;
            out_angles[keep]        = ang;
            ++keep;
        }
    }
    *num_dets = keep;
}

// ---- plugin ----------------------------------------------------------------

class YoloObbPostprocess: public nvinfer1::IPluginV2DynamicExt {
public:
    explicit YoloObbPostprocess(ObbParams p): params_(p) {}
    YoloObbPostprocess(const void* data, size_t length)
    {
        std::memcpy(&params_, data, sizeof(ObbParams));
        (void)length;
    }
    YoloObbPostprocess(const YoloObbPostprocess&) = default;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        auto* q = new YoloObbPostprocess(*this);
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
        else if (outputIndex == 1) {  // boxes [B, topk, 4] (cx,cy,w,h)
            out.nbDims = 3;
            out.d[0]   = batch;
            out.d[1]   = topk;
            out.d[2]   = expr.constant(4);
        }
        else if (outputIndex == 4) {  // angles [B, topk, 1]
            out.nbDims = 3;
            out.d[0]   = batch;
            out.d[1]   = topk;
            out.d[2]   = expr.constant(1);
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
        // 1 float input (pos 0); outputs: 1 num_dets(int32), 2 boxes, 3 scores, 4 classes(int32), 5 angles.
        if (pos == 1 || pos == 4) {
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
        return nms_workspace_bytes(inputs[0].dims.d[1]);
    }

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc*,
                    const void* const* inputs,
                    void* const*       outputs,
                    void*              workspace,
                    cudaStream_t       stream) noexcept override
    {
        const int B  = inputDesc[0].dims.d[0];
        const int A  = inputDesc[0].dims.d[1];
        const int C  = inputDesc[0].dims.d[2];
        const int nc = C - 5;  // 4 box + nc + 1 angle

        int* num_dets = static_cast<int*>(outputs[0]);
        if (nc < 1) {
            cudaMemsetAsync(num_dets, 0, static_cast<size_t>(B) * sizeof(int), stream);
            return 0;
        }

        const float* data        = static_cast<const float*>(inputs[0]);
        float*       out_boxes   = static_cast<float*>(outputs[1]);
        float*       out_scores  = static_cast<float*>(outputs[2]);
        int*         out_classes = static_cast<int*>(outputs[3]);
        float*       out_angles  = static_cast<float*>(outputs[4]);

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
            const float* d = data + static_cast<size_t>(b) * A * C;
            obb_argmax_kernel<<<gd, bk, 0, stream>>>(d, A, C, nc, keys_in, cls_buf);
            iota_kernel<<<gd, bk, 0, stream>>>(idx_in, A);
            sort_scores_desc(workspace, A, keys_in, keys_out, idx_in, idx_out, stream);
            obb_nms_kernel<<<1, 1, 0, stream>>>(d,
                                                A,
                                                C,
                                                nc,
                                                keys_out,
                                                idx_out,
                                                cls_buf,
                                                params_.score_threshold,
                                                params_.iou_threshold,
                                                t,
                                                num_dets + b,
                                                out_boxes + static_cast<size_t>(b) * t * 4,
                                                out_scores + static_cast<size_t>(b) * t,
                                                out_classes + static_cast<size_t>(b) * t,
                                                out_angles + static_cast<size_t>(b) * t);
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
        return sizeof(ObbParams);
    }
    void serialize(void* buffer) const noexcept override
    {
        std::memcpy(buffer, &params_, sizeof(ObbParams));
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
    ObbParams   params_;
    std::string namespace_;
};

class YoloObbPostprocessCreator: public nvinfer1::IPluginCreator {
public:
    YoloObbPostprocessCreator()
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
        ObbParams p;
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
        auto* plugin = new YoloObbPostprocess(p);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(const char*, const void* data, size_t length) noexcept override
    {
        auto* plugin = new YoloObbPostprocess(data, length);
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

REGISTER_TENSORRT_PLUGIN(YoloObbPostprocessCreator);
