// YoloObbPostprocess — oriented-bounding-box postprocess plugin. Like pose it takes the
// single transposed raw head tensor [B, A, C] (C = 4 + nc + 1: xc,yc,w,h | nc scores |
// angle in radians) and does argmax + NMS + top-k in-engine, but NMS uses *rotated* IoU
// (rotated-rect corners -> Sutherland-Hodgman polygon intersection -> shoelace area).
// Outputs center boxes + angle so the consumer can rebuild the rotated rect.

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

// ---- rotated IoU -----------------------------------------------------------

struct Pt {
    float x, y;
};

__device__ inline float cross_z(const Pt& o, const Pt& a, const Pt& b)
{
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

// Intersection of infinite lines AB and PQ (segments are guaranteed to cross when called).
__device__ inline Pt line_intersect(const Pt& a, const Pt& b, const Pt& p, const Pt& q)
{
    const float a1 = b.y - a.y, b1 = a.x - b.x, c1 = a1 * a.x + b1 * a.y;
    const float a2 = q.y - p.y, b2 = p.x - q.x, c2 = a2 * p.x + b2 * p.y;
    const float det = a1 * b2 - a2 * b1;
    if (fabsf(det) < 1e-9f) {
        return b;
    }
    return {(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det};
}

// Four corners of a rotated rect (cx,cy,w,h, angle in radians).
__device__ inline void rect_corners(const float* box, float ang, Pt p[4])
{
    const float c = cosf(ang), s = sinf(ang);
    const float dx = 0.5f * box[2], dy = 0.5f * box[3];
    const float rx[4] = {-dx, dx, dx, -dx};
    const float ry[4] = {-dy, -dy, dy, dy};
    for (int i = 0; i < 4; ++i) {
        p[i].x = box[0] + rx[i] * c - ry[i] * s;
        p[i].y = box[1] + rx[i] * s + ry[i] * c;
    }
}

__device__ inline float signed_area(const Pt* p, int n)
{
    float a = 0.f;
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        a += p[i].x * p[j].y - p[j].x * p[i].y;
    }
    return 0.5f * a;
}

__device__ inline void ensure_ccw(Pt p[4])
{
    if (signed_area(p, 4) < 0.f) {
        const Pt t = p[1];
        p[1]       = p[3];
        p[3]       = t;
    }
}

// Sutherland-Hodgman: clip the (CCW) subject polygon by the (CCW) convex clip polygon.
__device__ inline int clip_poly(const Pt* subj, int ns, const Pt* clip, int nc, Pt* out)
{
    Pt  buf_a[16], buf_b[16];
    Pt* cur = buf_a;
    int n   = ns;
    for (int i = 0; i < ns; ++i) {
        cur[i] = subj[i];
    }
    for (int e = 0; e < nc; ++e) {
        const Pt A   = clip[e];
        const Pt B   = clip[(e + 1) % nc];
        Pt*      nxt = (cur == buf_a) ? buf_b : buf_a;
        int      m   = 0;
        for (int k = 0; k < n && m < 15; ++k) {
            const Pt    P   = cur[k];
            const Pt    Q   = cur[(k + 1) % n];
            const float dP  = cross_z(A, B, P);
            const float dQ  = cross_z(A, B, Q);
            const bool  inP = dP >= 0.f;
            const bool  inQ = dQ >= 0.f;
            if (inQ) {
                if (!inP) {
                    nxt[m++] = line_intersect(A, B, P, Q);
                }
                nxt[m++] = Q;
            }
            else if (inP) {
                nxt[m++] = line_intersect(A, B, P, Q);
            }
        }
        cur = nxt;
        n   = m;
        if (n == 0) {
            break;
        }
    }
    for (int i = 0; i < n; ++i) {
        out[i] = cur[i];
    }
    return n;
}

__device__ inline float iou_rotated(const float* boxA, float angA, const float* boxB, float angB)
{
    Pt ca[4], cb[4];
    rect_corners(boxA, angA, ca);
    rect_corners(boxB, angB, cb);
    ensure_ccw(ca);
    ensure_ccw(cb);
    Pt          inter[16];
    const int   n   = clip_poly(ca, 4, cb, 4, inter);
    const float ai  = (n < 3) ? 0.f : fabsf(signed_area(inter, n));
    const float aA  = boxA[2] * boxA[3];
    const float aB  = boxB[2] * boxB[3];
    const float uni = aA + aB - ai;
    return uni > 0.f ? ai / uni : 0.f;
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
            if (out_classes[k] == cl && iou_rotated(box, ang, out_boxes + k * 4, out_angles[k]) > iou_thr) {
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
