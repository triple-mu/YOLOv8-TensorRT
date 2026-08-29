// GPU preprocessing: turn a raw uint8 HWC BGR image straight into the NCHW float
// blob the network expects (RGB, /255), with optional letterbox padding. This is
// the opt-in counterpart to the CPU path in preprocess.cpp; the host-side scalar
// math (ratio/dw/dh) is kept identical so pparam_ — and thus postprocess box
// rescaling — is unchanged. Bilinear sampling is not bit-identical to cv::resize,
// so results match within tolerance, not byte-for-byte.

#include "yolov8/preprocess.hpp"
#include <cmath>
#include <cuda_runtime.h>

namespace yolov8 {

namespace {

// One thread per destination pixel. Outside the resized ROI we write the 114 pad
// value; inside we bilinearly sample the source (cv::resize INTER_LINEAR mapping).
__global__ void blob_kernel(const unsigned char* src,
                            int                  src_w,
                            int                  src_h,
                            float*               dst,
                            int                  dst_w,
                            int                  dst_h,
                            int                  roi_x,
                            int                  roi_y,
                            int                  roi_w,
                            int                  roi_h,
                            float                scale_x,
                            float                scale_y)
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) {
        return;
    }

    const int plane = dst_w * dst_h;
    const int idx   = dy * dst_w + dx;
    const int rx    = dx - roi_x;
    const int ry    = dy - roi_y;

    if (rx < 0 || rx >= roi_w || ry < 0 || ry >= roi_h) {
        const float pad      = 114.0f / 255.0f;
        dst[0 * plane + idx] = pad;
        dst[1 * plane + idx] = pad;
        dst[2 * plane + idx] = pad;
        return;
    }

    float sx = (rx + 0.5f) * scale_x - 0.5f;
    float sy = (ry + 0.5f) * scale_y - 0.5f;
    sx       = sx < 0.0f ? 0.0f : (sx > src_w - 1 ? src_w - 1 : sx);
    sy       = sy < 0.0f ? 0.0f : (sy > src_h - 1 ? src_h - 1 : sy);

    const int   x0  = static_cast<int>(sx);
    const int   y0  = static_cast<int>(sy);
    const int   x1  = x0 + 1 < src_w ? x0 + 1 : x0;
    const int   y1  = y0 + 1 < src_h ? y0 + 1 : y0;
    const float ax  = sx - x0;
    const float ay  = sy - y0;
    const float w00 = (1.0f - ax) * (1.0f - ay);
    const float w01 = ax * (1.0f - ay);
    const float w10 = (1.0f - ax) * ay;
    const float w11 = ax * ay;

    const unsigned char* p00 = src + (y0 * src_w + x0) * 3;
    const unsigned char* p01 = src + (y0 * src_w + x1) * 3;
    const unsigned char* p10 = src + (y1 * src_w + x0) * 3;
    const unsigned char* p11 = src + (y1 * src_w + x1) * 3;

    // Source is BGR (channel c: 0=B,1=G,2=R); output plane order is RGB, so
    // channel c lands in plane (2 - c) — matching to_blob() in preprocess.cpp.
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float v              = w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
        dst[(2 - c) * plane + idx] = v / 255.0f;
    }
}

void launch(const unsigned char* src,
            int                  src_w,
            int                  src_h,
            float*               dst,
            int                  dst_w,
            int                  dst_h,
            int                  roi_x,
            int                  roi_y,
            int                  roi_w,
            int                  roi_h,
            float                scale_x,
            float                scale_y,
            cudaStream_t         stream)
{
    const dim3 block(16, 16);
    const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    blob_kernel<<<grid, block, 0, stream>>>(
        src, src_w, src_h, dst, dst_w, dst_h, roi_x, roi_y, roi_w, roi_h, scale_x, scale_y);
}

}  // namespace

void letterbox_cuda(
    const unsigned char* src, int src_w, int src_h, float* dst, int dst_w, int dst_h, PreParam& pparam, void* stream)
{
    // Identical scalar math to preprocess.cpp::letterbox so pparam stays the same.
    const float r    = std::min(static_cast<float>(dst_h) / src_h, static_cast<float>(dst_w) / src_w);
    const int   padw = static_cast<int>(std::round(src_w * r));
    const int   padh = static_cast<int>(std::round(src_h * r));
    const float dw   = (dst_w - padw) / 2.0f;
    const float dh   = (dst_h - padh) / 2.0f;
    const int   left = static_cast<int>(std::round(dw - 0.1f));
    const int   top  = static_cast<int>(std::round(dh - 0.1f));

    launch(src,
           src_w,
           src_h,
           dst,
           dst_w,
           dst_h,
           left,
           top,
           padw,
           padh,
           static_cast<float>(src_w) / padw,
           static_cast<float>(src_h) / padh,
           static_cast<cudaStream_t>(stream));

    pparam.ratio  = 1.0f / r;
    pparam.dw     = dw;
    pparam.dh     = dh;
    pparam.height = src_h;
    pparam.width  = src_w;
}

void resize_cuda(
    const unsigned char* src, int src_w, int src_h, float* dst, int dst_w, int dst_h, PreParam& pparam, void* stream)
{
    launch(src,
           src_w,
           src_h,
           dst,
           dst_w,
           dst_h,
           0,
           0,
           dst_w,
           dst_h,
           static_cast<float>(src_w) / dst_w,
           static_cast<float>(src_h) / dst_h,
           static_cast<cudaStream_t>(stream));

    pparam.ratio  = 1.0f;
    pparam.dw     = 0.0f;
    pparam.dh     = 0.0f;
    pparam.height = src_h;
    pparam.width  = src_w;
}

}  // namespace yolov8
