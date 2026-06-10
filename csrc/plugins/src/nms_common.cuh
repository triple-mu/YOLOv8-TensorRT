// Shared device/host helpers for the YOLOv8 postprocess plugins (det/seg/obb/pose):
// per-anchor class argmax, score sorting via cub (CUDA-graph/stream-capture safe —
// thrust is not), and axis-aligned IoU. Kernels are `static` so each translation unit
// in libyolov8_plugins.so gets its own copy without link clashes.
#pragma once

#include <cstddef>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

namespace yolov8_plugin {

__device__ inline float iou_xyxy(const float* a, const float* b)
{
    const float ix0    = fmaxf(a[0], b[0]);
    const float iy0    = fmaxf(a[1], b[1]);
    const float ix1    = fminf(a[2], b[2]);
    const float iy1    = fminf(a[3], b[3]);
    const float iw     = fmaxf(0.0f, ix1 - ix0);
    const float ih     = fmaxf(0.0f, iy1 - iy0);
    const float inter  = iw * ih;
    const float area_a = fmaxf(0.0f, a[2] - a[0]) * fmaxf(0.0f, a[3] - a[1]);
    const float area_b = fmaxf(0.0f, b[2] - b[0]) * fmaxf(0.0f, b[3] - b[1]);
    const float uni    = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// Per-anchor argmax over the class channels.
static __global__ void
argmax_kernel(const float* scores, int num_anchors, int num_classes, float* best_score, int* best_cls)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= num_anchors) {
        return;
    }
    const float* s    = scores + static_cast<size_t>(a) * num_classes;
    float        best = s[0];
    int          bi   = 0;
    for (int c = 1; c < num_classes; ++c) {
        if (s[c] > best) {
            best = s[c];
            bi   = c;
        }
    }
    best_score[a] = best;
    best_cls[a]   = bi;
}

static __global__ void iota_kernel(int* v, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] = i;
    }
}

// Scratch layout shared by the plugins: keys in/out (float) + idx in/out + class (int),
// all length A and 256-aligned, followed by cub's radix-sort temp storage.
inline size_t nms_arrays_bytes(int A)
{
    const size_t b = static_cast<size_t>(A) * (2 * sizeof(float) + 3 * sizeof(int));
    return (b + 255) & ~static_cast<size_t>(255);
}

inline size_t nms_cub_temp_bytes(int A)
{
    size_t t = 0;
    cub::DeviceRadixSort::SortPairsDescending<float, int>(
        nullptr, t, nullptr, nullptr, nullptr, nullptr, A, 0, sizeof(float) * 8, 0);
    return (t + 255) & ~static_cast<size_t>(255);
}

inline size_t nms_workspace_bytes(int A)
{
    return nms_arrays_bytes(A) + nms_cub_temp_bytes(A);
}

inline void sort_scores_desc(
    void* workspace, int A, float* keys_in, float* keys_out, int* idx_in, int* idx_out, cudaStream_t stream)
{
    void*  temp    = static_cast<char*>(workspace) + nms_arrays_bytes(A);
    size_t temp_sz = nms_cub_temp_bytes(A);
    cub::DeviceRadixSort::SortPairsDescending(
        temp, temp_sz, keys_in, keys_out, idx_in, idx_out, A, 0, sizeof(float) * 8, stream);
}

}  // namespace yolov8_plugin
