#pragma once
// RAII ownership for TensorRT objects and CUDA resources. Replaces the old raw
// pointers + manual delete/free, so cleanup is automatic and exception-safe.

#include "NvInfer.h"
#include "yolov8/check.hpp"
#include "yolov8/trt_compat.hpp"
#include <cuda_runtime_api.h>
#include <memory>
#include <utility>

namespace yolov8 {

// Deleter that routes through compat::trt_destroy (delete on TRT 10, destroy() on TRT 8).
struct TrtDeleter {
    template<typename T>
    void operator()(T* ptr) const
    {
        if (ptr) {
            compat::trt_destroy(ptr);
        }
    }
};

template<typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

// Owning, move-only wrapper for a CUDA stream.
class CudaStream {
public:
    CudaStream()
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    ~CudaStream()
    {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept: stream_(other.stream_)
    {
        other.stream_ = nullptr;
    }
    CudaStream& operator=(CudaStream&& other) noexcept
    {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_       = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const
    {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

// Owning, move-only byte buffer parameterised by its (de)allocation policy.
template<typename AllocPolicy, typename FreePolicy>
class GenericBuffer {
public:
    GenericBuffer() = default;
    explicit GenericBuffer(size_t bytes)
    {
        if (bytes > 0) {
            CUDA_CHECK(AllocPolicy{}(&ptr_, bytes));
            bytes_ = bytes;
        }
    }
    ~GenericBuffer()
    {
        FreePolicy{}(ptr_);
    }
    GenericBuffer(const GenericBuffer&)            = delete;
    GenericBuffer& operator=(const GenericBuffer&) = delete;
    GenericBuffer(GenericBuffer&& other) noexcept: ptr_(other.ptr_), bytes_(other.bytes_)
    {
        other.ptr_   = nullptr;
        other.bytes_ = 0;
    }
    GenericBuffer& operator=(GenericBuffer&& other) noexcept
    {
        if (this != &other) {
            FreePolicy{}(ptr_);
            ptr_         = other.ptr_;
            bytes_       = other.bytes_;
            other.ptr_   = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    void* data() const
    {
        return ptr_;
    }
    size_t bytes() const
    {
        return bytes_;
    }

private:
    void*  ptr_   = nullptr;
    size_t bytes_ = 0;
};

struct DeviceAlloc {
    cudaError_t operator()(void** ptr, size_t bytes) const
    {
        return cudaMalloc(ptr, bytes);
    }
};
struct DeviceFree {
    void operator()(void* ptr) const
    {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};
struct HostAlloc {
    cudaError_t operator()(void** ptr, size_t bytes) const
    {
        return cudaHostAlloc(ptr, bytes, cudaHostAllocDefault);
    }
};
struct HostFree {
    void operator()(void* ptr) const
    {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
};

using DeviceBuffer     = GenericBuffer<DeviceAlloc, DeviceFree>;
using HostPinnedBuffer = GenericBuffer<HostAlloc, HostFree>;

}  // namespace yolov8
