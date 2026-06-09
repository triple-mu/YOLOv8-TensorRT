#pragma once
// Error checking that survives release builds. Replaces the old assert()/exit(1)
// pattern: CUDA_CHECK and TRT_CHECK both throw TrtException on failure.

#include <cuda_runtime_api.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace yolov8 {

struct TrtException: public std::runtime_error {
    using std::runtime_error::runtime_error;
};

namespace detail {

inline void cuda_check(cudaError_t code, const char* expr, const char* file, int line)
{
    if (code != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error " << static_cast<int>(code) << " (" << cudaGetErrorString(code) << ") at " << file << ":"
            << line << "\n    " << expr;
        throw TrtException(oss.str());
    }
}

inline void trt_check(bool ok, const char* expr, const char* file, int line)
{
    if (!ok) {
        std::ostringstream oss;
        oss << "TensorRT check failed at " << file << ":" << line << "\n    " << expr;
        throw TrtException(oss.str());
    }
}

}  // namespace detail
}  // namespace yolov8

#define CUDA_CHECK(call) ::yolov8::detail::cuda_check((call), #call, __FILE__, __LINE__)
#define TRT_CHECK(cond) ::yolov8::detail::trt_check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
