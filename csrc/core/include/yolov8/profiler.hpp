#pragma once
// Per-layer inference profiler. Adapted from NVIDIA TensorRT samples
// (samples/common/common.h SimpleProfiler, Apache-2.0). See ACKNOWLEDGMENTS.md.

#include "NvInfer.h"
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace yolov8 {

// Accumulates layer timings across all enqueue() calls; print() emits a sorted table.
class Profiler: public nvinfer1::IProfiler {
public:
    void reportLayerTime(const char* layer_name, float ms) noexcept override;
    void print(std::ostream& os) const;
    bool empty() const
    {
        return order_.empty();
    }

private:
    struct Record {
        float time  = 0.0f;
        int   count = 0;
    };
    std::map<std::string, Record> records_;
    std::vector<std::string>      order_;  // first-seen order
};

}  // namespace yolov8
