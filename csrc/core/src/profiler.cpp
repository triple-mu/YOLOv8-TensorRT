#include "yolov8/profiler.hpp"
#include <algorithm>
#include <iomanip>

namespace yolov8 {

void Profiler::reportLayerTime(const char* layer_name, float ms) noexcept
{
    auto it = records_.find(layer_name);
    if (it == records_.end()) {
        order_.emplace_back(layer_name);
        records_[layer_name] = Record{ms, 1};
    }
    else {
        it->second.time += ms;
        it->second.count += 1;
    }
}

void Profiler::print(std::ostream& os) const
{
    float  total = 0.0f;
    size_t width = std::string("TensorRT layer").size();
    for (const auto& name : order_) {
        total += records_.at(name).time;
        width = std::max(width, name.size());
    }
    if (total <= 0.0f) {
        return;
    }

    const auto flags     = os.flags();
    const auto precision = os.precision();
    os << "\n========== per-layer profile ==========\n";
    os << std::setfill(' ') << std::left << std::setw(static_cast<int>(width)) << "TensorRT layer" << "  " << std::right
       << std::setw(8) << "%" << std::setw(14) << "time(ms)" << std::setw(8) << "calls" << "\n";
    for (const auto& name : order_) {
        const Record& r = records_.at(name);
        os << std::left << std::setw(static_cast<int>(width)) << name << "  " << std::right << std::fixed
           << std::setprecision(1) << std::setw(7) << (r.time * 100.0f / total) << "%" << std::setprecision(4)
           << std::setw(14) << r.time << std::setw(8) << r.count << "\n";
    }
    os << "total: " << std::fixed << std::setprecision(4) << total << " ms\n";
    os.flags(flags);
    os.precision(precision);
}

}  // namespace yolov8
