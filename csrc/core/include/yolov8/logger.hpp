#pragma once
#include "NvInfer.h"

namespace yolov8 {

// Minimal nvinfer1::ILogger that prints messages at or above a threshold.
class Logger: public nvinfer1::ILogger {
public:
    explicit Logger(Severity severity = Severity::kWARNING);
    void log(Severity severity, const char* msg) noexcept override;
    void set_severity(Severity severity)
    {
        reportable_severity_ = severity;
    }

private:
    Severity reportable_severity_;
};

// Shared logger instance used by the runtime and the plugin registry.
Logger& global_logger();

}  // namespace yolov8
