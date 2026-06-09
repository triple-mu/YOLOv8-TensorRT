#include "yolov8/logger.hpp"
#include <iostream>

namespace yolov8 {

Logger::Logger(Severity severity): reportable_severity_(severity) {}

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity > reportable_severity_) {
        return;
    }
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
    }
    std::cerr << msg << std::endl;
}

Logger& global_logger()
{
    static Logger logger{nvinfer1::ILogger::Severity::kERROR};
    return logger;
}

}  // namespace yolov8
