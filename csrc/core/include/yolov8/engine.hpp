#pragma once
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "yolov8/config.hpp"
#include "yolov8/profiler.hpp"
#include "yolov8/trt_raii.hpp"
#include "yolov8/types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace yolov8 {

// Base class holding all the version-agnostic plumbing: engine loading, binding
// enumeration, buffer management, preprocessing and inference. Task subclasses
// implement only postprocess() and draw().
class Engine {
public:
    Engine(const std::string& engine_path, const InferConfig& config);
    virtual ~Engine() = default;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    void make_pipe();                          // allocate device/host buffers + optional warmup
    void copy_from_mat(const cv::Mat& image);  // preprocess + host-to-device copy
    void infer();                              // enqueue + device-to-host copy + sync

    // Implemented per task.
    virtual void postprocess(std::vector<Object>& objs)                                          = 0;
    virtual void draw(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) const = 0;

    const InferConfig& config() const
    {
        return config_;
    }

    void print_profile() const;  // prints the per-layer report if --profile was set

protected:
    // Preprocess one image into an NCHW blob (CPU path). Default is letterbox; cls overrides it.
    virtual void preprocess(const cv::Mat& image, cv::Mat& blob);
    // Whether the GPU preprocess path should letterbox (true) or plain-resize (false, cls).
    virtual bool letterbox_preproc() const
    {
        return true;
    }

    InferConfig          config_;
    std::vector<Binding> input_bindings_;
    std::vector<Binding> output_bindings_;
    std::vector<void*>   host_ptrs_;  // host output buffers, one per output binding
    PreParam             pparam_;

private:
    void load_engine(const std::string& engine_path);

    // Declared so destruction runs context -> engine -> runtime -> stream -> buffers.
    std::vector<DeviceBuffer>                 device_buffers_;
    std::vector<HostPinnedBuffer>             host_buffers_;
    std::vector<void*>                        device_ptrs_;     // [inputs..., outputs...]
    DeviceBuffer                              raw_input_;       // uint8 device buffer for --gpu-preprocess
    HostPinnedBuffer                          raw_input_host_;  // pinned staging for a fast async H2D upload
    CudaStream                                stream_;
    TrtUniquePtr<nvinfer1::IRuntime>          runtime_;
    TrtUniquePtr<nvinfer1::ICudaEngine>       engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<Profiler>                 profiler_;  // non-null only when --profile
};

}  // namespace yolov8
