#include "yolov8/engine.hpp"
#include "NvInferPlugin.h"
#include "yolov8/check.hpp"
#include "yolov8/logger.hpp"
#include "yolov8/preprocess.hpp"
#include "yolov8/trt_compat.hpp"
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace yolov8 {

Engine::Engine(const std::string& engine_path, const InferConfig& config): config_(config)
{
    load_engine(engine_path);
}

void Engine::load_engine(const std::string& engine_path)
{
    std::ifstream file(engine_path, std::ios::binary);
    TRT_CHECK(file.good());
    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<size_t>(size));
    TRT_CHECK(file.read(blob.data(), size).good());
    file.close();

    // Load a custom plugin .so (if any) before plugin init / deserialize so its
    // creators are registered. Path from --plugin-lib or $YOLOV8_PLUGIN_LIB.
    const char* plugin_lib =
        !config_.plugin_lib.empty() ? config_.plugin_lib.c_str() : std::getenv("YOLOV8_PLUGIN_LIB");
    if (plugin_lib && *plugin_lib) {
        if (!dlopen(plugin_lib, RTLD_NOW | RTLD_GLOBAL)) {
            throw TrtException(std::string("failed to load plugin library '") + plugin_lib + "': " + dlerror());
        }
    }

    initLibNvInferPlugins(&global_logger(), "");
    runtime_.reset(nvinfer1::createInferRuntime(global_logger()));
    TRT_CHECK(runtime_ != nullptr);
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), static_cast<size_t>(size)));
    TRT_CHECK(engine_ != nullptr);
    context_.reset(engine_->createExecutionContext());
    TRT_CHECK(context_ != nullptr);

    if (config_.profile) {
        profiler_ = std::make_unique<Profiler>();
        context_->setProfiler(profiler_.get());
    }

    const int n = compat::num_io_tensors(engine_.get());
    for (int i = 0; i < n; ++i) {
        Binding           binding;
        const std::string name = compat::io_name(engine_.get(), i);
        binding.name           = name;
        binding.dsize          = dtype_size(compat::io_dtype(engine_.get(), i, name));

        if (compat::is_input(engine_.get(), i, name)) {
            const nvinfer1::Dims dims = compat::max_profile_dims(engine_.get(), i, name);
            compat::set_input_shape(context_.get(), i, name, dims);
            binding.dims = dims;
            binding.size = volume(dims);
            input_bindings_.push_back(binding);
        }
        else {
            const nvinfer1::Dims dims = compat::output_dims(context_.get(), i, name);
            binding.dims              = dims;
            binding.size              = volume(dims);
            output_bindings_.push_back(binding);
        }
    }

    // Adopt the engine's static input resolution (e.g. 640 for detect, 224 for cls).
    if (!input_bindings_.empty()) {
        const nvinfer1::Dims& d = input_bindings_[0].dims;
        if (d.nbDims == 4 && d.d[2] > 0 && d.d[3] > 0) {
            config_.input_size = cv::Size(static_cast<int>(d.d[3]), static_cast<int>(d.d[2]));
        }
    }
}

void Engine::make_pipe()
{
    device_buffers_.clear();
    host_buffers_.clear();
    device_ptrs_.clear();
    host_ptrs_.clear();
    device_buffers_.reserve(input_bindings_.size() + output_bindings_.size());
    host_buffers_.reserve(output_bindings_.size());

    for (const auto& b : input_bindings_) {
        device_buffers_.emplace_back(b.size * b.dsize);
        void* d = device_buffers_.back().data();
        device_ptrs_.push_back(d);
        compat::set_input_shape(context_.get(), 0, b.name, b.dims);
        compat::set_tensor_address(context_.get(), b.name, d);
    }
    for (const auto& b : output_bindings_) {
        device_buffers_.emplace_back(b.size * b.dsize);
        host_buffers_.emplace_back(b.size * b.dsize);
        void* d = device_buffers_.back().data();
        device_ptrs_.push_back(d);
        host_ptrs_.push_back(host_buffers_.back().data());
        compat::set_tensor_address(context_.get(), b.name, d);
    }

    if (config_.warmup) {
        std::vector<std::vector<char>> zeros;
        zeros.reserve(input_bindings_.size());
        for (size_t i = 0; i < input_bindings_.size(); ++i) {
            zeros.emplace_back(input_bindings_[i].size * input_bindings_[i].dsize, 0);
            CUDA_CHECK(cudaMemcpyAsync(
                device_ptrs_[i], zeros.back().data(), zeros.back().size(), cudaMemcpyHostToDevice, stream_.get()));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        for (int i = 0; i < 10; ++i) {
            infer();
        }
    }
}

void Engine::preprocess(const cv::Mat& image, cv::Mat& blob)
{
    letterbox(image, blob, config_.input_size, pparam_);
}

void Engine::copy_from_mat(const cv::Mat& image)
{
    const auto& in = input_bindings_[0];

    if (config_.gpu_preprocess) {
#ifdef YOLOV8_GPU_PREPROCESS
        // Upload the raw uint8 BGR image and let a CUDA kernel produce the NCHW float
        // blob directly in the input buffer — no CPU letterbox/blob.
        cv::Mat      src   = image.isContinuous() ? image : image.clone();
        const size_t bytes = src.total() * src.elemSize();
        if (raw_input_.bytes() < bytes) {
            raw_input_ = DeviceBuffer(bytes);  // grow (move-assign frees the old buffer)
        }
        CUDA_CHECK(cudaMemcpyAsync(raw_input_.data(), src.data, bytes, cudaMemcpyHostToDevice, stream_.get()));
        auto*       dst = static_cast<float*>(device_ptrs_[0]);
        const int   dw = config_.input_size.width, dh = config_.input_size.height;
        const auto* raw = static_cast<const unsigned char*>(raw_input_.data());
        if (letterbox_preproc()) {
            letterbox_cuda(raw, src.cols, src.rows, dst, dw, dh, pparam_, stream_.get());
        }
        else {
            resize_cuda(raw, src.cols, src.rows, dst, dw, dh, pparam_, stream_.get());
        }
        CUDA_CHECK(cudaGetLastError());
#else
        throw TrtException("--gpu-preprocess requires a CUDA-enabled build (no nvcc found at configure time)");
#endif
    }
    else {
        cv::Mat blob;
        preprocess(image, blob);
        CUDA_CHECK(cudaMemcpyAsync(
            device_ptrs_[0], blob.ptr<float>(), blob.total() * blob.elemSize(), cudaMemcpyHostToDevice, stream_.get()));
    }

    const nvinfer1::Dims4 shape{1, 3, config_.input_size.height, config_.input_size.width};
    compat::set_input_shape(context_.get(), 0, in.name, shape);
    compat::set_tensor_address(context_.get(), in.name, device_ptrs_[0]);
}

void Engine::infer()
{
    TRT_CHECK(compat::enqueue(context_.get(), device_ptrs_.data(), stream_.get()));
    for (size_t i = 0; i < output_bindings_.size(); ++i) {
        const size_t bytes = output_bindings_[i].size * output_bindings_[i].dsize;
        CUDA_CHECK(cudaMemcpyAsync(
            host_ptrs_[i], device_ptrs_[i + input_bindings_.size()], bytes, cudaMemcpyDeviceToHost, stream_.get()));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
}

void Engine::print_profile() const
{
    if (profiler_ && !profiler_->empty()) {
        profiler_->print(std::cout);
    }
}

}  // namespace yolov8
