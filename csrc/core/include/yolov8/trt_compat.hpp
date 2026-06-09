#pragma once
// The ONLY place in the project that branches on the TensorRT version.
//
// TensorRT 10 replaced the binding-index API with a tensor-name API and removed
// IObject::destroy() in favour of plain `delete`. These thin inline wrappers expose
// a single internal API (taking both the binding index and the tensor name) so the
// rest of the codebase never needs an #ifdef. TRT_10 is defined by cmake/TrtDefs.cmake.

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <string>

namespace yolov8::compat {

inline int num_io_tensors(nvinfer1::ICudaEngine* engine)
{
#ifdef TRT_10
    return engine->getNbIOTensors();
#else
    return engine->getNbBindings();
#endif
}

inline std::string io_name(nvinfer1::ICudaEngine* engine, int index)
{
#ifdef TRT_10
    return engine->getIOTensorName(index);
#else
    return engine->getBindingName(index);
#endif
}

inline nvinfer1::DataType io_dtype(nvinfer1::ICudaEngine* engine, int index, const std::string& name)
{
#ifdef TRT_10
    (void)index;
    return engine->getTensorDataType(name.c_str());
#else
    (void)name;
    return engine->getBindingDataType(index);
#endif
}

inline bool is_input(nvinfer1::ICudaEngine* engine, int index, const std::string& name)
{
#ifdef TRT_10
    (void)index;
    return engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
    (void)name;
    return engine->bindingIsInput(index);
#endif
}

inline nvinfer1::Dims max_profile_dims(nvinfer1::ICudaEngine* engine, int index, const std::string& name)
{
#ifdef TRT_10
    (void)index;
    return engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
#else
    (void)name;
    return engine->getProfileDimensions(index, 0, nvinfer1::OptProfileSelector::kMAX);
#endif
}

inline void
set_input_shape(nvinfer1::IExecutionContext* context, int index, const std::string& name, const nvinfer1::Dims& dims)
{
#ifdef TRT_10
    (void)index;
    context->setInputShape(name.c_str(), dims);
#else
    (void)name;
    context->setBindingDimensions(index, dims);
#endif
}

inline nvinfer1::Dims output_dims(nvinfer1::IExecutionContext* context, int index, const std::string& name)
{
#ifdef TRT_10
    (void)index;
    return context->getTensorShape(name.c_str());
#else
    (void)name;
    return context->getBindingDimensions(index);
#endif
}

// On TRT 10 the device address is bound to the context by name; on TRT 8 the address
// is passed positionally to enqueueV2, so this is a no-op there.
inline void set_tensor_address(nvinfer1::IExecutionContext* context, const std::string& name, void* ptr)
{
#ifdef TRT_10
    context->setTensorAddress(name.c_str(), ptr);
#else
    (void)context;
    (void)name;
    (void)ptr;
#endif
}

// device_ptrs must be ordered [inputs..., outputs...]; it is ignored on TRT 10
// (addresses were already bound via set_tensor_address).
inline bool enqueue(nvinfer1::IExecutionContext* context, void** device_ptrs, cudaStream_t stream)
{
#ifdef TRT_10
    (void)device_ptrs;
    return context->enqueueV3(stream);
#else
    return context->enqueueV2(device_ptrs, stream, nullptr);
#endif
}

// TRT 10 removed destroy(); these objects are destroyed with `delete`.
template<typename T>
inline void trt_destroy(T* ptr)
{
#ifdef TRT_10
    delete ptr;
#else
    ptr->destroy();
#endif
}

}  // namespace yolov8::compat
