"""TensorRT version compatibility helpers (Python side).

Centralises the TensorRT 8 (binding-index) vs 10+ (tensor-name) API differences
so the inference backends don't each branch on the version. Mirrors the role of
csrc/core/include/yolov8/trt_compat.hpp on the C++ side.
"""

import tensorrt as trt

tensorrt_version = trt.__version__
major_version = int(tensorrt_version.split(".")[0])
is_trt10 = major_version >= 10


def io_names(engine) -> tuple[list[str], list[str]]:
    """Return (input_names, output_names) for either API generation."""
    if is_trt10:
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    else:
        names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
        inputs = [names[i] for i in range(engine.num_bindings) if engine.binding_is_input(i)]
        outputs = [names[i] for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    return inputs, outputs


def trt_dtype(engine, name: str, index: int):
    """Raw TensorRT ``DataType`` for an IO tensor (use with a dtype mapping)."""
    return engine.get_tensor_dtype(name) if is_trt10 else engine.get_binding_dtype(index)


def np_dtype(engine, name: str, index: int):
    return trt.nptype(trt_dtype(engine, name, index))


def engine_shape(engine, name: str, index: int) -> tuple:
    return tuple(engine.get_tensor_shape(name) if is_trt10 else engine.get_binding_shape(index))


def context_shape(context, name: str, index: int) -> tuple:
    return tuple(context.get_tensor_shape(name) if is_trt10 else context.get_binding_shape(index))


def set_input_shape(context, name: str, index: int, shape) -> None:
    if is_trt10:
        context.set_input_shape(name, shape)
    else:
        context.set_binding_shape(index, shape)
