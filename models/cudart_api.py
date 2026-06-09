import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart
from numpy import ndarray

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
warnings.filterwarnings(action="ignore", category=DeprecationWarning)


@dataclass
class Tensor:
    name: str
    dtype: np.dtype
    shape: tuple
    cpu: ndarray
    gpu: int


class TRTEngine:
    def __init__(self, weight: str | Path) -> None:
        self.weight = Path(weight) if isinstance(weight, str) else weight
        status, self.stream = cudart.cudaStreamCreate()
        assert status.value == 0
        self.__init_engine()
        self.__init_bindings()
        self.__warm_up()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        names = [model.get_binding_name(i) for i in range(model.num_bindings)]
        self.num_bindings = model.num_bindings
        self.bindings: list[int] = [0] * self.num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(model.num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]

    def __init_bindings(self) -> None:
        dynamic = False
        inp_info = []
        out_info = []
        out_ptrs = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                dynamic |= True
            if not dynamic:
                cpu = np.empty(shape, dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream
                )
            else:
                cpu, gpu = np.empty(0), 0
            inp_info.append(Tensor(name, dtype, shape, cpu, gpu))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if not dynamic:
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream
                )
                out_ptrs.append(gpu)
            else:
                cpu, gpu = np.empty(0), 0
            out_info.append(Tensor(name, dtype, shape, cpu, gpu))

        self.is_dynamic = dynamic
        self.inp_info = inp_info
        self.out_info = out_info
        self.out_ptrs = out_ptrs

    def __warm_up(self) -> None:
        if self.is_dynamic:
            print("You engine has dynamic axes, please warm up by yourself !")
            return
        for _ in range(10):
            inputs = []
            for i in self.inp_info:
                inputs.append(i.cpu)
            self.__call__(inputs)

    def set_profiler(self, profiler: trt.IProfiler | None) -> None:
        self.context.profiler = profiler if profiler is not None else trt.Profiler()

    def __call__(self, *inputs) -> tuple | ndarray:

        assert len(inputs) == self.num_inputs
        contiguous_inputs: list[ndarray] = [np.ascontiguousarray(i) for i in inputs]

        for i in range(self.num_inputs):
            if self.is_dynamic:
                self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))
                status, self.inp_info[i].gpu = cudart.cudaMallocAsync(contiguous_inputs[i].nbytes, self.stream)
                assert status.value == 0
            cudart.cudaMemcpyAsync(
                self.inp_info[i].gpu,
                contiguous_inputs[i].ctypes.data,
                contiguous_inputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
            self.bindings[i] = self.inp_info[i].gpu

        output_gpu_ptrs: list[int] = []
        outputs: list[ndarray] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                dtype = self.out_info[i].dtype
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream
                )
            else:
                cpu = self.out_info[i].cpu
                gpu = self.out_info[i].gpu
            outputs.append(cpu)
            output_gpu_ptrs.append(gpu)
            self.bindings[j] = gpu

        self.context.execute_async_v2(self.bindings, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        for i, o in enumerate(output_gpu_ptrs):
            cudart.cudaMemcpyAsync(
                outputs[i].ctypes.data, o, outputs[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream
            )

        return tuple(outputs) if len(outputs) > 1 else outputs[0]
