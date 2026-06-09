"""Unified non-torch inference backends (cudart / pycuda).

Both backends share engine loading, binding enumeration (TensorRT 8/10 via
:mod:`models.compat`), warmup and the inference loop; subclasses implement only
the handful of CUDA primitives that differ (stream / malloc / memcpy / sync).
This replaces the duplicated TRT-8-only cudart_api.py / pycuda_api.py.
"""

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
from numpy import ndarray

from models import compat

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
warnings.filterwarnings(action="ignore", category=DeprecationWarning)


@dataclass
class Tensor:
    name: str
    dtype: np.dtype
    shape: tuple
    cpu: ndarray
    gpu: object  # device pointer (int) or pycuda DeviceAllocation


class Backend(ABC):
    """TensorRT inference over a NumPy interface, shared by the cudart/pycuda backends."""

    def __init__(self, weight: str | Path) -> None:
        self.weight = Path(weight)
        self.stream = self._create_stream()
        self.__init_engine()
        self.__init_bindings()
        self.__warm_up()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())
        if model is None:
            raise RuntimeError(f"failed to deserialize engine: {self.weight}")
        self.model = model
        self.context = model.create_execution_context()
        self.input_names, self.output_names = compat.io_names(model)
        self.num_inputs = len(self.input_names)
        self.num_outputs = len(self.output_names)

    def __init_bindings(self) -> None:
        self.is_dynamic = False
        self.inp_info: list[Tensor] = []
        self.out_info: list[Tensor] = []
        for index, name in enumerate(self.input_names + self.output_names):
            dtype = compat.np_dtype(self.model, name, index)
            shape = compat.engine_shape(self.model, name, index)
            self.is_dynamic |= -1 in shape
            target = self.inp_info if index < self.num_inputs else self.out_info
            if self.is_dynamic:
                target.append(Tensor(name, dtype, shape, np.empty(0), 0))
            else:
                cpu = np.empty(shape, dtype)
                gpu = self._malloc(cpu.nbytes)
                target.append(Tensor(name, dtype, shape, cpu, gpu))

    def __warm_up(self) -> None:
        if self.is_dynamic:
            print("Engine has dynamic axes; please warm up manually.")
            return
        for _ in range(10):
            self.__call__(*[t.cpu for t in self.inp_info])

    def set_profiler(self, profiler) -> None:
        self.context.profiler = profiler if profiler is not None else trt.Profiler()

    def __call__(self, *inputs) -> tuple | ndarray:
        assert len(inputs) == self.num_inputs
        contiguous = [np.ascontiguousarray(i) for i in inputs]
        bindings: list[int] = []

        for i, name in enumerate(self.input_names):
            arr = contiguous[i]
            if self.is_dynamic:
                compat.set_input_shape(self.context, name, i, tuple(arr.shape))
                self.inp_info[i].gpu = self._malloc(arr.nbytes)
            self._memcpy_h2d(self.inp_info[i].gpu, arr)
            self._bind(name, i, self.inp_info[i].gpu, bindings)

        outputs: list[ndarray] = []
        for i, name in enumerate(self.output_names):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = compat.context_shape(self.context, name, j)
                cpu = np.empty(shape, dtype=self.out_info[i].dtype)
                gpu = self._malloc(cpu.nbytes)
            else:
                cpu, gpu = self.out_info[i].cpu, self.out_info[i].gpu
            outputs.append(cpu)
            self._bind(name, j, gpu, bindings)

        self._execute(bindings)
        self._sync()
        for i, name in enumerate(self.output_names):
            gpu = self.out_info[i].gpu if not self.is_dynamic else bindings[i + self.num_inputs]
            self._memcpy_d2h(outputs[i], gpu)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def _bind(self, name: str, index: int, gpu, bindings: list) -> None:
        if compat.is_trt10:
            self.context.set_tensor_address(name, self._addr(gpu))
        else:
            bindings.append(self._addr(gpu))

    def _execute(self, bindings: list) -> None:
        if compat.is_trt10:
            ok = self.context.execute_async_v3(self._stream_ptr())
        else:
            ok = self.context.execute_async_v2(bindings, self._stream_ptr())
        if not ok:
            raise RuntimeError("TensorRT execution failed")

    # --- CUDA primitives implemented per backend ---
    @abstractmethod
    def _create_stream(self): ...
    @abstractmethod
    def _stream_ptr(self) -> int: ...
    @abstractmethod
    def _malloc(self, nbytes: int): ...
    @abstractmethod
    def _addr(self, gpu) -> int: ...
    @abstractmethod
    def _memcpy_h2d(self, gpu, arr: ndarray) -> None: ...
    @abstractmethod
    def _memcpy_d2h(self, arr: ndarray, gpu) -> None: ...
    @abstractmethod
    def _sync(self) -> None: ...


class CudartBackend(Backend):
    def _create_stream(self):
        from cuda import cudart

        self._cudart = cudart
        status, stream = cudart.cudaStreamCreate()
        assert status.value == 0
        return stream

    def _stream_ptr(self):
        return self.stream

    def _malloc(self, nbytes: int):
        status, gpu = self._cudart.cudaMallocAsync(nbytes, self.stream)
        assert status.value == 0
        return gpu

    def _addr(self, gpu) -> int:
        return int(gpu)

    def _memcpy_h2d(self, gpu, arr: ndarray) -> None:
        c = self._cudart
        c.cudaMemcpyAsync(gpu, arr.ctypes.data, arr.nbytes, c.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)

    def _memcpy_d2h(self, arr: ndarray, gpu) -> None:
        c = self._cudart
        c.cudaMemcpyAsync(arr.ctypes.data, gpu, arr.nbytes, c.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

    def _sync(self) -> None:
        self._cudart.cudaStreamSynchronize(self.stream)


class PycudaBackend(Backend):
    def _create_stream(self):
        import pycuda.autoinit  # noqa: F401  (initialises the CUDA context)
        import pycuda.driver as cuda

        self._cuda = cuda
        return cuda.Stream(0)

    def _stream_ptr(self):
        return self.stream.handle

    def _malloc(self, nbytes: int):
        return self._cuda.mem_alloc(nbytes)

    def _addr(self, gpu) -> int:
        return int(gpu)

    def _memcpy_h2d(self, gpu, arr: ndarray) -> None:
        self._cuda.memcpy_htod_async(gpu, arr, self.stream)

    def _memcpy_d2h(self, arr: ndarray, gpu) -> None:
        self._cuda.memcpy_dtoh_async(arr, gpu, self.stream)

    def _sync(self) -> None:
        self.stream.synchronize()
