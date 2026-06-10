import os
import pickle
from collections import defaultdict, namedtuple
from pathlib import Path

import onnx
import tensorrt as trt
import torch

from models import compat

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

tensorrt_version = trt.__version__
major_version = int(tensorrt_version.split(".")[0])
minor_version = int(tensorrt_version.split(".")[1])
device = torch.cuda.current_device()
total_memory = torch.cuda.get_device_properties(device).total_memory


class EngineBuilder:
    seg = False

    def __init__(self, checkpoint: str | Path, device: str | int | torch.device | None = None) -> None:
        checkpoint = Path(checkpoint) if isinstance(checkpoint, str) else checkpoint
        if not (checkpoint.exists() and checkpoint.suffix in (".onnx", ".pkl")):
            raise ValueError(f"weights must be an existing .onnx or .pkl file: {checkpoint}")
        self.api = checkpoint.suffix == ".pkl"
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")

        self.checkpoint = checkpoint
        self.device = device

    def __build_engine(
        self,
        fp16: bool = True,
        input_shape: list | tuple = (1, 3, 640, 640),
        iou_thres: float = 0.65,
        conf_thres: float = 0.25,
        topk: int = 100,
        with_profiling: bool = True,
    ) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace="")
        builder = trt.Builder(logger)
        config = builder.create_builder_config()

        if major_version >= 10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, total_memory // 2)
            config.set_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM, total_memory // 4)
        elif major_version >= 8:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, total_memory)
        else:
            config.max_workspace_size = total_memory

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)

        self.logger = logger
        self.builder = builder
        self.network = network
        if self.api:
            self.build_from_api(fp16, input_shape, iou_thres, conf_thres, topk)
        else:
            self.build_from_onnx(iou_thres, conf_thres, topk)
        if fp16 and self.builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        self.weight = self.checkpoint.with_suffix(".engine")

        if with_profiling:
            if major_version <= 8:
                config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE
            else:
                config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        if major_version >= 8:
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build serialized engine")
            with trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build engine")

        if engine is not None:
            self.weight.write_bytes(engine.serialize())
            self.logger.log(
                trt.Logger.WARNING, f"Build TensorRT engine finished.\nSaved to {str(self.weight.absolute())}"
            )
        else:
            raise RuntimeError("Engine creation failed")

    def build(
        self,
        fp16: bool = True,
        input_shape: list | tuple = (1, 3, 640, 640),
        iou_thres: float = 0.65,
        conf_thres: float = 0.25,
        topk: int = 100,
        with_profiling: bool = True,
    ) -> None:
        self.__build_engine(fp16, input_shape, iou_thres, conf_thres, topk, with_profiling)

    def build_from_onnx(self, iou_thres: float = 0.65, conf_thres: float = 0.25, topk: int = 100) -> None:
        parser = trt.OnnxParser(self.network, self.logger)
        onnx_model = onnx.load(str(self.checkpoint))
        if not self.seg:
            onnx_model.graph.node[-1].attribute[2].i = topk
            onnx_model.graph.node[-1].attribute[3].f = conf_thres
            onnx_model.graph.node[-1].attribute[4].f = iou_thres

        if not parser.parse(onnx_model.SerializeToString()):
            raise RuntimeError(f"Failed to load ONNX file: {str(self.checkpoint)}")
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        for inp in inputs:
            self.logger.log(trt.Logger.WARNING, f'input "{inp.name}" with shape: {inp.shape} dtype: {inp.dtype}')
        for out in outputs:
            self.logger.log(trt.Logger.WARNING, f'output "{out.name}" with shape: {out.shape} dtype: {out.dtype}')

    def build_from_api(
        self,
        fp16: bool = True,
        input_shape: list | tuple = (1, 3, 640, 640),
        iou_thres: float = 0.65,
        conf_thres: float = 0.25,
        topk: int = 100,
    ) -> None:
        if self.seg:
            raise ValueError("the TensorRT API builder does not support segmentation models")
        from .api import SPPF, C2f, Conv, Detect, get_depth, get_width

        with open(self.checkpoint, "rb") as f:
            state_dict = pickle.load(f)
        mapping = {0.25: 1024, 0.5: 1024, 0.75: 768, 1.0: 512, 1.25: 512}

        GW = state_dict["GW"]
        GD = state_dict["GD"]
        width_64 = get_width(64, GW)
        width_128 = get_width(128, GW)
        width_256 = get_width(256, GW)
        width_512 = get_width(512, GW)
        width_1024 = get_width(mapping[GW], GW)
        depth_3 = get_depth(3, GD)
        depth_6 = get_depth(6, GD)
        strides = state_dict["strides"]
        reg_max = state_dict["reg_max"]
        images = self.network.add_input(name="images", dtype=trt.float32, shape=trt.Dims4(input_shape))
        assert images, "Add input failed"

        Conv_0 = Conv(self.network, state_dict, images, width_64, 3, 2, 1, "Conv.0")
        Conv_1 = Conv(self.network, state_dict, Conv_0.get_output(0), width_128, 3, 2, 1, "Conv.1")
        C2f_2 = C2f(self.network, state_dict, Conv_1.get_output(0), width_128, depth_3, True, 1, 0.5, "C2f.2")
        Conv_3 = Conv(self.network, state_dict, C2f_2.get_output(0), width_256, 3, 2, 1, "Conv.3")
        C2f_4 = C2f(self.network, state_dict, Conv_3.get_output(0), width_256, depth_6, True, 1, 0.5, "C2f.4")
        Conv_5 = Conv(self.network, state_dict, C2f_4.get_output(0), width_512, 3, 2, 1, "Conv.5")
        C2f_6 = C2f(self.network, state_dict, Conv_5.get_output(0), width_512, depth_6, True, 1, 0.5, "C2f.6")
        Conv_7 = Conv(self.network, state_dict, C2f_6.get_output(0), width_1024, 3, 2, 1, "Conv.7")
        C2f_8 = C2f(self.network, state_dict, Conv_7.get_output(0), width_1024, depth_3, True, 1, 0.5, "C2f.8")
        SPPF_9 = SPPF(self.network, state_dict, C2f_8.get_output(0), width_1024, width_1024, 5, "SPPF.9")
        Upsample_10 = self.network.add_resize(SPPF_9.get_output(0))
        assert Upsample_10, "Add Upsample_10 failed"
        Upsample_10.resize_mode = trt.ResizeMode.NEAREST
        Upsample_10.shape = Upsample_10.get_output(0).shape[:2] + C2f_6.get_output(0).shape[2:]
        input_tensors11 = [Upsample_10.get_output(0), C2f_6.get_output(0)]
        Cat_11 = self.network.add_concatenation(input_tensors11)
        C2f_12 = C2f(self.network, state_dict, Cat_11.get_output(0), width_512, depth_3, False, 1, 0.5, "C2f.12")
        Upsample13 = self.network.add_resize(C2f_12.get_output(0))
        assert Upsample13, "Add Upsample13 failed"
        Upsample13.resize_mode = trt.ResizeMode.NEAREST
        Upsample13.shape = Upsample13.get_output(0).shape[:2] + C2f_4.get_output(0).shape[2:]
        input_tensors14 = [Upsample13.get_output(0), C2f_4.get_output(0)]
        Cat_14 = self.network.add_concatenation(input_tensors14)
        C2f_15 = C2f(self.network, state_dict, Cat_14.get_output(0), width_256, depth_3, False, 1, 0.5, "C2f.15")
        Conv_16 = Conv(self.network, state_dict, C2f_15.get_output(0), width_256, 3, 2, 1, "Conv.16")
        input_tensors17 = [Conv_16.get_output(0), C2f_12.get_output(0)]
        Cat_17 = self.network.add_concatenation(input_tensors17)
        C2f_18 = C2f(self.network, state_dict, Cat_17.get_output(0), width_512, depth_3, False, 1, 0.5, "C2f.18")
        Conv_19 = Conv(self.network, state_dict, C2f_18.get_output(0), width_512, 3, 2, 1, "Conv.19")
        input_tensors20 = [Conv_19.get_output(0), SPPF_9.get_output(0)]
        Cat_20 = self.network.add_concatenation(input_tensors20)
        C2f_21 = C2f(self.network, state_dict, Cat_20.get_output(0), width_1024, depth_3, False, 1, 0.5, "C2f.21")
        input_tensors22 = [C2f_15.get_output(0), C2f_18.get_output(0), C2f_21.get_output(0)]
        batched_nms = Detect(
            self.network, state_dict, input_tensors22, strides, "Detect.22", reg_max, fp16, iou_thres, conf_thres, topk
        )
        for o in range(batched_nms.num_outputs):
            self.network.mark_output(batched_nms.get_output(o))


class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32,
    }

    def __init__(self, weight: str | Path, device: torch.device | None) -> None:
        super().__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device("cuda:0")
        self.stream = torch.cuda.Stream(device=self.device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        self.input_names, self.output_names = compat.io_names(model)
        self.num_inputs = len(self.input_names)
        self.num_outputs = len(self.output_names)
        self.num_bindings = self.num_inputs + self.num_outputs
        self.bindings: list[int] = [0] * self.num_bindings  # positional bindings, used by TensorRT 8 enqueueV2
        self.model = model
        self.context = context
        self.idx = list(range(self.num_outputs))

    def __init_bindings(self) -> None:
        idynamic = odynamic = False
        Tensor = namedtuple("Tensor", ("name", "dtype", "shape"))
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            dtype = self.dtypeMapping[compat.trt_dtype(self.model, name, i)]
            shape = compat.engine_shape(self.model, name, i)
            if -1 in shape:
                idynamic = True
            inp_info.append(Tensor(name, dtype, shape))
        for i, name in enumerate(self.output_names):
            j = i + self.num_inputs
            dtype = self.dtypeMapping[compat.trt_dtype(self.model, name, j)]
            shape = compat.engine_shape(self.model, name, j)
            if -1 in shape:
                odynamic = True
            out_info.append(Tensor(name, dtype, shape))

        if not odynamic:
            self.output_tensor = [torch.empty(info.shape, dtype=info.dtype, device=self.device) for info in out_info]
        self.idynamic = idynamic
        self.odynamic = odynamic
        self.inp_info = inp_info
        self.out_info = out_info

    def set_profiler(self, profiler: trt.IProfiler | None):
        self.context.profiler = profiler if profiler is not None else trt.Profiler()

    def set_desired(self, desired: list | tuple | None):
        if isinstance(desired, (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs) -> tuple | torch.Tensor:
        if len(inputs) != self.num_inputs:
            raise ValueError(f"expected {self.num_inputs} input(s), got {len(inputs)}")
        contiguous_inputs: list[torch.Tensor] = [i.contiguous() for i in inputs]

        if major_version >= 10:
            for i, name in enumerate(self.input_names):
                self.context.set_tensor_address(name, contiguous_inputs[i].data_ptr())
                if self.idynamic:
                    self.context.set_input_shape(name, tuple(contiguous_inputs[i].shape))

            outputs: list[torch.Tensor] = []
            for i, name in enumerate(self.output_names):
                if self.odynamic:
                    shape = tuple(self.context.get_tensor_shape(name))
                    output = torch.empty(size=shape, dtype=self.out_info[i].dtype, device=self.device)
                else:
                    output = self.output_tensor[i]
                self.context.set_tensor_address(name, output.data_ptr())
                outputs.append(output)

            success = self.context.execute_async_v3(self.stream.cuda_stream)
            if not success:
                raise RuntimeError("TensorRT execution failed")
        else:
            for i in range(self.num_inputs):
                self.bindings[i] = contiguous_inputs[i].data_ptr()
                if self.idynamic:
                    self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))

            outputs: list[torch.Tensor] = []
            for i in range(self.num_outputs):
                j = i + self.num_inputs
                if self.odynamic:
                    shape = tuple(self.context.get_binding_shape(j))
                    output = torch.empty(size=shape, dtype=self.out_info[i].dtype, device=self.device)
                else:
                    output = self.output_tensor[i]
                self.bindings[j] = output.data_ptr()
                outputs.append(output)

            self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)

        self.stream.synchronize()

        return tuple(outputs[i] for i in self.idx) if len(outputs) > 1 else outputs[0]


class TRTProfilerV1(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.total_runtime = 0.0
        self.recorder = defaultdict(float)

    def report_layer_time(self, layer_name: str, ms: float):
        self.total_runtime += ms * 1000
        self.recorder[layer_name] += ms * 1000

    def report(self):
        fmt = "\t{:>40}\t\t\t\t{:>10.4f}"
        print("\t{:>40}\t\t\t\t{:>10}".format("layername", "cost(us)"))
        for name, cost in sorted(self.recorder.items(), key=lambda x: -x[1]):
            print(fmt.format(name if len(name) < 40 else name[:35] + " " + "*" * 4, cost))
        print(f"\nTotal Inference Time: {self.total_runtime:.4f}(us)")


class TRTProfilerV0(trt.IProfiler):
    def __init__(self):
        super().__init__()

    def report_layer_time(self, layer_name: str, ms: float):
        f = "\t%40s\t\t\t\t%10.4fms"
        print(f % (layer_name if len(layer_name) < 40 else layer_name[:35] + " " + "*" * 4, ms))
