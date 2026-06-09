# YOLOv8-TensorRT

`YOLOv8` using TensorRT accelerate !

______________________________________________________________________

[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![Python Version](https://img.shields.io/badge/Python-3.8--3.10-FFD43B?logo=python)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![img](https://badgen.net/badge/icon/tensorrt?icon=azurepipelines&label)](https://developer.nvidia.com/tensorrt)
[![C++](https://img.shields.io/badge/CPP-11%2F14-yellow)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![img](https://badgen.net/github/license/triple-Mu/YOLOv8-TensorRT)](https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/LICENSE)
[![img](https://badgen.net/github/prs/triple-Mu/YOLOv8-TensorRT)](https://github.com/triple-Mu/YOLOv8-TensorRT/pulls)
[![img](https://img.shields.io/github/stars/triple-Mu/YOLOv8-TensorRT?color=ccf)](https://github.com/triple-Mu/YOLOv8-TensorRT)

______________________________________________________________________

# Prepare the environment

1. Install `CUDA` follow [`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-the-nvidia-cuda-toolkit).

   🚀 RECOMMENDED `CUDA` >= 11.4

1. Install `TensorRT` follow [`TensorRT official website`](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

   🚀 RECOMMENDED `TensorRT` >= 8.4

1. Install python requirements.

   ```shell
   pip install -r requirements.txt
   ```

1. Install [`ultralytics`](https://github.com/ultralytics/ultralytics) package for ONNX export or TensorRT API building.

   ```shell
   pip install ultralytics
   ```

1. Prepare your own PyTorch weight such as `yolov8s.pt` or `yolov8s-seg.pt`.

***NOTICE:***

Please use the latest `CUDA` and `TensorRT`, so that you can achieve the fastest speed !

If you have to use a lower version of `CUDA` and `TensorRT`, please read the relevant issues carefully !

# Normal Usage

If you get ONNX from origin [`ultralytics`](https://github.com/ultralytics/ultralytics) repo, you should build engine by yourself.

You can only use the `c++` inference code to deserialize the engine and do inference.

You can get more information in [`Normal.md`](docs/Normal.md) !

Besides, other scripts won't work.

# Export End2End ONNX with NMS

You can export your onnx model by `ultralytics` API and add postprocess such as bbox decoder and `NMS` into ONNX model at the same time.

```shell
python3 export-det.py \
--weights yolov8s.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```

#### Description of all arguments

- `--weights` : The PyTorch model you trained.
- `--iou-thres` : IOU threshold for NMS plugin.
- `--conf-thres` : Confidence threshold for NMS plugin.
- `--topk` : Max number of detection bboxes.
- `--opset` : ONNX opset version, default is 11.
- `--sim` : Whether to simplify your onnx model.
- `--input-shape` : Input shape for you model, should be 4 dimensions.
- `--device` : The CUDA deivce you export engine .

You will get an onnx model whose prefix is the same as input weights.

# Build End2End Engine from ONNX

### 1. Build Engine by TensorRT ONNX Python api

You can export TensorRT engine from ONNX by [`build.py` ](build.py).

Usage:

```shell
python3 build.py \
--weights yolov8s.onnx \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--fp16  \
--device cuda:0
```

#### Description of all arguments

- `--weights` : The ONNX model you download.
- `--iou-thres` : IOU threshold for NMS plugin.
- `--conf-thres` : Confidence threshold for NMS plugin.
- `--topk` : Max number of detection bboxes.
- `--fp16` : Whether to export half-precision engine.
- `--device` : The CUDA deivce you export engine .

You can modify `iou-thres` `conf-thres` `topk` by yourself.

### 2. Export Engine by Trtexec Tools

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

```shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s.onnx \
--saveEngine=yolov8s.engine \
--fp16
```

**If you installed TensorRT by a debian package, then the installation path of `trtexec`
is `/usr/src/tensorrt/bin/trtexec`**

**If you installed TensorRT by a tar package, then the installation path of `trtexec` is under the `bin` folder in the path you decompressed**

# Build TensorRT Engine by TensorRT API

Please see more information in [`API-Build.md`](docs/API-Build.md)

***Notice !!!*** We don't support YOLOv8-seg model now !!!

# Inference

## 1. Infer with python script

You can infer images with the engine by [`infer-det.py`](infer-det.py) .

Usage:

```shell
python3 infer-det.py \
--engine yolov8s.engine \
--imgs data \
--show \
--out-dir outputs \
--device cuda:0
```

#### Description of all arguments

- `--engine` : The Engine you export.
- `--imgs` : The images path you want to detect.
- `--show` : Whether to show detection results.
- `--out-dir` : Where to save detection results images. It will not work when use `--show` flag.
- `--device` : The CUDA deivce you use.
- `--profile` : Profile the TensorRT engine.

## 2. Infer with C++

You can infer with c++ in [`csrc/detect/end2end`](csrc/detect/end2end) .

### Build:

Please set you own librarys in [`CMakeLists.txt`](csrc/detect/end2end/CMakeLists.txt) and modify `CLASS_NAMES` and `COLORS` in [`main.cpp`](csrc/detect/end2end/main.cpp).

```shell
export root=${PWD}
cd csrc/detect/end2end
mkdir -p build && cd build
cmake ..
make
mv yolov8 ${root}
cd ${root}
```

Usage:

```shell
# infer image
./yolov8 yolov8s.engine data/bus.jpg
# infer images
./yolov8 yolov8s.engine data
# infer video
./yolov8 yolov8s.engine data/test.mp4 # the video path
```

# TensorRT Segment Deploy

Please see more information in [`Segment.md`](docs/Segment.md)

# TensorRT Pose Deploy

Please see more information in [`Pose.md`](docs/Pose.md)

# TensorRT Cls Deploy

Please see more information in [`Cls.md`](docs/Cls.md)

# TensorRT Obb Deploy

Please see more information in [`Obb.md`](docs/Obb.md)

# DeepStream Detection Deploy

See more in [`README.md`](csrc/deepstream/README.md)

# Jetson Deploy

Only test on `Jetson-NX 4GB`.
See more in [`Jetson.md`](docs/Jetson.md)

# Profile you engine

If you want to profile the TensorRT engine:

Usage:

```shell
python3 trt-profile.py --engine yolov8s.engine --device cuda:0
```

# Refuse To Use PyTorch for Model Inference !!!

If you need to break away from pytorch and use tensorrt inference,
you can get more information in [`infer-det-without-torch.py`](infer-det-without-torch.py),
the usage is the same as the pytorch version, but its performance is much worse.

You can use `cuda-python` or `pycuda` for inference.
Please install by such command:

```shell
pip install cuda-python
# or
pip install pycuda
```

Usage:

```shell
python3 infer-det-without-torch.py \
--engine yolov8s.engine \
--imgs data \
--show \
--out-dir outputs \
--method cudart
```

#### Description of all arguments

- `--engine` : The Engine you export.
- `--imgs` : The images path you want to detect.
- `--show` : Whether to show detection results.
- `--out-dir` : Where to save detection results images. It will not work when use `--show` flag.
- `--method` : Choose `cudart` or `pycuda`, default is `cudart`.
