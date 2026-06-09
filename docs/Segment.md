# YOLOv8-seg Model with TensorRT

The yolov8-seg model conversion route is :
YOLOv8 PyTorch model -> ONNX -> TensorRT Engine

***Notice !!!*** We don't support TensorRT API building !!!

# Export Modified ONNX model

You can export your onnx model by `ultralytics` API and the onnx is also modify by this repo.

```shell
python3 export-seg.py \
--weights yolov8s-seg.pt \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```

#### Description of all arguments

- `--weights` : The PyTorch model you trained.
- `--opset` : ONNX opset version, default is 11.
- `--sim` : Whether to simplify your onnx model.
- `--input-shape` : Input shape for you model, should be 4 dimensions.
- `--device` : The CUDA deivce you export engine .

You will get an onnx model whose prefix is the same as input weights.

This onnx model doesn't contain postprocessing.

# Export Engine by TensorRT Python api

You can export TensorRT engine from ONNX by [`build.py` ](../build.py).

Usage:

```shell
python3 build.py \
--weights yolov8s-seg.onnx \
--fp16  \
--device cuda:0 \
--seg
```

#### Description of all arguments

- `--weights` : The ONNX model you download.
- `--fp16` : Whether to export half-precision engine.
- `--device` : The CUDA deivce you export engine.
- `--seg` : Whether to export seg engine.

# Export Engine by Trtexec Tools

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

```shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s-seg.onnx \
--saveEngine=yolov8s-seg.engine \
--fp16
```

# Inference

## Infer with python script

You can infer images with the engine by [`infer-seg.py`](../infer-seg.py) .

Usage:

```shell
python3 infer-seg.py \
--engine yolov8s-seg.engine \
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

## Infer with C++

You can infer segment engine with c++ in [`csrc/segment/simple`](../csrc/segment/simple) .

### Build:

Please set you own librarys in [`CMakeLists.txt`](../csrc/segment/simple/CMakeLists.txt) and modify you own config
in [`main.cpp`](../csrc/segment/simple/main.cpp) such as `CLASS_NAMES`, `COLORS`, `MASK_COLORS` and postprocess
parameters .

```c++
int topk = 100;
int seg_h = 160; // yolov8 model proto height
int seg_w = 160; // yolov8 model proto width
int seg_channels = 32; // yolov8 model proto channels
float score_thres = 0.25f;
float iou_thres = 0.65f;
```

```shell
export root=${PWD}
cd src/segment/simple
mkdir build
cmake ..
make
mv yolov8-seg ${root}
cd ${root}
```

***Notice !!!***

If you have build OpenCV(>=4.7.0) by yourself, it provides a new
api [`cv::dnn::NMSBoxesBatched`](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga977aae09fbf7c804e003cfea1d4e928c) .
It is a gread api about efficient in-class nms . It will be used by default!

***!!!***

Usage:

```shell
# infer image
./yolov8-seg yolov8s-seg.engine data/bus.jpg
# infer images
./yolov8-seg yolov8s-seg.engine data
# infer video
./yolov8-seg yolov8s-seg.engine data/test.mp4 # the video path
```

# Export Orin ONNX model by ultralytics

You can leave this repo and use the original `ultralytics` repo for onnx export.

### 1. ONNX -> TensorRT

You can export your onnx model by `ultralytics` API.

```shell
yolo export model=yolov8s-seg.pt format=onnx opset=11 simplify=True
```

or run this python script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-seg.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", opset=11, simplify=True)  # export the model to onnx format
assert success
```

Then build engine by Trtexec Tools.

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

```shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s-seg.onnx \
--saveEngine=yolov8s-seg.engine \
--fp16
```

### 2. Direct to TensorRT (NOT RECOMMAND!!)

Usage:

```shell
yolo export model=yolov8s-seg.pt format=engine device=0
```

or run python script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-seg.pt")  # load a pretrained model (recommended for training)
success = model.export(format="engine", device=0)  # export the model to engine format
assert success
```

After executing the above script, you will get an engine named `yolov8s-seg.engine` .

## Inference with c++

You can infer with c++ in [`csrc/segment/normal`](../csrc/segment/normal) .

### Build:

Please set you own librarys in [`CMakeLists.txt`](../csrc/segment/normal/CMakeLists.txt) and modify `CLASS_NAMES`
and `COLORS` in [`main.cpp`](../csrc/segment/normal/main.cpp).

Besides, you can modify the postprocess parameters such as `num_labels` and `score_thres` and `iou_thres` and `topk`
in [`main.cpp`](../csrc/segment/normal/main.cpp).

```c++
int topk = 100;
int seg_h = 160; // yolov8 model proto height
int seg_w = 160; // yolov8 model proto width
int seg_channels = 32; // yolov8 model proto channels
float score_thres = 0.25f;
float iou_thres = 0.65f;
```

And build:

```shell
export root=${PWD}
cd src/segment/normal
mkdir build
cmake ..
make
mv yolov8-seg ${root}
cd ${root}
```

Usage:

```shell
# infer image
./yolov8-seg yolov8s-seg.engine data/bus.jpg
# infer images
./yolov8-seg yolov8s-seg.engine data
# infer video
./yolov8-seg yolov8s-seg.engine data/test.mp4 # the video path
```

# Refuse To Use PyTorch for segment Model Inference !!!

It is the same as detection model.
you can get more information in [`infer-seg-without-torch.py`](../infer-seg-without-torch.py),

Usage:

```shell
python3 infer-seg-without-torch.py \
--engine yolov8s-seg.engine \
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
