# YOLOv8-pose Model with TensorRT

The yolov8-pose model conversion route is :
YOLOv8 PyTorch model -> ONNX -> TensorRT Engine

***Notice !!!*** We don't support TensorRT API building !!!

# Export Orin ONNX model by ultralytics

You can leave this repo and use the original `ultralytics` repo for onnx export.

### 1. ONNX -> TensorRT

You can export your onnx model by `ultralytics` API.

```shell
yolo export model=yolov8s-pose.pt format=onnx opset=11 simplify=True
```

or run this python script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-pose.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", opset=11, simplify=True)  # export the model to onnx format
assert success
```

Then build engine by Trtexec Tools.

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

```shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s-pose.onnx \
--saveEngine=yolov8s-pose.engine \
--fp16
```

### 2. Direct to TensorRT (NOT RECOMMAND!!)

Usage:

```shell
yolo export model=yolov8s-pose.pt format=engine device=0
```

or run python script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-pose.pt")  # load a pretrained model (recommended for training)
success = model.export(format="engine", device=0)  # export the model to engine format
assert success
```

After executing the above script, you will get an engine named `yolov8s-pose.engine` .

# Inference

## Infer with python script

You can infer images with the engine by [`infer-pose.py`](../infer-pose.py) .

Usage:

```shell
python3 infer-pose.py \
--engine yolov8s-pose.engine \
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

## Inference with c++

You can infer with c++ in [`csrc/pose/normal`](../csrc/pose/normal) .

### Build:

Please set you own librarys in [`CMakeLists.txt`](../csrc/pose/normal/CMakeLists.txt) and modify `KPS_COLORS`
and `SKELETON` and `LIMB_COLORS` in [`main.cpp`](../csrc/pose/normal/main.cpp).

Besides, you can modify the postprocess parameters such as `score_thres` and `iou_thres` and `topk`
in [`main.cpp`](../csrc/pose/normal/main.cpp).

```c++
int topk = 100;
float score_thres = 0.25f;
float iou_thres = 0.65f;
```

And build:

```shell
export root=${PWD}
cd src/pose/normal
mkdir build
cmake ..
make
mv yolov8-pose ${root}
cd ${root}
```

Usage:

```shell
# infer image
./yolov8-pose yolov8s-pose.engine data/bus.jpg
# infer images
./yolov8-pose yolov8s-pose.engine data
# infer video
./yolov8-pose yolov8s-pose.engine data/test.mp4 # the video path
```
