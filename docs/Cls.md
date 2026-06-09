# YOLOv8-cls Model with TensorRT

The yolov8-cls model conversion route is :
YOLOv8 PyTorch model -> ONNX -> TensorRT Engine

***Notice !!!*** We don't support TensorRT API building !!!

# Export Orin ONNX model by ultralytics

You can leave this repo and use the original `ultralytics` repo for onnx export.

### 1. ONNX -> TensorRT

You can export your onnx model by `ultralytics` API.

```shell
yolo export model=yolov8s-cls.pt format=onnx opset=11 simplify=True
```

or run this python script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-cls.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", opset=11, simplify=True)  # export the model to onnx format
assert success
```

Then build engine by Trtexec Tools.

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

```shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s-cls.onnx \
--saveEngine=yolov8s-cls.engine \
--fp16
```

### 2. Direct to TensorRT (NOT RECOMMAND!!)

Usage:

```shell
yolo export model=yolov8s-cls.pt format=engine device=0
```

or run python script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-cls.pt")  # load a pretrained model (recommended for training)
success = model.export(format="engine", device=0)  # export the model to engine format
assert success
```

After executing the above script, you will get an engine named `yolov8s-cls.engine` .

# Inference

## Infer with python script

You can infer images with the engine by [`infer-cls.py`](../infer-cls.py) .

Usage:

```shell
python3 infer-cls.py \
--engine yolov8s-cls.engine \
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

You can infer with c++ in [`csrc/cls/normal`](../csrc/cls/normal) .

### Build:

Please set you own librarys in [`CMakeLists.txt`](../csrc/cls/normal/CMakeLists.txt) and modify `CLASS_NAMES` in [`main.cpp`](../csrc/cls/normal/main.cpp).

And build:

```shell
export root=${PWD}
cd src/cls/normal
mkdir build
cmake ..
make
mv yolov8-cls ${root}
cd ${root}
```

Usage:

```shell
# infer image
./yolov8-cls yolov8s-cls.engine data/bus.jpg
# infer images
./yolov8-cls yolov8s-cls.engine data
# infer video
./yolov8-cls yolov8s-cls.engine data/test.mp4 # the video path
```
