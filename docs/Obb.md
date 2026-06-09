# YOLOv8-obb Model with TensorRT

The yolov8-obb model conversion route is :
YOLOv8 PyTorch model -> ONNX -> TensorRT Engine

***Notice !!!*** We don't support TensorRT API building !!!

# Export Orin ONNX model by ultralytics

You can leave this repo and use the original `ultralytics` repo for onnx export.

### 1. ONNX -> TensorRT

You can export your onnx model by `ultralytics` API.

```shell
yolo export model=yolov8s-obb.pt format=onnx opset=11 simplify=True
```

or run this python script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-obb.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", opset=11, simplify=True)  # export the model to onnx format
assert success
```

Then build engine by Trtexec Tools.

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

```shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s-obb.onnx \
--saveEngine=yolov8s-obb.engine \
--fp16
```

### 2. Direct to TensorRT (NOT RECOMMAND!!)

Usage:

```shell
yolo export model=yolov8s-obb.pt format=engine device=0
```

or run python script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-obb.pt")  # load a pretrained model (recommended for training)
success = model.export(format="engine", device=0)  # export the model to engine format
assert success
```

After executing the above script, you will get an engine named `yolov8s-obb.engine` .

# Inference

## Infer with python script

You can infer images with the engine by [`infer-obb.py`](../infer-obb.py) .

Usage:

```shell
python3 infer-obb.py \
--engine yolov8s-obb.engine \
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

You can infer with c++ in [`csrc/obb/normal`](../csrc/obb/normal) .

### Build:

Please set you own librarys in [`CMakeLists.txt`](../csrc/obb/normal/CMakeLists.txt) and modify `CLASS_NAMES`
and `COLORS` in [`main.cpp`](../csrc/obb/normal/main.cpp).

Besides, you can modify the postprocess parameters such as `num_labels` and `score_thres` and `iou_thres` and `topk`

And build:

```shell
export root=${PWD}
cd src/obb/normal
mkdir build
cmake ..
make
mv yolov8-obb ${root}
cd ${root}
```

Usage:

```shell
# infer image
./yolov8-obb yolov8s-obb.engine data/bus.jpg
# infer images
./yolov8-obb yolov8s-obb.engine data
# infer video
./yolov8-obb yolov8s-obb.engine data/test.mp4 # the video path
```
