# Normal Usage of [`ultralytics`](https://github.com/ultralytics/ultralytics)

## Export TensorRT Engine

### 1. ONNX -> TensorRT

You can export your onnx model by `ultralytics` API.

```shell
yolo export model=yolov8s.pt format=onnx opset=11 simplify=True
```

or run this python script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", opset=11, simplify=True)  # export the model to onnx format
assert success
```

Then build engine by Trtexec Tools.

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

```shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s.onnx \
--saveEngine=yolov8s.engine \
--fp16
```

### 2. Direct to TensorRT (NOT RECOMMAND!!)

Usage:

```shell
yolo export model=yolov8s.pt format=engine device=0
```

or run python script:

```python

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
success = model.export(format="engine", device=0)  # export the model to engine format
assert success
```

After executing the above script, you will get an engine named `yolov8s.engine` .

## Inference with c++

You can infer with c++ in [`csrc/detect/normal`](../csrc/detect/normal) .

### Build:

Please set you own librarys in [`CMakeLists.txt`](../csrc/detect/normal/CMakeLists.txt) and modify `CLASS_NAMES`
and `COLORS` in [`main.cpp`](../csrc/detect/normal/main.cpp).

Besides, you can modify the postprocess parameters such as `num_labels` and `score_thres` and `iou_thres` and `topk`
in [`main.cpp`](../csrc/detect/normal/main.cpp).

```c++
int num_labels = 80;
int topk = 100;
float score_thres = 0.25f;
float iou_thres = 0.65f;
```

And build:

```shell
export root=${PWD}
cd src/detect/normal
mkdir build && cd build
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
