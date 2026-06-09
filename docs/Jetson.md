# YOLOv8 on Jetson

Only test on `Jetson-NX 4GB`

ENVS:

- Jetpack 4.6.3
- CUDA-10.2
- CUDNN-8.2.1
- TensorRT-8.2.1
- DeepStream-6.0.1
- OpenCV-4.1.1
- CMake-3.10.2

If you have other environment-related issues, please discuss in issue.

## End2End Detection

### 1. Export Detection End2End ONNX

`yolov8s.pt` is your trained pytorch model, or the official pre-trained model.

Do not use any model other than pytorch model.
Do not use [`build.py`](../build.py) to export engine if you don't know how to install pytorch and other environments on
jetson.

***!!! Please use the PC to execute the following script !!!***

```shell
# Export yolov8s.pt to yolov8s.onnx
python3 export-det.py --weights yolov8s.pt --sim
```

***!!! Please use the Jetson to execute the following script !!!***

```shell
# Using trtexec tools for export engine
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s.onnx \
--saveEngine=yolov8s.engine
```

After executing the above command, you will get an engine named `yolov8s.engine` .

### 2. Inference with c++

It is highly recommended to use C++ inference on Jetson.
Here is a demo: [`csrc/jetson/detect`](../csrc/jetson/detect) .

#### Build:

Please modify `CLASS_NAMES` and `COLORS` in [`main.cpp`](../csrc/jetson/detect/main.cpp) for yourself.

And build:

```shell
export root=${PWD}
cd src/jetson/detect
mkdir build
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

## Speedup Segmention

### 1. Export Segmention Speedup ONNX

`yolov8s-seg.pt` is your trained pytorch model, or the official pre-trained model.

Do not use any model other than pytorch model.
Do not use [`build.py`](../build.py) to export engine if you don't know how to install pytorch and other environments on
jetson.

***!!! Please use the PC to execute the following script !!!***

```shell
# Export yolov8s-seg.pt to yolov8s-seg.onnx
python3 export-seg.py --weights yolov8s-seg.pt --sim
```

***!!! Please use the Jetson to execute the following script !!!***

```shell
# Using trtexec tools for export engine
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s-seg.onnx \
--saveEngine=yolov8s-seg.engine
```

After executing the above command, you will get an engine named `yolov8s-seg.engine` .

### 2. Inference with c++

It is highly recommended to use C++ inference on Jetson.
Here is a demo: [`csrc/jetson/segment`](../csrc/jetson/segment) .

#### Build:

Please modify `CLASS_NAMES` and `COLORS` and postprocess parameters in [`main.cpp`](../csrc/jetson/segment/main.cpp) for
yourself.

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
cd src/jetson/segment
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

## Normal Posture

### 1. Export Posture Normal ONNX

`yolov8s-pose.pt` is your trained pytorch model, or the official pre-trained model.

Do not use any model other than pytorch model.
Do not use [`build.py`](../build.py) to export engine if you don't know how to install pytorch and other environments on
jetson.

***!!! Please use the PC to execute the following script !!!***

```shell
# Export yolov8s-pose.pt to yolov8s-pose.onnx
yolo export model=yolov8s-pose.pt format=onnx simplify=True
```

***!!! Please use the Jetson to execute the following script !!!***

```shell
# Using trtexec tools for export engine
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s-pose.onnx \
--saveEngine=yolov8s-pose.engine
```

After executing the above command, you will get an engine named `yolov8s-pose.engine` .

### 2. Inference with c++

It is highly recommended to use C++ inference on Jetson.
Here is a demo: [`csrc/jetson/pose`](../csrc/jetson/pose) .

#### Build:

Please modify `KPS_COLORS` and `SKELETON` and `LIMB_COLORS` and postprocess parameters
in [`main.cpp`](../csrc/jetson/pose/main.cpp) for yourself.

```c++
int topk = 100;
float score_thres = 0.25f;
float iou_thres = 0.65f;
```

And build:

```shell
export root=${PWD}
cd src/jetson/pose
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
