## YOLOv7 tiny End2End using TensorRT

YOLOv7 tiny using TensorRT accelerate, using [WongKinYiu/yolov7.git](https://github.com/WongKinYiu/yolov7.git)

### Export ONNX with NMS

Pytorch to TensorRT with NMS (and inference)

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```

```shell
python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
```

### Export TensorRT Engine

using [Linaom1214/tensorrt-python.git](https://github.com/Linaom1214/tensorrt-python.git) to export engine

```shell
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

or export by `trtexec` tools.

Usage:

```shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov7-tiny.onnx \
--saveEngine=yolov7-tiny-nms.trt \
--fp16
```

and if on jeston, could set flag `--memPoolSize` lower:

```shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov7-tiny.onnx \
--saveEngine=yolov7-tiny-nms.trt \
--fp16 \
--memPoolSize=workspace:1024MiB
```

### Inference with c++

You can infer with c++ in [`csrc/detect/end2end`](https://github.com/nypyp/YOLOv8-TensorRT/blob/main/csrc/detect/end2end)  

#### Build:

Please set you own librarys in [`CMakeLists.txt`](detect/end2end/CMakeLists.txt) and modify `CLASS_NAMES` and `COLORS` in [`main.cpp`](detect/end2end/main.cpp).

```shell
export root=${PWD}
cd csrc/detect/end2end
mkdir -p build && cd build
cmake ..
make
mv yolov7 ${root}
cd ${root}
```

Usage:

```shell
# infer image
./yolov7 yolov7-tiny-nms.trt data/bus.jpg
# infer images
./yolov8 yolov7-tiny-nms.trt data
# infer video
./yolov8 yolov7-tiny-nms.trt data/test.mp4 # the video path
```