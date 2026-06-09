# Start DeepStream Using the engine build from [`YOLOv8-TensorRT`](https://github.com/triple-Mu/YOLOv8-TensorRT)

## 1. Build you own TensorRT engine from `trtexec` or [`build.py`](https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/build.py)

For example, if you have built an engine named `yolov8s.engine`.

## 2. Compile deepstream plugin

First, modify the [`CMakeLists.txt`](https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/csrc/deepstream/CMakeLists.txt)

```cmake
# Set your own TensorRT path
set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu)
# Set your own DeepStream path
set(DEEPSTREAM /opt/nvidia/deepstream/deepstream)
```

Second, build deepstream plugin

```shell
mkdir build
cd build
cmake ..
make
```

You will get a lib `libnvdsinfer_custom_bbox_yoloV8.so` in `build`.

## 3. Modify the deepstream config

The net config is [`config_yoloV8.txt`](config_yoloV8.txt). Please modify by your own model.

```text
net-scale-factor=0.0039215697906911373                              # the normalize param == 1/255
model-engine-file=./yolov8s.engine                                  # the engine path you build
labelfile-path=./labels.txt                                         # the class name path
num-detected-classes=80                                             # the number of classes
output-blob-names=num_dets;bboxes;scores;labels                     # the model output names
custom-lib-path=./build/libnvdsinfer_custom_bbox_yoloV8.so          # the deepstream plugin you build
```

The deepstream config is [`deepstream_app_config.txt`](deepstream_app_config.txt).

```text
****
[source0]
enable=1
#Type - 1=CameraV4L2 2=URI 3=MultiURI
type=3
uri=file://./sample_1080p_h264.mp4                                  # the video path or stream you want to detect
****
****
config-file=config_yoloV8.txt                                       # the net config path
```

You can get more information from [`deepstream offical`](https://developer.nvidia.com/deepstream-sdk).

## 4. Runing detector !

```shell
deepstream-app -c deepstream_app_config.txt
```
