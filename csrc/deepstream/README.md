# DeepStream deployment

Run a YOLOv8 detection engine built with [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT) inside NVIDIA DeepStream, using the custom bbox parser in this directory.

> Requires the **DeepStream SDK** and an **End2End** engine (built from `export-det.py`, output names `num_dets;bboxes;scores;labels`).

## 1. Build a TensorRT engine

See the project [README](../../README.md): export an End2End ONNX with `export-det.py`, then build it with `build.py` or `trtexec`, e.g. `yolov8s.engine`.

## 2. Build the parser plugin

The plugin is a standalone CMake project. Point it at your TensorRT and DeepStream installs (defaults shown):

```shell
cd csrc/deepstream
cmake -S . -B build \
    -DTensorRT_INCLUDE_DIRS=/usr/include/x86_64-linux-gnu \
    -DTensorRT_LIBRARIES=/usr/lib/x86_64-linux-gnu \
    -DDEEPSTREAM=/opt/nvidia/deepstream/deepstream
cmake --build build -j
# -> build/libnvdsinfer_custom_bbox_yoloV8.so
```

It is also wired into the top-level build via `cmake -S . -B build -DBUILD_DEEPSTREAM=ON` (off by default; needs the SDK).

## 3. Configure DeepStream

Edit [`config_yoloV8.txt`](config_yoloV8.txt) for your model:

```text
net-scale-factor=0.0039215697906911373                       # 1/255 normalization
model-engine-file=./yolov8s.engine                           # the engine you built
labelfile-path=./labels.txt                                  # class-name file
num-detected-classes=80
output-blob-names=num_dets;bboxes;scores;labels              # must match the engine outputs
custom-lib-path=./build/libnvdsinfer_custom_bbox_yoloV8.so   # the plugin built above
```

Set the input source in [`deepstream_app_config.txt`](deepstream_app_config.txt):

```text
[source0]
enable=1
type=3                                                       # 1=CameraV4L2 2=URI 3=MultiURI
uri=file://./sample_1080p_h264.mp4                           # video file or stream
...
config-file=config_yoloV8.txt
```

More options: [DeepStream SDK docs](https://developer.nvidia.com/deepstream-sdk).

## 4. Run

```shell
deepstream-app -c deepstream_app_config.txt
```
