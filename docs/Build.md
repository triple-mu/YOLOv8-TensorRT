# Building the C++ inference targets

The C++ side is one CMake project (`CMakeLists.txt` + `cmake/`) that builds a shared
core library (`libyolov8_core`) and one thin executable per task. The build adapts
itself to whatever TensorRT and OpenCV it finds — you only point it at the libraries.

## One command

```shell
cmake -S . -B build -DTensorRT_ROOT=/data/TensorRT-10.16.1.11
cmake --build build -j
# binaries in build/bin/: yolov8_detect, yolov8_detect_e2e, yolov8_seg,
#                         yolov8_seg_simple, yolov8_pose, yolov8_obb, yolov8_cls
```

At runtime, put the matching TensorRT (and CUDA) libraries on the loader path:

```shell
export LD_LIBRARY_PATH=/data/TensorRT-10.16.1.11/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
./build/bin/yolov8_detect model.engine data/bus.jpg --out-dir output
```

> A `.engine` is tied to the TensorRT version that built it. Build the engine
> (`trtexec` / `build.py`) with the same TensorRT you link the C++ against.

## What the build auto-detects

| Concern            | Handled by                                      | Behaviour                                                                                                                           |
| ------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| TensorRT location  | `cmake/FindTensorRT.cmake`                      | searches `TensorRT_ROOT`, then `/data/TensorRT-*`, then `/usr`                                                                      |
| TensorRT version   | `cmake/FindTensorRT.cmake`                      | parses `NvInferVersion.h`, including the **enterprise** indirection (`NV_TENSORRT_MAJOR → TRT_MAJOR_ENTERPRISE`)                    |
| TRT 8 vs 10 API    | `cmake/TrtDefs.cmake` → `-DTRT_10` (major ≥ 10) | one `trt_compat.hpp` switches binding-index vs tensor-name API, `enqueueV2` vs `enqueueV3`, `destroy()` vs `delete`                 |
| OpenCV ≥ 4.7       | `cmake/TrtDefs.cmake` → `-DBATCHED_NMS`         | uses class-aware `NMSBoxesBatched` instead of class-agnostic `NMSBoxes`                                                             |
| cuDNN for TRT < 10 | top `CMakeLists.txt`                            | finds cuDNN 8 (needed by `libnvinfer_plugin` on TRT 8); links `--allow-shlib-undefined` so a missing dev copy never breaks the link |

## Selecting a TensorRT version

```shell
cmake -S . -B build -DTensorRT_ROOT=/data/TensorRT-8.6.1.6   # or -10.8.0.43 / -11.0.0.114
```

The configure log reports the detected version and whether `-DTRT_10` is enabled.

### TensorRT 8.x needs cuDNN 8

TensorRT 8 plugins depend on `libcudnn.so.8`; TensorRT 10+ dropped that dependency.
Install cuDNN 8 (e.g. via conda) and point the build at it:

```shell
conda create -n cudnn8 -c conda-forge cudnn=8.9.7.29 -y
cmake -S . -B build -DTensorRT_ROOT=/data/TensorRT-8.6.1.6 -DCUDNN_ROOT=$HOME/miniconda3/envs/cudnn8
cmake --build build -j
# runtime also needs cuDNN 8 and TensorRT 8 on the loader path:
export LD_LIBRARY_PATH=/data/TensorRT-8.6.1.6/lib:$HOME/miniconda3/envs/cudnn8/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Selecting an OpenCV version

`find_package(OpenCV)` uses the system OpenCV by default. To test another build
(e.g. a conda one that crosses the 4.7 boundary), pass its CMake config dir:

```shell
conda create -n ocv411 -c conda-forge libopencv=4.11.0 -y
cmake -S . -B build -DOpenCV_DIR=$HOME/miniconda3/envs/ocv411/lib/cmake/opencv4 ...
# runtime: add $HOME/miniconda3/envs/ocv411/lib to LD_LIBRARY_PATH
```

## C++14 fallback (ghc::filesystem)

The default build is C++17 and uses `std::filesystem`. For older toolchains, build
at C++14 — `csrc/core/include/yolov8/fs.hpp` then selects the vendored
`ghc::filesystem` (`csrc/core/include/yolov8/3rdparty/ghc_filesystem.hpp`, MIT)
instead, with no source changes:

```shell
cmake -S . -B build -DTensorRT_ROOT=/path/to/TensorRT -DCMAKE_CXX_STANDARD=14
cmake --build build -j
```

## GPU preprocessing (opt-in)

By default preprocessing (letterbox + BGR→RGB + normalize) runs on the CPU via OpenCV.
Passing `--gpu-preprocess` at runtime instead uploads the raw uint8 image and does the
whole transform in a CUDA kernel, writing the network input directly — no CPU/OpenCV step.

It is compiled only when CMake finds a CUDA compiler (the configure log prints
`GPU preprocessing: enabled`); otherwise `--gpu-preprocess` throws and the CPU path is
unaffected. Set GPU architectures with `-DCMAKE_CUDA_ARCHITECTURES=86` (default
`75;86;89;90`). The kernel's bilinear resize is not bit-identical to `cv::resize`, so
detections match the CPU path within tolerance, not byte-for-byte.

## Custom postprocess plugins (opt-in)

The detection postprocess (decode + NMS) can run inside the engine as a custom
TensorRT plugin (`YoloDetPostprocess`, an alternative to the built-in
`EfficientNMS_TRT`). Build the plugin library — like the engine, it is tied to the
TensorRT version it is built against, so use the same `-DTensorRT_ROOT`:

```shell
cmake -S . -B build -DTensorRT_ROOT=/data/TensorRT-10.16.1.11 -DBUILD_PLUGINS=ON
cmake --build build -j           # -> build/bin/libyolov8_plugins.so
```

Export an ONNX that emits the plugin node, build an engine with the `.so` loaded,
then run — the C++ binary auto-detects the plugin/End2End outputs:

```shell
python export-det.py --weights yolov8s.pt --plugin --input-shape 1 3 640 640
# build with the plugin registered (trtexec --staticPlugins, or build.py with the env var):
trtexec --onnx=$PWD/yolov8s.onnx --saveEngine=$PWD/yolov8s.engine --fp16 \
    --staticPlugins=$PWD/build/bin/libyolov8_plugins.so
export YOLOV8_PLUGIN_LIB=$PWD/build/bin/libyolov8_plugins.so   # so build.py / infer.py / the C++ runtime can load it
./build/bin/yolov8_detect yolov8s.engine data/bus.jpg          # or pass --plugin-lib <path>
```

The C++ runtime loads the `.so` from `--plugin-lib` or `$YOLOV8_PLUGIN_LIB` before
deserializing. The plugin uses cub (not thrust) for its NMS sort, so it is safe under
CUDA-graph capture. `EfficientNMS_TRT` remains the default; the plugin is opt-in.

## Tests

```shell
cmake -S . -B build -DTensorRT_ROOT=/path/to/TensorRT -DBUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure   # self-contained core tests, no gtest needed
```

## Verified matrix (RTX 3080 Ti, CUDA 12.8)

Same `data/bus.jpg` detections (within fp16 tolerance) across every combination:

- TensorRT **8.6.1.6** (TRT 8 path), **10.8.0.43**, **10.16.1.11** (enterprise), **11.0.0.114** (enterprise)
- OpenCV **4.5.4** (system), **4.6.0** (conda, no BATCHED_NMS), **4.11.0** (conda, BATCHED_NMS)

> TensorRT 11.0's `trtexec` rejects relative `--onnx` paths — pass an absolute path.
