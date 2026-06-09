# YOLOv8-TensorRT

`YOLOv8` inference accelerated with `TensorRT` — detection, segmentation, pose, oriented boxes and classification, from Python and C++.

**English** | [简体中文](README.zh-CN.md)

______________________________________________________________________

[![Python](https://img.shields.io/badge/Python-3.8--3.10-FFD43B?logo=python)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![TensorRT](https://badgen.net/badge/icon/tensorrt?icon=azurepipelines&label)](https://developer.nvidia.com/tensorrt)
[![C++](https://img.shields.io/badge/CPP-14%2F17-yellow)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![License](https://badgen.net/github/license/triple-Mu/YOLOv8-TensorRT)](LICENSE)

## Highlights

- **One shared C++ core** (`libyolov8_core`): RAII-managed TensorRT/CUDA resources, exceptions instead of `assert`, and a single `trt_compat` layer that is the only place branching on the TensorRT version.
- **Version-agnostic build**: auto-detects TensorRT (8 ↔ 10/11, including enterprise headers) and OpenCV (`≥4.7` enables class-aware NMS); see [docs/Build.md](docs/Build.md). Verified on TensorRT 8.6 / 10.8 / 10.16 / 11.0.
- **C++14 fallback**: uses `std::filesystem` on C++17, otherwise a vendored `ghc::filesystem` (build with `-DCMAKE_CXX_STANDARD=14`).
- **One Python entry point**: `infer.py --task {det,seg,pose,obb,cls} --backend {torch,cudart,pycuda}` replaces the old per-task scripts; the cudart/pycuda backends now run on TensorRT 10.
- Unit tests (pytest + ctest), a `--profile` per-layer report and a `benchmark.py`.

## Layout

```
csrc/
├── core/        # libyolov8_core: engine, trt_compat, RAII, pre/post-process, profiler
├── apps/        # one thin executable per task (detect / segment / pose / obb / cls ...)
├── deepstream/  # DeepStream bbox parser plugin (optional)
└── tests/       # C++ unit tests (ctest)
models/          # Python: engine builder, backends, compat, labels, per-task handlers
data/labels/     # class names shared by Python and C++ (coco / imagenet / dota)
infer.py  build.py  export-det.py  export-seg.py  benchmark.py
```

## Setup

Install `CUDA` (≥ 11.4) and `TensorRT` (≥ 8.4), then:

```shell
pip install -r requirements.txt
pip install ultralytics            # for ONNX export
# optional non-torch inference backends:
pip install cuda-python            # --backend cudart
pip install pycuda                 # --backend pycuda
```

## Workflow

`.pt` → **export ONNX** → **build engine** → **infer**.

### 1. Export ONNX

End2End (NMS baked in, detection/segmentation):

```shell
python export-det.py --weights yolov8s.pt --sim --input-shape 1 3 640 640 \
    --iou-thres 0.65 --conf-thres 0.25 --topk 100 --device cuda:0
python export-seg.py --weights yolov8s-seg.pt --sim --device cuda:0
```

Pose / OBB / Cls (and "normal" detection without baked-in NMS) use the native ultralytics export, e.g. `yolo export model=yolov8s-pose.pt format=onnx opset=11 simplify`.

### 2. Build the engine

```shell
python build.py --weights yolov8s.onnx --fp16 --device cuda:0
# or
/path/to/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16
```

### 3. Inference (Python)

```shell
python infer.py --task det  --backend torch  --engine yolov8s.engine     --imgs data --out-dir output
python infer.py --task seg  --backend cudart --engine yolov8s-seg.engine  --imgs data --conf-thres 0.25 --iou-thres 0.65
python infer.py --task pose --backend pycuda --engine yolov8s-pose.engine --imgs data --show
```

| flag                         | meaning                                             |
| ---------------------------- | --------------------------------------------------- |
| `--task`                     | `det` / `seg` / `pose` / `obb` / `cls`              |
| `--backend`                  | `torch` (PyTorch), `cudart` (cuda-python), `pycuda` |
| `--engine` `--imgs`          | engine file; image file or directory                |
| `--show` / `--out-dir`       | display window, or save to a directory              |
| `--conf-thres` `--iou-thres` | thresholds (seg/pose/obb)                           |

### 4. Inference (C++)

```shell
cmake -S . -B build -DTensorRT_ROOT=/path/to/TensorRT
cmake --build build -j
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
./build/bin/yolov8_detect yolov8s.engine data/bus.jpg --out-dir output   # --show / --profile
```

Targets: `yolov8_detect`, `yolov8_detect_e2e`, `yolov8_seg`, `yolov8_seg_simple`, `yolov8_pose`, `yolov8_obb`, `yolov8_cls`. Build details, multiple TensorRT/OpenCV versions, cuDNN for TensorRT 8 and the C++14 fallback are in [docs/Build.md](docs/Build.md). Class names live in `data/labels/*.txt` (pass `--labels`).

## Profiling & benchmark

```shell
./build/bin/yolov8_detect yolov8s.engine data/bus.jpg --profile   # per-layer C++ timing
python benchmark.py --engine yolov8s.engine --runs 200            # latency / throughput
python trt-profile.py --engine yolov8s.engine --device cuda:0     # Python layer profile
```

## Development

```shell
pre-commit install                 # ruff + clang-format + mdformat on commit
python -m pytest tests/            # Python unit tests
cmake -S . -B build -DBUILD_TESTS=ON && ctest --test-dir build   # C++ unit tests
```

## Deployment notes

- **DeepStream**: parser plugin in [csrc/deepstream](csrc/deepstream/README.md) (`-DBUILD_DEEPSTREAM=ON`, needs the DeepStream SDK).
- **Jetson**: build the same targets on-device with `-DTensorRT_ROOT` pointing at the aarch64 TensorRT — no separate sources needed (see [docs/Build.md](docs/Build.md)).

## Acknowledgments

Bundled third-party code (ghc::filesystem, TensorRT samples) is credited in [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md). Licensed under [MIT](LICENSE).
