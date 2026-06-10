# YOLOv8-TensorRT

`YOLOv8` inference accelerated with `TensorRT` — detection, segmentation, pose, oriented boxes and classification, from Python **and** C++.

**English** | [简体中文](README.zh-CN.md)

______________________________________________________________________

[![Python](https://img.shields.io/badge/Python-3.8--3.10-FFD43B?logo=python)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![TensorRT](https://badgen.net/badge/icon/tensorrt?icon=azurepipelines&label)](https://developer.nvidia.com/tensorrt)
[![C++](https://img.shields.io/badge/CPP-14%2F17-yellow)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![License](https://badgen.net/github/license/triple-Mu/YOLOv8-TensorRT)](LICENSE)

Take a trained `ultralytics` YOLOv8 model, export it to ONNX, build a TensorRT engine, and run it from Python or a small C++ binary — for any of the five tasks. The Python and C++ sides share the same engines and class files; the build adapts itself to whatever TensorRT and OpenCV you have.

## Highlights

- **One shared C++ core** (`libyolov8_core`): RAII-managed TensorRT/CUDA resources, exceptions instead of `assert`, and a single `trt_compat` layer that is the *only* place branching on the TensorRT version.
- **Version-agnostic build**: auto-detects TensorRT (8 ↔ 10/11, including enterprise headers) and OpenCV (`≥4.7` switches to class-aware NMS); see [docs/Build.md](docs/Build.md). Verified on TensorRT **8.6 / 10.8 / 10.16 / 11.0** and OpenCV **4.5 / 4.6 / 4.11**.
- **C++14 fallback**: `std::filesystem` on C++17, otherwise a vendored `ghc::filesystem` (`-DCMAKE_CXX_STANDARD=14`).
- **One Python entry point**: `infer.py --task {det,seg,pose,obb,cls} --backend {torch,cudart,pycuda}` replaces ten per-task scripts; the cudart/pycuda backends now run on TensorRT 10.
- Unit tests (pytest + ctest), a `--profile` per-layer report, and `benchmark.py`.

## Supported tasks

| Task           | `infer.py --task` | C++ binary                                            | Export (ONNX)                  |
| -------------- | ----------------- | ----------------------------------------------------- | ------------------------------ |
| Detection      | `det`             | `yolov8_detect` (raw) · `yolov8_detect_e2e` (End2End) | `export-det.py` or ultralytics |
| Segmentation   | `seg`             | `yolov8_seg` · `yolov8_seg_simple`                    | `export-seg.py` or ultralytics |
| Pose           | `pose`            | `yolov8_pose`                                         | ultralytics                    |
| Oriented boxes | `obb`             | `yolov8_obb`                                          | ultralytics                    |
| Classification | `cls`             | `yolov8_cls`                                          | ultralytics                    |

> **Engine layouts.** `export-det.py` produces an **End2End** detection engine with NMS built in (outputs `num_dets, bboxes, scores, labels`); `export-seg.py` produces a segmentation engine (outputs `outputs, proto`); the native `ultralytics` export keeps the model's **raw** output (e.g. `[1, 84, anchors]`). Match the engine to its consumer: `infer.py --task det` and `yolov8_detect_e2e` need the End2End engine, `infer.py --task seg` needs the `export-seg.py` engine, while `yolov8_detect` and the pose/obb/cls paths take the raw ultralytics export.

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

> The repo is small; for the lightest checkout use a shallow clone: `git clone --depth 1 <url>` (latest code only, no history).

`CUDA` (≥ 11.4) and `TensorRT` (≥ 8.4) must already be installed system-wide — `nvidia-smi` and `trtexec --version` should both work. Then install the Python deps:

```shell
pip install -r requirements.txt
pip install ultralytics            # ONNX export
pip install cuda-python            # optional: infer.py --backend cudart
pip install pycuda                 # optional: infer.py --backend pycuda
```

## Workflow

`.pt` → **export ONNX** → **build engine** → **infer**.

### 1. Export ONNX

(`ultralytics` downloads pretrained weights such as `yolov8s.pt` automatically on first use.)

End2End (NMS built in — detection / segmentation):

```shell
python export-det.py --weights yolov8s.pt --sim --input-shape 1 3 640 640 \
    --iou-thres 0.65 --conf-thres 0.25 --topk 100 --device cuda:0
python export-seg.py --weights yolov8s-seg.pt --sim --device cuda:0
```

Raw export (pose / obb / cls, and detection/segmentation without built-in NMS) uses ultralytics:

```shell
yolo export model=yolov8s-pose.pt format=onnx opset=11 simplify
```

### 2. Build the engine

```shell
python build.py --weights yolov8s.onnx --fp16 --device cuda:0
# or
/path/to/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16
```

### 3. Inference — Python

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
| `--show` / `--out-dir`       | display in a window, or save to a directory         |
| `--conf-thres` `--iou-thres` | thresholds (seg / pose / obb)                       |
| `--device`                   | torch device, e.g. `cuda:0` (torch backend)         |
| `--batch`                    | images per engine call (dynamic-batch engines)      |

> **Batched inference.** Images are run in one engine call per batch and decoded
> per image. A fixed-batch engine (e.g. exported with `batch=2`) is driven at its
> own batch size; a dynamic-batch engine follows `--batch N`. With a single image
> or a batch-1 engine the behaviour is unchanged.

### 4. Inference — C++

```shell
cmake -S . -B build -DTensorRT_ROOT=/path/to/TensorRT
cmake --build build -j
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
./build/bin/yolov8_detect yolov8s.engine data/bus.jpg --out-dir output   # --show / --profile / --labels
```

Build details, multiple TensorRT/OpenCV versions, cuDNN for TensorRT 8 and the C++14 fallback are in **[docs/Build.md](docs/Build.md)**. Class names live in `data/labels/*.txt` (override with `--labels`).

## Performance

`benchmark.py` over the cudart backend (host-to-host: H2D + execute + D2H), `yolov8n` FP16, 640×640 (cls 224×224), on an RTX 3080 Ti Laptop / CUDA 12.8 / TensorRT 10.16:

| Task           | latency (mean) | throughput |
| -------------- | -------------- | ---------- |
| Detection      | 2.46 ms        | 406 qps    |
| Segmentation   | 3.43 ms        | 292 qps    |
| Pose           | 2.28 ms        | 439 qps    |
| Oriented boxes | 1.97 ms        | 507 qps    |
| Classification | 0.33 ms        | 3033 qps   |

```shell
python benchmark.py --engine yolov8s.engine --runs 200          # latency / throughput
./build/bin/yolov8_detect yolov8s.engine data/bus.jpg --profile  # per-layer C++ timing
python trt-profile.py --engine yolov8s.engine --device cuda:0    # Python layer profile
```

## Development

```shell
pre-commit install                 # ruff + clang-format + mdformat run on every commit
python -m pytest tests/            # Python unit tests
cmake -S . -B build -DBUILD_TESTS=ON && ctest --test-dir build   # C++ unit tests
```

## Troubleshooting

- **`libnvinfer.so: cannot open shared object file`** at runtime — add the TensorRT `lib/` (and `/usr/local/cuda/lib64`) to `LD_LIBRARY_PATH`.
- **Engine fails to deserialize** — a `.engine` is tied to the exact TensorRT version that built it; rebuild it with the same TensorRT you link/run against.
- **TensorRT 8 link error `undefined reference to cudnn*`** — TensorRT 8 plugins need cuDNN 8; pass `-DCUDNN_ROOT=<dir>` (e.g. a conda `cudnn=8` env) and put it on `LD_LIBRARY_PATH`. TensorRT 10+ dropped this dependency.
- **ONNX export gives a tiny / empty engine** — on PyTorch 2.x pass `dynamo=False` to `torch.onnx.export` (already set in `export-det.py` / `export-seg.py`).
- **TensorRT 11 `trtexec` says "model not found"** — pass an **absolute** `--onnx=` path.
- **`--show` does nothing on a headless box** — drop `--show` and use `--out-dir` to save annotated images.

More questions (batch, INT8, export tweaks, custom models, …) are answered in [docs/FAQ.md](docs/FAQ.md).

## Deployment

- **DeepStream** — bbox parser plugin in [csrc/deepstream](csrc/deepstream/README.md); build with `-DBUILD_DEEPSTREAM=ON` (needs the DeepStream SDK).
- **Jetson** — build the same targets on-device with `-DTensorRT_ROOT` pointing at the aarch64 TensorRT; no separate sources (see [docs/Build.md](docs/Build.md)).

## Acknowledgments

Bundled third-party code (`ghc::filesystem`, TensorRT samples) is credited in [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md). Licensed under [MIT](LICENSE).

[![Star History](https://api.star-history.com/svg?repos=triple-Mu/YOLOv8-TensorRT&type=Date)](https://star-history.com/#triple-Mu/YOLOv8-TensorRT&Date)
