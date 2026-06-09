# YOLOv8-TensorRT

用 `TensorRT` 加速 `YOLOv8` 推理 —— 检测、分割、姿态、旋转框、分类，支持 Python 与 C++。

[English](README.md) | **简体中文**

______________________________________________________________________

[![Python](https://img.shields.io/badge/Python-3.8--3.10-FFD43B?logo=python)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![TensorRT](https://badgen.net/badge/icon/tensorrt?icon=azurepipelines&label)](https://developer.nvidia.com/tensorrt)
[![C++](https://img.shields.io/badge/CPP-14%2F17-yellow)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![License](https://badgen.net/github/license/triple-Mu/YOLOv8-TensorRT)](LICENSE)

## 特性

- **统一的 C++ 核心库**（`libyolov8_core`）：RAII 管理 TensorRT/CUDA 资源、用异常替代 `assert`，TensorRT 版本差异全部收敛到唯一的 `trt_compat` 层。
- **构建自动适配版本**：自动识别 TensorRT（8 ↔ 10/11，含 enterprise 头）与 OpenCV（`≥4.7` 启用类感知 NMS），详见 [docs/Build.md](docs/Build.md)。已在 TensorRT 8.6 / 10.8 / 10.16 / 11.0 上验证。
- **C++14 回退**：C++17 用 `std::filesystem`，否则用内置的 `ghc::filesystem`（`-DCMAKE_CXX_STANDARD=14`）。
- **统一 Python 入口**：`infer.py --task {det,seg,pose,obb,cls} --backend {torch,cudart,pycuda}` 取代原先按任务拆分的脚本；cudart/pycuda 后端现已支持 TensorRT 10。
- 单元测试（pytest + ctest）、`--profile` 逐层耗时报告、`benchmark.py`。

## 目录结构

```
csrc/
├── core/        # libyolov8_core: engine、trt_compat、RAII、前后处理、profiler
├── apps/        # 每个任务一个薄可执行 (detect / segment / pose / obb / cls ...)
├── deepstream/  # DeepStream bbox 解析插件 (可选)
└── tests/       # C++ 单元测试 (ctest)
models/          # Python: engine 构建、后端、compat、labels、各任务处理
data/labels/     # Python 与 C++ 共用的类别名 (coco / imagenet / dota)
infer.py  build.py  export-det.py  export-seg.py  benchmark.py
```

## 环境准备

安装 `CUDA`（≥ 11.4）与 `TensorRT`（≥ 8.4），然后：

```shell
pip install -r requirements.txt
pip install ultralytics            # 导出 ONNX
# 可选的非 torch 推理后端:
pip install cuda-python            # --backend cudart
pip install pycuda                 # --backend pycuda
```

## 工作流

`.pt` → **导出 ONNX** → **构建 engine** → **推理**。

### 1. 导出 ONNX

End2End（NMS 烘焙进图，检测/分割）：

```shell
python export-det.py --weights yolov8s.pt --sim --input-shape 1 3 640 640 \
    --iou-thres 0.65 --conf-thres 0.25 --topk 100 --device cuda:0
python export-seg.py --weights yolov8s-seg.pt --sim --device cuda:0
```

Pose / OBB / Cls（以及不含 NMS 的 "normal" 检测）使用 ultralytics 原生导出，例如 `yolo export model=yolov8s-pose.pt format=onnx opset=11 simplify`。

### 2. 构建 engine

```shell
python build.py --weights yolov8s.onnx --fp16 --device cuda:0
# 或
/path/to/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16
```

### 3. 推理（Python）

```shell
python infer.py --task det  --backend torch  --engine yolov8s.engine     --imgs data --out-dir output
python infer.py --task seg  --backend cudart --engine yolov8s-seg.engine  --imgs data --conf-thres 0.25 --iou-thres 0.65
python infer.py --task pose --backend pycuda --engine yolov8s-pose.engine --imgs data --show
```

| 参数 | 含义 |
| --- | --- |
| `--task` | `det` / `seg` / `pose` / `obb` / `cls` |
| `--backend` | `torch`（PyTorch）、`cudart`（cuda-python）、`pycuda` |
| `--engine` `--imgs` | engine 文件；图片文件或目录 |
| `--show` / `--out-dir` | 弹窗显示，或保存到目录 |
| `--conf-thres` `--iou-thres` | 阈值（seg/pose/obb） |

### 4. 推理（C++）

```shell
cmake -S . -B build -DTensorRT_ROOT=/path/to/TensorRT
cmake --build build -j
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
./build/bin/yolov8_detect yolov8s.engine data/bus.jpg --out-dir output   # --show / --profile
```

目标：`yolov8_detect`、`yolov8_detect_e2e`、`yolov8_seg`、`yolov8_seg_simple`、`yolov8_pose`、`yolov8_obb`、`yolov8_cls`。构建细节、多版本 TensorRT/OpenCV、TensorRT 8 的 cuDNN 依赖与 C++14 回退见 [docs/Build.md](docs/Build.md)。类别名在 `data/labels/*.txt`（用 `--labels` 指定）。

## 性能分析与基准

```shell
./build/bin/yolov8_detect yolov8s.engine data/bus.jpg --profile   # C++ 逐层耗时
python benchmark.py --engine yolov8s.engine --runs 200            # 延迟 / 吞吐
python trt-profile.py --engine yolov8s.engine --device cuda:0     # Python 逐层分析
```

## 开发

```shell
pre-commit install                 # 提交时跑 ruff + clang-format + mdformat
python -m pytest tests/            # Python 单元测试
cmake -S . -B build -DBUILD_TESTS=ON && ctest --test-dir build   # C++ 单元测试
```

## 部署说明

- **DeepStream**：解析插件见 [csrc/deepstream](csrc/deepstream/README.md)（`-DBUILD_DEEPSTREAM=ON`，需 DeepStream SDK）。
- **Jetson**：在设备上用 `-DTensorRT_ROOT` 指向 aarch64 的 TensorRT 直接构建同样的目标，无需单独源码（见 [docs/Build.md](docs/Build.md)）。

## 致谢

内置的第三方代码（ghc::filesystem、TensorRT samples）在 [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) 中标注。基于 [MIT](LICENSE) 许可。
