# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

用 NVIDIA TensorRT 加速 YOLOv8 推理的部署框架，覆盖 5 类任务：检测(det)、分割(seg)、姿态(pose)、旋转框(obb)、分类(cls)。同时提供 Python 与 C++ 两套推理实现，以及 Jetson、DeepStream 部署路径。

## 三阶段核心工作流

整个仓库围绕一条流水线组织，理解它就理解了大部分代码：

```
.pt 权重  ──导出──▶  .onnx  ──构建──▶  .engine  ──推理──▶  结果
          export-*.py      build.py/trtexec       infer-*.py / C++
          (或 ultralytics)  (或 TensorRT API)
```

1. **导出 ONNX**：需要 PyTorch + `ultralytics`。
1. **构建 Engine**：`.onnx → .engine`，可用 `build.py`、`trtexec`，或 TensorRT Python API（`gen_pkl.py` + `build.py`，见 `docs/API-Build.md`）。
1. **推理**：加载 `.engine`，前处理 → 推理 → 后处理 → 可视化。

### 两条部署哲学（关键区分）

- **End2End**：把 BBox 解码 + `EfficientNMS_TRT` 插件**烘焙进 ONNX/engine**，引擎直接输出 4 个张量 `num_dets / bboxes / scores / labels`，推理端无需后处理。**仅检测任务有此模式**。
- **Normal**：使用 ultralytics 标准导出的 ONNX，引擎输出原始特征图，**后处理由消费方完成**（需传 `conf_thres/iou_thres/topk`）。seg/pose/obb/cls 均为 Normal 风格。

## 常用命令

```shell
# 安装依赖（注意：numpy 被钉在 <=1.23.5）
pip install -r requirements.txt
pip install ultralytics

# 1) 导出 End2End ONNX（仅 det 与 seg 有专用脚本）
python3 export-det.py --weights yolov8s.pt --opset 11 --sim --input-shape 1 3 640 640 \
        --iou-thres 0.65 --conf-thres 0.25 --topk 100 --device cuda:0
python3 export-seg.py --weights yolov8s-seg.pt --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0
#   pose/obb/cls 没有导出脚本，用 ultralytics 原生导出，见 docs/Pose.md、docs/Obb.md、docs/Cls.md

# 2) 构建 Engine
python3 build.py --weights yolov8s.onnx --iou-thres 0.65 --conf-thres 0.25 --topk 100 --fp16 --device cuda:0
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16   # 等价替代

# 3) Python 推理（每个任务都有「含 torch」与「无 torch」两个版本）
python3 infer-det.py  --engine yolov8s.engine --imgs data --show --out-dir outputs --device cuda:0
python3 infer-det-without-torch.py --engine yolov8s.engine --imgs data --method cudart   # 或 --method pycuda
#   其余任务：infer-{seg,pose,obb,cls}.py，seg/pose/obb 推理支持 --conf-thres/--iou-thres

# 性能分析
python3 trt-profile.py --engine yolov8s.engine --device cuda:0

# C++ 推理（各任务通用：进入对应目录 cmake + make）
export root=${PWD}
cd csrc/detect/end2end && mkdir -p build && cd build && cmake .. && make && mv yolov8 ${root} && cd ${root}
./yolov8 yolov8s.engine data/bus.jpg        # 单图 / 目录 / 视频均可
```

C++ 各任务目录：`csrc/detect/{end2end,normal}`、`csrc/segment/{normal,simple}`、`csrc/{pose,obb,cls}/normal`、`csrc/jetson/{detect,pose,segment}`、`csrc/deepstream`。

## 代码架构

### Python 模型层 `models/`

- **`engine.py`** — 核心 `TRTModule`（torch 后端）。`models/engine.py:13` 读取 `trt.__version__`，据此在 **TensorRT 8 的 binding API** 与 **TensorRT 10 的 tensor API**（`get_tensor_name`/`execute_async_v3` 等）之间自动切换。还含 `EngineBuilder`（ONNX→engine）。
- **`cudart_api.py` / `pycuda_api.py`** — 两个「不依赖 PyTorch」的推理后端，接口为 NumPy。`cudart` 最轻量，`pycuda` 自动初始化 GPU。被 `infer-*-without-torch.py` 使用。
- **`common.py`** — ONNX 导出时插入的后处理模块（BBox 解码、`EfficientNMS_TRT` 等），即 End2End 的来源。
- **`torch_utils.py`** — 基于 torch 的后处理：`det/seg/pose/obb` 各自的解码与 NMS。
- **`utils.py`** — 前处理（`letterbox` + `blob` 归一化，返回缩放比与 padding 偏移）和 cv2 可视化。坐标逆变换必须**先减 padding 再除缩放比**。

### 入口脚本（仓库根）

`export-det.py`/`export-seg.py`（导出）、`build.py`（构建）、`gen_pkl.py`（API 构建用，提取权重为 pickle）、`infer-*.py` 与 `infer-*-without-torch.py`（推理）、`trt-profile.py`（性能）、`config.py`（配置中心）。

### `config.py`

各任务的类别名与颜色、Pose 17 关键点/骨骼连接、各任务推理配置。注意类别集互不通用：det/seg 为 COCO 80 类、cls 为 ImageNet 1000 类、obb 为 DOTA 遥感 15 类。

### C++ 部署层 `csrc/`

- 每个任务一个目录，结构一致：`CMakeLists.txt` + `cmake/{FindTensorRT,Function}.cmake` + `include/*.hpp` + `main.cpp`。
- **版本兼容**：`CMakeLists.txt` 通过 `FindTensorRT.cmake` 自动检测 `TensorRT_VERSION_MAJOR >= 10` 并 `add_definitions(-DTRT_10)`；头文件中 `#ifdef TRT_10` 切换 `getNbIOTensors`/`enqueueV3`/`setTensorAddress`（TRT10）与 `getNbBindings`/`enqueueV2`（TRT8）。**通常无需手改宏**。
- **`CLASS_NAMES`、`COLORS`（seg/pose 还有 `MASK_COLORS`/`KPS_COLORS`/`SKELETON`）硬编码在各 `main.cpp`**，改类别须改源码并重新编译。
- `csrc/deepstream` 是 DeepStream 自定义 BBox 解析插件（编译为 `.so`），配 `config_yoloV8.txt`。

## 关键约定与易错点

- **阈值烘焙进图**：det End2End 的 `--conf-thres/--iou-thres/--topk` 会写入 ONNX 输出形状与 NMS 插件，**改这些值必须重新导出 ONNX**，不能只在推理端改。
- **TensorRT 版本必须前后一致**：TRT8 编译/序列化的 engine 与 TRT10 不兼容，二进制也不互通。
- **首次推理抖动**：含动态维（-1）的张量会禁用显存预分配，需 `warm_up()` 规避首帧延迟。
- **模型/权重不入库**：`.gitignore` 忽略 `*.pt/*.onnx/*.engine` 及图片、`build/` 等。
- **可选依赖**：`requirements.txt` 中 `tensorrt`/`cuda-python`/`pycuda` 默认注释，按推理后端选装。

## 代码风格（pre-commit）

提交前会跑 `.pre-commit-config.yaml`：

- **Python**：`ruff` lint + format，`line-length=120`，目标 `py310`，规则 `E9/F63/F7/F82/I/UP`。
- **C/C++/CUDA**：`clang-format`（`.clang-format`：C++17、Stroustrup 风格、列宽 120、4 空格缩进、指针左对齐）。
- **Markdown**：`mdformat` 仅作用于 `README.md` 与 `docs/*.md`。
- 通用 hook：去尾空格、统一行尾、`check-ast/json/yaml`、禁止 `debug-statements`、大文件(>2MB)拦截、私钥检测。

本地执行：`pre-commit run --all-files`。
