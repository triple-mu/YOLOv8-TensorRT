# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

用 NVIDIA TensorRT 加速 YOLOv8 推理的部署框架，覆盖 5 类任务：检测(det)、分割(seg)、姿态(pose)、旋转框(obb)、分类(cls)。Python 与 C++ 两套推理实现，构建系统自动适配多版本 TensorRT(8/10/11) 与 OpenCV。

## 三阶段核心工作流

```
.pt 权重  ──导出──▶  .onnx  ──构建──▶  .engine  ──推理──▶  结果
          export-*.py /      build.py /            infer.py / C++ build/bin
          ultralytics        trtexec
```

- **End2End vs Normal**：End2End 把 NMS(`EfficientNMS_TRT`) 内置到引擎，直出 `num_dets/bboxes/scores/labels`（仅 det/seg 的 `export-*.py` 走此路）；Normal 是 ultralytics 原生 ONNX，后处理在消费端做。
- `export-det.py`/`export-seg.py` 用 `torch.onnx.export(..., dynamo=False)`（torch 2.x 的 dynamo 导出器会丢主干）。

## 常用命令

```shell
# Python 推理（统一入口，替代了旧的 infer-*.py）
python infer.py --task {det,seg,pose,obb,cls} --backend {torch,cudart,pycuda} \
    --engine model.engine --imgs data --out-dir output [--show --conf-thres --iou-thres --device]

# C++ 构建（一条命令，产物在 build/bin/yolov8_*）
cmake -S . -B build -DTensorRT_ROOT=/path/to/TensorRT
cmake --build build -j
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
./build/bin/yolov8_detect model.engine data/bus.jpg --out-dir output   # --show / --profile

# 开发：格式化、测试、基准
pre-commit install && pre-commit run --all-files       # ruff + clang-format + mdformat
python -m pytest tests/                                # Python 单测
cmake -S . -B build -DBUILD_TESTS=ON && ctest --test-dir build   # C++ 单测
python benchmark.py --engine model.engine --runs 200
```

C++ 目标：`yolov8_detect / _detect_e2e / _seg / _seg_simple / _pose / _obb / _cls`（源文件在 `csrc/apps/*.cpp`）。

## 代码架构

### C++ 核心库 `csrc/core/`（libyolov8_core）

每个任务（`csrc/apps/<task>.cpp`）是继承 `Engine` 的子类，只实现 `postprocess()`/`draw()`；公共流程全在 core：

- **`trt_compat.hpp`** — **全仓库唯一**带 `#ifdef TRT_10` 的文件，统一 TRT 8/10 的 binding-index vs tensor-name API（`enqueueV2`↔`enqueueV3`、`destroy()`↔`delete`）。
- **`trt_raii.hpp`** — `TrtUniquePtr`/`CudaStream`/`GenericBuffer` 管理 TRT/CUDA 资源（替代裸指针+手动释放）。
- **`check.hpp`** — `CUDA_CHECK`/`TRT_CHECK` 抛 `TrtException`（替代 Release 下失效的 `assert`）。
- **`engine.hpp/cpp`** — 基类：加载/binding 枚举(经 compat)/`make_pipe`/`copy_from_mat`(letterbox)/`infer`。
- `fs.hpp`（std::filesystem 或 `3rdparty/ghc_filesystem.hpp` 回退）、`labels`/`config`/`preprocess`/`draw`/`profiler`/`runner`。

### 构建系统（顶层 `CMakeLists.txt` + `cmake/`）

- `cmake/FindTensorRT.cmake` 搜 `TensorRT_ROOT → /data/TensorRT-* → /usr`，解析数字与 **enterprise 间接版本头**(`NV_TENSORRT_MAJOR→TRT_MAJOR_ENTERPRISE`)。
- `cmake/TrtDefs.cmake` 据版本设 `-DTRT_10`(TRT≥10) 与 `-DBATCHED_NMS`(OpenCV≥4.7)。
- TRT\<10 的 plugin 依赖 cuDNN 8（`-DCUDNN_ROOT`，并用 `--allow-shlib-undefined` 容错）。
- 默认 C++17；`-DCMAKE_CXX_STANDARD=14` 走 ghc 回退。`-DBUILD_TESTS=ON` 启用 ctest。详见 `docs/Build.md`。

### Python `models/`

- `engine.py` — `TRTModule`(torch 后端，已版本自适应) + `EngineBuilder`(ONNX→engine)。
- `compat.py` — TRT 8/10 版本助手（`io_names`/`np_dtype`/`engine_shape`）。
- `backend.py` — `Backend` 基类 + `CudartBackend`/`PycudaBackend`（统一三后端的非 torch 路径，**新增了 cudart/pycuda 原先缺失的 TRT10 支持**）；`cudart_api.py`/`pycuda_api.py` 是其别名。
- `labels.py` — 从 `data/labels/*.txt` 读类别 + 颜色/骨架；`config.py` 重导出它（向后兼容）。
- `tasks/{det,seg,pose,obb,cls}.py` — 各任务 `process()`，复用 `utils.py`(numpy) 与 `torch_utils.py`(torch) 的纯函数。
- `infer.py`(根) 统一入口；`utils.py` 纯函数(letterbox/blob/NMS/postprocess) 是单测对象。

## 关键约定与易错点

- **阈值写入引擎**：det End2End 的 conf/iou/topk 写入 ONNX，改值须重新导出。
- **TensorRT 版本前后一致**：engine 与链接/运行的 TRT 必须同版本；engine 不跨版本通用。
- **运行时 `LD_LIBRARY_PATH`** 必须含所链接 TRT 的 lib（TRT8 还需 cuDNN 8）。
- **TRT 11 的 trtexec** 不接受相对 `--onnx` 路径，用绝对路径。
- 权重/模型(`*.pt/*.onnx/*.engine`)、`build*/`、`work/` 均被 .gitignore 忽略。

## 代码风格（pre-commit，提交时自动跑）

- Python：`ruff` lint+format（line-length=120, py310, 规则 E9/F63/F7/F82/I/UP）。
- C/C++/CUDA：`clang-format`（`.clang-format`：C++17、Stroustrup、列宽 120、4 空格），覆盖 `csrc/`（vendored `3rdparty/` 除外）。
- Markdown：`mdformat`（README\*.md、docs/、CLAUDE.md 等）。
