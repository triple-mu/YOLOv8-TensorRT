# YOLOv8-TensorRT

用 `TensorRT` 加速 `YOLOv8` 推理 —— 检测、分割、姿态、旋转框、分类，**同时支持** Python 与 C++。

[English](README.md) | **简体中文**

______________________________________________________________________

[![Python](https://img.shields.io/badge/Python-3.8--3.10-FFD43B?logo=python)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![TensorRT](https://badgen.net/badge/icon/tensorrt?icon=azurepipelines&label)](https://developer.nvidia.com/tensorrt)
[![C++](https://img.shields.io/badge/CPP-14%2F17-yellow)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![License](https://badgen.net/github/license/triple-Mu/YOLOv8-TensorRT)](LICENSE)

把训练好的 `ultralytics` YOLOv8 模型导出为 ONNX、构建 TensorRT 引擎，然后用 Python 或一个小巧的 C++ 程序运行五类任务中的任意一种。Python 与 C++ 两侧共用同一份引擎与类别文件；构建过程会自动适配你机器上的 TensorRT 与 OpenCV。

## 特性

- **统一的 C++ 核心库**（`libyolov8_core`）：RAII 管理 TensorRT/CUDA 资源、用异常替代 `assert`，TensorRT 版本差异全部集中在*唯一*的 `trt_compat` 层。
- **构建自动适配版本**：自动识别 TensorRT（8 ↔ 10/11，含 enterprise 头）与 OpenCV（`≥4.7` 切换为类感知 NMS），详见 [docs/Build.md](docs/Build.md)。已在 TensorRT **8.6 / 10.8 / 10.16 / 11.0** 与 OpenCV **4.5 / 4.6 / 4.11** 上验证。
- **C++14 回退**：C++17 用 `std::filesystem`，否则用内置的 `ghc::filesystem`（`-DCMAKE_CXX_STANDARD=14`）。
- **统一 Python 入口**：`infer.py --task {det,seg,pose,obb,cls} --backend {torch,cudart,pycuda}` 取代了原先按任务拆分的十个脚本；cudart/pycuda 后端现已支持 TensorRT 10。
- 单元测试（pytest + ctest）、`--profile` 逐层耗时报告、`benchmark.py`。

## 支持的任务

| 任务 | `infer.py --task` | C++ 可执行 | 导出 ONNX |
| --- | --- | --- | --- |
| 检测 | `det` | `yolov8_detect`（raw）· `yolov8_detect_e2e`（End2End） | `export-det.py` 或 ultralytics |
| 分割 | `seg` | `yolov8_seg` · `yolov8_seg_simple` | `export-seg.py` 或 ultralytics |
| 姿态 | `pose` | `yolov8_pose` | ultralytics |
| 旋转框 | `obb` | `yolov8_obb` | ultralytics |
| 分类 | `cls` | `yolov8_cls` | ultralytics |

> **引擎布局**：`export-det.py` 产出 **End2End** 检测引擎（NMS 内置到引擎，输出 `num_dets, bboxes, scores, labels`）；`export-seg.py` 产出分割引擎（输出 `outputs, proto`）；`ultralytics` 原生导出保留模型 **raw** 输出（如 `[1, 84, anchors]`）。消费端必须匹配引擎：`infer.py --task det` 与 `yolov8_detect_e2e` 用 End2End 引擎，`infer.py --task seg` 用 `export-seg.py` 引擎，`yolov8_detect` 与 pose/obb/cls 路径用 raw ultralytics 导出。

## 目录结构

```
csrc/
├── core/        # libyolov8_core: engine、trt_compat、RAII、前后处理、profiler
├── apps/        # 每个任务一个轻量可执行 (detect / segment / pose / obb / cls ...)
├── deepstream/  # DeepStream bbox 解析插件 (可选)
└── tests/       # C++ 单元测试 (ctest)
models/          # Python: engine 构建、后端、compat、labels、各任务处理
data/labels/     # Python 与 C++ 共用的类别名 (coco / imagenet / dota)
infer.py  build.py  export-det.py  export-seg.py  benchmark.py
```

## 环境准备

> 仓库很小；想要最轻的检出可用浅克隆：`git clone --depth 1 <url>`（只取最新代码、不含历史）。

`CUDA`（≥ 11.4）与 `TensorRT`（≥ 8.4）需已在系统中安装 —— `nvidia-smi` 与 `trtexec --version` 都应可用。然后安装 Python 依赖：

```shell
pip install -r requirements.txt
pip install ultralytics            # 导出 ONNX
pip install cuda-python            # 可选: infer.py --backend cudart
pip install pycuda                 # 可选: infer.py --backend pycuda
```

## 工作流

`.pt` → **导出 ONNX** → **构建引擎** → **推理**。

### 1. 导出 ONNX

（`ultralytics` 会在首次使用时自动下载 `yolov8s.pt` 等预训练权重。）

End2End（NMS 内置到引擎 —— 检测 / 分割）：

```shell
python export-det.py --weights yolov8s.pt --sim --input-shape 1 3 640 640 \
    --iou-thres 0.65 --conf-thres 0.25 --topk 100 --device cuda:0
python export-seg.py --weights yolov8s-seg.pt --sim --device cuda:0
```

Raw 导出（pose / obb / cls，以及不内置 NMS 的检测/分割）用 ultralytics：

```shell
yolo export model=yolov8s-pose.pt format=onnx opset=11 simplify
```

### 2. 构建引擎

```shell
python build.py --weights yolov8s.onnx --fp16 --device cuda:0
# 或
/path/to/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16
```

### 3. 推理 —— Python

```shell
python infer.py --task det  --backend torch  --engine yolov8s.engine     --imgs data --out-dir output
python infer.py --task seg  --backend cudart --engine yolov8s-seg.engine  --imgs data --conf-thres 0.25 --iou-thres 0.65
python infer.py --task pose --backend pycuda --engine yolov8s-pose.engine --imgs data --show
```

| 参数 | 含义 |
| --- | --- |
| `--task` | `det` / `seg` / `pose` / `obb` / `cls` |
| `--backend` | `torch`（PyTorch）、`cudart`（cuda-python）、`pycuda` |
| `--engine` `--imgs` | 引擎文件；图片文件或目录 |
| `--show` / `--out-dir` | 弹窗显示，或保存到目录 |
| `--conf-thres` `--iou-thres` | 阈值（seg / pose / obb） |
| `--device` | torch 设备，如 `cuda:0`（torch 后端） |
| `--batch` | 每次引擎调用的图片数（dynamic-batch 引擎） |

> **批量推理**：同一批图片合并为一次引擎调用，再逐图解码。固定 batch 的引擎（如导出时 `batch=2`）按自身 batch 运行；dynamic-batch 引擎按 `--batch N` 运行。单张图片或 batch=1 引擎行为不变。

### 4. 推理 —— C++

```shell
cmake -S . -B build -DTensorRT_ROOT=/path/to/TensorRT
cmake --build build -j
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
./build/bin/yolov8_detect yolov8s.engine data/bus.jpg --out-dir output   # --show / --profile / --labels / --gpu-preprocess
```

`--gpu-preprocess` 用 CUDA kernel 做 letterbox + 归一化（原始 uint8 图直接变成网络输入，无 CPU/OpenCV 步骤）；opt-in，仅在找到 CUDA 编译器时构建。结果与 CPU 路径在容差内一致（GPU 双线性 ≠ `cv::resize`）。

构建细节、多版本 TensorRT/OpenCV、TensorRT 8 的 cuDNN 依赖与 C++14 回退见 **[docs/Build.md](docs/Build.md)**。类别名在 `data/labels/*.txt`（用 `--labels` 覆盖）。

## 性能

`benchmark.py` 经 cudart 后端测量（host 到 host：H2D + execute + D2H），`yolov8n` FP16，640×640（cls 224×224），RTX 3080 Ti Laptop / CUDA 12.8 / TensorRT 10.16：

| 任务 | 延迟(均值) | 吞吐 |
| --- | --- | --- |
| 检测 | 2.46 ms | 406 qps |
| 分割 | 3.43 ms | 292 qps |
| 姿态 | 2.28 ms | 439 qps |
| 旋转框 | 1.97 ms | 507 qps |
| 分类 | 0.33 ms | 3033 qps |

```shell
python benchmark.py --engine yolov8s.engine --runs 200          # 延迟 / 吞吐
./build/bin/yolov8_detect yolov8s.engine data/bus.jpg --profile  # C++ 逐层耗时
python trt-profile.py --engine yolov8s.engine --device cuda:0    # Python 逐层分析
```

## 开发

```shell
pre-commit install                 # 每次提交跑 ruff + clang-format + mdformat
python -m pytest tests/            # Python 单元测试
cmake -S . -B build -DBUILD_TESTS=ON && ctest --test-dir build   # C++ 单元测试
```

## 故障排查

- **运行时 `libnvinfer.so: cannot open shared object file`** —— 把 TensorRT 的 `lib/`（和 `/usr/local/cuda/lib64`）加入 `LD_LIBRARY_PATH`。
- **引擎反序列化失败** —— `.engine` 与构建它的 TensorRT 版本严格绑定；用链接/运行所用的同版本 TensorRT 重新构建。
- **TensorRT 8 链接报 `undefined reference to cudnn*`** —— TensorRT 8 的 plugin 依赖 cuDNN 8；传 `-DCUDNN_ROOT=<dir>`（如 conda `cudnn=8` 环境）并加入 `LD_LIBRARY_PATH`。TensorRT 10+ 已去除该依赖。
- **导出的 ONNX 构出的引擎极小/为空** —— PyTorch 2.x 下给 `torch.onnx.export` 传 `dynamo=False`（`export-det.py` / `export-seg.py` 已设置）。
- **TensorRT 11 的 `trtexec` 报 "model not found"** —— `--onnx=` 用**绝对路径**。
- **无显示环境下 `--show` 无效** —— 去掉 `--show`，用 `--out-dir` 保存标注图。

更多问题（batch、INT8、导出参数、自定义模型等）见 [docs/FAQ.md](docs/FAQ.md)。

## 部署

- **DeepStream** —— bbox 解析插件见 [csrc/deepstream](csrc/deepstream/README.md)；用 `-DBUILD_DEEPSTREAM=ON` 构建（需 DeepStream SDK）。
- **Jetson** —— 在设备上用 `-DTensorRT_ROOT` 指向 aarch64 的 TensorRT 直接构建同样的目标，无需单独源码（见 [docs/Build.md](docs/Build.md)）。

## 致谢

内置的第三方代码（`ghc::filesystem`、TensorRT samples）在 [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) 中标注。基于 [MIT](LICENSE) 许可。

[![Star History](https://api.star-history.com/svg?repos=triple-Mu/YOLOv8-TensorRT&type=Date)](https://star-history.com/#triple-Mu/YOLOv8-TensorRT&Date)
