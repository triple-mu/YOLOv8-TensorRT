# FAQ

Common questions from the issue tracker. See also [docs/Build.md](Build.md) and the [README](../README.md).

## Build & TensorRT versions

**`ICudaEngine has no member getNbBindings` / C++ doesn't work on TensorRT 10/11.**
Fixed — the C++ core has one `trt_compat` layer that supports TensorRT 8 / 10 / 11, and the build auto-detects the version. Rebuild from `main`.

**Windows build fails (`unistd.h`, `cudaMallocAsync` not found, CMake paths).**
`unistd.h` came from the old vendored filesystem header (removed); the core now uses `cudaMalloc` (not `cudaMallocAsync`), which fixes the "cannot locate `cudaMallocAsync`" loader error. Pass `-DTensorRT_ROOT=...`. Linux/Jetson are the primarily tested platforms; Windows is best-effort.

**`deserializeCudaEngine` returns null.**
An `.engine` only loads on the **same** TensorRT version that built it. Rebuild the engine with the TensorRT you link/run against, and keep `LD_LIBRARY_PATH` pointing at that TensorRT's `lib`.

**Which TensorRT under `/data` (or elsewhere) is used?**
`-DTensorRT_ROOT=...` wins; otherwise the build searches `/data/TensorRT-*` then `/usr`. See [docs/Build.md](Build.md).

## Export & engine

**Wrong predictions / export fails / `EfficientNMS_TRT` shape-inference warning on recent PyTorch.**
Fixed — `export-det.py` / `export-seg.py` pass `dynamo=False` to `torch.onnx.export` (PyTorch 2.x's dynamo exporter mishandled the graph). Re-export from `main`.

**`_pickle.UnpicklingError: Weights only load failed` (PyTorch ≥ 2.6).**
PyTorch changed `torch.load(weights_only=True)` by default. Use a matching `ultralytics` version, or `torch.serialization.add_safe_globals([...])` / `weights_only=False` for trusted checkpoints.

**Change conf / iou / topk.**
For **End2End** engines these are baked into the graph at export — re-run `export-det.py --conf-thres ... --iou-thres ... --topk ...` and rebuild. For **raw** engines, pass `--conf-thres/--iou-thres` to `infer.py` at runtime.

**INT8.**
Not built in. Build an INT8 engine with `trtexec --int8` plus a calibration cache, or extend `EngineBuilder` with an `IInt8Calibrator`. FP16 (`--fp16`) is the supported fast path.

**Inference with ONNX Runtime fails (missing `EfficientNMS_TRT`).**
`EfficientNMS_TRT` is a TensorRT plugin, not an ONNX Runtime op. For ORT, export a **raw** model (native ultralytics export) and do NMS yourself.

## Inference

**Batch size > 1.**
Export with a dynamic batch axis and build the engine with an optimization profile that covers your batch; the input is `[N, 3, H, W]`. (The default examples use batch 1.)

**Use a USB camera / video in C++.**
Pass a video file path to the binary (the runner opens it with `cv::VideoCapture`); for a camera, adapt `csrc/core/src/runner.cpp` to open a device index.

**Export box coordinates to a text file.**
The boxes are available after `postprocess`; add a small writer in the runner loop (each `Object` carries `rect`, `label`, `prob`).

**Tracking (ByteTrack, etc.).**
Out of scope — the engine outputs detections; feed them to your tracker of choice.

## Tasks

**Multi-class pose.**
Stock YOLOv8-pose is single-class (person). A custom multi-class pose model outputs `4 + nc + 17*3` channels per anchor; the postprocess takes the top score over the `nc` class channels as the score and its index as the label (`nc = 1` is the usual single-class case).

**Segmentation output is only an axis-aligned box + mask.**
That is the current output; rotated rectangles / polygons would need an extra step (fit a `minAreaRect` / contour on the mask).

**YOLOv11 / YOLOv10.**
YOLOv11 detect/pose use the same head family as v8 and generally work after a normal export. YOLOv10 has an NMS-free head and is not supported as-is.

## Performance

**Preprocess / mask resize is slow.**
`letterbox` and the per-object mask resize are CPU-bound and scale with resolution and object count. Lower the input size, reuse buffers, or move resize to the GPU. Use `--profile` (C++) or `benchmark.py` to find the real bottleneck.

## No detections on a custom model

Check, in order: the class count (`nc`) matches your model; the engine was built from the matching ONNX/TensorRT version; `--conf-thres` isn't too high; and the engine layout matches its consumer (End2End vs raw — see the README). If it still fails, open an issue with the export command, the model's `nc`, and a sample image.
