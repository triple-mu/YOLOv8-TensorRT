# Acknowledgments

This project bundles or adapts the following third-party work.

## gulrak/filesystem (ghc::filesystem)

- File: `csrc/core/include/yolov8/3rdparty/ghc_filesystem.hpp` (vendored, v1.5.14)
- Source: https://github.com/gulrak/filesystem
- License: MIT — © 2018 Steffen Schümann
- Use: `std::filesystem`-compatible fallback for pre-C++17 toolchains, selected at
  compile time by `csrc/core/include/yolov8/fs.hpp`.

## NVIDIA TensorRT OSS samples

- Adapted into: `csrc/core/include/yolov8/profiler.hpp` + `csrc/core/src/profiler.cpp`
  (per-layer `IProfiler`, modeled on `samples/common/common.h` `SimpleProfiler`).
- Source: https://github.com/NVIDIA/TensorRT (`samples/common`)
- License: Apache-2.0 — © NVIDIA Corporation
