#pragma once
// Filesystem compatibility: use std::filesystem on C++17+, otherwise fall back to
// the vendored ghc::filesystem (a std::filesystem-compatible implementation for
// C++11/14). Both are exposed as yolov8::fs, so the rest of the code is agnostic.

#if __cplusplus >= 201703L && defined(__has_include) && __has_include(<filesystem>)
#include <filesystem>
namespace yolov8 {
namespace fs = std::filesystem;
}  // namespace yolov8
#else
#include <ghc_filesystem.hpp>  // SYSTEM include dir: csrc/core/include/yolov8/3rdparty
namespace yolov8 {
namespace fs = ghc::filesystem;
}  // namespace yolov8
#endif
