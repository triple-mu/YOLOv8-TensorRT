#pragma once
#include "yolov8/config.hpp"
#include "yolov8/engine.hpp"
#include <string>
#include <vector>

namespace yolov8 {

// Expands a path into a list of image files. A single image yields one entry; a
// directory is globbed for images; a video path yields an empty list with is_video=true.
std::vector<std::string> collect_sources(const std::string& path, bool& is_video);

// Drives `engine` over the CLI source (image / directory / video): preprocess ->
// infer -> postprocess -> draw, then show or save per args.config. Returns process code.
int run(Engine& engine, const CliArgs& args);

}  // namespace yolov8
