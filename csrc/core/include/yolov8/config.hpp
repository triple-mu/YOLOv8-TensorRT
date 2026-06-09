#pragma once
#include "opencv2/core.hpp"
#include <string>

namespace yolov8 {

// Runtime knobs that used to be hard-coded in every main.cpp. Populated from the CLI.
struct InferConfig {
    cv::Size    input_size{640, 640};
    float       score_thres  = 0.25f;
    float       iou_thres    = 0.65f;
    int         topk         = 100;
    int         num_labels   = 80;
    int         seg_channels = 32;  // segmentation prototype channels
    int         seg_h        = 160;
    int         seg_w        = 160;
    std::string labels_path;  // empty -> built-in COCO names
    bool        warmup  = true;
    bool        show    = false;  // off by default (works headless); otherwise save
    bool        profile = false;  // attach a per-layer IProfiler and print a report
    std::string out_dir = "output";
};

struct CliArgs {
    std::string engine;
    std::string source;  // image file, directory, or video
    InferConfig config;
};

// Parses `<engine> <source> [options]`. Throws TrtException on bad input; prints
// usage and exits(0) on -h/--help. `defaults` seeds task-specific defaults.
CliArgs parse_args(int argc, char** argv, const InferConfig& defaults = {});

}  // namespace yolov8
