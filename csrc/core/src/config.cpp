#include "yolov8/config.hpp"
#include "yolov8/check.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace yolov8 {

namespace {

[[noreturn]] void print_usage_and_exit(const char* prog, const InferConfig& d, int code)
{
    std::cout << "Usage: " << prog << " <engine> <source> [options]\n"
              << "  <engine>          path to a serialized TensorRT engine\n"
              << "  <source>          image file, image directory, or video file\n"
              << "Options:\n"
              << "  --size <n>        square input size (default " << d.input_size.width << ")\n"
              << "  --score <f>       score threshold (default " << d.score_thres << ")\n"
              << "  --iou <f>         NMS IoU threshold (default " << d.iou_thres << ")\n"
              << "  --topk <n>        max detections (default " << d.topk << ")\n"
              << "  --labels <path>   class-names file, one per line (default: built-in COCO)\n"
              << "  --seg-channels <n> mask prototype channels (default " << d.seg_channels << ")\n"
              << "  --seg-hw <h> <w>  mask prototype size (default " << d.seg_h << " " << d.seg_w << ")\n"
              << "  --out-dir <dir>   output directory (default \"" << d.out_dir << "\")\n"
              << "  --show            display results in a window instead of saving\n"
              << "  --profile         print a per-layer timing report\n"
              << "  --gpu-preprocess  preprocess on the GPU (CUDA kernel) instead of CPU/OpenCV\n"
              << "  --no-warmup       skip warmup iterations\n"
              << "  -h, --help        show this help\n";
    std::exit(code);
}

float to_float(const char* s, const char* opt)
{
    char* end = nullptr;
    float v   = std::strtof(s, &end);
    TRT_CHECK(end != s && *end == '\0');
    (void)opt;
    return v;
}

int to_int(const char* s, const char* opt)
{
    char* end = nullptr;
    long  v   = std::strtol(s, &end, 10);
    TRT_CHECK(end != s && *end == '\0');
    (void)opt;
    return static_cast<int>(v);
}

}  // namespace

CliArgs parse_args(int argc, char** argv, const InferConfig& defaults)
{
    CliArgs args;
    args.config = defaults;
    std::vector<std::string> positionals;

    for (int i = 1; i < argc; ++i) {
        const std::string a    = argv[i];
        auto              next = [&](const char* opt) -> const char* {
            TRT_CHECK(i + 1 < argc);
            (void)opt;
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            print_usage_and_exit(argv[0], defaults, 0);
        }
        else if (a == "--size") {
            const int n            = to_int(next("--size"), "--size");
            args.config.input_size = cv::Size(n, n);
        }
        else if (a == "--score") {
            args.config.score_thres = to_float(next("--score"), "--score");
        }
        else if (a == "--iou") {
            args.config.iou_thres = to_float(next("--iou"), "--iou");
        }
        else if (a == "--topk") {
            args.config.topk = to_int(next("--topk"), "--topk");
        }
        else if (a == "--labels") {
            args.config.labels_path = next("--labels");
        }
        else if (a == "--seg-channels") {
            args.config.seg_channels = to_int(next("--seg-channels"), "--seg-channels");
        }
        else if (a == "--seg-hw") {
            args.config.seg_h = to_int(next("--seg-hw"), "--seg-hw");
            args.config.seg_w = to_int(next("--seg-hw"), "--seg-hw");
        }
        else if (a == "--out-dir") {
            args.config.out_dir = next("--out-dir");
        }
        else if (a == "--show") {
            args.config.show = true;
        }
        else if (a == "--profile") {
            args.config.profile = true;
        }
        else if (a == "--gpu-preprocess") {
            args.config.gpu_preprocess = true;
        }
        else if (a == "--no-warmup") {
            args.config.warmup = false;
        }
        else if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage_and_exit(argv[0], defaults, 1);
        }
        else {
            positionals.push_back(a);
        }
    }

    if (positionals.size() != 2) {
        print_usage_and_exit(argv[0], defaults, 1);
    }
    args.engine = positionals[0];
    args.source = positionals[1];
    return args;
}

}  // namespace yolov8
