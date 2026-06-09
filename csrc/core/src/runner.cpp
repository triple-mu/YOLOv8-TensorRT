#include "yolov8/runner.hpp"
#include "opencv2/opencv.hpp"
#include "yolov8/fs.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace fs = yolov8::fs;

namespace yolov8 {

namespace {

bool has_suffix(const std::string& ext, std::initializer_list<const char*> suffixes)
{
    return std::any_of(suffixes.begin(), suffixes.end(), [&](const char* s) { return ext == s; });
}

}  // namespace

std::vector<std::string> collect_sources(const std::string& path, bool& is_video)
{
    is_video = false;
    std::vector<std::string> images;
    const fs::path           p{path};

    if (fs::is_directory(p)) {
        for (const auto& entry : fs::directory_iterator(p)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (has_suffix(ext, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})) {
                images.push_back(entry.path().string());
            }
        }
        std::sort(images.begin(), images.end());
    }
    else if (fs::exists(p)) {
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (has_suffix(ext, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})) {
            images.push_back(path);
        }
        else if (has_suffix(ext, {".mp4", ".avi", ".m4v", ".mpeg", ".mov", ".mkv"})) {
            is_video = true;
        }
        else {
            throw TrtException("unsupported source extension: " + ext);
        }
    }
    else {
        throw TrtException("source does not exist: " + path);
    }
    return images;
}

namespace {

void emit(const cv::Mat& res, const std::string& name, const InferConfig& cfg)
{
    if (cfg.show) {
        cv::imshow("result", res);
        cv::waitKey(0);
    }
    else {
        fs::create_directories(cfg.out_dir);
        const std::string out = (fs::path(cfg.out_dir) / fs::path(name).filename()).string();
        cv::imwrite(out, res);
        std::cout << "saved " << out << "\n";
    }
}

double infer_once(Engine& engine, const cv::Mat& image, cv::Mat& res, std::vector<Object>& objs)
{
    objs.clear();
    engine.copy_from_mat(image);
    const auto start = std::chrono::steady_clock::now();
    engine.infer();
    const auto end = std::chrono::steady_clock::now();
    engine.postprocess(objs);
    engine.draw(image, res, objs);
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Optional per-object dump (YOLOV8_VERBOSE=1), used for cross-checking against references.
// Prints a comparable signature per task: rotated box for obb, plus a keypoint
// checksum / mask pixel count when present.
void dump_objects(const std::vector<Object>& objs)
{
    if (std::getenv("YOLOV8_VERBOSE") == nullptr) {
        return;
    }
    for (const auto& o : objs) {
        if (o.has_rrect) {
            std::printf("  label=%d prob=%.4f rrect=[%.2f, %.2f, %.2f, %.2f, %.2f]",
                        o.label,
                        o.prob,
                        o.rrect.center.x,
                        o.rrect.center.y,
                        o.rrect.size.width,
                        o.rrect.size.height,
                        o.rrect.angle);
        }
        else {
            std::printf("  label=%d prob=%.4f box=[%.2f, %.2f, %.2f, %.2f]",
                        o.label,
                        o.prob,
                        o.rect.x,
                        o.rect.y,
                        o.rect.x + o.rect.width,
                        o.rect.y + o.rect.height);
        }
        if (!o.kps.empty()) {
            double s = 0;
            for (float v : o.kps) {
                s += v;
            }
            std::printf(" kps_sum=%.2f", s);
        }
        if (!o.boxMask.empty()) {
            std::printf(" mask_px=%d", cv::countNonZero(o.boxMask));
        }
        std::printf("\n");
    }
}

}  // namespace

int run(Engine& engine, const CliArgs& args)
{
    bool       is_video = false;
    const auto images   = collect_sources(args.source, is_video);

    if (args.config.show) {
        cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
    }

    cv::Mat             res;
    std::vector<Object> objs;

    if (is_video) {
        cv::VideoCapture cap(args.source);
        if (!cap.isOpened()) {
            std::cerr << "cannot open video: " << args.source << "\n";
            return 1;
        }
        cv::Mat frame;
        while (cap.read(frame)) {
            const double ms = infer_once(engine, frame, res, objs);
            std::printf("cost %.4lf ms\n", ms);
            if (args.config.show) {
                cv::imshow("result", res);
                if (cv::waitKey(10) == 'q') {
                    break;
                }
            }
        }
    }
    else {
        for (const auto& path : images) {
            const cv::Mat image = cv::imread(path);
            if (image.empty()) {
                std::cerr << "skip unreadable image: " << path << "\n";
                continue;
            }
            const double ms = infer_once(engine, image, res, objs);
            std::printf("[%s] cost %.4lf ms, %zu objects\n", path.c_str(), ms, objs.size());
            dump_objects(objs);
            emit(res, path, args.config);
        }
    }

    if (args.config.show) {
        cv::destroyAllWindows();
    }
    engine.print_profile();
    return 0;
}

}  // namespace yolov8
