#include "opencv2/opencv.hpp"
#include "yolov8/engine.hpp"
#include "yolov8/labels.hpp"
#include "yolov8/preprocess.hpp"
#include "yolov8/runner.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace yolov8;

// YOLOv8 classification: plain resize (no letterbox), output is per-class logits.
class ClsEngine: public Engine {
public:
    ClsEngine(const std::string& engine_path, const InferConfig& config): Engine(engine_path, config)
    {
        if (!config.labels_path.empty()) {
            class_names_ = load_labels(config.labels_path);
        }
    }

    void postprocess(std::vector<Object>& objs) override
    {
        objs.clear();
        const int num_cls = output_bindings_[0].dims.d[1];
        float*    data    = static_cast<float*>(host_ptrs_[0]);
        float*    max_ptr = std::max_element(data, data + num_cls);

        Object obj;
        obj.label = static_cast<int>(std::distance(data, max_ptr));
        obj.prob  = *max_ptr;
        objs.push_back(obj);
    }

    void draw(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) const override
    {
        res = image.clone();
        if (objs.empty()) {
            return;
        }
        const Object&     obj = objs[0];
        const std::string name =
            obj.label < static_cast<int>(class_names_.size()) ? class_names_[obj.label] : std::to_string(obj.label);
        char text[256];
        std::snprintf(text, sizeof(text), "%s %.1f%%", name.c_str(), obj.prob * 100);

        int      base_line  = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &base_line);
        cv::rectangle(res, cv::Rect(10, 10, label_size.width, label_size.height + base_line), {0, 0, 255}, -1);
        cv::putText(
            res, text, cv::Point(10, 10 + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }

protected:
    void preprocess(const cv::Mat& image, cv::Mat& blob) override
    {
        resize_blob(image, blob, config_.input_size, pparam_);
    }

    bool letterbox_preproc() const override
    {
        return false;  // cls uses plain resize on the GPU path too
    }

private:
    std::vector<std::string> class_names_;
};

int main(int argc, char** argv)
{
    try {
        const CliArgs args = parse_args(argc, argv);
        ClsEngine     engine(args.engine, args.config);
        engine.make_pipe();
        return run(engine, args);
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
