#include "opencv2/opencv.hpp"
#include "yolov8/draw.hpp"
#include "yolov8/engine.hpp"
#include "yolov8/labels.hpp"
#include "yolov8/runner.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace yolov8;

// End2End detection: NMS is baked into the engine, which emits four tensors
// (num_dets, bboxes, scores, labels). Postprocess only rescales the boxes.
class End2EndEngine: public Engine {
public:
    End2EndEngine(const std::string& engine_path, const InferConfig& config): Engine(engine_path, config)
    {
        class_names_ = config.labels_path.empty() ? coco_labels() : load_labels(config.labels_path);
    }

    void postprocess(std::vector<Object>& objs) override
    {
        objs.clear();
        const int*   num_dets = static_cast<const int*>(host_ptrs_[0]);
        const float* boxes    = static_cast<const float*>(host_ptrs_[1]);
        const float* scores   = static_cast<const float*>(host_ptrs_[2]);
        const int*   labels   = static_cast<const int*>(host_ptrs_[3]);
        const float  dw = pparam_.dw, dh = pparam_.dh;
        const float  width = pparam_.width, height = pparam_.height, ratio = pparam_.ratio;

        for (int i = 0; i < num_dets[0]; ++i) {
            const float* ptr = boxes + i * 4;
            const float  x0  = clamp((ptr[0] - dw) * ratio, 0.f, width);
            const float  y0  = clamp((ptr[1] - dh) * ratio, 0.f, height);
            const float  x1  = clamp((ptr[2] - dw) * ratio, 0.f, width);
            const float  y1  = clamp((ptr[3] - dh) * ratio, 0.f, height);

            Object obj;
            obj.rect  = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
            obj.prob  = scores[i];
            obj.label = labels[i];
            objs.push_back(obj);
        }
    }

    void draw(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) const override
    {
        draw_detections(image, res, objs, class_names_);
    }

private:
    std::vector<std::string> class_names_;
};

int main(int argc, char** argv)
{
    try {
        const CliArgs args = parse_args(argc, argv);
        End2EndEngine engine(args.engine, args.config);
        engine.make_pipe();
        return run(engine, args);
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
