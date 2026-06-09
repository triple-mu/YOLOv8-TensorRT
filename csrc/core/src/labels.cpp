#include "yolov8/labels.hpp"
#include "yolov8/check.hpp"
#include <fstream>

namespace yolov8 {

std::vector<std::string> load_labels(const std::string& path)
{
    std::ifstream file(path);
    TRT_CHECK(file.good());
    std::vector<std::string> labels;
    std::string              line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    return labels;
}

const std::vector<std::string>& coco_labels()
{
    static const std::vector<std::string> names = {"person",        "bicycle",      "car",
                                                   "motorcycle",    "airplane",     "bus",
                                                   "train",         "truck",        "boat",
                                                   "traffic light", "fire hydrant", "stop sign",
                                                   "parking meter", "bench",        "bird",
                                                   "cat",           "dog",          "horse",
                                                   "sheep",         "cow",          "elephant",
                                                   "bear",          "zebra",        "giraffe",
                                                   "backpack",      "umbrella",     "handbag",
                                                   "tie",           "suitcase",     "frisbee",
                                                   "skis",          "snowboard",    "sports ball",
                                                   "kite",          "baseball bat", "baseball glove",
                                                   "skateboard",    "surfboard",    "tennis racket",
                                                   "bottle",        "wine glass",   "cup",
                                                   "fork",          "knife",        "spoon",
                                                   "bowl",          "banana",       "apple",
                                                   "sandwich",      "orange",       "broccoli",
                                                   "carrot",        "hot dog",      "pizza",
                                                   "donut",         "cake",         "chair",
                                                   "couch",         "potted plant", "bed",
                                                   "dining table",  "toilet",       "tv",
                                                   "laptop",        "mouse",        "remote",
                                                   "keyboard",      "cell phone",   "microwave",
                                                   "oven",          "toaster",      "sink",
                                                   "refrigerator",  "book",         "clock",
                                                   "vase",          "scissors",     "teddy bear",
                                                   "hair drier",    "toothbrush"};
    return names;
}

const Palette& palette()
{
    static const Palette colors = {
        {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
        {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
        {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
        {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
        {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
        {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
        {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
        {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
        {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
        {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
        {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
        {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
        {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
        {80, 183, 189},  {128, 128, 0}};
    return colors;
}

namespace pose {

const Palette& kps_colors()
{
    static const Palette colors = {{0, 255, 0},
                                   {0, 255, 0},
                                   {0, 255, 0},
                                   {0, 255, 0},
                                   {0, 255, 0},
                                   {255, 128, 0},
                                   {255, 128, 0},
                                   {255, 128, 0},
                                   {255, 128, 0},
                                   {255, 128, 0},
                                   {255, 128, 0},
                                   {51, 153, 255},
                                   {51, 153, 255},
                                   {51, 153, 255},
                                   {51, 153, 255},
                                   {51, 153, 255},
                                   {51, 153, 255}};
    return colors;
}

const Palette& limb_colors()
{
    static const Palette colors = {{51, 153, 255},
                                   {51, 153, 255},
                                   {51, 153, 255},
                                   {51, 153, 255},
                                   {255, 51, 255},
                                   {255, 51, 255},
                                   {255, 51, 255},
                                   {255, 128, 0},
                                   {255, 128, 0},
                                   {255, 128, 0},
                                   {255, 128, 0},
                                   {255, 128, 0},
                                   {0, 255, 0},
                                   {0, 255, 0},
                                   {0, 255, 0},
                                   {0, 255, 0},
                                   {0, 255, 0},
                                   {0, 255, 0},
                                   {0, 255, 0}};
    return colors;
}

const std::vector<std::vector<unsigned int>>& skeleton()
{
    static const std::vector<std::vector<unsigned int>> sk = {{16, 14},
                                                              {14, 12},
                                                              {17, 15},
                                                              {15, 13},
                                                              {12, 13},
                                                              {6, 12},
                                                              {7, 13},
                                                              {6, 7},
                                                              {6, 8},
                                                              {7, 9},
                                                              {8, 10},
                                                              {9, 11},
                                                              {2, 3},
                                                              {1, 2},
                                                              {1, 3},
                                                              {2, 4},
                                                              {3, 5},
                                                              {4, 6},
                                                              {5, 7}};
    return sk;
}

}  // namespace pose
}  // namespace yolov8
