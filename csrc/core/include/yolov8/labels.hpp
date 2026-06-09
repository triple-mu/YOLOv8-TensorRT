#pragma once
#include <string>
#include <vector>

namespace yolov8 {

using Color   = std::vector<unsigned int>;  // {r, g, b}
using Palette = std::vector<Color>;

// Reads one label per line. Throws TrtException if the file cannot be opened.
std::vector<std::string> load_labels(const std::string& path);

// Built-in COCO 80 names, used when no labels file is supplied.
const std::vector<std::string>& coco_labels();

// 80-entry color palette (cycled if more classes are requested).
const Palette& palette();

namespace pose {
const Palette&                                kps_colors();   // 17 keypoint colors
const Palette&                                limb_colors();  // 19 limb colors
const std::vector<std::vector<unsigned int>>& skeleton();     // 19 keypoint-index pairs
}  // namespace pose

}  // namespace yolov8
