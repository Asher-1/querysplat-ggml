// Image input for the QuerySplat chain (pure C++, no python): decoding
// (libjpeg / libwebp / stb — the decoders PIL itself uses where it matters),
// the official 512 preprocessing (torchvision ImageTransform(max_crop=True):
// center square crop of side min(h,w), then Resize), and the
// F.interpolate(mode="bilinear", align_corners=False, antialias=True) resize
// it relies on (also the evaluator's metric-resolution GT resize).
#pragma once

#include <string>
#include <vector>

namespace qs {

// Directory -> sorted files with an infer.py IMAGE_EXTENSIONS suffix; anything
// else -> comma-separated file list. False when nothing was found.
bool collect_images(const std::string& spec, std::vector<std::string>* out);

// Decode + center-crop + resize every path into (S,3,H,W) f32 in [0,1].
bool load_images_as_frames(const std::vector<std::string>& paths, int H, int W,
                           std::vector<float>* frames);

// (sh,sw,3) interleaved -> (H,W,3) interleaved, F.interpolate bilinear with
// antialias=True: plain bilinear (ATen upsample_bilinear2d accumulation order)
// when neither axis shrinks, the separable antialias triangle filter otherwise.
void resize_rgb_hwc(const float* src, int sh, int sw, int H, int W, float* dst);

// The same resize on (n,3,h,w) planes -> (n,3,H,W).
std::vector<float> resize_planes(const float* src, int n, int h, int w, int H, int W);

}  // namespace qs
