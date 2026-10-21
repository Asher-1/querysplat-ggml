// Official-output parity: 1:1 ports of the scripts/infer.py output layout and
// the scripts/output_utils.py writers. Pure C++ — no Python, no matplotlib.
//
// Contract: given the same tensors, every writer produces the same file the
// official writer produces —
//   * JSON: python `json.dump(obj, f, indent=2)` + "\n" byte layout, floats in
//     python repr (shortest round-trip), keys in the official insertion order;
//   * statistics: numpy semantics (float32 pairwise sums, even-length median =
//     mean of the two middle values, NEP-50 float32 comparisons against python
//     scalars, np.percentile linear interpolation, np.rint);
//   * 8-bit images: numpy `astype(np.uint8)` truncation (rgb) / np.rint
//     (depth visualizations);
//   * .npz: a real zip of numpy-format .npy members (byte-identical .npy
//     headers); stored instead of deflated — np.load reads both identically.
// Known, documented residuals: c2w is the exact inverse rounded once (torch's
// f32 LU is ISA/library dependent and lands a few ulp away), the depth point
// cloud's world points inherit that plus cuBLAS's einsum accumulation, and the
// distribution PNGs are a C++ re-plot of the same data/layout rather than a
// pixel copy of matplotlib's rasterizer.
#pragma once

#include <string>
#include <vector>

namespace qs {

// Everything the official writers consume from the camera/depth stage.
struct OfficialInputs {
    int S = 0, H = 0, W = 0;
    const float* frames = nullptr;      // (S,3,H,W) preprocessed rgb in [0,1]
    const float* w2c = nullptr;         // S*16 row-major [[R,T],[0,1]]
    const float* intrinsics = nullptr;  // S*4 [fx, fy, cx, cy]
    const float* pose_enc = nullptr;    // S*9 raw camera-head output
    const float* depth = nullptr;       // (S, depth_h, depth_w)
    const float* conf = nullptr;        // (S, depth_h, depth_w)
    int depth_h = 0, depth_w = 0;
    std::vector<std::string> names;     // os.path.splitext(basename(src))[0]
};

// os.path.splitext(os.path.basename(path))[0]
std::string source_stem(const std::string& path);
// GGUF metadata stores config floats as f32; the official chain reads the YAML
// decimal as a python float. The shortest f32 repr parsed as double recovers
// that decimal (0.075f -> 0.075).
double config_value(float v);
// output_utils.opacity_threshold_suffix: f"_opacity{threshold:g}"
std::string opacity_threshold_suffix(double threshold);

// (3,H,W) [0,1] planes -> 8-bit RGB PNG, numpy
// `(x.clip(0, 1) * 255).astype(np.uint8)` truncation
bool write_rgb_planes_png(const std::string& path, const float* chw, int H, int W);
// input_frames/view{i}_{name}_preprocessed.png (save_rgb_tensor_image)
bool write_preprocessed_frames(const std::string& out_dir,
                               const OfficialInputs& in);
// rendered/render_view{i}.png from (S,3,H,W) [0,1] render planes
bool write_rendered_views(const std::string& out_dir, const float* images,
                          int S, int H, int W);
// predicted_input_cameras.json + .npz (save_predicted_input_cameras)
bool write_predicted_input_cameras(const std::string& out_dir,
                                   const OfficialInputs& in);
// vggt_input_depths/{vggt_input_depths.npz,json, depth_viewNN.png,
// depth_confidence_viewNN.png} (save_vggt_input_depths)
bool write_vggt_input_depths(const std::string& out_dir,
                             const OfficialInputs& in);
// vggt_depth_pointcloud.ply + .json (save_vggt_depth_pointcloud); exact
// mean nearest-neighbour distance (kd-tree, same as scipy cKDTree k=2)
bool write_vggt_depth_pointcloud(const std::string& out_dir,
                                 const OfficialInputs& in, long long target_points);
// gaussian_alpha_distribution.json + .png (save_gaussian_alpha_distribution)
bool write_alpha_distribution(const std::string& out_dir, const float* g,
                              long long N, int stride);
// gaussian_scale_distribution.json + .png (save_gaussian_scale_distribution)
bool write_scale_distribution(const std::string& out_dir, const float* g,
                              long long N, int stride, double max_scale);

// inference_timing.json + .txt (save_inference_timing; TTO fields fixed to
// the no-TTO values — the cpp chain has no TTO).
struct InferenceTiming {
    std::string device;  // "cuda" / "vulkan" / "cpu" (str(torch.device))
    int num_input_views = 0;
    double camera_and_depth_seconds = 0;
    double image_to_3dgs_seconds = 0;
    double render_seconds = 0;
    std::vector<double> opacity_thresholds;
    // TTO (use_tto=false keeps the historical non-TTO output byte-compatible)
    bool use_tto = false;
    long long tto_n_steps = 0;
    double tto_lr = 0;
    bool tto_lr_set = false;
    double tto_lpips_weight = 0;
    bool tto_lpips_set = false;
    std::vector<long long> tto_save_steps;
};
bool write_inference_timing(const std::string& out_dir,
                            const InferenceTiming& t);
// camera_info.txt
bool write_camera_info(const std::string& out_dir, int num_input_views);
// save_tto_losses (scripts/output_utils.py): tto_losses.json (indent 2) +
// .txt ("TTO final losses", "key: value:.6f" / "key: None")
bool write_tto_losses(const std::string& out_dir, double l1, double ssim,
                      const double* lpips);

}  // namespace qs
