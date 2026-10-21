// The paper's per-view image metrics (evaluations/metrics.py ImageMetrics),
// host side: PSNR and skimage's Gaussian-window SSIM. LPIPS-vgg runs as a
// ggml graph (lpips_graph.hpp).
//
// Images are (V,3,H,W) f32, already clamped to [0,1] by the caller (the
// official __call__ clamps both before scoring).
#pragma once

#include <vector>

namespace qs {

// -10 * log10(mean((p - t)^2)) per view, torch f32 semantics (the mean is
// accumulated in f64 and rounded once — torch's CUDA reduction order is not
// reproducible anyway; the difference is < 1 f32 ulp of the mse).
std::vector<float> psnr_views(const float* pred, const float* target, int V, int H, int W);

// skimage.metrics.structural_similarity(gt, pred, win_size=11,
// gaussian_weights=True, channel_axis=0, data_range=1.0) per view, op-for-op
// in its float32 semantics: scipy gaussian_filter (sigma 1.5, truncate 3.5 ->
// radius 5, mode "reflect", axis 0 then axis 1, f64 line accumulation in
// scipy's symmetric-kernel order, f32 intermediates), sample-covariance
// normalization 121/120, f32 SSIM map, f64 mean over the pad-5 crop rounded to
// f32 per channel, f32 mean over channels.
std::vector<float> ssim_views(const float* pred, const float* target, int V, int H, int W);

}  // namespace qs
