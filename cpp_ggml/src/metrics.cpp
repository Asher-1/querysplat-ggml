#include "metrics.hpp"

#include <cmath>
#include <cstddef>

namespace qs {

std::vector<float> psnr_views(const float* pred, const float* target, int V, int H, int W) {
    const size_t n = (size_t)3 * H * W;
    std::vector<float> out((size_t)V);
    for (int v = 0; v < V; v++) {
        const float* p = pred + (size_t)v * n;
        const float* t = target + (size_t)v * n;
        double acc = 0;
        for (size_t i = 0; i < n; i++) {
            const float d = p[i] - t[i];
            acc += (double)(d * d);  // .square() is an f32 op
        }
        const float mse = (float)(acc / (double)n);
        out[(size_t)v] = -10.0f * std::log10(mse);
    }
    return out;
}

namespace {

constexpr int kRadius = 5;  // int(truncate * sigma + 0.5) = int(3.5 * 1.5 + 0.5)

// scipy _gaussian_kernel1d(1.5, 0, 5): exp(-0.5 / sigma2 * x**2), divided by
// its numpy sum (pairwise_sum for n = 11: eight lanes folded as
// ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)), then the tail added in order).
// Returned as w[d] for the tap at distance d (the kernel is symmetric).
void gaussian_weights(double w[kRadius + 1]) {
    const double sigma = 1.5, sigma2 = sigma * sigma;
    double phi[2 * kRadius + 1];
    for (int i = 0; i < 2 * kRadius + 1; i++) {
        const double x = i - kRadius;
        phi[i] = std::exp(-0.5 / sigma2 * (x * x));
    }
    double sum = ((phi[0] + phi[1]) + (phi[2] + phi[3])) + ((phi[4] + phi[5]) + (phi[6] + phi[7]));
    for (int i = 8; i < 2 * kRadius + 1; i++) sum += phi[i];
    for (int d = 0; d <= kRadius; d++) w[d] = phi[kRadius + d] / sum;
}

// scipy mode "reflect" (half-sample symmetric: d c b a | a b c d | d c b a)
inline int reflect(int i, int n) {
    if (i < 0) return -i - 1;
    if (i >= n) return 2 * n - i - 1;
    return i;
}

// One scipy correlate1d pass of the symmetric kernel along `axis` of an
// (H,W) f32 plane: the line is read into f64, out = l[0]*w0 then
// += (l[-d] + l[d]) * w_d for d = 5..1 (NI_Correlate1D's symmetric loop),
// stored back as f32. O(H*W*radius).
void correlate_axis(const float* src, float* dst, int H, int W, int axis, const double* w) {
    const int len = axis == 0 ? H : W, lines = axis == 0 ? W : H;
    const size_t stride = axis == 0 ? (size_t)W : 1, lstride = axis == 0 ? 1 : (size_t)W;
    std::vector<double> line((size_t)len + 2 * kRadius);
    for (int l = 0; l < lines; l++) {
        const float* s = src + (size_t)l * lstride;
        float* o = dst + (size_t)l * lstride;
        for (int i = -kRadius; i < len + kRadius; i++)
            line[(size_t)(i + kRadius)] = s[(size_t)reflect(i, len) * stride];
        const double* c = line.data() + kRadius;
        for (int i = 0; i < len; i++) {
            double acc = c[i] * w[0];
            for (int d = kRadius; d >= 1; d--) acc += (c[i - d] + c[i + d]) * w[d];
            o[(size_t)i * stride] = (float)acc;
        }
    }
}

// skimage gaussian(image, sigma=1.5, truncate=3.5, mode="reflect") ==
// ndi.gaussian_filter: axis 0 into an f32 output, then axis 1 over it.
std::vector<float> gaussian_filter(const std::vector<float>& img, int H, int W, const double* w) {
    std::vector<float> tmp(img.size()), out(img.size());
    correlate_axis(img.data(), tmp.data(), H, W, 0, w);
    correlate_axis(tmp.data(), out.data(), H, W, 1, w);
    return out;
}

// structural_similarity for one 2-D f32 channel (im1 = gt, im2 = pred, the
// official argument order).
float ssim_channel(const float* im1, const float* im2, int H, int W, const double* w) {
    const size_t n = (size_t)H * W;
    std::vector<float> x(im1, im1 + n), y(im2, im2 + n), xx(n), yy(n), xy(n);
    for (size_t i = 0; i < n; i++) {
        xx[i] = x[i] * x[i];
        yy[i] = y[i] * y[i];
        xy[i] = x[i] * y[i];
    }
    const std::vector<float> ux = gaussian_filter(x, H, W, w), uy = gaussian_filter(y, H, W, w),
                             uxx = gaussian_filter(xx, H, W, w),
                             uyy = gaussian_filter(yy, H, W, w),
                             uxy = gaussian_filter(xy, H, W, w);
    // python-float scalars meet f32 arrays as f32 (numpy weak scalars)
    const float cov_norm = (float)(121.0 / 120.0);
    const float C1 = (float)((0.01 * 1.0) * (0.01 * 1.0));
    const float C2 = (float)((0.03 * 1.0) * (0.03 * 1.0));
    const int pad = kRadius;
    double acc = 0;
    for (int r = pad; r < H - pad; r++) {
        for (int c = pad; c < W - pad; c++) {
            const size_t i = (size_t)r * W + c;
            const float vx = cov_norm * (uxx[i] - ux[i] * ux[i]);
            const float vy = cov_norm * (uyy[i] - uy[i] * uy[i]);
            const float vxy = cov_norm * (uxy[i] - ux[i] * uy[i]);
            const float A1 = 2.0f * ux[i] * uy[i] + C1;
            const float A2 = 2.0f * vxy + C2;
            const float B1 = ux[i] * ux[i] + uy[i] * uy[i] + C1;
            const float B2 = vx + vy + C2;
            const float D = B1 * B2;
            const float S = (A1 * A2) / D;
            acc += (double)S;
        }
    }
    return (float)(acc / (double)((size_t)(H - 2 * pad) * (W - 2 * pad)));
}

}  // namespace

std::vector<float> ssim_views(const float* pred, const float* target, int V, int H, int W) {
    double w[kRadius + 1];
    gaussian_weights(w);
    const size_t plane = (size_t)H * W;
    std::vector<float> out((size_t)V);
    for (int v = 0; v < V; v++) {
        float ch[3];
        for (int c = 0; c < 3; c++) {
            const size_t off = ((size_t)v * 3 + c) * plane;
            ch[c] = ssim_channel(target + off, pred + off, H, W, w);
        }
        out[(size_t)v] = ((ch[0] + ch[1]) + ch[2]) / 3.0f;  // f32 mssim.mean()
    }
    return out;
}

}  // namespace qs
