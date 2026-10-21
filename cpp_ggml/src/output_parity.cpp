// Official-output parity writers — implementation. See output_parity.hpp.
// Every writer is a 1:1 port of the matching function in
// scripts/output_utils.py; the numpy/python semantics each one depends on are
// spelled out next to the code that reproduces them.

#include "output_parity.hpp"

#include "camera_math.hpp"
#include "dist_plot.hpp"
#include "pyio.hpp"
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace qs {

namespace {

// ---- numpy reduction semantics ----------------------------------------------

// numpy pairwise_sum (loops_utils.h.src), accumulating in T. A contiguous
// (or plain strided, no-cast) 1-D reduction hands the whole array to one
// inner-loop call, so this is np.sum / np.mean's exact summation tree.
template <typename T>
T pairwise_sum(const T* a, size_t n, size_t stride) {
    if (n < 8) {
        T r = 0;
        for (size_t k = 0; k < n; k++) r += a[k * stride];
        return r;
    }
    if (n <= 128) {
        T r[8];
        for (int k = 0; k < 8; k++) r[k] = a[(size_t)k * stride];
        size_t k = 8;
        for (; k < n - (n % 8); k += 8)
            for (int m = 0; m < 8; m++) r[m] += a[(k + (size_t)m) * stride];
        T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; k < n; k++) res += a[k * stride];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2, stride) + pairwise_sum(a + n2 * stride, n - n2, stride);
}

// float32 ndarray.mean(): f32 pairwise sum, then f32(sum / np.intp(N)) —
// the scalar division promotes to f64 before the cast back.
double mean_f32(const std::vector<float>& v) {
    const float s = pairwise_sum(v.data(), v.size(), 1);
    return (double)(float)((double)s / (double)v.size());
}

// np.median: middle element, or mean of the two middle ones (a 2-element
// mean: f32/f64 0 + a + b, then / 2)
template <typename T>
double median_of(std::vector<T> v) {
    const size_t n = v.size(), h = n / 2;
    std::nth_element(v.begin(), v.begin() + (long)h, v.end());
    const T hi = v[h];
    if (n % 2) return (double)hi;
    const T lo = *std::max_element(v.begin(), v.begin() + (long)h);
    T s = 0;
    s += lo;
    s += hi;
    return (double)(T)((double)s / 2.0);
}

// np.percentile(f32 samples, q) (method="linear"): virtual index
// (n-1)*(q/100) in f64, diff = f32(b - a), np._lerp in f64.
double percentile_f32(const std::vector<float>& sorted, double q) {
    const size_t n = sorted.size();
    const double vi = (double)(n - 1) * (q / 100.0);
    size_t prev = (size_t)std::floor(vi), next = prev + 1;
    double g = vi - std::floor(vi);
    if (vi >= (double)(n - 1)) { prev = next = n - 1; }
    const float a = sorted[prev], b = sorted[std::min(next, n - 1)];
    const float d = b - a;
    return g < 0.5 ? (double)a + (double)d * g : (double)b - (double)d * (1.0 - g);
}

// bool_mask.mean() * 100.0
double percent(long long count, long long total) {
    return total > 0 ? ((double)count / (double)total) * 100.0 : 0.0;
}

// ---- shared tensor helpers --------------------------------------------------

void mkdirs(const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
}

// numpy `(x.clip(0, 1) * 255).astype(np.uint8)`: truncation, not rounding
inline unsigned char u8_trunc(float v) {
    return (unsigned char)(std::clamp(v, 0.0f, 1.0f) * 255.0f);
}

void transpose4(const float* m, float* o) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) o[r * 4 + c] = m[c * 4 + r];
}

// c2w = torch.linalg.inv(w2c): general inverse (qs::invert4), not R^T —
// the f32 quaternion R is orthonormal only to ~1e-7. Torch's own f32 LU
// (MKL on CPU, cuSOLVER/MAGMA on CUDA, ISA-dependent) lands within a few ulp
// of the exact inverse that invert4 rounds to — a documented residual.
struct FrameCams {
    std::vector<float> cam_view, w2c, c2w, K;  // S*16, S*16, S*16, S*9
};

FrameCams frame_cams(const OfficialInputs& in) {
    FrameCams c;
    c.cam_view.resize((size_t)in.S * 16);
    c.w2c.assign(in.w2c, in.w2c + (size_t)in.S * 16);
    c.c2w.resize((size_t)in.S * 16);
    c.K.assign((size_t)in.S * 9, 0.f);
    for (int i = 0; i < in.S; i++) {
        transpose4(in.w2c + i * 16, &c.cam_view[(size_t)i * 16]);
        invert4(in.w2c + i * 16, &c.c2w[(size_t)i * 16]);  // torch.linalg.inv
        const float* k4 = in.intrinsics + i * 4;
        float* K = &c.K[(size_t)i * 9];  // _intrinsics_4_to_k
        K[0] = k4[0]; K[4] = k4[1]; K[2] = k4[2]; K[5] = k4[3]; K[8] = 1.f;
    }
    return c;
}

std::string frame_source(int i, const std::string& name) {
    return "input_frames/view" + std::to_string(i) + "_" + name + "_preprocessed.png";
}

std::string view2(const char* prefix, int i) {
    char b[64];
    std::snprintf(b, sizeof(b), "%s%02d.png", prefix, i);
    return b;
}

}  // namespace

bool write_rgb_planes_png(const std::string& path, const float* chw, int H, int W) {
    const size_t plane = (size_t)H * W;
    std::vector<unsigned char> px(plane * 3);
    for (size_t p = 0; p < plane; p++)
        for (int c = 0; c < 3; c++) px[p * 3 + c] = u8_trunc(chw[(size_t)c * plane + p]);
    return stbi_write_png(path.c_str(), W, H, 3, px.data(), W * 3) != 0;
}

namespace {

// ---- exact nearest neighbour (scipy cKDTree(points).query(points, k=2)) ----

// cKDTree works on float64 copies; p=2 squared distance is a plain
// sequential sum over the 3 dims (its 4-way unroll does not trigger at
// m=3), sqrt applied to the final k-th distance.
struct KdTree {
    const std::vector<std::array<double, 3>>& p;
    std::vector<int> idx;
    std::vector<int> axis_of;  // per node (node == position in idx)

    explicit KdTree(const std::vector<std::array<double, 3>>& pts) : p(pts) {
        idx.resize(p.size());
        axis_of.resize(p.size());
        for (size_t k = 0; k < p.size(); k++) idx[k] = (int)k;
        build(0, (int)p.size());
    }
    // O(n log^2 n) build: median split on the widest axis per subtree
    void build(int lo, int hi) {
        if (hi - lo <= 1) { if (hi > lo) axis_of[lo] = 0; return; }
        double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
        for (int k = lo; k < hi; k++)
            for (int d = 0; d < 3; d++) {
                mn[d] = std::min(mn[d], p[idx[k]][d]);
                mx[d] = std::max(mx[d], p[idx[k]][d]);
            }
        int ax = 0;
        for (int d = 1; d < 3; d++) if (mx[d] - mn[d] > mx[ax] - mn[ax]) ax = d;
        const int mid = (lo + hi) / 2;
        std::nth_element(idx.begin() + lo, idx.begin() + mid, idx.begin() + hi,
                         [&](int a, int b) { return p[a][ax] < p[b][ax]; });
        axis_of[mid] = ax;
        build(lo, mid);
        build(mid + 1, hi);
    }
    static double sqdist(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        double s = 0;
        for (int d = 0; d < 3; d++) { const double t = a[d] - b[d]; s += t * t; }
        return s;
    }
    void nearest(int lo, int hi, int self, double& best) const {
        if (hi <= lo) return;
        const int mid = (lo + hi) / 2;
        const int node = idx[mid];
        const auto& q = p[self];
        if (node != self) best = std::min(best, sqdist(q, p[node]));
        if (hi - lo == 1) return;
        const int ax = axis_of[mid];
        const double diff = q[ax] - p[node][ax];
        const bool left_first = diff < 0;
        if (left_first) nearest(lo, mid, self, best); else nearest(mid + 1, hi, self, best);
        if (diff * diff <= best) {
            if (left_first) nearest(mid + 1, hi, self, best); else nearest(lo, mid, self, best);
        }
    }
};

// mean_pointcloud_nn_distance_np: float(dists[:, 1].mean()) — f64 pairwise
double mean_nn_distance(const std::vector<std::array<float, 3>>& xyz) {
    std::vector<std::array<double, 3>> p(xyz.size());
    for (size_t k = 0; k < xyz.size(); k++)
        p[k] = {(double)xyz[k][0], (double)xyz[k][1], (double)xyz[k][2]};
    KdTree tree(p);
    std::vector<double> d(p.size());
    for (size_t k = 0; k < p.size(); k++) {
        double best = std::numeric_limits<double>::infinity();
        tree.nearest(0, (int)p.size(), (int)k, best);
        d[k] = std::sqrt(best);
    }
    return pairwise_sum(d.data(), d.size(), 1) / (double)d.size();
}

// F.interpolate(mode="bilinear", align_corners=False) of (3,h,w) planes —
// only used when the depth map resolution differs from the input frames.
std::vector<float> resize_bilinear(const float* src, int h, int w, int H, int W) {
    std::vector<float> out((size_t)3 * H * W);
    const float sy = (float)h / (float)H, sx = (float)w / (float)W;
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < H; y++) {
            const float fy = std::max(0.f, sy * ((float)y + 0.5f) - 0.5f);
            const int y0 = (int)fy, y1 = std::min(y0 + 1, h - 1);
            const float ly = fy - (float)y0;
            for (int x = 0; x < W; x++) {
                const float fx = std::max(0.f, sx * ((float)x + 0.5f) - 0.5f);
                const int x0 = (int)fx, x1 = std::min(x0 + 1, w - 1);
                const float lx = fx - (float)x0;
                const float* p = src + (size_t)c * h * w;
                out[((size_t)c * H + y) * W + x] =
                    (1 - ly) * ((1 - lx) * p[y0 * w + x0] + lx * p[y0 * w + x1]) +
                    ly * ((1 - lx) * p[y1 * w + x0] + lx * p[y1 * w + x1]);
            }
        }
    return out;
}

// _depth_visualization_range + _save_scalar_visualization
std::pair<double, double> save_scalar_visualization(const float* v, int H, int W,
                                                    bool positive_only,
                                                    const std::string& path) {
    const size_t n = (size_t)H * W;
    const auto valid = [&](float x) { return std::isfinite(x) && (!positive_only || x > 0); };
    std::vector<float> s;
    for (size_t k = 0; k < n; k++) if (valid(v[k])) s.push_back(v[k]);
    double low = 0.0, high = 1.0;
    if (!s.empty()) {
        std::sort(s.begin(), s.end());
        low = percentile_f32(s, 1.0);
        high = percentile_f32(s, 99.0);
        if (!std::isfinite(low) || !std::isfinite(high) || high <= low) {
            low = (double)s.front();
            high = (double)s.back();
        }
        if (high <= low) high = low + std::max(std::abs(low) * 1e-6, 1e-6);
    }
    // NEP 50: f32 array op python float -> f32 arithmetic throughout
    const float lowf = (float)low, spanf = (float)(high - low);
    std::vector<unsigned char> img(n, 0);
    for (size_t k = 0; k < n; k++) {
        if (!valid(v[k])) continue;
        const float sc = std::clamp((v[k] - lowf) / spanf, 0.0f, 1.0f);
        img[k] = (unsigned char)std::rint(sc * 255.0f);
    }
    stbi_write_png(path.c_str(), W, H, 1, img.data(), W);
    return {low, high};
}

}  // namespace

// ---- public API ---------------------------------------------------------------

std::string source_stem(const std::string& path) {
    // os.path.basename then genericpath._splitext: the last dot starts an
    // extension only if a non-dot character precedes it (".hidden" has none)
    const size_t slash = path.find_last_of('/');
    const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = base.find_last_of('.');
    if (dot != std::string::npos)
        for (size_t k = 0; k < dot; k++)
            if (base[k] != '.') return base.substr(0, dot);
    return base;
}

double config_value(float v) {
    char b[32];
    auto r = std::to_chars(b, b + sizeof(b), v);  // shortest f32 repr
    *r.ptr = '\0';
    return std::strtod(b, nullptr);
}

std::string opacity_threshold_suffix(double threshold) {
    return "_opacity" + py_g(threshold);
}

bool write_preprocessed_frames(const std::string& out_dir, const OfficialInputs& in) {
    mkdirs(out_dir + "/input_frames");
    const size_t plane = (size_t)in.H * in.W;
    for (int i = 0; i < in.S; i++)
        if (!write_rgb_planes_png(out_dir + "/" + frame_source(i, in.names[(size_t)i]),
                                  in.frames + (size_t)i * 3 * plane, in.H, in.W))
            return false;
    return true;
}

bool write_rendered_views(const std::string& out_dir, const float* images, int S,
                          int H, int W) {
    mkdirs(out_dir + "/rendered");
    for (int i = 0; i < S; i++)
        if (!write_rgb_planes_png(out_dir + "/rendered/render_view" + std::to_string(i) + ".png",
                                  images + (size_t)i * 3 * H * W, H, W))
            return false;
    return true;
}

bool write_predicted_input_cameras(const std::string& out_dir, const OfficialInputs& in) {
    const FrameCams c = frame_cams(in);
    Json conv = jobj();
    conv.add("cam_view", jstr("QuerySplat renderer convention: transposed OpenCV world-to-camera matrix, shape [V,4,4]."))
        .add("w2c", jstr("OpenCV world-to-camera matrix, shape [V,4,4]."))
        .add("c2w", jstr("Inverse of w2c, shape [V,4,4]."))
        .add("intrinsics", jstr("[fx, fy, cx, cy] in rendered/preprocessed image pixels, shape [V,4]."))
        .add("K", jstr("3x3 pinhole intrinsics matrix in rendered/preprocessed image pixels, shape [V,3,3]."))
        .add("pose_enc", jstr("Raw VGGT camera-head output, shape [V,9]."));
    Json frames = jarr();
    std::vector<std::string> sources;
    for (int i = 0; i < in.S; i++) {
        sources.push_back(frame_source(i, in.names[(size_t)i]));
        Json f = jobj();
        f.add("index", jint(i))
            .add("name", jstr(in.names[(size_t)i]))
            .add("source", jstr(sources.back()))
            .add("cam_view", mat_list(&c.cam_view[(size_t)i * 16], 4, 4))
            .add("w2c", mat_list(&c.w2c[(size_t)i * 16], 4, 4))
            .add("c2w", mat_list(&c.c2w[(size_t)i * 16], 4, 4))
            .add("intrinsics", f32_list(in.intrinsics + i * 4, 4))
            .add("K", mat_list(&c.K[(size_t)i * 9], 3, 3))
            .add("pose_enc", f32_list(in.pose_enc + i * 9, 9));
        frames.push(std::move(f));
    }
    Json root = jobj();
    root.add("convention", std::move(conv)).add("frames", std::move(frames));
    if (!write_json(out_dir + "/predicted_input_cameras.json", root)) return false;
    const long long S = in.S;
    return write_npz(out_dir + "/predicted_input_cameras.npz",
                     {{"cam_view", npy_f32(c.cam_view.data(), {S, 4, 4})},
                      {"w2c", npy_f32(c.w2c.data(), {S, 4, 4})},
                      {"c2w", npy_f32(c.c2w.data(), {S, 4, 4})},
                      {"intrinsics", npy_f32(in.intrinsics, {S, 4})},
                      {"K", npy_f32(c.K.data(), {S, 3, 3})},
                      {"pose_enc", npy_f32(in.pose_enc, {S, 9})},
                      {"frame_names", npy_str(in.names)},
                      {"frame_sources", npy_str(sources)}});
}

bool write_vggt_input_depths(const std::string& out_dir, const OfficialInputs& in) {
    const std::string dir = out_dir + "/vggt_input_depths";
    mkdirs(dir);
    const int S = in.S, H = in.depth_h, W = in.depth_w;
    const size_t plane = (size_t)H * W;
    const FrameCams c = frame_cams(in);
    if (!write_npz(dir + "/vggt_input_depths.npz",
                   {{"depth", npy_f32(in.depth, {S, H, W})},
                    {"depth_confidence", npy_f32(in.conf, {S, H, W})},
                    {"cam_view", npy_f32(c.cam_view.data(), {S, 4, 4})},
                    {"intrinsics", npy_f32(in.intrinsics, {S, 4})},
                    {"pose_enc", npy_f32(in.pose_enc, {S, 9})}}))
        return false;
    Json frames = jarr();
    for (int i = 0; i < S; i++) {
        const float* d = in.depth + (size_t)i * plane;
        const float* cf = in.conf + (size_t)i * plane;
        const std::string dname = view2("depth_view", i);
        const std::string cname = view2("depth_confidence_view", i);
        const auto [dlo, dhi] = save_scalar_visualization(d, H, W, true, dir + "/" + dname);
        const auto [clo, chi] = save_scalar_visualization(cf, H, W, false, dir + "/" + cname);
        long long npos = 0, nconf = 0;
        float dmin = 0, dmax = 0, cmin = 0, cmax = 0;
        for (size_t k = 0; k < plane; k++) {
            if (std::isfinite(d[k]) && d[k] > 0) {
                dmin = npos ? std::min(dmin, d[k]) : d[k];
                dmax = npos ? std::max(dmax, d[k]) : d[k];
                npos++;
            }
            if (std::isfinite(cf[k])) {
                cmin = nconf ? std::min(cmin, cf[k]) : cf[k];
                cmax = nconf ? std::max(cmax, cf[k]) : cf[k];
                nconf++;
            }
        }
        Json f = jobj();
        f.add("index", jint(i))
            .add("name", jstr(in.names[(size_t)i]))
            .add("source", jstr(frame_source(i, in.names[(size_t)i])))
            .add("depth_image", jstr(dname))
            .add("depth_confidence_image", jstr(cname))
            .add("depth_visualization_range", jarr().push(jnum(dlo)).push(jnum(dhi)))
            .add("depth_confidence_visualization_range", jarr().push(jnum(clo)).push(jnum(chi)))
            .add("positive_finite_depth_pixels", jint(npos))
            .add("depth_min", npos ? jnum(dmin) : jnull())
            .add("depth_max", npos ? jnum(dmax) : jnull())
            .add("confidence_min", nconf ? jnum(cmin) : jnull())
            .add("confidence_max", nconf ? jnum(cmax) : jnull())
            .add("cam_view", mat_list(&c.cam_view[(size_t)i * 16], 4, 4))
            .add("w2c", mat_list(&c.w2c[(size_t)i * 16], 4, 4))
            .add("c2w", mat_list(&c.c2w[(size_t)i * 16], 4, 4))
            .add("intrinsics", f32_list(in.intrinsics + i * 4, 4))
            .add("pose_enc", f32_list(in.pose_enc + i * 9, 9));
        frames.push(std::move(f));
    }
    Json conv = jobj();
    conv.add("depth", jstr("VGGT input-view depth in the VGGT-predicted camera coordinate system."))
        .add("depth_confidence", jstr("Raw VGGT depth confidence; it is not normalized to [0, 1]."))
        .add("depth_images", jstr("8-bit visualizations clipped to each view's 1st-99th percentile positive finite depth range; invalid and non-positive depths are black."))
        .add("depth_confidence_images", jstr("8-bit visualizations clipped to each view's 1st-99th percentile finite confidence range; invalid values are black."))
        .add("cam_view", jstr("QuerySplat renderer convention: transposed OpenCV world-to-camera matrix."))
        .add("intrinsics", jstr("[fx, fy, cx, cy] in preprocessed/render image pixels."));
    Json root = jobj();
    root.add("raw_tensors", jstr("vggt_input_depths.npz"))
        .add("depth_shape", int_list({S, H, W}))
        .add("depth_confidence_shape", int_list({S, H, W}))
        .add("convention", std::move(conv))
        .add("frames", std::move(frames));
    return write_json(dir + "/vggt_input_depths.json", root);
}

bool write_vggt_depth_pointcloud(const std::string& out_dir, const OfficialInputs& in,
                                 long long target_points) {
    const int S = in.S, H = in.depth_h, W = in.depth_w;
    const size_t plane = (size_t)H * W;
    // sample_depth_image_grid: python int((max(V*H*W, 1) / target) ** 0.5)
    const int stride = std::max(
        1, (int)std::pow((double)std::max((long long)S * H * W, 1LL) / (double)target_points, 0.5));
    const int cells_y = (H + stride - 1) / stride, cells_x = (W + stride - 1) / stride;
    std::vector<float> rgb;  // (S,3,H,W) at the depth resolution
    if (in.H == H && in.W == W) {
        rgb.assign(in.frames, in.frames + (size_t)S * 3 * plane);
    } else {
        for (int i = 0; i < S; i++) {
            auto r = resize_bilinear(in.frames + (size_t)i * 3 * in.H * in.W, in.H, in.W, H, W);
            rgb.insert(rgb.end(), r.begin(), r.end());
        }
    }
    const FrameCams c = frame_cams(in);
    std::vector<std::array<float, 3>> xyz;
    std::vector<std::array<unsigned char, 3>> col;
    std::vector<int> view_of;
    std::vector<long long> valid_count((size_t)S, 0);
    for (int v = 0; v < S; v++) {
        const float* d = in.depth + (size_t)v * plane;
        const float* cf = in.conf + (size_t)v * plane;
        const float* k4 = in.intrinsics + v * 4;
        const float* m = &c.c2w[(size_t)v * 16];
        for (int gy = 0; gy < cells_y; gy++)
            for (int gx = 0; gx < cells_x; gx++) {
                // argmax of the cell's confidence (row-major, first max; the
                // -inf padding never wins over a real pixel; a NaN wins and
                // then fails the finite-confidence test below)
                int best = -1;
                float bc = 0;
                bool nan = false;
                for (int yy = gy * stride; yy < std::min(gy * stride + stride, H) && !nan; yy++)
                    for (int xx = gx * stride; xx < std::min(gx * stride + stride, W); xx++) {
                        const float cv = cf[(size_t)yy * W + xx];
                        if (std::isnan(cv)) { nan = true; break; }
                        if (best < 0 || cv > bc) { bc = cv; best = yy * W + xx; }
                    }
                if (nan || best < 0) continue;
                const float z = d[best];
                const float px = (float)(best % W), py = (float)(best / W);
                // world_points_from_sampled_depth (f32)
                const float cp[3] = {(px - k4[2]) / std::max(k4[0], 1e-6f) * z,
                                     (py - k4[3]) / std::max(k4[1], 1e-6f) * z, z};
                float w[3];
                for (int r = 0; r < 3; r++)
                    w[r] = cp[0] * m[r * 4 + 0] + cp[1] * m[r * 4 + 1] + cp[2] * m[r * 4 + 2] + m[r * 4 + 3];
                if (!(std::isfinite(w[0]) && std::isfinite(w[1]) && std::isfinite(w[2]) &&
                      std::isfinite(z) && z > 0 && std::isfinite(bc)))
                    continue;
                xyz.push_back({w[0], w[1], w[2]});
                const float* rv = &rgb[(size_t)v * 3 * plane + (size_t)best];
                col.push_back({u8_trunc(rv[0]), u8_trunc(rv[plane]), u8_trunc(rv[2 * plane])});
                view_of.push_back(v);
                valid_count[(size_t)v]++;
            }
    }
    // plyfile layout: no comment lines, x/y/z f4, red/green/blue u1, view_index i4
    const std::string ply_path = out_dir + "/vggt_depth_pointcloud.ply";
    std::string ply = "ply\nformat binary_little_endian 1.0\nelement vertex " +
                      std::to_string(xyz.size()) +
                      "\nproperty float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                      "property int view_index\nend_header\n";
    for (size_t k = 0; k < xyz.size(); k++) {
        ply.append(reinterpret_cast<const char*>(xyz[k].data()), 12);
        ply.append(reinterpret_cast<const char*>(col[k].data()), 3);
        ply.append(reinterpret_cast<const char*>(&view_of[k]), 4);
    }
    if (!write_text(ply_path, ply)) return false;

    Json frames = jarr();
    for (int i = 0; i < S; i++) {
        Json f = jobj();
        f.add("index", jint(i))
            .add("name", jstr(in.names[(size_t)i]))
            .add("source", jstr(frame_source(i, in.names[(size_t)i])))
            .add("valid_points", jint(valid_count[(size_t)i]))
            .add("cam_view", mat_list(&c.cam_view[(size_t)i * 16], 4, 4))
            .add("w2c", mat_list(&c.w2c[(size_t)i * 16], 4, 4))
            .add("c2w", mat_list(&c.c2w[(size_t)i * 16], 4, 4))
            .add("intrinsics", f32_list(in.intrinsics + i * 4, 4));
        frames.push(std::move(f));
    }
    const long long sh = stride == 1 ? H : cells_y, sw = stride == 1 ? W : cells_x;
    Json root = jobj();
    root.add("pointcloud", jstr("vggt_depth_pointcloud.ply"))
        .add("sampling", jstr("image_grid"))
        .add("target_points_per_scene", jint(target_points))
        .add("sampling_stride", jint(stride))
        .add("total_points", jint((long long)xyz.size()))
        .add("mean_nearest_neighbor_distance",
             xyz.size() < 2 ? jnull() : jnum(mean_nn_distance(xyz)))
        .add("original_depth_shape", int_list({1, S, 1, H, W}))
        .add("depth_shape", int_list({1, S, 1, sh, sw}))
        .add("frames", std::move(frames));
    return write_json(out_dir + "/vggt_depth_pointcloud.json", root);
}

bool write_alpha_distribution(const std::string& out_dir, const float* g, long long N,
                              int stride) {
    static const double kThr[] = {5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3,
                                  5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6};
    std::vector<float> alpha((size_t)N);  // .float().clamp(0, 1) (f32)
    for (long long i = 0; i < N; i++)
        alpha[(size_t)i] = std::clamp(g[(size_t)i * stride + 3], 0.0f, 1.0f);
    bool has_pos = false;
    float minpos = 0, amin = 0, amax = 0;
    long long zero = 0;
    for (size_t k = 0; k < alpha.size(); k++) {
        const float a = alpha[k];
        amin = k ? std::min(amin, a) : a;
        amax = k ? std::max(amax, a) : a;
        if (a > 0) { minpos = has_pos ? std::min(minpos, a) : a; has_pos = true; }
        if (a <= 0) zero++;
    }
    const double floor_ = std::max(has_pos ? (double)minpos : 1e-8, 1e-8);
    const double bin_min = std::min(floor_, 1e-6);  // min(thresholds)
    const int log_min_exp = std::min((int)std::floor(std::log10(bin_min)), -6);
    std::vector<double> edges;
    for (int e = log_min_exp; e < 0; e++) {
        edges.push_back(std::pow(10.0, (double)e));
        edges.push_back(5.0 * std::pow(10.0, (double)e));
    }
    edges.push_back(1.0);
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    // NEP 50: np.clip(f32 alpha, np.float64 edge, 1.0) promotes to float64
    // (the edge is a strong numpy scalar), so values clipped up land exactly
    // on edges[0] and are counted by bin 0's `>= left` test.
    std::vector<double> clipped(alpha.size());
    for (size_t k = 0; k < alpha.size(); k++)
        clipped[k] = std::clamp((double)alpha[k], edges[0], 1.0);

    Json thr = jobj();
    AlphaPlot plot;
    plot.edges = edges;
    plot.log_min_exp = log_min_exp;
    plot.total = N;
    plot.zero_percent = percent(zero, N);
    for (double t : kThr) {
        const float tf = (float)t;  // NEP 50: f32 array > python float
        long long cnt = 0;
        for (float a : alpha) cnt += a > tf;
        thr.add(py_g(t), jobj().add("count", jint(cnt)).add("percent", jnum(percent(cnt, N))));
        plot.thresholds.push_back(t);
        plot.threshold_percent.push_back(percent(cnt, N));
    }
    Json bins = jarr();
    for (size_t b = 0; b + 1 < edges.size(); b++) {
        const double l = edges[b], r = edges[b + 1];
        const bool last = r == edges.back();
        long long cnt = 0;
        for (double x : clipped) cnt += x >= l && (last ? x <= r : x < r);
        bool below = false;
        if (b == 0)
            for (float a : alpha) if ((double)a < l) { below = true; break; }
        bins.push(jobj()
                      .add("range", jstr("[" + py_g(l) + ", " + py_g(r) + (last ? "]" : ")")))
                      .add("contains_alpha_below_range", jbool(below))
                      .add("count", jint(cnt))
                      .add("percent", jnum(percent(cnt, N))));
        plot.percent.push_back(percent(cnt, N));
    }
    Json stats = jobj();
    stats.add("min", N ? jnum(amin) : jnull())
        .add("min_positive", has_pos ? jnum(minpos) : jnull())
        .add("max", N ? jnum(amax) : jnull())
        .add("mean", N ? jnum(mean_f32(alpha)) : jnull())
        .add("median", N ? jnum(median_of(alpha)) : jnull())
        .add("zero_count", jint(zero))
        .add("zero_percent", jnum(percent(zero, N)));
    Json root = jobj();
    root.add("total_gaussians", jint(N))
        .add("alpha_channel", jstr("gaussians[..., 3]"))
        .add("statistics", std::move(stats))
        .add("thresholds_alpha_greater_than", std::move(thr))
        .add("log_bins", std::move(bins));
    if (!write_json(out_dir + "/gaussian_alpha_distribution.json", root)) return false;
    return plot_alpha_distribution(out_dir + "/gaussian_alpha_distribution.png", plot);
}

bool write_scale_distribution(const std::string& out_dir, const float* g, long long N,
                              int stride, double max_scale) {
    if (!(max_scale > 0)) return false;
    const int num_bins = 50;
    // .float().clamp_min(0).numpy().astype(np.float64)
    std::vector<double> scale((size_t)N * 3);
    for (long long i = 0; i < N; i++)
        for (int c = 0; c < 3; c++)
            scale[(size_t)i * 3 + c] = (double)std::max(g[(size_t)i * stride + 4 + c], 0.0f);
    const long long total = (long long)scale.size();
    const double eps = std::max(std::abs(max_scale) * 1e-6, 1e-12);
    // np.linspace(0, max, 51): i * (max / 50) + 0.0, last element = max
    const double step = max_scale / (double)num_bins;
    std::vector<double> edges((size_t)num_bins + 1);
    for (int k = 0; k <= num_bins; k++) edges[(size_t)k] = (double)k * step + 0.0;
    edges.back() = max_scale;
    // np.histogram(np.clip(scale, 0, max), bins=edges): [l, r) bins, last
    // one closed (searchsorted path) — a binary search per value.
    std::vector<long long> counts((size_t)num_bins, 0);
    bool has_pos = false;
    double minpos = 0, smin = 0, smax = 0;
    long long zero = 0, above = 0;
    for (size_t k = 0; k < scale.size(); k++) {
        const double s = scale[k];
        const double x = std::min(std::max(s, 0.0), max_scale);
        const size_t b = x == edges.back()
                             ? (size_t)num_bins - 1
                             : (size_t)(std::upper_bound(edges.begin(), edges.end(), x) - edges.begin()) - 1;
        counts[b]++;
        smin = k ? std::min(smin, s) : s;
        smax = k ? std::max(smax, s) : s;
        if (s > 0) { minpos = has_pos ? std::min(minpos, s) : s; has_pos = true; }
        zero += s <= 0;
        above += s > max_scale + eps;
    }
    Json bins = jarr(), cum = jarr();
    ScalePlot plot;
    plot.max_scale = max_scale;
    plot.edges = edges;
    plot.total = total;
    long long run = 0;
    for (int b = 0; b < num_bins; b++) {
        const double l = edges[(size_t)b], r = edges[(size_t)b + 1];
        run += counts[(size_t)b];
        const double pct = (double)counts[(size_t)b] / (double)std::max(total, 1LL) * 100.0;
        const double cpct = (double)run / (double)std::max(total, 1LL) * 100.0;
        const bool last = r == edges.back();
        bins.push(jobj()
                      .add("range", jstr("[" + py_g(l) + ", " + py_g(r) + (last ? "]" : ")")))
                      .add("contains_scale_above_range", jbool(b == num_bins - 1 && above > 0))
                      .add("count", jint(counts[(size_t)b]))
                      .add("percent", jnum(pct)));
        cum.push(jobj().add("scale_lte", jnum(r)).add("count", jint(run)).add("percent", jnum(cpct)));
        plot.percent.push_back(pct);
        plot.cumulative_percent.push_back(cpct);
    }
    plot.zero_percent = percent(zero, total);
    plot.above_percent = percent(above, total);
    // component stats over the (N,3) view: axis-0 reductions run row by row
    // (sequential per column), np.median per column.
    Json cmin = jarr(), cmax = jarr(), cmean = jarr(), cmed = jarr();
    for (int c = 0; c < 3; c++) {
        std::vector<double> col((size_t)N);
        double acc = 0;
        for (long long i = 0; i < N; i++) {
            col[(size_t)i] = scale[(size_t)i * 3 + c];
            acc += col[(size_t)i];
        }
        cmin.push(jnum(*std::min_element(col.begin(), col.end())));
        cmax.push(jnum(*std::max_element(col.begin(), col.end())));
        cmean.push(jnum(acc / (double)N));
        cmed.push(jnum(median_of(std::move(col))));
    }
    Json stats = jobj();
    stats.add("min", jnum(smin))
        .add("min_positive", has_pos ? jnum(minpos) : jnull())
        .add("max", jnum(smax))
        .add("mean", jnum(pairwise_sum(scale.data(), scale.size(), 1) / (double)total))
        .add("median", jnum(median_of(scale)))
        .add("zero_count", jint(zero))
        .add("zero_percent", jnum(percent(zero, total)))
        .add("above_max_scale_count", jint(above))
        .add("above_max_scale_percent", jnum(percent(above, total)))
        .add("component_min", std::move(cmin))
        .add("component_max", std::move(cmax))
        .add("component_mean", std::move(cmean))
        .add("component_median", std::move(cmed));
    Json root = jobj();
    root.add("total_gaussians", jint(N))
        .add("total_scale_values", jint(total))
        .add("scale_channels", jstr("gaussians[..., 4:7]"))
        .add("scale_metric", jstr("all three scale components flattened"))
        .add("max_scale_from_config", jnum(max_scale))
        .add("bin_count", jint(num_bins))
        .add("statistics", std::move(stats))
        .add("linear_bins", std::move(bins))
        .add("cumulative_scale_less_equal", std::move(cum));
    if (!write_json(out_dir + "/gaussian_scale_distribution.json", root)) return false;
    return plot_scale_distribution(out_dir + "/gaussian_scale_distribution.png", plot);
}

bool write_inference_timing(const std::string& out_dir, const InferenceTiming& t) {
    Json thr = jarr();
    std::string thr_repr = "[";
    for (size_t k = 0; k < t.opacity_thresholds.size(); k++) {
        thr.push(jnum(t.opacity_thresholds[k]));
        thr_repr += (k ? ", " : "") + py_repr(t.opacity_thresholds[k]);
    }
    thr_repr += "]";
    Json root = jobj();
    root.add("device", jstr(t.device))
        .add("num_input_views", jint(t.num_input_views))
        .add("camera_and_depth_seconds", jnum(t.camera_and_depth_seconds))
        .add("image_to_3dgs_seconds", jnum(t.image_to_3dgs_seconds))
        .add("render_seconds", jnum(t.render_seconds))
        .add("use_tto", jbool(t.use_tto))
        .add("tto_n_steps", jint(t.use_tto ? t.tto_n_steps : 0))
        .add("tto_lr", t.use_tto && t.tto_lr_set ? jnum(t.tto_lr) : jnull())
        .add("tto_lpips_weight",
             t.use_tto && t.tto_lpips_set ? jnum(t.tto_lpips_weight) : jnull())
        .add("tto_save_steps", [&] {
            Json a = jarr();
            if (t.use_tto)
                for (long long v : t.tto_save_steps) a.push(jint(v));
            return a;
        }())
        .add("gaussian_save_opacity_thresholds", std::move(thr));
    if (!write_json(out_dir + "/inference_timing.json", root)) return false;
    // save_inference_timing's text form: f"{key}: {value:.6f} s" for float
    // *_seconds, f"{key}: {value}" (python str) otherwise
    char b[128];
    std::string txt = "Inference timing\n";
    txt += "device: " + t.device + "\n";
    txt += "num_input_views: " + std::to_string(t.num_input_views) + "\n";
    std::snprintf(b, sizeof(b), "camera_and_depth_seconds: %.6f s\n", t.camera_and_depth_seconds);
    txt += b;
    std::snprintf(b, sizeof(b), "image_to_3dgs_seconds: %.6f s\n", t.image_to_3dgs_seconds);
    txt += b;
    std::snprintf(b, sizeof(b), "render_seconds: %.6f s\n", t.render_seconds);
    txt += b;
    char tb[256];
    if (t.use_tto) {
        std::string steps = "[";
        for (size_t k = 0; k < t.tto_save_steps.size(); k++)
            steps += (k ? ", " : "") + std::to_string(t.tto_save_steps[k]);
        steps += "]";
        std::snprintf(tb, sizeof(tb), "use_tto: True\ntto_n_steps: %lld\n",
                      t.tto_n_steps);
        txt += tb;
        txt += t.tto_lr_set ? "tto_lr: " + py_repr(t.tto_lr) + "\n"
                            : "tto_lr: None\n";
        txt += t.tto_lpips_set
                   ? "tto_lpips_weight: " + py_repr(t.tto_lpips_weight) + "\n"
                   : "tto_lpips_weight: None\n";
        txt += "tto_save_steps: " + steps + "\n";
    } else {
        txt += "use_tto: False\ntto_n_steps: 0\ntto_lr: None\ntto_lpips_weight: None\n";
    }
    txt += "tto_save_steps: []\n";
    txt += "gaussian_save_opacity_thresholds: " + thr_repr + "\n";
    return write_text(out_dir + "/inference_timing.txt", txt);
}

bool write_tto_losses(const std::string& out_dir, double l1, double ssim,
                      const double* lpips) {
    Json root = jobj();
    root.add("L1", jnum(l1)).add("SSIM", jnum(ssim));
    root.add("LPIPS", lpips ? jnum(*lpips) : jnull());
    std::string txt = "TTO final losses\n";
    auto line = [&](const char* k, double v) {
        char b[128];
        std::snprintf(b, sizeof(b), "%s: %.6f\n", k, v);
        return std::string(b);
    };
    txt += line("L1", l1) + line("SSIM", ssim);
    txt += lpips ? line("LPIPS", *lpips) : std::string("LPIPS: None\n");
    if (!write_json(out_dir + "/tto_losses.json", root)) return false;
    return write_text(out_dir + "/tto_losses.txt", txt);
}

bool write_camera_info(const std::string& out_dir, int num_input_views) {
    return write_text(out_dir + "/camera_info.txt",
                      "Input and render cameras are predicted by the frozen VGGT-Omega camera head.\n"
                      "num_input_views: " + std::to_string(num_input_views) + "\n");
}

}  // namespace qs
