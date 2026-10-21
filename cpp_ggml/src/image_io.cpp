// Image input — implementation. See image_io.hpp.

#include "image_io.hpp"

#include "common.hpp"
#include "gguf_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef QS_USE_LIBJPEG
// PIL (the official chain) decodes JPEGs through libjpeg-turbo, whose ISLOW
// IDCT is bit-compatible with libjpeg's; stb_image's IDCT differs by ±1-2 per
// channel, which the model amplifies. JPEG-content files (whatever their
// extension) are routed here.
#include <setjmp.h>
#include <jpeglib.h>
#endif
#ifdef QS_USE_LIBWEBP
// PIL decodes WebP through libwebp itself; same library, same pixels.
#include <webp/decode.h>
#endif

// stb: the PNG/BMP/fallback decoder (pure C++ preprocessing, no python).
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace qs {

using mapggml::log_error;
using mapggml::log_info;

// Upscaling on both axes: bit-exact replication of ATen's CPU
// upsample_bilinear2d (what F.interpolate(antialias=True) reduces to when
// nothing shrinks — verified 1.2e-7 on the DL3DV 270->512 geometry):
//   src = scale*(dst+0.5)-0.5, clamped to 0; i1 = i0+1 clamped;
//   val = h0*(w0*p00 + w1*p01) + h1*(w0*p10 + w1*p11) — the exact
// accumulation order of ATen's out_frame loop.
// Any axis shrinking (e.g. a 6205x4135 photo cropped to 4135x4135 -> 512, or
// the evaluator's 512 -> 256 GT): ATen's antialias path, two separable passes
// of a triangle filter of half-width support = max(scale, 1) with
// per-output-pixel normalization (matches F.interpolate(antialias=True) to
// ~2.4e-7 on the 4135 -> 512 geometry; impulse-response-derived). Plain
// bilinear there produces materially different model inputs (courtyard
// cpp-vs-py was 30 dB for exactly this reason).
void resize_rgb_hwc(const float* src, int sh, int sw, int H, int W, float* dst) {
    const float scale_x = (float)sw / (float)W;
    const float scale_y = (float)sh / (float)H;
    if (scale_x > 1.0f || scale_y > 1.0f) {
        const float sup_x = std::max(scale_x, 1.0f), sup_y = std::max(scale_y, 1.0f);
        // pass 1: horizontal sw -> W (rows unchanged)
        std::vector<float> tmp((size_t)sh * W * 3);
        for (int y = 0; y < sh; y++) {
            for (int x = 0; x < W; x++) {
                const float c = scale_x * ((float)x + 0.5f) - 0.5f;
                const int lo = std::max(0, (int)std::floor(c - sup_x));
                const int hi = std::min(sw - 1, (int)std::ceil(c + sup_x));
                float wsum = 0.f;
                for (int j = lo; j <= hi; j++)
                    wsum += std::max(0.f, 1.f - std::fabs((float)j - c) / sup_x);
                if (wsum <= 0.f) wsum = 1.f;
                float* d = &tmp[((size_t)y * W + x) * 3];
                d[0] = d[1] = d[2] = 0.f;
                for (int j = lo; j <= hi; j++) {
                    const float w = std::max(0.f, 1.f - std::fabs((float)j - c) / sup_x) / wsum;
                    const float* p = &src[((size_t)y * sw + j) * 3];
                    d[0] += w * p[0];
                    d[1] += w * p[1];
                    d[2] += w * p[2];
                }
            }
        }
        // pass 2: vertical sh -> H
        for (int y = 0; y < H; y++) {
            const float c = scale_y * ((float)y + 0.5f) - 0.5f;
            const int lo = std::max(0, (int)std::floor(c - sup_y));
            const int hi = std::min(sh - 1, (int)std::ceil(c + sup_y));
            float wsum = 0.f;
            for (int j = lo; j <= hi; j++)
                wsum += std::max(0.f, 1.f - std::fabs((float)j - c) / sup_y);
            if (wsum <= 0.f) wsum = 1.f;
            for (int x = 0; x < W; x++) {
                float* d = &dst[((size_t)y * W + x) * 3];
                d[0] = d[1] = d[2] = 0.f;
                for (int j = lo; j <= hi; j++) {
                    const float w = std::max(0.f, 1.f - std::fabs((float)j - c) / sup_y) / wsum;
                    const float* p = &tmp[((size_t)j * W + x) * 3];
                    d[0] += w * p[0];
                    d[1] += w * p[1];
                    d[2] += w * p[2];
                }
            }
        }
        return;
    }
    for (int y = 0; y < H; y++) {
        float sy = scale_y * ((float)y + 0.5f) - 0.5f;
        sy = sy < 0.0f ? 0.0f : sy;
        const int y0 = (int)sy;
        const int y1 = y0 < sh - 1 ? y0 + 1 : y0;
        const float ly = sy - (float)y0;
        const float h0 = 1.0f - ly, h1 = ly;
        for (int x = 0; x < W; x++) {
            float sx = scale_x * ((float)x + 0.5f) - 0.5f;
            sx = sx < 0.0f ? 0.0f : sx;
            const int x0 = (int)sx;
            const int x1 = x0 < sw - 1 ? x0 + 1 : x0;
            const float lx = sx - (float)x0;
            const float w0 = 1.0f - lx, w1 = lx;
            const float* p00 = &src[((size_t)y0 * sw + x0) * 3];
            const float* p01 = &src[((size_t)y0 * sw + x1) * 3];
            const float* p10 = &src[((size_t)y1 * sw + x0) * 3];
            const float* p11 = &src[((size_t)y1 * sw + x1) * 3];
            float* d = &dst[((size_t)y * W + x) * 3];
            for (int c = 0; c < 3; c++)
                d[c] = h0 * (w0 * p00[c] + w1 * p01[c]) + h1 * (w0 * p10[c] + w1 * p11[c]);
        }
    }
}

std::vector<float> resize_planes(const float* src, int n, int h, int w, int H, int W) {
    std::vector<float> out((size_t)n * 3 * H * W);
    std::vector<float> a((size_t)h * w * 3), b((size_t)H * W * 3);
    for (int v = 0; v < n; v++) {
        const float* s = src + (size_t)v * 3 * h * w;
        for (int c = 0; c < 3; c++)
            for (size_t k = 0; k < (size_t)h * w; k++) a[k * 3 + c] = s[(size_t)c * h * w + k];
        resize_rgb_hwc(a.data(), h, w, H, W, b.data());
        float* d = out.data() + (size_t)v * 3 * H * W;
        for (int c = 0; c < 3; c++)
            for (size_t k = 0; k < (size_t)H * W; k++) d[(size_t)c * H * W + k] = b[k * 3 + c];
    }
    return out;
}

// Extension set = infer.py IMAGE_EXTENSIONS (name.lower().endswith(...)).
namespace {

constexpr const char* kImageExts[] = {".png", ".jpg", ".jpeg", ".webp", ".bmp"};

bool has_image_ext(const std::string& name) {
    const auto dot = name.rfind('.');
    if (dot == std::string::npos) return false;
    std::string ext = name.substr(dot);
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    for (const char* e : kImageExts)
        if (ext == e) return true;
    return false;
}

}  // namespace

bool collect_images(const std::string& spec, std::vector<std::string>* out) {
    std::error_code ec;
    if (std::filesystem::is_directory(spec, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(spec, ec))
            if (entry.is_regular_file(ec) &&
                has_image_ext(entry.path().filename().string()))
                out->push_back(entry.path().string());
        std::sort(out->begin(), out->end());  // sorted, like scripts/infer.py
        return !out->empty();
    }
    std::string item;
    for (size_t pos = 0; pos <= spec.size(); pos++) {
        if (pos == spec.size() || spec[pos] == ',') {
            if (!item.empty()) out->push_back(item);
            item.clear();
        } else {
            item += spec[pos];
        }
    }
    return !out->empty();
}

bool load_images_as_frames(const std::vector<std::string>& paths, int H, int W,
                           std::vector<float>* frames) {
    const size_t plane = (size_t)H * W;
    frames->assign(paths.size() * 3 * plane, 0.f);
    std::vector<float> rgb((size_t)W * H * 3);
    for (size_t s = 0; s < paths.size(); s++) {
        int w = 0, h = 0, n = 0;
        std::vector<unsigned char> jpeg_pixels;
        std::vector<unsigned char> webp_pixels;
        uint8_t* data = nullptr;
        bool from_stb = false;
        {
            // content sniff: JPEG starts FF D8 FF (the courtyard inputs are
            // JPEGs with a .png extension — trust the bytes, not the name);
            // WebP is "RIFF....WEBP"
            std::ifstream f(paths[s], std::ios::binary);
            unsigned char magic[12] = {0};
            f.read(reinterpret_cast<char*>(magic), 12);
            const bool is_jpeg =
                magic[0] == 0xFF && magic[1] == 0xD8 && magic[2] == 0xFF;
            const bool is_webp = std::memcmp(magic, "RIFF", 4) == 0 &&
                                 std::memcmp(magic + 8, "WEBP", 4) == 0;
#ifdef QS_USE_LIBJPEG
            if (is_jpeg) {
                // PIL (the official chain) decodes JPEGs through
                // libjpeg-turbo, whose ISLOW IDCT is bit-compatible with
                // libjpeg's; stb_image's IDCT differs by ±1-2 per channel,
                // which the model amplifies.
                struct jpeg_decompress_struct cinfo;
                struct jpeg_error_mgr jerr;
                FILE* fp = fopen(paths[s].c_str(), "rb");
                if (!fp) {
                    log_error("failed to open %s", paths[s].c_str());
                    return false;
                }
                cinfo.err = jpeg_std_error(&jerr);
                jpeg_create_decompress(&cinfo);
                jpeg_stdio_src(&cinfo, fp);
                jpeg_read_header(&cinfo, TRUE);
                // defaults match PIL's JpegImagePlugin (ISLOW IDCT, fancy
                // upsampling)
                cinfo.out_color_space = JCS_RGB;
                jpeg_start_decompress(&cinfo);
                w = (int)cinfo.output_width;
                h = (int)cinfo.output_height;
                n = 3;
                jpeg_pixels.resize((size_t)w * h * 3);
                while (cinfo.output_scanline < (unsigned)h) {
                    unsigned char* row = jpeg_pixels.data() +
                        (size_t)cinfo.output_scanline * w * 3;
                    JSAMPROW rows[1] = {row};
                    jpeg_read_scanlines(&cinfo, rows, 1);
                }
                jpeg_finish_decompress(&cinfo);
                jpeg_destroy_decompress(&cinfo);
                fclose(fp);
                data = jpeg_pixels.data();
            }
#endif
#ifdef QS_USE_LIBWEBP
            if (is_webp) {
                // PIL: WebPAnimDecoder (non-premultiplied RGBA) -> convert("RGB")
                // drops alpha; still images decode straight into the canvas
                const std::vector<uint8_t> bytes = mapggml::read_file_bytes(paths[s]);
                int ww = 0, hh = 0;
                uint8_t* px = WebPDecodeRGBA(bytes.data(), bytes.size(), &ww, &hh);
                if (px) {
                    w = ww;
                    h = hh;
                    n = 3;
                    webp_pixels.resize((size_t)w * h * 3);
                    for (size_t k = 0; k < (size_t)w * h; k++)
                        for (int c = 0; c < 3; c++) webp_pixels[k * 3 + c] = px[k * 4 + c];
                    WebPFree(px);
                    data = webp_pixels.data();
                }
            }
#endif
            if (!data) {
                (void)is_jpeg;
                (void)is_webp;
                data = stbi_load(paths[s].c_str(), &w, &h, &n, 3);
                from_stb = data != nullptr;
            }
        }
        if (!data) {
            log_error("failed to decode %s: %s", paths[s].c_str(),
                      stbi_failure_reason());
            return false;
        }
        const int side = std::min(w, h);
        const int x0 = (w - side) / 2, y0 = (h - side) / 2;
        std::vector<float> sq((size_t)side * side * 3);
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                const uint8_t* p = data + ((size_t)(y0 + y) * w + (x0 + x)) * 3;
                float* q = &sq[((size_t)y * side + x) * 3];
                q[0] = p[0] / 255.f;
                q[1] = p[1] / 255.f;
                q[2] = p[2] / 255.f;
            }
        }
        if (from_stb) stbi_image_free(data);
        resize_rgb_hwc(sq.data(), side, side, H, W, rgb.data());
        float* dst = frames->data() + s * 3 * plane;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    dst[(size_t)c * plane + (size_t)y * W + x] =
                        rgb[((size_t)y * W + x) * 3 + c];
        log_info("image %zu: %s (%dx%d) -> crop %d -> %dx%d", s,
                 paths[s].c_str(), w, h, side, W, H);
    }
    return true;
}

}  // namespace qs
