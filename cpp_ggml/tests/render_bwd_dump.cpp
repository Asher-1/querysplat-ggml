// render_backward dump tool — parity harness for benchmarks/render_backward_parity.py.
//
// Reads a raw gaussian bin (N x (11+color_dim) f32, the querysplat layout) and
// a cameras bin (S*16 row-major w2c + S*4 intrinsics), renders the input views
// with render_gaussians, builds a DETERMINISTIC pseudo-random d_images
// (identical LCG on the python side), runs render_backward, and dumps:
//   <prefix>_images.bin    (S,3,H,W) f32 forward render
//   <prefix>_dimages.bin   (S,3,H,W) f32 loss gradient seed
//   <prefix>_grads.bin     (N,11+color_dim) f32, packed like the gaussian
//                          buffer: [dpos3 | dop | dscale3 | drot4 | dcolor3]
//   <prefix>_meta.txt      "N S H W color_dim"
//
// Usage: test_render_bwd <gaussians.bin> <cameras.bin> S H W <prefix>
#include "render/rasterizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace {

// Deterministic LCG shared with the python parity script: float32 in [0,1).
float lcg_next(uint64_t* state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((*state >> 40) & 0xFFFFFF) / (float)0x1000000;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
                     "usage: %s <gaussians.bin> <cameras.bin> S H W <prefix>\n",
                     argv[0]);
        return 2;
    }
    const std::string gpath = argv[1], cpath = argv[2];
    const int S = std::atoi(argv[3]), H = std::atoi(argv[4]),
              W = std::atoi(argv[5]);
    const std::string prefix = argv[6];

    auto read_bin = [](const std::string& p, std::vector<float>* out) {
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        const long bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        out->resize((size_t)bytes / 4);
        const size_t got = fread(out->data(), 4, out->size(), f);
        fclose(f);
        return got == out->size();
    };
    std::vector<float> graw, craw;
    if (!read_bin(gpath, &graw) || !read_bin(cpath, &craw)) {
        std::fprintf(stderr, "cannot read %s or %s\n", gpath.c_str(),
                     cpath.c_str());
        return 1;
    }
    // stride = 11 + color_dim, solved from the byte count (23 sh0 / 14 sh1)
    const int64_t total = (int64_t)graw.size();
    const int stride = total % 23 == 0 ? 23 : 14;
    const int64_t N = total / stride;

    qs::GaussianOutput g;
    g.N = N;
    g.color_dim = stride - 11;
    g.data = std::move(graw);
    g.opacity.resize((size_t)N);
    for (int64_t i = 0; i < N; i++)
        g.opacity[(size_t)i] = g.data[(size_t)i * stride + 3];

    qs::RenderConfig cfg;
    cfg.width = W;
    cfg.height = H;
    cfg.sh_degree = g.color_dim == 3 ? 0 : 1;
    const float* w2c = craw.data();
    const float* intr = craw.data() + (size_t)S * 16;

    qs::RenderedViews views = qs::render_gaussians(g, w2c, intr, S, cfg);

    // deterministic d_images seed (identical LCG in the python script)
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    std::vector<float> dimages((size_t)S * 3 * H * W);
    for (float& x : dimages) x = lcg_next(&st);

    qs::GaussianGrads grads;
    if (!qs::render_backward(g, w2c, intr, S, cfg, views, dimages.data(),
                             &grads)) {
        std::fprintf(stderr, "render_backward failed (sh0 only)\n");
        return 1;
    }

    auto write_bin = [](const std::string& p, const void* data, size_t bytes) {
        FILE* f = fopen(p.c_str(), "wb");
        if (!f) return false;
        const size_t got = fwrite(data, 1, bytes, f);
        fclose(f);
        return got == bytes;
    };
    const std::string ip = prefix + "_images.bin";
    const std::string dp = prefix + "_dimages.bin";
    const std::string gp = prefix + "_grads.bin";
    const std::string mp = prefix + "_meta.txt";
    if (!write_bin(ip, views.images.data(),
                   views.images.size() * sizeof(float)) ||
        !write_bin(dp, dimages.data(), dimages.size() * sizeof(float))) {
        std::fprintf(stderr, "write failed under %s\n", prefix.c_str());
        return 1;
    }
    // pack grads into the gaussian layout (N, stride)
    std::vector<float> packed((size_t)N * stride, 0.0f);
    for (int64_t i = 0; i < N; i++) {
        float* row = &packed[(size_t)i * stride];
        for (int c = 0; c < 3; c++) row[c] = grads.pos[(size_t)i * 3 + c];
        row[3] = grads.opacity[(size_t)i];
        for (int c = 0; c < 3; c++) row[4 + c] = grads.scale[(size_t)i * 3 + c];
        for (int c = 0; c < 4; c++) row[7 + c] = grads.rot[(size_t)i * 4 + c];
        for (int c = 0; c < grads.color_dim; c++)
            row[11 + c] = grads.color[(size_t)i * grads.color_dim + c];
    }
    if (!write_bin(gp, packed.data(), packed.size() * sizeof(float))) return 1;
    FILE* fm = fopen(mp.c_str(), "w");
    if (fm) {
        std::fprintf(fm, "%lld %d %d %d %d\n", (long long)N, S, H, W,
                     g.color_dim);
        fclose(fm);
    }
    std::printf("N=%lld S=%d H=%d W=%d color_dim=%d -> %s*\n", (long long)N, S,
                H, W, g.color_dim, prefix.c_str());
    return 0;
}
