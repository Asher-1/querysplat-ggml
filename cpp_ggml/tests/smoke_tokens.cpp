// M0 acceptance smoke: run the embedded VGGT-Omega encoder with
// RuntimeOptions::expose_tokens and verify the cached-layer token export.
//
// Usage: test_smoke_tokens <vggt-omega.gguf> [S] [H] [W] [device]
//
// Layout rules under test (per the repo memory-order contract):
//   cached_tokens = 4 blocks appended in cached_layer_idx order; block i is
//   ggml ne {2048, Nagg, S} == torch (S, Nagg, 2048) via
//   reshape(d, c, b, a); patch_token_start == 1 + num_register_tokens.
#include "backend.hpp"
#include "gguf_loader.hpp"
#include "vggt_graph.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace mapggml;

namespace {

struct Stats {
    double mean = 0.0;
    float mn = 0.0f;
    float mx = 0.0f;
    bool finite = true;
};

Stats stats_of(const std::vector<float>& v, size_t offset, size_t count) {
    Stats s;
    double acc = 0.0;
    s.mn = v[offset];
    s.mx = v[offset];
    for (size_t i = 0; i < count; i++) {
        const float x = v[offset + i];
        if (!std::isfinite(x)) s.finite = false;
        acc += x;
        s.mn = x < s.mn ? x : s.mn;
        s.mx = x > s.mx ? x : s.mx;
    }
    s.mean = acc / double(count);
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <vggt-omega.gguf> [S=2] [H=512] [W=512] "
                     "[device=auto]\n",
                     argv[0]);
        return 2;
    }
    const int S = argc > 2 ? std::atoi(argv[2]) : 2;
    const int H = argc > 3 ? std::atoi(argv[3]) : 512;
    const int W = argc > 4 ? std::atoi(argv[4]) : 512;
    const char* device = argc > 5 ? argv[5] : "auto";

    auto model = load_gguf(argv[1]);
    if (!model) {
        log_error("failed to load %s", argv[1]);
        return 1;
    }
    const int nreg = model->meta.num_register_tokens;
    const int C2 = 2 * model->meta.embed_dim;
    const int P = (H / model->meta.patch_size) * (W / model->meta.patch_size);
    const int Nagg = 1 + nreg + P;

    RuntimeOptions opts;
    opts.expose_tokens = true;
    Backend be = init_backend(device, (int)std::thread::hardware_concurrency());
    VGGTRuntime rt(*model, be, S, H, W, opts);
    if (!rt.valid()) {
        log_error("runtime init failed");
        return 1;
    }

    // deterministic synthetic frames in [0,1]
    std::vector<float> frames((size_t)S * 3 * H * W);
    for (size_t i = 0; i < frames.size(); i++)
        frames[i] = float((i * 2654435761u % 1000u) / 1000.0);

    VGGTOutputs out;
    if (!rt.run(frames.data(), out)) {
        log_error("run failed");
        return 1;
    }

    // ---- contract checks ----
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
        if (!ok) failures++;
    };

    std::printf("cached_layer_idx: ");
    for (int idx : out.cached_layer_idx) std::printf("%d ", idx);
    std::printf("\n");

    check(out.cached_layer_idx.size() == 4 &&
              out.cached_layer_idx[0] == 4 && out.cached_layer_idx[1] == 11 &&
              out.cached_layer_idx[2] == 17 && out.cached_layer_idx[3] == 23,
          "cached_layer_idx == {4, 11, 17, 23}");
    check(out.patch_token_start == 1 + nreg,
          "patch_token_start == 1 + num_register_tokens");
    const size_t per_layer = (size_t)C2 * Nagg * S;
    check(out.cached_tokens.size() == per_layer * 4,
          "cached_tokens size == 4 * 2C * Nagg * S");

    for (int i = 0; i < 4; i++) {
        const Stats st = stats_of(out.cached_tokens, i * per_layer, per_layer);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "layer %d: mean=%.6f min=%.4f max=%.4f finite=%s",
                      out.cached_layer_idx[i], st.mean, st.mn, st.mx,
                      st.finite ? "yes" : "no");
        check(st.finite, buf);
    }
    check(!out.pose_enc.empty() && out.pose_enc.size() == (size_t)9 * S,
          "pose_enc == (S, 9)");
    check(!out.depth.empty() &&
              out.depth.size() == (size_t)S * out.out_h * out.out_w,
          "depth == (S, out_h, out_w)");

    std::printf("%s (%d failures)\n", failures == 0 ? "SMOKE PASS" : "SMOKE FAIL",
                failures);
    return failures == 0 ? 0 : 1;
}
