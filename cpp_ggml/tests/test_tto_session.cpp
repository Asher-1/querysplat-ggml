// TTO session verification (t4): linear-probe loss check.
//
// L(mem) = sum(d .* raw(mem)). AdamW's FIRST step is a sign update (m_hat
// and v_hat normalize to +-1), so the analytic per-step drop is not
// lr*||d||^2; the meaningful gate is a MONOTONE decreasing loss over 3
// steps — a systematically biased decoder backward shows up as a step-up
// or a collapse. This exercises the full session: graph F (forward), the
// loss-side components' d(raw) -> the decoder backward (graph B) and the
// in-place AdamW on the KV memory.
//
// Usage: test_tto_session <decoder.gguf> [S=1]
#include "backend.hpp"
#include "common.hpp"
#include "querysplat_graph.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using mapggml::log_info;
using mapggml::log_error;

namespace {

std::vector<float> randn(size_t n, std::mt19937& rng, float scale) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <decoder.gguf> [S=1]\n", argv[0]);
        return 2;
    }
    const int S = argc > 2 ? std::atoi(argv[2]) : 1;
    const int H = 512, W = 512;

    mapggml::Backend be = mapggml::init_backend("cpu", 4);
    qs::DecoderConfig cfg;
    if (!qs::read_decoder_config(argv[1], &cfg)) return 1;
    qs::DecoderRuntimeOptions opts;
    qs::QuerySplatDecoder dec(argv[1], be, opts);
    if (!dec.valid()) return 1;
    if (!dec.ensure_shape(S, H, W, 17)) return 1;

    std::mt19937 rng(42);
    const int Nagg = 17 + (H / cfg.patch_size) * (W / cfg.patch_size);
    const int C = cfg.enc_embed_dim;
    // synthetic encoder tokens / inputs (values don't matter for the check)
    std::vector<float> layers((size_t)4 * C * Nagg * S);
    for (auto& x : layers) x = (float)((rng() % 2000) - 1000) / 1000.f;
    std::vector<float> rgb((size_t)S * 3 * H * W, 0.5f);
    std::vector<float> plk((size_t)S * 6 * H * W);
    for (auto& x : plk) x = (float)((rng() % 2000) - 1000) / 1000.f;

    const float lr = 1.0f;
    if (!dec.tto_begin(S, H, W, 17, layers.data(), rgb.data(), plk.data(), lr))
        return 1;
    log_info("tto session built; running probe forward");

    // Linear probe: L(mem) = sum(d .* raw(mem)). AdamW's FIRST step is a
    // sign update (m_hat and v_hat normalize to +-1), so the analytic
    // per-step drop is not lr*||d||^2; the meaningful gate is a MONOTONE
    // decreasing loss over 3 steps — a systematically biased decoder
    // backward shows up as a step-up or a collapse.
    auto probe = [&](const std::vector<float>& a, const std::vector<float>& b) {
        double acc = 0;
        for (size_t i = 0; i < a.size(); i++) acc += (double)a[i] * b[i];
        return acc;
    };
    std::vector<float> dd, dc;
    double L_prev = 0;
    bool ok = true;
    for (int step = 0; step < 3; step++) {
        std::vector<float> deconv_raw, color_raw;
        if (!dec.tto_forward(deconv_raw, color_raw)) return 1;
        if (step == 0) {
            dd = randn(deconv_raw.size(), rng, 0.05f);
            dc = randn(color_raw.size(), rng, 0.05f);
        }
        const double L = probe(dd, deconv_raw) + probe(dc, color_raw);
        log_info("probe step %d: L=%.6f", step, L);
        if (step > 0 && L >= L_prev) {
            log_error("probe: loss did not decrease at step %d", step);
            ok = false;
        }
        L_prev = L;
        if (!dec.tto_backward(dd.data(), dc.data(), step)) return 1;
    }
    log_info("TTO PROBE %s (final L=%.4f)", ok ? "PASS" : "FAIL", L_prev);
    return ok ? 0 : 1;
}
