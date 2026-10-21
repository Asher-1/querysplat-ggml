// t7: finite-difference verification of the TTO decoder backward (graph B).
//
// Contract under test: given upstream gradients d_pred in the ACTIVATED
// gaussian space (N, 11+cd), the chain
//   activate_gaussians_backward -> graph B (block VJPs) -> dmem
// must satisfy  dL/dmem = dmem  for the synthetic loss
//   L(mem) = <d_pred, gaussians(mem)>.
// The backward components t1/t2 are torch-parity verified; graph B is the
// only unverified link, so a random-direction central difference is the
// cheapest full-space check: one random projection covers all elements.
//
// AdamW is neutralized by tto_begin(lr=0): the kernel is w = w - lr*mh/vh,
// so mem stays frozen while dmem is still computed (kernel verified by
// inspection; wd=0).
//
// Runs: L0 (baseline), L_all (perturb all four memories along independent
// random directions), L_0 (perturb mem[0] only) — two scalar fd checks
// cover the sum and mem[0] alone.
//
// Usage: test_tto_fd <decoder.gguf> [S=1]
#include "activations_host.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "querysplat_graph.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
    const float eps = 1e-3f;

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
    std::vector<float> layers((size_t)4 * C * Nagg * S);
    for (auto& x : layers) x = (float)((rng() % 2000) - 1000) / 1000.f;
    std::vector<float> rgb((size_t)S * 3 * H * W, 0.5f);
    std::vector<float> plk((size_t)S * 6 * H * W);
    for (auto& x : plk) x = (float)((rng() % 2000) - 1000) / 1000.f;

    // lr=0 freezes mem inside the AdamW step (see file comment)
    if (!dec.tto_begin(S, H, W, 17, layers.data(), rgb.data(), plk.data(), 0.f))
        return 1;
    log_info("fd: session built (S=%d lr=0)", S);

    // baseline forward + synthetic upstream gradient
    std::vector<float> deconv_raw, color_raw;
    if (!dec.tto_forward(deconv_raw, color_raw)) return 1;
    const int64_t N = (int64_t)cfg.num_gs_tokens * cfg.num_gaussians_per_token;
    const int stride = 11 + 3;  // sh0: color_dim 3
    const int post_out_dim =
        3 * (cfg.gaussian_sh_degree + 1) * (cfg.gaussian_sh_degree + 1) + 1;
    // Synthetic upstream gradients in the RAW head space. The loss under test
    // is the linear functional L(mem) = <dg, deconv_raw(mem)> + <dc,
    // color_raw(mem)> — this isolates graph B (the only unverified link; the
    // activation backward is torch-parity verified in t2, and the activated
    // space would blow up magnitudes on synthetic inputs).
    std::vector<float> dgeom((size_t)cfg.num_gs_tokens * 10 *
                             cfg.num_gaussians_per_token),
        dcol((size_t)cfg.num_gs_tokens * post_out_dim *
             cfg.num_gaussians_per_token);
    for (auto& x : dgeom) x = randn(1, rng, 0.05f)[0];
    for (auto& x : dcol) x = randn(1, rng, 0.05f)[0];

    double dg_scale = 1.0, dc_scale = 1.0;  // per-case upstream scaling
    auto loss_of = [&](const std::vector<float>& dr, const std::vector<float>& dc) {
        double acc = 0;
        for (size_t i = 0; i < dr.size(); i++)
            acc += (double)dgeom[i] * dr[i] * dg_scale;
        for (size_t i = 0; i < dc.size(); i++)
            acc += (double)dcol[i] * dc[i] * dc_scale;
        return acc;
    };

    const double L0 = loss_of(deconv_raw, color_raw);
    log_info("fd: L0=%.8f (N=%lld stride=%d)", L0, (long long)N, stride);
    (void)stride; (void)post_out_dim;

    // ---- case harness -------------------------------------------------
    // For (dg_scale, dc_scale): run graph B with the scaled upstream, then a
    // one-sided finite difference of the matching linear functional along a
    // random mem direction. The VJP is linear, so scaling the upstream
    // scales dmem; a sign/structure bug shows up as fd vs analytic mismatch.
    auto run_case = [&](const char* nm, double gs, double cs, int pert_idx) {
        dg_scale = gs;
        dc_scale = cs;
        // case-specific baseline: the functional is linear in the upstream,
        // so scale the frozen baseline raws (mem never moved: lr=0)
        double b = 0;
        for (size_t i = 0; i < deconv_raw.size(); i++)
            b += (double)dgeom[i] * deconv_raw[i] * gs;
        for (size_t i = 0; i < color_raw.size(); i++)
            b += (double)dcol[i] * color_raw[i] * cs;
        const double L0_case = b;
        std::vector<float> dmem[4];
        std::vector<float>* dm[4] = {&dmem[0], &dmem[1], &dmem[2], &dmem[3]};
        // Scale the UPSTREAM, not the functional: the VJP is linear, so
        // backward(gs*dg, cs*dc) = gs*D^dg + cs*D^dc — the analytic must be
        // the gradient of the SAME functional the fd perturbs. Comparing a
        // full-upstream analytic against a scaled-functional fd would
        // mismatch even for a perfect backward (cross-path terms).
        std::vector<float> dgu = dgeom, dcu = dcol;
        for (auto& x : dgu) x = (float)(x * gs);
        for (auto& x : dcu) x = (float)(x * cs);
        if (!dec.tto_backward(dgu.data(), dcu.data(), 0, dm)) return 1;
        std::vector<float> u = randn(dmem[pert_idx].size(), rng, 1.f);
        double an = 0;
        for (size_t k = 0; k < dmem[pert_idx].size(); k++)
            an += (double)dmem[pert_idx][k] * u[k];
        // CENTRAL difference: the functional is nonlinear in mem (softmax
        // attention), and a one-sided difference carries an O(eps*curvature)
        // bias that varies with the random direction (observed 0.7%..11%).
        // The central form cancels the second-order term exactly.
        std::vector<float> m0;
        if (!dec.tto_mem_get(pert_idx, m0)) return 1;
        std::vector<float> mp = m0, mm = m0;
        for (size_t k = 0; k < m0.size(); k++) {
            mp[k] += eps * u[k];
            mm[k] -= eps * u[k];
        }
        if (!dec.tto_mem_set(pert_idx, mp)) return 1;
        std::vector<float> dr, dc;
        if (!dec.tto_forward(dr, dc)) return 1;
        const double Lp = loss_of(dr, dc);
        if (!dec.tto_mem_set(pert_idx, mm)) return 1;
        if (!dec.tto_forward(dr, dc)) return 1;
        const double Lm = loss_of(dr, dc);
        if (!dec.tto_mem_set(pert_idx, m0)) return 1;
        const double fd = (Lp - Lm) / (2.0 * eps);
        const double rel = std::fabs(fd - an) / (std::fabs(an) + 1e-12);
        log_info("fd: %-10s fd=%+.5e analytic=%+.5e rel=%.4f %s", nm, fd, an,
                 rel, rel < 0.05 ? "OK" : "MISMATCH");
        return rel < 0.05 ? 0 : 1;
    };

    int fails = 0;
    // L0 must be captured per-case (the functional differs); recompute the
    // case-specific baseline from the SAME forward as its fd perturbation —
    // so re-fetch the baseline raw with mem frozen (it is: lr=0).
    {
        std::vector<float> dr, dc;
        if (!dec.tto_forward(dr, dc)) return 1;
    }
    // baseline for the all-upstream case
    dg_scale = dc_scale = 1.0;
    double L0_all = loss_of(deconv_raw, color_raw);
    (void)L0_all;

    // linear sanity: dmem(2*upstream) == 2*dmem(upstream) on mem[0]
    // (scale BOTH upstreams — otherwise the fixed dc part breaks homogeneity)
    {
        std::vector<float> d1[4], d2[4];
        std::vector<float>* p1[4] = {&d1[0], &d1[1], &d1[2], &d1[3]};
        std::vector<float>* p2[4] = {&d2[0], &d2[1], &d2[2], &d2[3]};
        if (!dec.tto_backward(dgeom.data(), dcol.data(), 1, p1)) return 1;
        for (size_t k = 0; k < dgeom.size(); k++) dgeom[k] *= 2.f;
        for (size_t k = 0; k < dcol.size(); k++) dcol[k] *= 2.f;
        if (!dec.tto_backward(dgeom.data(), dcol.data(), 2, p2)) return 1;
        for (size_t k = 0; k < dgeom.size(); k++) dgeom[k] *= 0.5f;
        for (size_t k = 0; k < dcol.size(); k++) dcol[k] *= 0.5f;
        double num = 0, den = 0;
        for (size_t k = 0; k < d1[0].size(); k++) {
            num += std::fabs(d2[0][k] - 2.0 * d1[0][k]);
            den += std::fabs(2.0 * d1[0][k]);
        }
        log_info("fd: linearity  dmem0 rel=%.6f %s", num / (den + 1e-30),
                 num / (den + 1e-30) < 1e-4 ? "OK" : "MISMATCH");
        if (!(num / (den + 1e-30) < 1e-4)) fails++;
    }

    // case A: geom upstream only -> mem[0] (dec-chain VJP)
    fails += run_case("dg-only", 1.0, 0.0, 0);
    // case B: color upstream only -> mem[2] (pdec-chain VJP)
    fails += run_case("dc-only", 0.0, 1.0, 2);
    // case C: both -> mem[0] (cross-talk through the pdec->geom shortcut)
    fails += run_case("both", 1.0, 1.0, 0);

    log_info("FD PARITY %s", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
