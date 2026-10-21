// TTO component dump tool — parity harness for benchmarks/tto_components_parity.py.
//
//   test_tto_components ssim <pred.bin> <gt.bin> V H W <prefix>
//     Writes <prefix>_ssim_val.txt (fused_ssim mean) and
//     <prefix>_ssim_grad.bin (V,3,H,W) for the loss l1 + 0.2*(1-s)/2 with
//     lpips_weight = 0 (the LPIPS term is verified separately in t3).
//
//   test_tto_components act <geom_raw.bin> <color_raw.bin> <d_gaussians.bin>
//                         n_tokens gpt post_out scale_cap z_offset sh_deg
//                         sh_rest <prefix>
//     Writes <prefix>_act_dgeom.bin / <prefix>_act_dcolor.bin.
#include "activations_host.hpp"
#include "backend.hpp"
#include "lpips_graph.hpp"
#include "tto_loss.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool read_bin(const std::string& p, std::vector<float>* out) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->resize((size_t)bytes / 4);
    const size_t got = fread(out->data(), 4, out->size(), f);
    fclose(f);
    return got == out->size();
}

bool write_bin(const std::string& p, const void* data, size_t bytes) {
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) return false;
    const size_t got = fwrite(data, 1, bytes, f);
    fclose(f);
    return got == bytes;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s ssim <pred.bin> <gt.bin> V H W <prefix>\n"
                     "       %s lpips <gguf> <pred.bin> <gt.bin> V H W <prefix>\n"
                     "       %s act <geom.bin> <color.bin> <dgauss.bin> "
                     "n_tokens gpt post_out cap zoff sh rest <prefix>\n",
                     argv[0], argv[0], argv[0]);
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "ssim" && argc == 8) {
        std::vector<float> pred, gt;
        if (!read_bin(argv[2], &pred) || !read_bin(argv[3], &gt)) return 1;
        const int V = std::atoi(argv[4]), H = std::atoi(argv[5]),
                  W = std::atoi(argv[6]);
        const std::string prefix = argv[7];
        qs::TtoLossOut out;
        if (!qs::tto_loss_value_and_grad(pred.data(), gt.data(), V, H, W,
                                         0.0f, nullptr, &out))
            return 1;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.8f %.8f\n", out.l1, out.ssim);
        FILE* f = fopen((prefix + "_ssim_val.txt").c_str(), "w");
        if (!f) return 1;
        fputs(buf, f);
        fclose(f);
        if (!write_bin(prefix + "_ssim_grad.bin", out.d_pred.data(),
                       out.d_pred.size() * 4))
            return 1;
        std::printf("l1=%.8f ssim=%.8f grad=%zu\n", out.l1, out.ssim,
                    out.d_pred.size());
        return 0;
    }
    if (cmd == "lpips" && argc == 9) {
        // lpips value_and_grad parity: pred/gt bins + the lpips gguf
        std::vector<float> pred, gt;
        if (!read_bin(argv[3], &pred) || !read_bin(argv[4], &gt)) return 1;
        const int V = std::atoi(argv[5]), H = std::atoi(argv[6]),
                  W = std::atoi(argv[7]);
        const std::string prefix = argv[8];
        mapggml::Backend be = mapggml::init_backend("cpu", 1);
        qs::LpipsVgg lp(argv[2], be);
        if (!lp.valid()) return 1;
        std::vector<float> val, grad;
        if (!lp.value_and_grad(pred.data(), gt.data(), V, H, W, &val, &grad))
            return 1;
        if (!write_bin(prefix + "_lpips_val.bin", val.data(), val.size() * 4))
            return 1;
        if (!write_bin(prefix + "_lpips_grad.bin", grad.data(),
                       grad.size() * 4))
            return 1;
        std::printf("lpips val=%.6f grad=%zu\n", val[0], grad.size());
        return 0;
    }
    if (cmd == "act" && argc == 13) {
        std::vector<float> geom, color, dg;
        if (!read_bin(argv[2], &geom) || !read_bin(argv[3], &color) ||
            !read_bin(argv[4], &dg))
            return 1;
        const long long n_tokens = std::atoll(argv[5]);
        const int gpt = std::atoi(argv[6]);
        const int post_out = std::atoi(argv[7]);
        const float cap = (float)std::atof(argv[8]);
        const float zoff = (float)std::atof(argv[9]);
        const int sh = std::atoi(argv[10]);
        const float rest = (float)std::atof(argv[11]);
        const std::string prefix = argv[12];
        std::vector<float> dgeom((size_t)n_tokens * 10 * gpt),
            dcolor((size_t)n_tokens * post_out * gpt);
        qs::activate_gaussians_backward(
            geom.data(), color.data(), dg.data(), n_tokens, gpt, post_out,
            cap, zoff, sh, rest, dgeom.data(), dcolor.data());
        if (!write_bin(prefix + "_act_dgeom.bin", dgeom.data(),
                       dgeom.size() * 4))
            return 1;
        if (!write_bin(prefix + "_act_dcolor.bin", dcolor.data(),
                       dcolor.size() * 4))
            return 1;
        std::printf("dgeom=%zu dcolor=%zu\n", dgeom.size(), dcolor.size());
        return 0;
    }
    std::fprintf(stderr, "unknown subcommand\n");
    return 2;
}
