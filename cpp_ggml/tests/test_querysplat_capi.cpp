// C API smoke: load the paper f16 pair through querysplat_load_opts, run one
// S=2 inference from preprocessed frames, and check the result contract
// (gaussian count/stride/opacity range, pose stride, timings populated).
#include "querysplat/capi.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const char* enc = argc > 1 ? argv[1]
                               : "models/gguf/vggt-omega-1b-512-f16.gguf";
    const char* dec = argc > 2 ? argv[2]
                               : "models/gguf/querysplat-paper-f16.gguf";

    if (querysplat_abi_version() != QUERYSPLAT_CAPI_ABI_VERSION) {
        fprintf(stderr, "ABI version mismatch\n");
        return 1;
    }

    querysplat_options* opts = querysplat_options_new();
    querysplat_options_set_device(opts, "cpu");
    querysplat_options_set_threads(opts, 0);  // hardware default
    querysplat_ctx* ctx = querysplat_load_opts(enc, dec, opts);
    querysplat_options_free(opts);
    if (!ctx) {
        fprintf(stderr, "load failed: %s\n", querysplat_last_error());
        return 1;
    }
    if (!querysplat_is_ready(ctx)) {
        fprintf(stderr, "context not ready: %s\n", querysplat_last_error());
        querysplat_free(ctx);
        return 1;
    }

    const int S = 2;
    const int32_t nominal = querysplat_default_image_size(ctx);
    if (nominal <= 0) {
        fprintf(stderr, "default image size unavailable\n");
        querysplat_free(ctx);
        return 1;
    }
    const int H = nominal, W = nominal;
    std::vector<float> frames((size_t)S * 3 * H * W, 0.5f);

    querysplat_result r{};
    const int rc = querysplat_run_frames(ctx, frames.data(), S, H, W, &r);
    if (rc != 0) {
        fprintf(stderr, "run_frames failed (rc=%d): %s\n", rc,
                querysplat_last_error());
        querysplat_free(ctx);
        return 1;
    }

    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        printf("  %-42s %s\n", what, ok ? "OK" : "FAIL");
        if (!ok) failures++;
    };
    printf("querysplat C API smoke (S=%d %dx%d):\n", S, W, H);
    check(r.n_views == S, "n_views echoes the request");
    check(r.height == H && r.width == W, "resolution echoes the request");
    check(r.pose != nullptr && r.pose_stride == 9, "pose [n*9] present");
    // paper checkpoint: 8192 gs tokens x 64 gaussians/token, sh0 -> RGB
    check(r.n_gaussians == 8192 * 64, "n_gaussians == 524288 (paper layout)");
    check(r.gaussians_stride == 14, "gaussians_stride == 14 (sh0)");
    check(r.gaussians != nullptr, "gaussians buffer present");
    if (r.gaussians && r.n_gaussians > 0) {
        // opacity (stride index 3) is sigmoid-activated: strictly in (0,1);
        // positions/rotations finite (a NaN here means a broken graph)
        bool opacity_ok = true, finite_ok = true;
        const size_t sample = (size_t)r.n_gaussians < 10000 ? r.n_gaussians
                                                            : 10000;
        for (size_t i = 0; i < sample; i++) {
            const float* g = r.gaussians + i * r.gaussians_stride;
            if (!(g[3] > 0.0f && g[3] < 1.0f)) opacity_ok = false;
            for (int c = 0; c < 11; c++)
                if (!std::isfinite(g[c])) finite_ok = false;
        }
        check(opacity_ok, "opacity in (0,1) over the sample");
        check(finite_ok, "gaussian channels finite over the sample");
    }
    check(r.timings.e2e_ms > 0.0 && r.timings.inference_ms > 0.0,
          "timings populated");

    // second run on the same geometry must reuse the staged runtime
    querysplat_result r2{};
    if (querysplat_run_frames(ctx, frames.data(), S, H, W, &r2) != 0) {
        fprintf(stderr, "second run failed: %s\n", querysplat_last_error());
        failures++;
    } else {
        check(r2.timings.e2e_ms < r.timings.e2e_ms,
              "warm rerun not slower (runtime reuse)");
    }

    querysplat_free(ctx);
    printf("%s\n", failures == 0 ? "C API SMOKE PASS" : "C API SMOKE FAIL");
    return failures == 0 ? 0 : 1;
}
