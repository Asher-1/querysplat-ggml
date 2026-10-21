// Component parity probe for querysplat-eval: the metric and camera-alignment
// ports on caller-supplied raw f32 inputs, so each can be diffed against the
// official evaluations/{metrics,cameras}.py on identical data
// (benchmarks/eval_parity_unit.py drives both sides).
//
//   test_eval_parity metrics <dir> V H W <lpips.gguf> [device]
//       reads <dir>/pred.f32, <dir>/gt.f32 (V,3,H,W) -> <dir>/cpp_metrics.f32
//       (3*V: psnr | ssim | lpips per view)
//   test_eval_parity align <dir> n_views n_ref
//       reads <dir>/w2c.f32 (n_views*16), <dir>/ref_w2c.f32 (n_ref*16), both
//       row-major w2c -> <dir>/cpp_aligned.f32 (n_views*16 row-major w2c)
#include "backend.hpp"
#include "camera_math.hpp"
#include "lpips_graph.hpp"
#include "metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string& path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(float)));
    if ((size_t)f.gcount() != n * sizeof(float)) {
        std::fprintf(stderr, "%s: expected %zu floats\n", path.c_str(), n);
        std::exit(1);
    }
    return v;
}

void write_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(v.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 2 ? argv[1] : "";
    if (mode == "metrics" && argc >= 7) {
        const std::string dir = argv[2];
        const int V = std::atoi(argv[3]), H = std::atoi(argv[4]), W = std::atoi(argv[5]);
        const size_t n = (size_t)V * 3 * H * W;
        const std::vector<float> pred = read_f32(dir + "/pred.f32", n);
        const std::vector<float> gt = read_f32(dir + "/gt.f32", n);
        mapggml::Backend be = mapggml::init_backend(argc > 7 ? argv[7] : "auto", 0);
        qs::LpipsVgg lp(argv[6], be);
        if (!lp.valid()) return 1;
        std::vector<float> l;
        if (!lp.score(pred.data(), gt.data(), V, H, W, &l)) return 1;
        std::vector<float> out = qs::psnr_views(pred.data(), gt.data(), V, H, W);
        const std::vector<float> s = qs::ssim_views(pred.data(), gt.data(), V, H, W);
        out.insert(out.end(), s.begin(), s.end());
        out.insert(out.end(), l.begin(), l.end());
        write_f32(dir + "/cpp_metrics.f32", out);
        return 0;
    }
    if (mode == "align" && argc >= 5) {
        const std::string dir = argv[2];
        const int n = std::atoi(argv[3]), m = std::atoi(argv[4]);
        const std::vector<float> w2c = read_f32(dir + "/w2c.f32", (size_t)n * 16);
        const std::vector<float> ref = read_f32(dir + "/ref_w2c.f32", (size_t)m * 16);
        write_f32(dir + "/cpp_aligned.f32",
                  qs::align_cameras_to_reference(w2c.data(), n, ref.data(), m));
        return 0;
    }
    std::fprintf(stderr,
                 "usage: %s metrics <dir> V H W <lpips.gguf> [device]\n"
                 "       %s align <dir> n_views n_ref\n",
                 argv[0], argv[0]);
    return 2;
}
