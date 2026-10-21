// querysplat-eval — a pure C++ port of evaluations/evaluate_jsons.py (no TTO).
//
// Per case (one DL3DV split JSON sample), exactly the official protocol:
//   inputs (model resolution, official preprocessing)
//     -> encoder (S = N, expose_tokens) -> input cameras -> decoder -> gaussians
//        [reconstruction_seconds: case start .. backend sync, image decode included]
//   per group in (input, interp, extrap):
//     inputs + targets -> encoder camera pass (S = N + T, own runtime)
//       -> align_cameras_to_reference(cams, input cams, N) -> cams[N:]
//       -> intrinsics * [tw/sw, th/sh, tw/sw, th/sh]
//       -> render at the metric resolution (default 256x256), clamp [0,1]
//     targets clamp + bilinear antialias resize -> PSNR / skimage SSIM /
//     VGG LPIPS (ggml graph) per view
//   writes, with the official names/key order/float repr:
//     <out>/<split>/<group>/case_<idx:06d>_<scene[:12]>/{metrics.json,
//       timing.json, cameras.npz[, renders/, gt/]}
//     <out>/<split>/summary.json, <out>/summary.json, <out>/summary.md,
//     <out>/run_config.json
//
// Usage:
//   querysplat-eval --encoder models/gguf/vggt-omega-1b-512-f32.gguf
//                   --decoder models/gguf/querysplat-paper-f32.gguf
//                   [--lpips models/gguf/lpips-vgg-f32.gguf]
//                   --dataset-root ../data/DL3DV-Evaluation
//                   --eval-jsons ../evaluations/jsons_dl3dv/dl3dv_val_4views_medium.json
//                   --output-dir OUT [--limit N | --case-indices i ...]
//                   [--input-resolution 256x256] [--save-images]
//                   [--device auto] [--threads 0] [--cuda-fast] [--vk-fast]
#include "activations_host.hpp"
#include "backend.hpp"
#include "camera_math.hpp"
#include "common.hpp"
#include "gguf_loader.hpp"
#include "image_io.hpp"
#include "lpips_graph.hpp"
#include "metrics.hpp"
#include "output_parity.hpp"
#include "pyio.hpp"
#include "querysplat_graph.hpp"
#include "render/rasterizer.hpp"
#include "tto_loss.hpp"
#include "vggt_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

using mapggml::log_error;
using mapggml::log_info;
using qs::Json;

constexpr const char* kGroups[3] = {"input", "interp", "extrap"};
constexpr const char* kMetrics[3] = {"psnr", "ssim", "lpips"};

struct Args {
    std::string encoder, decoder, lpips = "models/gguf/lpips-vgg-f32.gguf";
    std::string dataset_root = "data/DL3DV-Evaluation";
    std::vector<std::string> eval_jsons;
    std::string output_dir;
    bool has_limit = false;
    long long limit = 0;
    bool has_case_indices = false;
    std::vector<long long> case_indices;
    int metric_h = 256, metric_w = 256;
    bool save_images = false;
    std::string device = "auto";
    int threads = 0;
    bool cuda_fast = false, vk_fast = false;
    // TTO (official evaluate_jsons.py flags; kv target, save_steps=[])
    bool use_tto = false;
    long long tto_n_steps = 20;
    double tto_lr = 5e-3, tto_lpips_weight = 0.05;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --encoder <omega.gguf> --decoder <querysplat.gguf>\n"
                 "              --output-dir DIR (new or empty)\n"
                 "              [--lpips models/gguf/lpips-vgg-f32.gguf]\n"
                 "              [--dataset-root data/DL3DV-Evaluation]\n"
                 "              [--eval-jsons a.json ...] (default: sorted\n"
                 "               evaluations/jsons_dl3dv/*views*.json)\n"
                 "              [--limit N | --case-indices i ...]\n"
                 "              [--input-resolution 256x256] [--save-images]\n"
                 "              [--device auto] [--threads 0] [--cuda-fast] [--vk-fast]\n"
                 "\n"
                 "  --use-tto  test-time optimization of the KV memory\n"
                 "             (--tto-n-steps/--tto-lr/--tto-lpips-weight; kv\n"
                 "             target, sh0).\n",
                 argv0);
}

bool parse_int(const std::string& s, long long* v) {
    char* end = nullptr;
    *v = std::strtoll(s.c_str(), &end, 10);
    return !s.empty() && *end == '\0';
}

// argparse resolution(): "HxW" (case-insensitive), both >= 11
bool parse_resolution(std::string s, int* h, int* w) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    const size_t x = s.find('x');
    long long a = 0, b = 0;
    if (x == std::string::npos || s.find('x', x + 1) != std::string::npos ||
        !parse_int(s.substr(0, x), &a) || !parse_int(s.substr(x + 1), &b) ||
        std::min(a, b) < 11)
        return false;
    *h = (int)a;
    *w = (int)b;
    return true;
}

bool parse_args(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                log_error("%s expects a value", arg.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        // nargs="+": consume until the next option-looking token
        auto values = [&]() {
            std::vector<std::string> v;
            while (i + 1 < argc && argv[i + 1][0] != '-') v.push_back(argv[++i]);
            if (v.empty()) {
                log_error("%s expects at least one value", arg.c_str());
                std::exit(2);
            }
            return v;
        };
        if (arg == "--encoder") a->encoder = next();
        else if (arg == "--decoder") a->decoder = next();
        else if (arg == "--lpips") a->lpips = next();
        else if (arg == "--dataset-root") a->dataset_root = next();
        else if (arg == "--eval-jsons") a->eval_jsons = values();
        else if (arg == "--output-dir") a->output_dir = next();
        else if (arg == "--limit") {
            if (!parse_int(next(), &a->limit)) { log_error("--limit: invalid int"); return false; }
            a->has_limit = true;
        } else if (arg == "--case-indices") {
            for (const std::string& s : values()) {
                long long v = 0;
                if (!parse_int(s, &v)) { log_error("--case-indices: invalid int '%s'", s.c_str()); return false; }
                a->case_indices.push_back(v);
            }
            a->has_case_indices = true;
        } else if (arg == "--input-resolution") {
            if (!parse_resolution(next(), &a->metric_h, &a->metric_w)) {
                log_error("Expected HEIGHTxWIDTH, with both dimensions >= 11");
                return false;
            }
        } else if (arg == "--save-images") a->save_images = true;
        else if (arg == "--device") a->device = next();
        else if (arg == "--threads") {
            long long t = 0;
            if (!parse_int(next(), &t)) return false;
            a->threads = (int)t;
        } else if (arg == "--cuda-fast") a->cuda_fast = true;
        else if (arg == "--vk-fast") a->vk_fast = true;
        else if (arg == "--use-tto") a->use_tto = true;
        else if (arg == "--tto-n-steps") {
            if (!parse_int(next(), &a->tto_n_steps)) {
                log_error("--tto-n-steps: invalid int");
                return false;
            }
        } else if (arg == "--tto-lr") {
            a->tto_lr = std::strtod(next().c_str(), nullptr);
        } else if (arg == "--tto-lpips-weight") {
            a->tto_lpips_weight = std::strtod(next().c_str(), nullptr);
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            log_error("unknown argument: %s", arg.c_str());
            return false;
        }
    }
    if (a->has_limit && a->has_case_indices) {
        log_error("argument --case-indices: not allowed with argument --limit");
        return false;
    }
    if (a->has_limit && a->limit < 1) {
        log_error("--limit must be positive");
        return false;
    }
    if (a->encoder.empty() || a->decoder.empty() || a->output_dir.empty()) {
        usage(argv[0]);
        return false;
    }
    if (a->use_tto && (a->tto_n_steps < 0 || !(a->tto_lr > 0) ||
                       !(a->tto_lpips_weight >= 0))) {
        log_error("TTO steps/LPIPS weight must be non-negative and TTO LR "
                  "must be positive");
        return false;
    }
    if (a->eval_jsons.empty()) {
        // default=sorted(glob("evaluations/jsons_dl3dv/*views*.json"))
        std::error_code ec;
        for (const auto& e : fs::directory_iterator("evaluations/jsons_dl3dv", ec)) {
            const std::string n = e.path().filename().string();
            if (e.path().extension() == ".json" && n.find("views") != std::string::npos)
                a->eval_jsons.push_back(e.path().string());
        }
        std::sort(a->eval_jsons.begin(), a->eval_jsons.end());
    }
    return true;
}

// ---- SHA-256 (FIPS 180-4), streamed: run_config's split/checkpoint digests ----

struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char buf[64];
    size_t len = 0;
    uint64_t total = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const unsigned char* p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 |
                   (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) +
                                ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) +
                                ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const unsigned char* p, size_t n) {
        total += n;
        while (n) {
            const size_t take = std::min(n, 64 - len);
            std::memcpy(buf + len, p, take);
            len += take; p += take; n -= take;
            if (len == 64) { block(buf); len = 0; }
        }
    }
    std::string hex() {
        const uint64_t bits = total * 8;
        const unsigned char pad = 0x80, zero = 0;
        update(&pad, 1);
        while (len != 56) update(&zero, 1);
        unsigned char lb[8];
        for (int i = 0; i < 8; i++) lb[i] = (unsigned char)(bits >> (56 - 8 * i));
        update(lb, 8);
        char out[65];
        for (int i = 0; i < 8; i++) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return out;
    }
};

std::string file_sha256(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    Sha256 s;
    std::vector<unsigned char> chunk(16u << 20);
    while (f) {
        f.read(reinterpret_cast<char*>(chunk.data()), (std::streamsize)chunk.size());
        s.update(chunk.data(), (size_t)f.gcount());
    }
    return s.hex();
}

bool read_text(const std::string& path, std::string* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

// numpy pairwise_sum (loops_utils.h.src) in f64: np.mean / np.std over a
// python float list reduce with exactly this tree.
double pairwise_sum(const double* a, size_t n) {
    if (n < 8) {
        double r = 0;
        for (size_t k = 0; k < n; k++) r += a[k];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; k++) r[k] = a[k];
        size_t k = 8;
        for (; k < n - (n % 8); k += 8)
            for (int m = 0; m < 8; m++) r[m] += a[k + (size_t)m];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; k < n; k++) res += a[k];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

double np_mean(const std::vector<double>& v) {
    return pairwise_sum(v.data(), v.size()) / (double)v.size();
}

// stats(): {mean, std (population: sqrt(mean((x - mean)**2))), count}
Json stats(const std::vector<double>& v) {
    const double m = np_mean(v);
    std::vector<double> d2(v.size());
    for (size_t k = 0; k < v.size(); k++) d2[k] = (v[k] - m) * (v[k] - m);
    Json j = qs::jobj();
    j.add("mean", qs::jnum(m))
        .add("std", qs::jnum(std::sqrt(np_mean(d2))))
        .add("count", qs::jint((long long)v.size()));
    return j;
}

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---- one split JSON ----

struct Split {
    std::string path, stem, sha256;
    Json config;
    std::vector<long long> indices;
};

// selected_indices(): declared nonempty sample list, distinct in-range
// indices, 2/4/12 input views, 4 interp + 4 extrap views per case
bool selected_indices(const Args& a, Split* s) {
    const Json* samples = s->config.get("samples");
    const Json* total = s->config.get("total_samples");
    if (!samples || samples->kind != Json::Arr || samples->arr.empty() || !total ||
        total->kind != Json::Int || total->i != (long long)samples->arr.size()) {
        log_error("%s: Evaluation JSON must contain its declared, nonempty sample list",
                  s->path.c_str());
        return false;
    }
    const long long n = (long long)samples->arr.size();
    if (a.has_case_indices) {
        s->indices = a.case_indices;
    } else {
        s->indices.clear();
        for (long long i = 0; i < n && (!a.has_limit || i < a.limit); i++) s->indices.push_back(i);
    }
    std::set<long long> uniq(s->indices.begin(), s->indices.end());
    if (uniq.size() != s->indices.size() || (!uniq.empty() && (*uniq.begin() < 0 || *uniq.rbegin() >= n))) {
        log_error("%s: Case indices must be distinct and within the JSON sample list",
                  s->path.c_str());
        return false;
    }
    const Json* nv = s->config.get("num_input_views");
    const long long expected = nv ? (long long)nv->number() : 0;
    if (expected != 2 && expected != 4 && expected != 12) {
        log_error("%s: Supported evaluation input view counts are 2, 4 and 12", s->path.c_str());
        return false;
    }
    for (long long index : s->indices) {
        const Json& sample = samples->arr[(size_t)index];
        for (int g = 0; g < 3; g++) {
            const long long count = g == 0 ? expected : 4;
            const Json* v = sample.get(std::string(kGroups[g]) + "_views");
            if (!v || v->kind != Json::Arr || (long long)v->arr.size() != count) {
                log_error("%s: Case %lld: expected %lld %s views", s->path.c_str(), index,
                          count, kGroups[g]);
                return false;
            }
        }
    }
    return true;
}

std::vector<std::string> split_path(const std::string& p) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : p) {
        if (c == '/') {
            if (!cur.empty() && cur != ".") parts.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty() && cur != ".") parts.push_back(cur);
    return parts;
}

// image_path(): JSON paths are relative to the parent of DL3DV-Evaluation;
// the first component must be Path(dataset_path).name, replaced by the root
bool image_path(const Json& entry, const Json& config, const std::string& root,
                std::string* out) {
    const Json* p = entry.get("path");
    const Json* dp = config.get("dataset_path");
    if (!p || p->kind != Json::Str || !dp || dp->kind != Json::Str) {
        log_error("view entry without a path");
        return false;
    }
    const std::vector<std::string> parts = split_path(p->s);
    const std::vector<std::string> dparts = split_path(dp->s);
    const std::string prefix = dparts.empty() ? std::string() : dparts.back();
    const bool dotdot = std::find(parts.begin(), parts.end(), "..") != parts.end();
    if (p->s.empty() || p->s[0] == '/' || dotdot || parts.empty() || parts[0] != prefix) {
        log_error("Unexpected dataset-relative image path: %s", p->s.c_str());
        return false;
    }
    fs::path r(root);
    for (size_t k = 1; k < parts.size(); k++) r /= parts[k];
    *out = r.string();
    return true;
}

std::vector<long long> frame_indices(const Json& views) {
    std::vector<long long> v;
    for (const Json& e : views.arr) {
        const Json* f = e.get("frame_index");
        v.push_back(f ? (long long)f->number() : 0);
    }
    return v;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// cam_view = w2c^T per view (the npz stores the torch tensors)
std::vector<float> cam_view_of(const float* w2c, int n) {
    std::vector<float> o((size_t)n * 16);
    for (int v = 0; v < n; v++)
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) o[(size_t)v * 16 + r * 4 + c] = w2c[v * 16 + c * 4 + r];
    return o;
}

struct Evaluator {
    Args args;
    mapggml::Backend be;
    std::unique_ptr<mapggml::GGUFModel> enc_model;
    std::unique_ptr<qs::QuerySplatDecoder> dec;
    std::unique_ptr<qs::LpipsVgg> lpips;
    qs::DecoderConfig cfg;
    int H = 0, W = 0;  // model resolution (options.img_size)
    // VGGTRuntime fixes S: the reconstruction graph (S = N, tokens exposed)
    // and one camera-only graph per input+target count, all on ONE staged
    // weight buffer (owned by the first runtime built)
    std::map<std::pair<int, bool>, std::unique_ptr<mapggml::VGGTRuntime>> rts;
    const mapggml::VGGTRuntime* owner = nullptr;

    void sync() { ggml_backend_synchronize(be.handle); }

    mapggml::VGGTRuntime* runtime(int S, bool tokens) {
        auto& r = rts[{S, tokens}];
        if (!r) {
            mapggml::RuntimeOptions o;
            o.expose_tokens = tokens;
            // the target-camera pass is predict_cameras (no depth decoding)
            o.cameras_only = !tokens;
            o.f32_accuracy = !args.cuda_fast;
            r = owner ? std::make_unique<mapggml::VGGTRuntime>(*owner, S, H, W, o)
                      : std::make_unique<mapggml::VGGTRuntime>(*enc_model, be, S, H, W, o);
            if (!r->valid()) return nullptr;
            if (!owner) owner = r.get();
        }
        return r.get();
    }

    // drop every graph but the weight owner's (view counts change per split)
    void release_graphs() {
        for (auto it = rts.begin(); it != rts.end();)
            it = it->second.get() == owner ? std::next(it) : rts.erase(it);
    }

    bool load_views(const Split& s, const Json& views, std::vector<float>* frames) {
        std::vector<std::string> paths;
        for (const Json& e : views.arr) {
            std::string p;
            if (!image_path(e, s.config, args.dataset_root, &p)) return false;
            paths.push_back(p);
        }
        return qs::load_images_as_frames(paths, H, W, frames);
    }

    // per-case per-view metric values, kept for the summaries
    struct CaseMetrics {
        std::vector<double> per_view[3];
        double mean[3];
    };

    bool evaluate_case(const Split& s, long long index, std::map<std::string, std::vector<CaseMetrics>>* rows) {
        const Json& sample = s.config.get("samples")->arr[(size_t)index];
        const Json& inputs = *sample.get("input_views");
        const int N = (int)inputs.arr.size();
        const std::string scene = sample.get("scene_id")->s;

        // ---- reconstruction (input views only) ----
        mapggml::VGGTRuntime* recon = runtime(N, true);
        if (!recon) return false;
        const auto start = std::chrono::steady_clock::now();
        std::vector<float> in_frames;
        if (!load_views(s, inputs, &in_frames)) return false;
        mapggml::VGGTOutputs enc_out;
        if (!recon->run(in_frames.data(), enc_out)) return false;
        const qs::CameraData in_cams = qs::decode_cameras(enc_out.pose_enc.data(), N, H, W);
        const std::vector<float> plucker = qs::plucker_from_cameras(in_cams, H, W);
        const std::vector<float> rgb_norm =
            qs::imagenet_normalize(in_frames.data(), N, in_frames.size());
        if (!dec->ensure_shape(N, H, W, enc_out.patch_token_start)) return false;
        qs::DecoderOutputs dec_out;
        if (!dec->run(enc_out.cached_tokens.data(), rgb_norm.data(), plucker.data(), dec_out))
            return false;
        if (args.use_tto) {
            // official evaluate_jsons: forward_tto_memory(kv, save_steps=[])
            // inside the reconstruction timing
            std::unique_ptr<qs::LpipsVgg> lp;
            if (args.tto_lpips_weight > 0) {
                lp = std::make_unique<qs::LpipsVgg>(args.lpips, be);
                if (!lp->valid()) return false;
            }
            if (!dec->tto_begin(N, H, W, enc_out.patch_token_start,
                                enc_out.cached_tokens.data(), rgb_norm.data(),
                                plucker.data(), args.tto_lr))
                return false;
            double last_l1 = 0, last_ssim = 0, last_lp = 0;
            bool has_lp = false;
            qs::RenderConfig trcfg;
            trcfg.width = W;
            trcfg.height = H;
            trcfg.near_plane = cfg.znear;
            trcfg.far_plane = cfg.zfar;
            trcfg.sh_degree = cfg.gaussian_sh_degree;
            std::memcpy(trcfg.bg_color, cfg.bg_color, sizeof(trcfg.bg_color));
            const int stride_t = 11 + dec_out.color_dim;
            const int post_out_dim =
                3 * (cfg.gaussian_sh_degree + 1) * (cfg.gaussian_sh_degree + 1) + 1;
            for (long long step = 1; step <= args.tto_n_steps; step++) {
                std::vector<float> deconv_raw, color_raw;
                if (!dec->tto_forward(deconv_raw, color_raw)) return false;
                qs::DecoderOutputs to = dec->tto_gaussians_from(deconv_raw, color_raw);
                qs::GaussianOutput g;
                g.N = to.N;
                g.color_dim = to.color_dim;
                g.data = std::move(to.gaussians);
                g.opacity.resize((size_t)g.N);
                for (int64_t i = 0; i < g.N; i++)
                    g.opacity[(size_t)i] = g.data[(size_t)i * stride_t + 3];
                qs::RenderedViews views = qs::render_gaussians(
                    g, in_cams.w2c.data(), in_cams.intrinsics.data(), N, trcfg);
                std::vector<float> d_lpips;
                if (args.tto_lpips_weight > 0) {
                    std::vector<float> lv;
                    if (!lp->value_and_grad(views.images.data(),
                                            in_frames.data(), N, H, W, &lv,
                                            &d_lpips))
                        return false;
                    last_lp = lv.empty() ? 0.0 : (double)lv[0];
                    has_lp = true;
                }
                qs::TtoLossOut lo;
                if (!qs::tto_loss_value_and_grad(
                        views.images.data(), in_frames.data(), N, H, W,
                        args.tto_lpips_weight,
                        args.tto_lpips_weight > 0 ? d_lpips.data() : nullptr,
                        &lo))
                    return false;
                qs::GaussianGrads gg;
                if (!qs::render_backward(g, in_cams.w2c.data(),
                                         in_cams.intrinsics.data(), N, trcfg,
                                         views, lo.d_pred.data(), &gg))
                    return false;
                std::vector<float> dg((size_t)gg.N * stride_t);
                for (int64_t i = 0; i < gg.N; i++) {
                    float* r = &dg[(size_t)i * stride_t];
                    for (int c = 0; c < 3; c++) r[c] = gg.pos[(size_t)i * 3 + c];
                    r[3] = gg.opacity[(size_t)i];
                    for (int c = 0; c < 3; c++)
                        r[4 + c] = gg.scale[(size_t)i * 3 + c];
                    for (int c = 0; c < 4; c++)
                        r[7 + c] = gg.rot[(size_t)i * 4 + c];
                    for (int c = 0; c < gg.color_dim; c++)
                        r[11 + c] = gg.color[(size_t)i * gg.color_dim + c];
                }
                std::vector<float> dgeom((size_t)cfg.num_gs_tokens * 10 *
                                         cfg.num_gaussians_per_token),
                    dcol((size_t)cfg.num_gs_tokens * post_out_dim *
                         cfg.num_gaussians_per_token);
                qs::activate_gaussians_backward(
                    deconv_raw.data(), color_raw.data(), dg.data(),
                    cfg.num_gs_tokens, cfg.num_gaussians_per_token, post_out_dim,
                    cfg.gaussian_scale_cap, cfg.gaussian_z_offset,
                    cfg.gaussian_sh_degree, cfg.gaussian_sh_rest_scale,
                    dgeom.data(), dcol.data());
                if (!dec->tto_backward(dgeom.data(), dcol.data(), (int)step - 1))
                    return false;
                last_l1 = lo.l1;
                last_ssim = lo.ssim;
            }
            std::vector<float> fr, cr;
            if (!dec->tto_forward(fr, cr)) return false;
            dec_out = dec->tto_gaussians_from(fr, cr);
            // official: write_json(<split>/case_xxxxxx_tto_losses.json,
            // model.last_tto_losses) at the output root
            char tl[128];
            std::snprintf(tl, sizeof(tl), "%s/%s/case_%06lld_tto_losses.json",
                          args.output_dir.c_str(), s.stem.c_str(), index);
            Json tj = qs::jobj();
            tj.add("L1", qs::jnum(last_l1))
                .add("SSIM", qs::jnum(last_ssim))
                .add("LPIPS", has_lp ? qs::jnum(last_lp) : qs::jnull());
            if (!qs::write_json(tl, tj)) return false;
        }
        if (!all_finite(dec_out.gaussians)) {
            log_error("%s case %lld: non-finite Gaussians", s.stem.c_str(), index);
            return false;
        }
        sync();
        const double reconstruction_seconds = since(start);

        qs::GaussianOutput g;
        g.N = dec_out.N;
        g.color_dim = dec_out.color_dim;
        g.data = std::move(dec_out.gaussians);
        const int stride = 11 + g.color_dim;
        g.opacity.resize((size_t)g.N);
        for (int64_t i = 0; i < g.N; i++) g.opacity[(size_t)i] = g.data[(size_t)i * stride + 3];

        const int mh = args.metric_h, mw = args.metric_w;
        // render_at_resolution: scale = new_tensor([tw/sw, th/sh, tw/sw, th/sh]) (f32)
        const float sx = (float)((double)mw / W), sy = (float)((double)mh / H);
        qs::RenderConfig rcfg;
        rcfg.width = mw;
        rcfg.height = mh;
        rcfg.near_plane = cfg.znear;
        rcfg.far_plane = cfg.zfar;
        rcfg.sh_degree = cfg.gaussian_sh_degree;
        std::memcpy(rcfg.bg_color, cfg.bg_color, sizeof(rcfg.bg_color));

        for (int gi = 0; gi < 3; gi++) {
            const auto group_start = std::chrono::steady_clock::now();
            const char* group = kGroups[gi];
            const Json& targets = *sample.get(std::string(group) + "_views");
            const int T = (int)targets.arr.size();
            // the historical input+input pass for the input group
            Json all = qs::jarr();
            for (const Json& e : inputs.arr) all.push(e);
            for (const Json& e : targets.arr) all.push(e);
            std::vector<float> frames;
            if (!load_views(s, all, &frames)) return false;
            const double t_load = since(group_start);
            mapggml::VGGTRuntime* cr = runtime(N + T, false);
            if (!cr) return false;
            mapggml::VGGTOutputs cam_out;
            if (!cr->run(frames.data(), cam_out)) return false;
            const double t_cams = since(group_start);
            const qs::CameraData pc = qs::decode_cameras(cam_out.pose_enc.data(), N + T, H, W);
            const std::vector<float> aligned =
                qs::align_cameras_to_reference(pc.w2c.data(), N + T, in_cams.w2c.data(), N);
            const float* tw2c = aligned.data() + (size_t)N * 16;
            std::vector<float> tk(pc.intrinsics.begin() + (size_t)N * 4, pc.intrinsics.end());
            for (int v = 0; v < T; v++) {
                tk[(size_t)v * 4 + 0] *= sx;
                tk[(size_t)v * 4 + 1] *= sy;
                tk[(size_t)v * 4 + 2] *= sx;
                tk[(size_t)v * 4 + 3] *= sy;
            }
            qs::RenderedViews rv = qs::render_gaussians(g, tw2c, tk.data(), T, rcfg);
            const double t_render = since(group_start);
            std::vector<float>& prediction = rv.images;
            for (float& x : prediction) x = std::clamp(x, 0.0f, 1.0f);
            // metric_images(): clamp, then bilinear antialias resize when needed
            const size_t src_n = (size_t)3 * H * W;
            std::vector<float> tgt(frames.begin() + (size_t)N * src_n, frames.end());
            for (float& x : tgt) x = std::clamp(x, 0.0f, 1.0f);
            if (mh != H || mw != W) tgt = qs::resize_planes(tgt.data(), T, H, W, mh, mw);
            if (!all_finite(prediction) || !all_finite(tgt)) {
                log_error("Non-finite evaluation images");
                return false;
            }
            const double t_gt = since(group_start);
            const std::vector<float> psnr = qs::psnr_views(prediction.data(), tgt.data(), T, mh, mw);
            const std::vector<float> ssim = qs::ssim_views(prediction.data(), tgt.data(), T, mh, mw);
            const double t_ssim = since(group_start);
            std::vector<float> lp;
            if (!lpips->score(prediction.data(), tgt.data(), T, mh, mw, &lp)) return false;
            const double t_lpips = since(group_start);

            CaseMetrics cm;
            const std::vector<float>* pv[3] = {&psnr, &ssim, &lp};
            Json per_view = qs::jobj();
            for (int m = 0; m < 3; m++) {
                for (float x : *pv[m]) {
                    if (!std::isfinite(x)) {
                        log_error("%s case %lld/%s: non-finite metrics", s.stem.c_str(), index, group);
                        return false;
                    }
                    cm.per_view[m].push_back((double)x);
                }
                cm.mean[m] = np_mean(cm.per_view[m]);
                per_view.add(kMetrics[m], qs::f32_list(pv[m]->data(), T));
            }
            char leaf[64];
            std::snprintf(leaf, sizeof(leaf), "case_%06lld_%s", index, scene.substr(0, 12).c_str());
            const fs::path dst = fs::path(args.output_dir) / s.stem / group / leaf;
            std::error_code ec;
            fs::create_directories(dst, ec);
            Json mj = qs::jobj();
            for (int m = 0; m < 3; m++) mj.add(kMetrics[m], qs::jnum(cm.mean[m]));
            mj.add("case_index", qs::jint(index))
                .add("scene_id", qs::jstr(scene))
                .add("target_group", qs::jstr(group))
                .add("per_view", std::move(per_view))
                .add("per_view_frames", qs::int_list(frame_indices(targets)))
                .add("input_frames", qs::int_list(frame_indices(inputs)))
                .add("metric_image_hw", qs::int_list({mh, mw}))
                .add("model_image_hw", qs::int_list({H, W}));
            if (!qs::write_json((dst / "metrics.json").string(), mj)) return false;
            sync();
            Json tj = qs::jobj();
            tj.add("reconstruction_seconds", qs::jnum(reconstruction_seconds))
                .add("target_camera_render_metrics_seconds", qs::jnum(since(group_start)));
            if (!qs::write_json((dst / "timing.json").string(), tj)) return false;
            const std::vector<float> icv = cam_view_of(in_cams.w2c.data(), N);
            const std::vector<float> tcv = cam_view_of(tw2c, T);
            if (!qs::write_npz((dst / "cameras.npz").string(),
                               {{"input_cam_view", qs::npy_f32(icv.data(), {1, N, 4, 4})},
                                {"target_cam_view", qs::npy_f32(tcv.data(), {1, T, 4, 4})},
                                {"intrinsics", qs::npy_f32(tk.data(), {1, T, 4})}}))
                return false;
            if (args.save_images) {
                const std::vector<long long> fi = frame_indices(targets);
                for (const auto& [sub, img] : {std::pair<const char*, const std::vector<float>*>{"renders", &prediction},
                                               {"gt", &tgt}}) {
                    fs::create_directories(dst / sub, ec);
                    for (int v = 0; v < T; v++) {
                        char name[64];
                        std::snprintf(name, sizeof(name), "%02d_frame_%06lld.png", v, fi[(size_t)v]);
                        if (!qs::write_rgb_planes_png((dst / sub / name).string(),
                                                      img->data() + (size_t)v * 3 * mh * mw, mh, mw))
                            return false;
                    }
                }
            }
            log_info("%s case %lld %s: psnr=%.4f ssim=%.4f lpips=%.4f  [load %.2fs cams %.2fs "
                     "render %.2fs gt %.2fs psnr+ssim %.2fs lpips %.2fs]",
                     s.stem.c_str(), index, group, cm.mean[0], cm.mean[1], cm.mean[2], t_load,
                     t_cams - t_load, t_render - t_cams, t_gt - t_render, t_ssim - t_gt,
                     t_lpips - t_ssim);
            (*rows)[group].push_back(std::move(cm));
        }
        return true;
    }
};

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) return 2;
    std::error_code ec;
    if (!fs::is_directory(args.dataset_root, ec)) {
        log_error("dataset root not found: %s", args.dataset_root.c_str());
        return 1;
    }
    // unique split names; every JSON validated before any work
    std::vector<Split> splits;
    std::set<std::string> stems;
    for (const std::string& p : args.eval_jsons) {
        Split s;
        s.path = p;
        s.stem = fs::path(p).stem().string();
        std::string text, err;
        if (!read_text(p, &text) || !qs::parse_json(text, &s.config, &err)) {
            log_error("failed to read %s %s", p.c_str(), err.c_str());
            return 1;
        }
        Sha256 h;
        h.update(reinterpret_cast<const unsigned char*>(text.data()), text.size());
        s.sha256 = h.hex();
        if (!selected_indices(args, &s)) return 1;
        stems.insert(s.stem);
        splits.push_back(std::move(s));
    }
    if (splits.empty() || stems.size() != splits.size()) {
        log_error("Provide evaluation JSONs with unique split names");
        return 1;
    }
    if (fs::exists(args.output_dir, ec) && !fs::is_empty(args.output_dir, ec)) {
        log_error("Use a new output directory to avoid mixing runs: %s", args.output_dir.c_str());
        return 1;
    }
    fs::create_directories(args.output_dir, ec);

    Evaluator ev;
    ev.args = args;
    if (!qs::read_decoder_config(args.decoder, &ev.cfg)) return 1;
    ev.H = ev.cfg.img_size[0];
    ev.W = ev.cfg.img_size[1];

    // ---- run_config.json (evaluate_jsons main(): argument values, split
    // digests, protocol strings, checkpoint identity — here the GGUF pair) ----
    {
        Json ej = qs::jarr(), ci = qs::jarr(), sha = qs::jobj();
        for (const Split& s : splits) {
            ej.push(qs::jstr(fs::absolute(s.path).lexically_normal().string()));
            sha.add(fs::path(s.path).filename().string(), qs::jstr(s.sha256));
        }
        for (long long i : args.case_indices) ci.push(qs::jint(i));
        Json model = qs::jobj();
        model.add("img_size", qs::int_list({ev.H, ev.W}))
            .add("bg_color", qs::f32_list(ev.cfg.bg_color, 3))
            .add("znear", qs::jnum(qs::config_value(ev.cfg.znear)))
            .add("zfar", qs::jnum(qs::config_value(ev.cfg.zfar)));
        Json bytes = qs::jobj(), digest = qs::jobj();
        for (const auto& [k, p] : {std::pair<const char*, std::string>{"encoder", args.encoder},
                                   {"decoder", args.decoder},
                                   {"lpips", args.lpips}}) {
            bytes.add(k, qs::jint((long long)fs::file_size(p, ec)));
            log_info("sha256 %s ...", p.c_str());
            digest.add(k, qs::jstr(file_sha256(p)));
        }
        Json rc = qs::jobj();
        rc.add("encoder", qs::jstr(args.encoder))
            .add("decoder", qs::jstr(args.decoder))
            .add("lpips", qs::jstr(args.lpips))
            .add("dataset_root", qs::jstr(args.dataset_root))
            .add("eval_jsons", std::move(ej))
            .add("output_dir", qs::jstr(args.output_dir))
            .add("device", qs::jstr(args.device))
            .add("batch_size", qs::jint(1))
            .add("limit", args.has_limit ? qs::jint(args.limit) : qs::jnull())
            .add("case_indices", args.has_case_indices ? std::move(ci) : qs::jnull())
            .add("input_resolution", qs::int_list({args.metric_h, args.metric_w}))
            .add("use_tto", qs::jbool(args.use_tto))
            .add("save_images", qs::jbool(args.save_images))
            .add("split_sha256", std::move(sha))
            .add("model_config", std::move(model))
            .add("camera_protocol", qs::jstr("input-only reconstruction; input+target prediction per group; similarity alignment to input cameras; no dataset poses"))
            .add("metric_protocol", qs::jstr("model-resolution preprocessing/reconstruction/cameras/TTO; direct metric-resolution rendering with scaled intrinsics; GT bilinear antialias resize; PSNR, skimage Gaussian SSIM, VGG LPIPS"))
            .add("reconstruct_at_model_resolution", qs::jbool(true))
            .add("input_resolution_behavior", qs::jstr("metric_render_resolution"))
            .add("checkpoint_bytes", std::move(bytes))
            .add("checkpoint_sha256", std::move(digest))
            .add("runtime", qs::jstr(args.cuda_fast || args.vk_fast ? "ggml (fast kernels)"
                                                                     : "ggml (f32-accuracy kernels)"));
        if (!qs::write_json(args.output_dir + "/run_config.json", rc)) return 1;
    }

    // ---- models ----
    // accuracy profiles ride the explicit BackendOptions (formerly the
    // process-global set_*_f32_accuracy_mode calls); f32-accuracy by default
    mapggml::BackendOptions bopts;
    bopts.device = args.device;
    bopts.cpu_threads = args.threads > 0
                            ? args.threads
                            : (int)std::thread::hardware_concurrency();
    bopts.vulkan_f32_accuracy = !args.vk_fast;
    bopts.cuda_f32_accuracy = !args.cuda_fast;
    ev.be = mapggml::init_backend(bopts);
    ev.enc_model = mapggml::load_gguf(args.encoder);
    if (!ev.enc_model) return 1;
    qs::DecoderRuntimeOptions dopts;
    dopts.f32_accuracy = !args.cuda_fast;
    ev.dec = std::make_unique<qs::QuerySplatDecoder>(args.decoder, ev.be, dopts);
    if (!ev.dec->valid()) return 1;
    ev.cfg = ev.dec->config();
    ev.lpips = std::make_unique<qs::LpipsVgg>(args.lpips, ev.be);
    if (!ev.lpips->valid()) return 1;

    // ---- evaluate ----
    std::vector<Json> summaries;
    struct SplitAgg {
        long long views;
        std::string scale;
        long long cases;
        double means[3][3];  // group x metric: metrics[name]["mean"]
    };
    std::vector<SplitAgg> aggs;
    for (const Split& s : splits) {
        const int N = (int)s.config.get("num_input_views")->number();
        ev.release_graphs();
        std::map<std::string, std::vector<Evaluator::CaseMetrics>> rows;
        for (long long index : s.indices) {
            if (!ev.evaluate_case(s, index, &rows)) {
                const Json& sample = s.config.get("samples")->arr[(size_t)index];
                log_error("Failed %s, case %lld, scene %s",
                          fs::path(s.path).filename().string().c_str(), index,
                          sample.get("scene_id")->s.c_str());
                return 1;
            }
        }
        // ---- <split>/summary.json ----
        SplitAgg agg;
        agg.views = N;
        const Json* sc = s.config.get("scale");
        agg.scale = sc && sc->kind == Json::Str ? sc->s : "";
        agg.cases = (long long)s.indices.size();
        Json groups = qs::jobj();
        for (int gi = 0; gi < 3; gi++) {
            const auto& r = rows[kGroups[gi]];
            Json mt = qs::jobj(), cmt = qs::jobj();
            for (int m = 0; m < 3; m++) {
                std::vector<double> all, per_case;
                for (const auto& c : r) {
                    all.insert(all.end(), c.per_view[m].begin(), c.per_view[m].end());
                    per_case.push_back(c.mean[m]);
                }
                agg.means[gi][m] = np_mean(all);
                mt.add(kMetrics[m], stats(all));
                cmt.add(kMetrics[m], stats(per_case));
            }
            Json gj = qs::jobj();
            gj.add("num_cases", qs::jint((long long)r.size()))
                .add("metrics", std::move(mt))
                .add("case_metrics", std::move(cmt));
            groups.add(kGroups[gi], std::move(gj));
        }
        Json sj = qs::jobj();
        sj.add("split", qs::jstr(s.stem))
            .add("num_input_views", *s.config.get("num_input_views"))
            .add("scale", sc ? *sc : qs::jnull())
            .add("num_cases", qs::jint(agg.cases))
            .add("groups", std::move(groups));
        fs::create_directories(fs::path(args.output_dir) / s.stem, ec);
        if (!qs::write_json(args.output_dir + "/" + s.stem + "/summary.json", sj)) return 1;
        summaries.push_back(std::move(sj));
        aggs.push_back(agg);
    }

    // ---- summary.json (splits + complete small/medium/large averages) ----
    Json root_splits = qs::jarr(), averages = qs::jarr();
    for (Json& sj : summaries) root_splits.push(sj);
    struct Avg {
        long long views, cases;
        double v[3][3];
    };
    std::vector<Avg> avgs;
    for (long long views : {2LL, 4LL, 12LL}) {
        std::set<std::string> scales;
        std::vector<const SplitAgg*> sel;
        for (const SplitAgg& a : aggs)
            if (a.views == views) {
                sel.push_back(&a);
                scales.insert(a.scale);
            }
        if (scales != std::set<std::string>{"small", "medium", "large"}) continue;
        Avg av{views, 0, {}};
        for (const SplitAgg* a : sel) av.cases += a->cases;
        Json groups = qs::jobj();
        for (int gi = 0; gi < 3; gi++) {
            Json gj = qs::jobj();
            for (int m = 0; m < 3; m++) {
                std::vector<double> means;
                for (const SplitAgg* a : sel) means.push_back(a->means[gi][m]);
                av.v[gi][m] = np_mean(means);
                gj.add(kMetrics[m], qs::jnum(av.v[gi][m]));
            }
            groups.add(kGroups[gi], std::move(gj));
        }
        Json aj = qs::jobj();
        aj.add("num_input_views", qs::jint(views))
            .add("num_cases", qs::jint(av.cases))
            .add("groups", std::move(groups));
        averages.push(std::move(aj));
        avgs.push_back(av);
    }
    Json root = qs::jobj();
    root.add("splits", std::move(root_splits)).add("scale_averages", std::move(averages));
    if (!qs::write_json(args.output_dir + "/summary.json", root)) return 1;

    // ---- summary.md ----
    std::string md = "# Evaluation results\n\nPSNR/SSIM: higher is better; LPIPS: lower is better.\n\n"
                     "| Split | Cases | Target | PSNR | SSIM | LPIPS |\n"
                     "| --- | ---: | --- | ---: | ---: | ---: |\n";
    char line[512];
    for (size_t k = 0; k < splits.size(); k++)
        for (int gi = 0; gi < 3; gi++) {
            std::snprintf(line, sizeof(line), "| %s | %lld | %s | %.4f | %.4f | %.4f |\n",
                          splits[k].stem.c_str(), aggs[k].cases, kGroups[gi], aggs[k].means[gi][0],
                          aggs[k].means[gi][1], aggs[k].means[gi][2]);
            md += line;
        }
    for (const Avg& a : avgs)
        for (int gi = 0; gi < 3; gi++) {
            std::snprintf(line, sizeof(line),
                          "| %lldviews / scale average | %lld | %s | %.4f | %.4f | %.4f |\n", a.views,
                          a.cases, kGroups[gi], a.v[gi][0], a.v[gi][1], a.v[gi][2]);
            md += line;
        }
    if (!qs::write_text(args.output_dir + "/summary.md", md)) return 1;
    std::fputs(md.c_str(), stdout);
    return 0;
}
