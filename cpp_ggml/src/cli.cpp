// querysplat-cli — single-process dual-graph QuerySplat inference on ggml.
//
// Pipeline (mirrors scripts/infer.py, no TTO):
//   frames f32 (S,3,H,W) [0,1]
//     -> VGGT-Omega encoder (embedded mapggml runtime, expose_tokens)
//        pose_enc (S,9) + depth/conf + cached-layer tokens
//     -> host: pose_enc -> w2c/c2w/intrinsics -> Plucker rays;
//        ImageNet normalization of rgb (feeds patch_embed only)
//     -> QuerySplat decoder graph (this project)
//     -> host activation heads -> gaussians (N, 11+color_dim)
//     -> CPU gsplat-aligned rasterizer (input views)
//     -> --output_dir: the official infer.py output tree (output_parity.cpp);
//        --out-prefix: raw bins + gaussians PLY for the parity/bench tooling
//
// Parity dump: pass --dump-stage <dir> (routed into DecoderRuntimeOptions::
// dump_dir — no environment variables in the querysplat chain) to write the
// seams (cpp_tokens / cpp_fused_patches / cpp_geom_kv_k/v / cpp_post_kv_k/v /
// cpp_geom_tokens / cpp_deconv_raw / cpp_color_raw) as raw f32 bins, paired
// with scripts/spy_torch_querysplat.py on the Python side.
//
// Usage (official scripts/infer.py layout under --output_dir):
//   querysplat-cli --encoder models/gguf/vggt-omega-1b-512-f16.gguf
//                  --decoder models/gguf/querysplat-paper-f16.gguf
//                  --input_folder DIR --output_dir OUT
//                  [--gaussian_save_opacity_threshold 0.05 ...] [--save_* ...]
// Usage (tooling layout: raw bins + PLY + renders under a prefix):
//   querysplat-cli ... --bin frames.bin --S 4 --out-prefix /tmp/qs [--render]
#include "activations_host.hpp"
#include "backend.hpp"
#include "camera_math.hpp"
#include "common.hpp"
#include "gguf_loader.hpp"
#include "image_io.hpp"
#include "lpips_graph.hpp"
#include "output_parity.hpp"
#include "tto_loss.hpp"
#include "ply_writer.hpp"
#include "querysplat_graph.hpp"
#include "render/rasterizer.hpp"
#include "vggt_graph.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using mapggml::log_error;
using mapggml::log_info;

struct Args {
    std::string encoder_gguf;
    std::string decoder_gguf;
    std::string bin_path;
    std::string out_prefix;  // tooling layout (bins/PLY/renders/timing)
    std::string output_dir;  // official scripts/infer.py layout
    std::string device = "auto";
    std::string dump_stage;
    std::string images;  // real-image input: dir or comma-separated files
    // 0 = the decoder GGUF's querysplat.img_size (infer.py: options.img_size)
    int S = 0, H = 0, W = 0;
    int threads = 0;  // 0 = hardware_concurrency
    bool render = false;
    bool timing = false;
    bool vk_fast = false;    // opt out of the Vulkan f32-accuracy mode
    bool cuda_fast = false;  // opt out of the CUDA f32-accuracy mode
    // infer.py flags (same names, same validation)
    bool save_cameras_json = false;      // --save_predicted_input_cameras
    bool save_input_depths = false;      // --save_vggt_input_depths
    bool save_alpha_dist = false;        // --save_gaussian_alpha_distribution
    bool save_scale_dist = false;        // --save_gaussian_scale_distribution
    bool save_depth_pointcloud = false;  // --save_vggt_depth_pointcloud
    bool depth_pointcloud_target_set = false;
    long long depth_pointcloud_target = 0;  // --vggt_depth_pointcloud_target_points
    bool use_tto = false;
    long long tto_n_steps = 20;
    double tto_lr = 5e-3, tto_lpips_weight = 0.05;
    std::vector<std::string> tto_save_step_raw;
    std::vector<std::string> threshold_raw{"0.05"};  // nargs="+", default
    std::string render_from;  // render-only: prefix with _gaussians/_cameras bins
    std::string render_cameras;  // render-only: explicit cameras.bin override
    std::vector<double> opacity_thresholds;  // sorted(set(parse_number_list))
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --encoder <omega.gguf> --decoder <querysplat.gguf>\n"
                 "                 (--input_folder DIR | --bin frames.f32.bin --S N)\n"
                 "                 (--output_dir DIR | --out-prefix PREFIX)\n"
                 "                 [--H img_size] [--W img_size] [--device auto] [--threads 0]\n"
                 "                 [--gaussian_save_opacity_threshold 0.05 ...]\n"
                 "                 [--save_gaussian_alpha_distribution]\n"
                 "                 [--save_gaussian_scale_distribution]\n"
                 "                 [--save_vggt_depth_pointcloud\n"
                 "                  --vggt_depth_pointcloud_target_points N]\n"
                 "                 [--save_vggt_input_depths] [--save_predicted_input_cameras]\n"
                 "                 [--render] [--timing] [--dump-stage DIR]\n"
                 "                 [--vk-fast] [--cuda-fast]\n"
                 "\n"
                 "  --output_dir     the official scripts/infer.py output layout\n"
                 "            (input_frames/, rendered/, pointcloud.ply, gaussians*.ply,\n"
                 "            inference_timing.json/.txt, camera_info.txt and the\n"
                 "            opt-in --save_* exports), byte-compatible writers.\n"
                 "  --out-prefix     tooling layout: PREFIX_gaussians.bin/.ply,\n"
                 "            PREFIX_cameras.bin, with --render PREFIX_render_view*.png\n"
                 "            + raw render bins, with --timing PREFIX_timing.json.\n"
                 "            Both may be given together.\n"
                 "  --input_folder / --images  png/jpg/jpeg/webp/bmp (dir, sorted;\n"
                 "            or a comma-separated file list): center square crop +\n"
                 "            resize to --H x --W, the official preprocessing.\n"
                 "  --gaussian_save_opacity_threshold  one or more values in [0,1]\n"
                 "            (infer.py parse_number_list); several -> one\n"
                 "            gaussians_opacity<t>.ply each.\n"

                 "  --vk-fast use ggml-vulkan's default fp16/coop-mat shader\n"
                 "            paths; by default the f32-accuracy mode is enabled\n"
                 "            because the fast paths drift past the parity gate.\n"
                 "  --cuda-fast  use ggml-cuda's TF32 GEMMs and the fp16 K/V\n"
                 "            flash-attention conversion; by default the\n"
                 "            f32-accuracy mode is enabled (exact F32 GEMMs +\n"
                 "            exact attention) because both drift past the\n"
                 "            parity gate.\n",
                 argv0);
}

// infer.py parse_number_list(raw, float|int): strip whitespace, strip the
// "{}[]()" characters, commas -> spaces, split; each item through python
// float()/int() (decimal digits with '_' only between digits).
bool parse_number_list(const std::vector<std::string>& raw, bool integer,
                       const char* option, std::vector<double>* out) {
    for (const std::string& r : raw) {
        size_t b = 0, e = r.size();
        while (b < e && std::isspace((unsigned char)r[b])) b++;
        while (e > b && std::isspace((unsigned char)r[e - 1])) e--;
        const std::string brackets = "{}[]()";
        while (b < e && brackets.find(r[b]) != std::string::npos) b++;
        while (e > b && brackets.find(r[e - 1]) != std::string::npos) e--;
        std::string text = r.substr(b, e - b);
        for (char& c : text)
            if (c == ',') c = ' ';
        std::istringstream ss(text);
        std::string item;
        while (ss >> item) {
            std::string num;
            bool ok = item.find_first_of("xX") == std::string::npos;  // no hex
            for (size_t k = 0; k < item.size() && ok; k++) {
                if (item[k] != '_') { num += item[k]; continue; }
                ok = k > 0 && k + 1 < item.size() &&
                     std::isdigit((unsigned char)item[k - 1]) &&
                     std::isdigit((unsigned char)item[k + 1]);
            }
            char* end = nullptr;
            errno = 0;
            double v = 0;
            if (ok) {
                v = integer ? (double)std::strtoll(num.c_str(), &end, 10)
                            : std::strtod(num.c_str(), &end);
                ok = end != num.c_str() && *end == '\0' && !(integer && errno);
            }
            if (!ok) {
                log_error("Invalid value for %s: '%s'", option, item.c_str());
                return false;
            }
            out->push_back(v);
        }
    }
    return true;
}

// sorted(set(values)): set keeps the first-inserted of equal values, then a
// stable sort
std::vector<double> sorted_set(std::vector<double> v) {
    std::vector<double> u;
    for (double x : v)
        if (std::find(u.begin(), u.end(), x) == u.end()) u.push_back(x);
    std::stable_sort(u.begin(), u.end());
    return u;
}

bool parse_args(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        // argparse "--opt=value" form
        std::string inline_value;
        bool has_inline = false;
        if (arg.rfind("--", 0) == 0) {
            const size_t eq = arg.find('=');
            if (eq != std::string::npos) {
                inline_value = arg.substr(eq + 1);
                arg = arg.substr(0, eq);
                has_inline = true;
            }
        }
        auto next = [&](const char* what) -> std::string {
            if (has_inline) { has_inline = false; return inline_value; }
            if (i + 1 >= argc) {
                log_error("%s expects a value", what);
                exit(2);
            }
            return argv[++i];
        };
        // argparse nargs="+"/"*": consume until the next option-looking token
        // (a leading '-' that is not a negative number)
        auto values = [&](const char* what, bool at_least_one) {
            std::vector<std::string> v;
            if (has_inline) v.push_back(inline_value);
            while (i + 1 < argc) {
                const char* t = argv[i + 1];
                const bool negnum = t[0] == '-' &&
                    (std::isdigit((unsigned char)t[1]) ||
                     (t[1] == '.' && std::isdigit((unsigned char)t[2])));
                if (t[0] == '-' && !negnum) break;
                v.push_back(argv[++i]);
            }
            if (at_least_one && v.empty()) {
                log_error("%s expects at least one value", what);
                exit(2);
            }
            return v;
        };
        auto to_ll = [&](const std::string& s, const char* what) {
            char* end = nullptr;
            const long long v = std::strtoll(s.c_str(), &end, 10);
            if (s.empty() || *end) {
                log_error("%s: invalid int value '%s'", what, s.c_str());
                exit(2);
            }
            return v;
        };
        auto to_d = [&](const std::string& s, const char* what) {
            char* end = nullptr;
            const double v = std::strtod(s.c_str(), &end);
            if (s.empty() || *end) {
                log_error("%s: invalid float value '%s'", what, s.c_str());
                exit(2);
            }
            return v;
        };
        if (arg == "--encoder") a->encoder_gguf = next("--encoder");
        else if (arg == "--decoder") a->decoder_gguf = next("--decoder");
        else if (arg == "--bin") a->bin_path = next("--bin");
        else if (arg == "--S") a->S = std::atoi(next("--S").c_str());
        else if (arg == "--H") a->H = std::atoi(next("--H").c_str());
        else if (arg == "--W") a->W = std::atoi(next("--W").c_str());
        else if (arg == "--out-prefix") a->out_prefix = next("--out-prefix");
        else if (arg == "--output_dir" || arg == "--output-dir") a->output_dir = next("--output_dir");
        else if (arg == "--device") a->device = next("--device");
        else if (arg == "--threads") a->threads = std::atoi(next("--threads").c_str());
        else if (arg == "--dump-stage") a->dump_stage = next("--dump-stage");
        else if (arg == "--images" || arg == "--input_folder" || arg == "--input-folder")
            a->images = next("--input_folder");
        else if (arg == "--timing") a->timing = true;
        else if (arg == "--vk-fast") a->vk_fast = true;
        else if (arg == "--cuda-fast") a->cuda_fast = true;
        else if (arg == "--save_predicted_input_cameras" || arg == "--save-predicted-input-cameras")
            a->save_cameras_json = true;
        else if (arg == "--save_vggt_input_depths" || arg == "--save-vggt-input-depths")
            a->save_input_depths = true;
        else if (arg == "--save_gaussian_alpha_distribution" || arg == "--save-alpha-dist")
            a->save_alpha_dist = true;
        else if (arg == "--save_gaussian_scale_distribution" || arg == "--save-scale-dist")
            a->save_scale_dist = true;
        else if (arg == "--save_vggt_depth_pointcloud") a->save_depth_pointcloud = true;
        else if (arg == "--vggt_depth_pointcloud_target_points") {
            a->depth_pointcloud_target_set = true;
            a->depth_pointcloud_target = to_ll(next(arg.c_str()), arg.c_str());
        } else if (arg == "--save-depth-pointcloud") {  // legacy: flag + target
            a->save_depth_pointcloud = a->depth_pointcloud_target_set = true;
            a->depth_pointcloud_target = to_ll(next(arg.c_str()), arg.c_str());
        } else if (arg == "--gaussian_save_opacity_threshold" || arg == "--opacity-threshold")
            a->threshold_raw = values("--gaussian_save_opacity_threshold", true);
        else if (arg == "--use_tto") a->use_tto = true;
        else if (arg == "--tto_n_steps") a->tto_n_steps = to_ll(next(arg.c_str()), arg.c_str());
        else if (arg == "--tto_lr") a->tto_lr = to_d(next(arg.c_str()), arg.c_str());
        else if (arg == "--tto_lpips_weight") a->tto_lpips_weight = to_d(next(arg.c_str()), arg.c_str());
        else if (arg == "--tto_save_step") a->tto_save_step_raw = values("--tto_save_step", false);
        else if (arg == "--render") a->render = true;
        else if (arg == "--render-only") { a->render = true; a->render_from = next("--render-only"); }
        else if (arg == "--render-from-cameras") a->render_cameras = next("--render-from-cameras");
        else {
            log_error("unknown argument: %s", arg.c_str());
            return false;
        }
    }
    if (a->render_from.empty()) {
        // full pipeline: exactly one of --bin / --images (--S only with --bin;
        // with --images the view count comes from the file list)
        if (a->encoder_gguf.empty() || a->decoder_gguf.empty()) return false;
        if (a->bin_path.empty() == a->images.empty()) return false;
        if (!a->bin_path.empty() && a->S <= 0) return false;
    }
    if (!a->render_from.empty() && a->S <= 0) return false;
    if (!a->render_from.empty() && a->out_prefix.empty() && !a->output_dir.empty()) {
        log_error("--render-only writes the tooling layout: pass --out-prefix");
        return false;
    }
    if (a->out_prefix.empty() && a->output_dir.empty()) a->out_prefix = "/tmp/querysplat";

    // infer.py main(): thresholds/tto steps parsed, then validate_args()
    std::vector<double> thr, steps;
    if (!parse_number_list(a->threshold_raw, false, "--gaussian_save_opacity_threshold", &thr))
        return false;
    if (!parse_number_list(a->tto_save_step_raw, true, "--tto_save_step", &steps))
        return false;
    a->opacity_thresholds = sorted_set(thr);
    steps = sorted_set(steps);
    auto fail = [](const char* msg) { log_error("%s", msg); return false; };
    if (a->opacity_thresholds.empty()) return fail("Opacity thresholds must be in [0, 1]");
    for (double v : a->opacity_thresholds)
        if (!(0 <= v && v <= 1)) return fail("Opacity thresholds must be in [0, 1]");
    if (a->depth_pointcloud_target_set) {
        if (!a->save_depth_pointcloud)
            return fail("--vggt_depth_pointcloud_target_points requires --save_vggt_depth_pointcloud");
        if (a->depth_pointcloud_target <= 0)
            return fail("--vggt_depth_pointcloud_target_points must be positive");
    }
    if (a->save_depth_pointcloud && !a->depth_pointcloud_target_set)
        return fail("--save_vggt_depth_pointcloud requires --vggt_depth_pointcloud_target_points");
    if (a->tto_n_steps < 0 || !(a->tto_lr > 0) || !(a->tto_lpips_weight >= 0))
        return fail("TTO steps/LPIPS weight must be non-negative and TTO LR must be positive");
    if (!steps.empty() && !a->use_tto) return fail("--tto_save_step requires --use_tto");
    for (double s : steps)
        if (s < 0 || s > (double)a->tto_n_steps)
            return fail("TTO save steps must fall between 0 and --tto_n_steps");
    return true;
}


// ---- stage timing (--timing) ------------------------------------------------

struct Timing {
    double model_load = 0, input = 0, camera_and_depth = 0, camera_math = 0;
    double image_to_3dgs = 0, render = 0, write = 0, total = 0;
    // camera_math split along the official stage boundary: pose decoding
    // belongs to predict_cameras_and_depths, plucker/normalize to
    // forward_gaussians
    double cam_decode = 0, host_prep = 0;
};

double since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
}

long peak_rss_kb() {
#if defined(__linux__)
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long kb = 0;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, "VmHWM: %ld kB", &kb) == 1) break;
    std::fclose(f);
    return kb;
#else
    return 0;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 2;
    }
    // Backend options carry the accuracy profiles explicitly (the former
    // process-global set_*_f32_accuracy_mode calls): f32-accuracy kernels by
    // default; --vk-fast / --cuda-fast opt out per backend.
    const auto bopts_from_args = [&args]() {
        mapggml::BackendOptions bopts;
        bopts.device = args.device;
        bopts.cpu_threads = args.threads > 0
                                ? args.threads
                                : (int)std::thread::hardware_concurrency();
        bopts.vulkan_f32_accuracy = !args.vk_fast;
        bopts.cuda_f32_accuracy = !args.cuda_fast;
        return bopts;
    };
    // model resolution + background from the decoder metadata (infer.py
    // reads both from the options YAML the GGUF was converted from)
    qs::DecoderConfig meta_cfg;
    if (!args.decoder_gguf.empty() && !qs::read_decoder_config(args.decoder_gguf, &meta_cfg))
        return 1;
    if (args.H <= 0) args.H = meta_cfg.img_size[0];
    if (args.W <= 0) args.W = meta_cfg.img_size[1];
    if (!args.dump_stage.empty())
        // dump writers fopen() without mkdir — create the tree up front
        std::filesystem::create_directories(args.dump_stage);
    {
        // same for the output prefix's parent (PLY/bin/renders/timing json)
        const auto parent = std::filesystem::path(args.out_prefix).parent_path();
        if (!args.out_prefix.empty() && !parent.empty())
            std::filesystem::create_directories(parent);
    }
    if (!args.render_from.empty()) {
        // render-only path: rasterize previously predicted gaussians without
        // running the models (parity tooling / re-rendering).
        const std::string gpath = args.render_from + "_gaussians.bin";
        // cameras: explicit bin (--render-from-cameras, e.g. the official DL3DV
        // protocol's target cameras) or the prefix's own predicted cameras
        const std::string cpath = args.render_cameras.empty()
                                      ? args.render_from + "_cameras.bin"
                                      : args.render_cameras;
        auto graw = mapggml::read_file_bytes(gpath);
        auto craw = mapggml::read_file_bytes(cpath);
        const int64_t total = (int64_t)(graw.size() / 4);
        qs::GaussianOutput g;
        // stride = 11 + color_dim; solve from the total with stride 23/14
        const int stride_guess = total % 23 == 0 ? 23 : 14;
        g.N = total / stride_guess;
        g.color_dim = stride_guess - 11;
        g.data.resize(total);
        std::memcpy(g.data.data(), graw.data(), graw.size());
        g.opacity.resize((size_t)g.N);
        for (int64_t i = 0; i < g.N; i++)
            g.opacity[(size_t)i] = g.data[(size_t)i * stride_guess + 3];
        mapggml::Backend be = mapggml::init_backend(bopts_from_args());
        qs::RenderConfig rcfg;
        rcfg.width = args.W;
        rcfg.height = args.H;
        rcfg.sh_degree = g.color_dim == 3 ? 0 : 1;
        rcfg.near_plane = meta_cfg.znear;
        rcfg.far_plane = meta_cfg.zfar;
        std::memcpy(rcfg.bg_color, meta_cfg.bg_color, sizeof(rcfg.bg_color));
        const float* w2c = reinterpret_cast<const float*>(craw.data());
        const float* k4 = reinterpret_cast<const float*>(craw.data()) + args.S * 16;
        qs::RenderedViews views =
            qs::render_gaussians(g, w2c, k4, args.S, rcfg);
        for (int v = 0; v < args.S; v++) {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s_render_view%d.png",
                          args.out_prefix.c_str(), v);
            qs::save_png(path, &views.images[(size_t)v * 3 * args.H * args.W],
                         args.W, args.H);
        }
        FILE* fi = fopen((args.out_prefix + "_render_images.bin").c_str(), "wb");
        if (fi) { fwrite(views.images.data(), 4, views.images.size(), fi); fclose(fi); }
        FILE* fa = fopen((args.out_prefix + "_alphas.bin").c_str(), "wb");
        if (fa) { fwrite(views.alphas.data(), 4, views.alphas.size(), fa); fclose(fa); }
        log_info("render-only done: %s*", args.out_prefix.c_str());
        return 0;
    }

    mapggml::Backend be = mapggml::init_backend(bopts_from_args());
    const auto sync_be = [&be]() { ggml_backend_synchronize(be.handle); };
    Timing tm;

    // ---- input: --images (stb decode + preprocess) or --bin ----
    std::vector<float> frames;
    std::vector<std::string> image_paths;
    if (!args.images.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::string>& paths = image_paths;
        if (!qs::collect_images(args.images, &paths)) {
            log_error("No supported images found in: %s (png/jpg/jpeg/webp/bmp)",
                      args.images.c_str());
            return 1;
        }
        if (!qs::load_images_as_frames(paths, args.H, args.W, &frames)) return 1;
        if (!args.dump_stage.empty()) {
            // parity debugging: the pre-normalization [0,1] frames (S,3,H,W)
            std::filesystem::path fp = std::filesystem::path(args.dump_stage) /
                                       "cpp_input_frames.f32";
            std::ofstream f(fp, std::ios::binary);
            f.write(reinterpret_cast<const char*>(frames.data()),
                    (std::streamsize)(frames.size() * sizeof(float)));
            log_info("dumped input frames to %s", fp.string().c_str());
        }
        args.S = (int)paths.size();
        tm.input = since(t0);
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> raw = mapggml::read_file_bytes(args.bin_path);
        const size_t expect = (size_t)args.S * 3 * args.H * args.W * sizeof(float);
        if (raw.size() != expect) {
            log_error("frames.bin size mismatch: got %zu bytes, expected %zu",
                      raw.size(), expect);
            return 1;
        }
        frames.resize(expect / sizeof(float));
        std::memcpy(frames.data(), raw.data(), raw.size());
        tm.input = since(t0);
    }
    const size_t frame_elems = frames.size();

    // ---- encoder (VGGT-Omega via the embedded runtime) ----
    const auto t_load_enc = std::chrono::steady_clock::now();
    auto enc_model = mapggml::load_gguf(args.encoder_gguf);
    if (!enc_model) return 1;
    mapggml::RuntimeOptions ropts;
    ropts.expose_tokens = true;
    ropts.f32_accuracy = !args.cuda_fast;
    mapggml::VGGTRuntime enc(*enc_model, be, args.S, args.H, args.W, ropts);
    if (!enc.valid()) return 1;
    tm.model_load += since(t_load_enc);

    // ---- encoder run (official timing stage: camera_and_depth) ----
    const auto t_enc = std::chrono::steady_clock::now();
    mapggml::VGGTOutputs enc_out;
    if (!enc.run(frames.data(), enc_out)) return 1;
    sync_be();
    tm.camera_and_depth = since(t_enc);
    log_info("encoder: tokens=%zu pose=%zu depth=%zu (out %dx%d) token_start=%d",
             enc_out.cached_tokens.size(), enc_out.pose_enc.size(),
             enc_out.depth.size(), enc_out.out_h, enc_out.out_w,
             enc_out.patch_token_start);

    // ---- host camera math ----
    const auto t_cam = std::chrono::steady_clock::now();
    qs::CameraData cams =
        qs::decode_cameras(enc_out.pose_enc.data(), args.S, args.H, args.W);
    tm.cam_decode = since(t_cam);
    const auto t_prep = std::chrono::steady_clock::now();
    std::vector<float> plucker = qs::plucker_from_cameras(cams, args.H, args.W);
    std::vector<float> rgb_norm = qs::imagenet_normalize(
        frames.data(), args.S, frame_elems);
    tm.host_prep = since(t_prep);
    tm.camera_math = tm.cam_decode + tm.host_prep;
    if (!args.dump_stage.empty()) {
        // host seams: pose_enc (S,9), w2c (S*16 row-major), intrinsics (S*4),
        // plucker (S,6,H,W), depth/depth_conf (S,out_h,out_w)
        auto bin = [&](const std::string& name, const void* p, size_t bytes) {
            FILE* d = fopen((args.dump_stage + "/" + name + ".bin").c_str(), "wb");
            if (d) {
                fwrite(p, 1, bytes, d);
                fclose(d);
            }
        };
        bin("cpp_pose_enc", enc_out.pose_enc.data(),
            enc_out.pose_enc.size() * sizeof(float));
        bin("cpp_w2c", cams.w2c.data(), cams.w2c.size() * sizeof(float));
        bin("cpp_intrinsics", cams.intrinsics.data(),
            cams.intrinsics.size() * sizeof(float));
        bin("cpp_plucker", plucker.data(), plucker.size() * sizeof(float));
        bin("cpp_depth", enc_out.depth.data(), enc_out.depth.size() * sizeof(float));
        bin("cpp_depth_conf", enc_out.depth_conf.data(),
            enc_out.depth_conf.size() * sizeof(float));
    }

    // ---- decoder ----
    const auto t_load_dec = std::chrono::steady_clock::now();
    qs::DecoderRuntimeOptions decopts;
    decopts.dump_dir = args.dump_stage;
    decopts.f32_accuracy = !args.cuda_fast;
    qs::QuerySplatDecoder dec(args.decoder_gguf, be, decopts);
    if (!dec.valid()) return 1;
    const auto& cfg = dec.config();
    log_info("decoder: C=%d heads=%d dec_depth=%d post_depth=%d gs_tokens=%d "
             "gpt=%d sh=%d",
             cfg.enc_embed_dim, cfg.enc_num_heads, cfg.dec_depth,
             cfg.post_decoder_depth, cfg.num_gs_tokens,
             cfg.num_gaussians_per_token, cfg.gaussian_sh_degree);
    if (!dec.ensure_shape(args.S, args.H, args.W, enc_out.patch_token_start))
        return 1;
    tm.model_load += since(t_load_dec);

    // ---- decoder run (official timing stage: image_to_3dgs) ----
    const auto t_dec = std::chrono::steady_clock::now();
    qs::DecoderOutputs dec_out;
    if (!dec.run(enc_out.cached_tokens.data(), rgb_norm.data(), plucker.data(),
                 dec_out))
        return 1;
    sync_be();
    tm.image_to_3dgs = since(t_dec);
    log_info("gaussians: N=%lld stride=%d", (long long)dec_out.N,
             11 + dec_out.color_dim);

    // ---- TTO (t4/t5): optimize the KV memory against the input views ----
    // Mirrors QuerySplat._optimize_memory: probe render at the input views,
    // L = L1 + 0.2*(1-fused_ssim)/2 + w*LPIPS(gt, pred, normalize=False),
    // AdamW on the KV memory. The encoder/cameras stay frozen.
    std::vector<long long> tto_steps;
    if (args.use_tto) {
        std::vector<double> tstep_d;
        if (!parse_number_list(args.tto_save_step_raw, true, "--tto_save_step",
                               &tstep_d))
            return 1;
        for (double v : tstep_d) tto_steps.push_back((long long)v);
        // LPIPS-vgg metric graph (t3) for the w-weighted loss term
        std::unique_ptr<qs::LpipsVgg> lpips;
        if (args.tto_lpips_weight > 0) {
            lpips = std::make_unique<qs::LpipsVgg>(
                "models/gguf/lpips-vgg-f32.gguf", be);
            if (!lpips->valid()) {
                log_error("tto: missing LPIPS metric weights "
                          "models/gguf/lpips-vgg-f32.gguf (convert with "
                          "scripts/convert_lpips_to_gguf.py)");
                return 1;
            }
        }
        if (!dec.tto_begin(args.S, args.H, args.W, enc_out.patch_token_start,
                           enc_out.cached_tokens.data(), rgb_norm.data(),
                           plucker.data(), args.tto_lr))
            return 1;
        const double lw = args.tto_lpips_weight;
        const auto t_tto = std::chrono::steady_clock::now();
        double last_l1 = 0, last_ssim = 0, last_lp = 0;
        bool has_lp = false;
        qs::RenderConfig rcfg;
        rcfg.width = args.W;
        rcfg.height = args.H;
        rcfg.near_plane = cfg.znear;
        rcfg.far_plane = cfg.zfar;
        rcfg.sh_degree = cfg.gaussian_sh_degree;
        std::memcpy(rcfg.bg_color, cfg.bg_color, sizeof(rcfg.bg_color));
        const int post_out_dim =
            3 * (cfg.gaussian_sh_degree + 1) * (cfg.gaussian_sh_degree + 1) + 1;
        for (long long step = 1; step <= args.tto_n_steps; step++) {
            std::vector<float> deconv_raw, color_raw;
            if (!dec.tto_forward(deconv_raw, color_raw)) return 1;
            qs::DecoderOutputs to = dec.tto_gaussians_from(deconv_raw, color_raw);
            qs::GaussianOutput g;
            g.N = to.N;
            g.color_dim = to.color_dim;
            g.data = std::move(to.gaussians);
            g.opacity.resize((size_t)g.N);
            const int stride = 11 + g.color_dim;
            for (int64_t i = 0; i < g.N; i++)
                g.opacity[(size_t)i] = g.data[(size_t)i * stride + 3];
            qs::RenderedViews views = qs::render_gaussians(
                g, cams.w2c.data(), cams.intrinsics.data(), args.S, rcfg);
            // LPIPS first (its grad feeds the combined loss backward)
            std::vector<float> d_lpips;
            double lp_val = 0;
            if (lw > 0) {
                std::vector<float> lv;
                if (!lpips->value_and_grad(views.images.data(), frames.data(),
                                           args.S, args.H, args.W, &lv,
                                           &d_lpips))
                    return 1;
                lp_val = lv.empty() ? 0.0 : (double)lv[0];
                has_lp = true;
                last_lp = lp_val;
            }
            qs::TtoLossOut lo;
            if (!qs::tto_loss_value_and_grad(
                    views.images.data(), frames.data(), args.S, args.H, args.W,
                    lw, lw > 0 ? d_lpips.data() : nullptr, &lo))
                return 1;
            qs::GaussianGrads gg;
            if (!qs::render_backward(g, cams.w2c.data(), cams.intrinsics.data(),
                                     args.S, rcfg, views, lo.d_pred.data(),
                                     &gg))
                return 1;
            // pack the gaussian grads into the raw layout (N, 11+color_dim)
            std::vector<float> dg((size_t)gg.N * stride);
            for (int64_t i = 0; i < gg.N; i++) {
                float* r = &dg[(size_t)i * stride];
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
            if (!dec.tto_backward(dgeom.data(), dcol.data(), (int)step - 1))
                return 1;
            last_l1 = lo.l1;
            last_ssim = lo.ssim;
            log_info("TTO [%lld/%lld] L1=%.6f SSIM=%.6f LPIPS=%s", step,
                     args.tto_n_steps, lo.l1, lo.ssim,
                     lw > 0 ? std::to_string(lp_val).c_str() : "nan");
            // save_steps capture the POST-step decode (official capture())
            bool want = false;
            for (long long v : tto_steps)
                if (v == step) want = true;
            if (want) {
                std::vector<float> sr, cr;
                if (!dec.tto_forward(sr, cr)) return 1;
                qs::DecoderOutputs so = dec.tto_gaussians_from(sr, cr);
                qs::GaussianOutput gs;
                gs.N = so.N;
                gs.color_dim = so.color_dim;
                gs.data = std::move(so.gaussians);
                gs.opacity.resize((size_t)gs.N);
                for (int64_t i = 0; i < gs.N; i++)
                    gs.opacity[(size_t)i] =
                        gs.data[(size_t)i * stride + 3];
                char nm[64];
                std::snprintf(nm, sizeof(nm), "gaussians_tto_step%04lld.ply",
                              step);
                const std::string where =
                    args.output_dir.empty()
                        ? args.out_prefix + "_" + nm
                        : args.output_dir + "/" + nm;
                if (!qs::save_gaussian_ply(gs, where, 0.005f))
                    log_error("tto: failed to write %s", where.c_str());
                else
                    log_info("tto: wrote %s", where.c_str());
            }
        }
        // final decode from the optimized memory (official returns decode())
        std::vector<float> fr, cr;
        if (!dec.tto_forward(fr, cr)) return 1;
        dec_out = dec.tto_gaussians_from(fr, cr);
        log_info("tto done in %.1fs (final L1=%.6f SSIM=%.6f%s)", since(t_tto),
                 last_l1, last_ssim,
                 has_lp ? (" LPIPS=" + std::to_string(last_lp)).c_str() : "");
        if (args.output_dir.empty() && args.out_prefix.empty()) {
            // nothing to write to, but the losses still go somewhere useful
        }
        if (!args.output_dir.empty()) {
            // the official outputs section creates the tree only later —
            // create it here first (tto_losses lands before the run ends)
            std::filesystem::create_directories(args.output_dir);
            if (!qs::write_tto_losses(args.output_dir, last_l1, last_ssim,
                                      has_lp ? &last_lp : nullptr))
                return 1;
        }
        sync_be();
    }

    // ---- outputs ----
    qs::GaussianOutput g;
    g.N = dec_out.N;
    g.color_dim = dec_out.color_dim;
    g.data = dec_out.gaussians;
    g.opacity.resize((size_t)g.N);
    const int stride = 11 + g.color_dim;
    for (int64_t i = 0; i < g.N; i++) g.opacity[(size_t)i] = g.data[(size_t)i * stride + 3];

    const bool official = !args.output_dir.empty();
    const bool prefix_mode = !args.out_prefix.empty();
    // infer.py: one PLY per threshold, suffixed only when there are several
    const bool multi = args.opacity_thresholds.size() > 1;

    // ---- forward rasterization (official chain: always renders) ----
    qs::RenderedViews views;
    if (official || args.render) {
        const auto t_rend = std::chrono::steady_clock::now();
        qs::RenderConfig rcfg;
        rcfg.width = args.W;
        rcfg.height = args.H;
        rcfg.near_plane = cfg.znear;
        rcfg.far_plane = cfg.zfar;
        rcfg.sh_degree = cfg.gaussian_sh_degree;
        std::memcpy(rcfg.bg_color, cfg.bg_color, sizeof(rcfg.bg_color));
        views = qs::render_gaussians(g, cams.w2c.data(), cams.intrinsics.data(),
                                     args.S, rcfg);
        tm.render = since(t_rend);
    }
    const auto t_out = std::chrono::steady_clock::now();

    // ---- tooling layout (--out-prefix) ----
    if (prefix_mode) {
        for (double thr : args.opacity_thresholds) {
            const std::string path = args.out_prefix + "_gaussians" +
                (multi ? qs::opacity_threshold_suffix(thr) : std::string()) + ".ply";
            if (!qs::save_gaussian_ply(g, path, (float)thr)) {
                log_error("failed to write %s", path.c_str());
                return 1;
            }
            log_info("wrote %s", path.c_str());
        }
        std::string bin_path = args.out_prefix + "_gaussians.bin";
        if (!qs::save_gaussians_bin(g, bin_path)) return 1;
        log_info("wrote %s", bin_path.c_str());
        // cameras for downstream consumers (--render-only, eval tooling):
        // w2c (S*16) then intrinsics (S*4), row-major, f32
        std::string cam_path = args.out_prefix + "_cameras.bin";
        FILE* fp = fopen(cam_path.c_str(), "wb");
        if (fp) {
            fwrite(cams.w2c.data(), 4, cams.w2c.size(), fp);
            fwrite(cams.intrinsics.data(), 4, cams.intrinsics.size(), fp);
            fclose(fp);
            log_info("wrote %s", cam_path.c_str());
        }
        if (args.render) {
            for (int v = 0; v < args.S; v++) {
                char path[1024];
                std::snprintf(path, sizeof(path), "%s_render_view%d.png",
                              args.out_prefix.c_str(), v);
                if (!qs::save_png(path, &views.images[(size_t)v * 3 * args.H * args.W],
                                  args.W, args.H)) {
                    log_error("failed to write %s", path);
                    return 1;
                }
                log_info("wrote %s", path);
            }
            auto raw = [&](const char* suffix, const std::vector<float>& v) {
                FILE* f = fopen((args.out_prefix + suffix).c_str(), "wb");
                if (f) {
                    fwrite(v.data(), 4, v.size(), f);
                    fclose(f);
                }
            };
            raw("_render_images.bin", views.images);
            raw("_alphas.bin", views.alphas);
            raw("_depths.bin", views.depths);
        }
    }
    if (!args.dump_stage.empty()) {
        // parity seam copy of the final gaussians
        qs::save_gaussians_bin(g, args.dump_stage + "/cpp_gaussians.bin");
    }

    // ---- official layout (--output_dir): scripts/infer.py main(), same
    // files in the same order, writers in output_parity.cpp ----
    if (official) {
        const std::string& od = args.output_dir;
        std::error_code ec;
        std::filesystem::create_directories(od, ec);
        qs::OfficialInputs in;
        in.S = args.S;
        in.H = args.H;
        in.W = args.W;
        in.frames = frames.data();
        in.w2c = cams.w2c.data();
        in.intrinsics = cams.intrinsics.data();
        in.pose_enc = enc_out.pose_enc.data();
        in.depth = enc_out.depth.data();
        in.conf = enc_out.depth_conf.data();
        in.depth_h = enc_out.out_h;
        in.depth_w = enc_out.out_w;
        for (int i = 0; i < args.S; i++)
            in.names.push_back(i < (int)image_paths.size()
                                   ? qs::source_stem(image_paths[(size_t)i])
                                   : "frame" + std::to_string(i));  // --bin input
        bool ok = qs::write_preprocessed_frames(od, in) &&
                  qs::write_rendered_views(od, views.images.data(), args.S, args.H, args.W);
        if (ok && args.save_cameras_json) ok = qs::write_predicted_input_cameras(od, in);
        if (ok && args.save_input_depths) ok = qs::write_vggt_input_depths(od, in);
        if (ok && args.save_depth_pointcloud)
            ok = qs::write_vggt_depth_pointcloud(od, in, args.depth_pointcloud_target);
        // pointcloud.ply threshold = options.gaussian_save_opacity_threshold
        // (config), not the CLI list
        ok = ok && qs::save_colored_pointcloud_ply(g, od + "/pointcloud.ply",
                                                   cfg.gaussian_save_opacity_threshold);
        if (ok && args.save_alpha_dist)
            ok = qs::write_alpha_distribution(od, g.data.data(), g.N, stride);
        if (ok && args.save_scale_dist)
            ok = qs::write_scale_distribution(od, g.data.data(), g.N, stride,
                                              qs::config_value(cfg.gaussian_scale_cap));
        std::string saved;
        for (double thr : args.opacity_thresholds) {
            if (!ok) break;
            const std::string name =
                multi ? "gaussians" + qs::opacity_threshold_suffix(thr) + ".ply"
                      : std::string("gaussians.ply");
            ok = qs::save_gaussian_ply(g, od + "/" + name, (float)thr);
            saved += (saved.empty() ? "" : ", ") + name;
        }
        qs::InferenceTiming it;
        // str(torch.device): backend name minus the ordinal, lower-cased
        // ("CUDA0" -> "cuda", "Vulkan0" -> "vulkan", "CPU" -> "cpu")
        for (char c : be.name)
            if (!std::isdigit((unsigned char)c)) it.device += (char)std::tolower((unsigned char)c);
        it.num_input_views = args.S;
        it.camera_and_depth_seconds = tm.camera_and_depth + tm.cam_decode;
        it.image_to_3dgs_seconds = tm.host_prep + tm.image_to_3dgs;
        it.render_seconds = tm.render;
        it.opacity_thresholds = args.opacity_thresholds;
        it.use_tto = args.use_tto;
        it.tto_n_steps = args.use_tto ? args.tto_n_steps : 0;
        it.tto_lr = args.tto_lr;
        it.tto_lr_set = args.use_tto;
        it.tto_lpips_weight = args.tto_lpips_weight;
        it.tto_lpips_set = args.use_tto && args.tto_lpips_weight > 0;
        for (long long v : tto_steps) it.tto_save_steps.push_back(v);
        ok = ok && qs::write_inference_timing(od, it) && qs::write_camera_info(od, args.S);
        if (!ok) {
            log_error("failed to write the official outputs under %s", od.c_str());
            return 1;
        }
        log_info("Results saved to %s", od.c_str());
        log_info("3DGS files: %s", saved.c_str());
        log_info("Timing: camera/depth=%.6fs, image-to-3DGS=%.6fs, render=%.6fs",
                 it.camera_and_depth_seconds, it.image_to_3dgs_seconds,
                 it.render_seconds);
    }
    tm.write = since(t_out);

    if (args.timing && prefix_mode) {
        tm.total = tm.model_load + tm.input + tm.camera_and_depth +
                   tm.camera_math + tm.image_to_3dgs + tm.render + tm.write;
        const long rss = peak_rss_kb();
        log_info("timing: model_load=%.2fs input=%.3fs "
                 "camera_and_depth=%.2fs camera_math=%.3fs "
                 "image_to_3dgs=%.2fs render=%.2fs write=%.2fs total=%.2fs "
                 "peak_rss=%.0f MB",
                 tm.model_load, tm.input, tm.camera_and_depth, tm.camera_math,
                 tm.image_to_3dgs, tm.render, tm.write, tm.total,
                 rss / 1024.0);
        char tpath[1024];
        std::snprintf(tpath, sizeof(tpath), "%s_timing.json",
                      args.out_prefix.c_str());
        FILE* ft = fopen(tpath, "w");
        if (ft) {
            std::fprintf(ft,
                         "{\n"
                         "  \"arch\": \"querysplat\",\n"
                         "  \"device\": \"%s\",\n"
                         "  \"num_input_views\": %d,\n"
                         "  \"image_size\": [%d, %d],\n"
                         "  \"model_load_seconds\": %.4f,\n"
                         "  \"input_seconds\": %.4f,\n"
                         "  \"camera_and_depth_seconds\": %.4f,\n"
                         "  \"camera_math_seconds\": %.4f,\n"
                         "  \"image_to_3dgs_seconds\": %.4f,\n"
                         "  \"render_seconds\": %.4f,\n"
                         "  \"write_seconds\": %.4f,\n"
                         "  \"total_seconds\": %.4f,\n"
                         "  \"peak_rss_kb\": %ld\n"
                         "}\n",
                         be.name.c_str(), args.S, args.H, args.W,
                         tm.model_load, tm.input, tm.camera_and_depth,
                         tm.camera_math, tm.image_to_3dgs, tm.render,
                         tm.write, tm.total, rss);
            std::fclose(ft);
            log_info("wrote %s", tpath);
        }
    }
    return 0;
}
