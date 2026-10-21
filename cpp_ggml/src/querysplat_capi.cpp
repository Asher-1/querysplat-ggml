// ----------------------------------------------------------------------------
// querysplat C API implementation — one Session owns the dual-graph chain and
// mirrors querysplat-cli's flow: mapggml runtime (encoder, expose_tokens) ->
// host camera math (Plucker + ImageNet normalize) -> QuerySplat decoder ->
// host activation heads (inside decoder run()). All state is per-context;
// errors are thread-local strings; no environment variables are read.
// ----------------------------------------------------------------------------
#include "querysplat/capi.h"

#include "backend.hpp"
#include "camera_math.hpp"
#include "gguf_loader.hpp"
#include "image_io.hpp"
#include "querysplat_graph.hpp"
#include "vggt_graph.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace mapggml;
using namespace qs;

namespace {

thread_local std::string t_last_error;

void set_last_error(const char* msg) { t_last_error = msg ? msg : ""; }

double steady_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// C++-side snapshot of the opaque options handle.
struct Options {
    std::string device = "auto";
    int threads = 0;
    int f32_accuracy = 1;  // GPU f32-accuracy profile (vulkan + cuda + graph)
    int height = 0;        // 0 = decoder GGUF metadata resolution
    int width = 0;
};

// Session: both graphs + the backend + the lazily (re)built encoder runtime
// for the current (S, H, W) + the borrowed result store.
struct Session {
    Backend be;
    std::unique_ptr<GGUFModel> enc_model;
    std::unique_ptr<QuerySplatDecoder> dec;
    std::unique_ptr<VGGTRuntime> enc;
    RuntimeOptions ropts;    // expose_tokens + graph-level f32_accuracy
    DecoderConfig meta_cfg;  // decoder GGUF metadata (resolution, gs layout)
    int cur_s = 0, cur_h = 0, cur_w = 0;

    // result store — querysplat_result points into these
    DecoderOutputs out;
    std::vector<float> pose_store;
    querysplat_timings timings{};

    // (Re)builds the encoder runtime when the (S, H, W) geometry changes;
    // weights re-stage (the mapggml capi.cpp contract: callers sweeping
    // resolutions should expect that cost).
    bool ensure_encoder(int s, int h, int w, std::string& err) {
        if (enc && cur_s == s && cur_h == h && cur_w == w) return true;
        enc.reset();
        cur_s = cur_h = cur_w = 0;
        enc.reset(new VGGTRuntime(*enc_model, be, s, h, w, ropts));
        if (!enc->valid()) {
            enc.reset();
            err = "encoder runtime failed to build for the requested geometry";
            return false;
        }
        cur_s = s;
        cur_h = h;
        cur_w = w;
        return true;
    }
};

// Shared inference tail. preprocess_ms and the t0 anchor come from the
// entry point (frames and image-paths preprocess differently); e2e covers
// entry until the result is ready.
int run_common(Session* s, std::vector<float>&& input, int n, int h, int w,
               double preprocess_ms, double t0, querysplat_result* out) {
    if (!s || input.size() != (size_t)n * 3 * h * w || n < 1 || h <= 0 ||
        w <= 0 || !out) {
        set_last_error("invalid arguments (ctx/frames/n_views/h/w/out)");
        return -1;
    }
    if (!s->enc_model || !s->dec || !s->dec->valid()) {
        set_last_error("context is not ready (encoder or decoder failed to load)");
        return -1;
    }
    s->timings = {};
    s->timings.preprocess_ms = preprocess_ms;

    const double t1 = steady_ms();
    std::string err;
    if (!s->ensure_encoder(n, h, w, err)) {
        set_last_error(err.c_str());
        return -2;
    }
    VGGTOutputs enc_out;
    if (!s->enc->run(input.data(), enc_out)) {
        set_last_error("encoder run failed (see the process log)");
        return -2;
    }
    // host camera math (official stage: camera_math)
    CameraData cams = decode_cameras(enc_out.pose_enc.data(), n, h, w);
    std::vector<float> plucker = plucker_from_cameras(cams, h, w);
    std::vector<float> rgb_norm =
        imagenet_normalize(input.data(), n, input.size());
    if (!s->dec->ensure_shape(n, h, w, enc_out.patch_token_start)) {
        set_last_error("decoder graph build failed for the requested geometry");
        return -2;
    }
    if (!s->dec->run(enc_out.cached_tokens.data(), rgb_norm.data(),
                     plucker.data(), s->out)) {
        set_last_error("decoder run failed (see the process log)");
        return -2;
    }
    s->timings.inference_ms = steady_ms() - t1;

    const double t2 = steady_ms();
    s->pose_store = enc_out.pose_enc;
    querysplat_result r{};
    r.n_views = n;
    r.height = h;
    r.width = w;
    r.pose = s->pose_store.data();
    r.pose_stride = 9;
    r.n_gaussians = s->out.N;
    r.gaussians_stride = 11 + s->out.color_dim;
    r.gaussians = s->out.gaussians.data();
    r.timings = s->timings;
    r.timings.postprocess_ms = steady_ms() - t2;
    r.timings.e2e_ms = steady_ms() - t0;
    *out = r;
    t_last_error.clear();
    return 0;
}

}  // namespace

extern "C" {

int querysplat_abi_version(void) { return QUERYSPLAT_CAPI_ABI_VERSION; }

// ---- lifecycle ----

querysplat_options* querysplat_options_new(void) {
    try {
        return (querysplat_options*)(new Options());
    } catch (...) {
        return nullptr;
    }
}

void querysplat_options_free(querysplat_options* opts) {
    delete (Options*)opts;
}

void querysplat_options_set_device(querysplat_options* opts,
                                   const char* device) {
    if (!opts || !device || !*device) return;
    ((Options*)opts)->device = device;
}

void querysplat_options_set_threads(querysplat_options* opts, int n_threads) {
    if (!opts || n_threads < 0) return;
    ((Options*)opts)->threads = n_threads;
}

void querysplat_options_set_f32_accuracy(querysplat_options* opts,
                                         int enable) {
    if (!opts || enable < 0) return;
    ((Options*)opts)->f32_accuracy = enable ? 1 : 0;
}

void querysplat_options_set_image_size(querysplat_options* opts, int height,
                                       int width) {
    if (!opts || height < 0 || width < 0) return;
    ((Options*)opts)->height = height;  // 0 = decoder metadata resolution
    ((Options*)opts)->width = width;
}

querysplat_ctx* querysplat_load_opts(const char* encoder_gguf,
                                     const char* decoder_gguf,
                                     const querysplat_options* opts) {
    if (!encoder_gguf || !*encoder_gguf || !decoder_gguf || !*decoder_gguf) {
        set_last_error("encoder_gguf/decoder_gguf is NULL or empty");
        return nullptr;
    }
    try {
        Session* s = new Session();
        Options o;
        if (opts) o = *(const Options*)opts;
        // decoder metadata first: the model resolution + gaussian layout are
        // needed to resolve the options defaults (infer.py reads the same
        // fields from the options YAML)
        if (!read_decoder_config(decoder_gguf, &s->meta_cfg)) {
            set_last_error("failed to read the decoder GGUF metadata");
            delete s;
            return nullptr;
        }
        BackendOptions bopts;
        bopts.device = o.device;
        bopts.cpu_threads = o.threads;
        bopts.vulkan_f32_accuracy = o.f32_accuracy != 0;
        bopts.cuda_f32_accuracy = o.f32_accuracy != 0;
        s->be = init_backend(bopts);
        s->ropts.expose_tokens = true;
        s->ropts.f32_accuracy = o.f32_accuracy != 0;
        s->enc_model = load_gguf(encoder_gguf);
        if (!s->enc_model) {
            set_last_error("failed to load the encoder GGUF (see the log)");
            delete s;
            return nullptr;
        }
        DecoderRuntimeOptions dopts;
        dopts.f32_accuracy = o.f32_accuracy != 0;
        s->dec.reset(new QuerySplatDecoder(decoder_gguf, s->be, dopts));
        if (!s->dec->valid()) {
            set_last_error("failed to load the decoder GGUF (see the log)");
            delete s;
            return nullptr;
        }
        if (o.height > 0 && o.width > 0) {
            s->cur_h = o.height;  // resolved geometry marker for callers;
            s->cur_w = o.width;   // the runtime itself builds on first run
        }
        t_last_error.clear();
        return (querysplat_ctx*)s;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("unknown error loading the querysplat chain");
        return nullptr;
    }
}

void querysplat_free(querysplat_ctx* ctx) { delete (Session*)ctx; }

int querysplat_is_ready(const querysplat_ctx* ctx) {
    const Session* s = (const Session*)ctx;
    return s && s->enc_model && s->dec && s->dec->valid() ? 1 : 0;
}

const char* querysplat_last_error(void) { return t_last_error.c_str(); }

void querysplat_free_buffer(void* p) { free(p); }

int32_t querysplat_default_image_size(const querysplat_ctx* ctx) {
    const Session* s = (const Session*)ctx;
    return s && s->dec && s->dec->valid() ? s->meta_cfg.img_size[0] : 0;
}

// ---- typed inference ----

int querysplat_run_frames(querysplat_ctx* ctx, const float* frames_f32,
                          int32_t n_views, int32_t height, int32_t width,
                          querysplat_result* out) {
    try {
        if (!ctx || !frames_f32 || n_views < 1 || height <= 0 || width <= 0 ||
            !out) {
            set_last_error("invalid arguments (ctx/frames/n_views/h/w/out)");
            return -1;
        }
        // frames arrive [0,1]-normalized (S,3,H,W); copied so the result
        // store owns nothing it does not need
        const double t0 = steady_ms();
        const size_t expect = (size_t)n_views * 3 * height * width;
        std::vector<float> input(frames_f32, frames_f32 + expect);
        const double pp = steady_ms() - t0;
        return run_common((Session*)ctx, std::move(input), n_views, height,
                          width, pp, t0, out);
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -2;
    } catch (...) {
        set_last_error("unknown runtime error in run_frames");
        return -2;
    }
}

int querysplat_run_image_paths(querysplat_ctx* ctx, const char* const* paths,
                               int32_t n_paths, querysplat_result* out) {
    try {
        Session* s = (Session*)ctx;
        if (!s || !paths || n_paths < 1 || !out) {
            set_last_error("invalid arguments (ctx/paths/n_paths/out)");
            return -1;
        }
        if (!s->dec || !s->dec->valid()) {
            set_last_error("context is not ready");
            return -1;
        }
        // resolved resolution: explicit options override, else the decoder
        // GGUF metadata (the official preprocess target)
        int h = s->cur_h > 0 ? s->cur_h : s->meta_cfg.img_size[0];
        int w = s->cur_w > 0 ? s->cur_w : s->meta_cfg.img_size[1];
        std::vector<std::string> files;
        for (int32_t i = 0; i < n_paths; i++) {
            if (!paths[i] || !*paths[i]) {
                set_last_error("paths contains a NULL/empty entry");
                return -1;
            }
            files.emplace_back(paths[i]);
        }
        const double t0 = steady_ms();
        std::vector<float> frames;
        if (!load_images_as_frames(files, h, w, &frames)) {
            set_last_error("image decode/preprocess failed (see the log)");
            return -2;
        }
        const double pp = steady_ms() - t0;
        return run_common(s, std::move(frames), (int)files.size(), h, w, pp,
                          t0, out);
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -2;
    } catch (...) {
        set_last_error("unknown runtime error in run_image_paths");
        return -2;
    }
}

}  /* extern "C" */
