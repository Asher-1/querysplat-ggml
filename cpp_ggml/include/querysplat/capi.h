// ----------------------------------------------------------------------------
// querysplat C ABI — integration surface for the QuerySplat dual-graph chain
// (VGGT-Omega encoder via the embedded mapggml runtime + this project's ggml
// decoder + host activation heads). Mirrors the mapggml / AICore C-API
// contract: opaque contexts, explicit options with defaults, typed results
// with documented ownership, thread-local last_error, and NO environment
// variables for logic control. No STL, Qt, OpenCV, exceptions, or ggml types
// cross this boundary.
//
// The result of one inference is the scene's 3D-Gaussian splat set; rendering
// joins with the rasterizer port (M4) — hosts render the gaussians with their
// own pipeline for now.
// ----------------------------------------------------------------------------
#ifndef QUERYSPLAT_CAPI_H
#define QUERYSPLAT_CAPI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(QS_CAPI_SHARED)
#ifdef QS_CAPI_BUILD
#define QS_CAPI __declspec(dllexport)
#else
#define QS_CAPI __declspec(dllimport)
#endif
#elif defined(QS_CAPI_BUILD)
#define QS_CAPI __attribute__((visibility("default")))
#else
#define QS_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Binary contract of this header. The implementation returns it; bump when
 * any struct layout or function signature changes. */
#define QUERYSPLAT_CAPI_ABI_VERSION 1
QS_CAPI int querysplat_abi_version(void);

typedef struct querysplat_ctx querysplat_ctx;
typedef struct querysplat_options querysplat_options;

/* ---- options (opaque, heap-allocated; setters are no-ops on NULL and
 * ignore invalid values, keeping the previous state) ---- */

QS_CAPI querysplat_options* querysplat_options_new(void);
QS_CAPI void querysplat_options_free(querysplat_options* opts);

/* "auto|cpu|cuda|vulkan|metal" or an instance name such as "CUDA0".
 * NULL/empty keeps the current value. "auto" picks the first available
 * compiled-in backend (CUDA -> Metal -> Vulkan -> CPU). */
QS_CAPI void querysplat_options_set_device(querysplat_options* opts,
                                           const char* device);
/* CPU thread count; <= 0 picks the hardware default. */
QS_CAPI void querysplat_options_set_threads(querysplat_options* opts,
                                            int n_threads);
/* GPU f32-accuracy profile (default 1, torch parity): disables ggml-vulkan's
 * fp16/coop-mat shader paths and selects exact-F32 cuBLAS GEMMs, and builds
 * the decoder attention from explicit F32 ops on CUDA. 0 opts into the fast
 * (drift-prone) kernels. Consumed once at the first backend device init. */
QS_CAPI void querysplat_options_set_f32_accuracy(querysplat_options* opts,
                                                 int enable);
/* Preprocess target resolution before inference. 0,0 (the default) uses the
 * resolution stored in the decoder GGUF metadata (512x512 for the paper
 * checkpoints; querysplat_default_image_size reports it). */
QS_CAPI void querysplat_options_set_image_size(querysplat_options* opts,
                                               int height, int width);

/* ---- context lifecycle ---- */

/* Load BOTH graphs of the chain: the VGGT-Omega encoder GGUF (run on the
 * embedded mapggml runtime with token export) and the QuerySplat decoder
 * GGUF (this project's graph). The backend is initialized from the options.
 * Returns a context whose is_ready() may still be 0; inspect
 * querysplat_last_error(). NULL only on invalid input or out-of-memory. */
QS_CAPI querysplat_ctx* querysplat_load_opts(const char* encoder_gguf,
                                             const char* decoder_gguf,
                                             const querysplat_options* opts);
/* Releases a context; safe on NULL. Contexts are not thread-safe: serialize
 * run calls, distinct contexts may run concurrently. */
QS_CAPI void querysplat_free(querysplat_ctx* ctx);
/* Returns 1 only when the context owns successfully loaded encoder + decoder
 * models. */
QS_CAPI int querysplat_is_ready(const querysplat_ctx* ctx);
/* Last error on the calling thread (empty string when none). The pointer
 * stays valid until the next querysplat call on that thread. */
QS_CAPI const char* querysplat_last_error(void);
/* Releases any malloc'd buffer handed out by this API. Safe on NULL. */
QS_CAPI void querysplat_free_buffer(void* p);

/* Checkpoint-nominal preprocessing resolution from the decoder GGUF metadata
 * (512 for the paper checkpoints; 0 when not ready). */
QS_CAPI int32_t querysplat_default_image_size(const querysplat_ctx* ctx);

/* ---- typed inference ---- */

/* Common wall-clock timings (semantics follow AICore's pipeline timings:
 * preprocess = image ingest / validation; inference = encoder + host camera
 * math + decoder (weight re-staging on a resolution change included);
 * postprocess = typed-result construction; e2e = entry until the result is
 * ready). */
typedef struct querysplat_timings {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double e2e_ms;
} querysplat_timings;

/* Typed result of one inference. Buffer ownership: the pointers are BORROWED
 * views of the context's result store — valid until the next run call or
 * querysplat_free. Never freed by the caller. */
typedef struct querysplat_result {
    int32_t n_views;
    int32_t height;   /* input resolution the gaussians were predicted at */
    int32_t width;
    /* Camera head output [n*9] per view: pose_enc = absT 3 + quaternion 4
     * (scalar-last xyzw) + FoV 2 (vertical/horizontal radians,
     * FoV = relu(x) + 0.01). Borrowed. */
    const float* pose;
    int32_t pose_stride; /* floats per view: 9 */
    /* Final gaussians [n_gaussians * gaussians_stride], torch (N, stride)
     * C-order: [pos(3) | opacity | scale(3) | rot(4) | color(color_dim)].
     * Borrowed. */
    int64_t n_gaussians;      /* num_gs_tokens * gaussians_per_token
                                 (524288 for the paper checkpoints) */
    int32_t gaussians_stride; /* 11 + color_dim (14 for sh0 / RGB) */
    const float* gaussians;
    querysplat_timings timings;
} querysplat_result;

/* Multi-view inference from PREPROCESSED frames — the bit-exact parity path
 * (same contract as the CLI --bin feed): frames_f32 is S*3*H*W floats in
 * [0,1], torch (S,3,H,W) order. S >= 1. Returns 0 ok, -1 contract/validation
 * error, -2 runtime/inference error; inspect querysplat_last_error. */
QS_CAPI int querysplat_run_frames(querysplat_ctx* ctx,
                                  const float* frames_f32, int32_t n_views,
                                  int32_t height, int32_t width,
                                  querysplat_result* out);

/* Multi-view inference from image FILES — stb/libjpeg-decoded, then the
 * official center-square-crop + antialiased-resize preprocessing runs inside
 * (the same semantics as the official python ImageTransform). Every path
 * must be non-NULL and decodable (png/jpg/jpeg/webp/bmp). Same return and
 * result-store contract as run_frames. */
QS_CAPI int querysplat_run_image_paths(querysplat_ctx* ctx,
                                       const char* const* paths,
                                       int32_t n_paths, querysplat_result* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QUERYSPLAT_CAPI_H */
