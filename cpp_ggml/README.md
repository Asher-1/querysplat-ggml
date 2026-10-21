# cpp_ggml — QuerySplat in pure C++ (ggml, single-process dual-graph)

Pure C++ ggml inference for QuerySplat (image sequence -> 3D Gaussians ->
rendered views), with zero PyTorch dependency at runtime. The frozen
VGGT-Omega encoder runs on the embedded [map-anything-ggml](third_party/map-anything-ggml)
runtime (`RuntimeOptions::expose_tokens`); the QuerySplat decoder is this
project's own ggml graph; the activation heads run on the host; rendering
uses a forward-only CPU rasterizer aligned with gsplat semantics.

## Chain (mirrors `scripts/infer.py`, TTO excluded)

```
images -> center-crop/resize 512 -> frames f32 [0,1]
      -> VGGT-Omega encoder (mapggml, expose_tokens)
           pose_enc (S,9), depth, depth_conf,
           cached tokens: layers {4,11,17,23}, per layer {2048, Nagg, S}
      -> host: pose_enc -> c2w/intrinsics -> Plucker rays -> ImageNet-normalized rgb
      -> decoder graph (this project):
           patch_embed x2 (conv16x16 + LN) -> rgb_tokens
           layer fusion: softmax(MLP(eye_token)) weighted sum of 4 layers
           concat [camera|register|patch] + output_norm
           kv projections (geometry + post_rgb)
           gs_tokens (8192x2048) + eye -> 12 DecoderBlocks -> geometry tokens
           deconv Linear 2048->640
      -> host activations: sign*expm1 | exp(clamp) | normalize | tanh | sigmoid
      -> gaussians (524288, 14) -> .gaussians.bin / gaussians.ply
      -> forward rasterizer (c2w->viewmats, K) -> rendered PNGs
```

## Build

```bash
cd cpp_ggml
git submodule update --init --recursive   # map-anything-ggml + nested ggml
cmake -B build-cpu && cmake --build build-cpu -j
# GPU: -DMAP_GGML_CUDA=ON (or VULKAN/METAL/HIPBLAS; one per build dir)
```

## Models

| File | Path | Source |
|---|---|---|
| VGGT-Omega 1B/512 encoder | `models/gguf/vggt-omega-1b-512-{f32,f16,q8_0}.gguf` | upstream `convert_vggt_omega_to_gguf.py` on `checkpoints/vggt_omega_1b_512.pt` |
| QuerySplat decoder (paper) | `models/gguf/querysplat-paper-{f32,f16,q8_0}.gguf` | `scripts/convert_querysplat_to_gguf.py --outtype {f32,f16,q8_0}` on the inspatio/querysplat paper safetensors |

Torch checkpoints land in `models/pytorch/`; per-model cards in
`models/MODEL_CARDS.md`. Published GGUFs: the decoder family +
`lpips-vgg-f32.gguf` at
[Asher-1/querysplat_vggto](https://huggingface.co/Asher-1/querysplat_vggto),
the encoder at
[Asher-1/map_anything_vggt](https://huggingface.co/Asher-1/map_anything_vggt)
(digest notes in the model cards).

## Parity methodology

Same discipline as map-anything-ggml (AGENTS.md): every seam is gated
spy(torch hooks) -> dump -> compare, thresholds calibrated to the measured
noise floor with >= 1.7x headroom. Layout contract: ggml ne {a,b,c,d} dumps
are numpy `reshape(d,c,b,a)` (ne[0] fastest); torch (B,nh,T,dh) and ggml
{dh,nh,T,B} are memory-identical.

- `scripts/spy_torch_querysplat.py` — hooks the official Python graph
  (fused tokens / latent KV / geometry tokens / gaussians); `--timing` gives
  same-boundary stage times.
- `scripts/cmp_querysplat.py` — per-seam comparison.
- `scripts/e2e_querysplat.sh` — one-click gate: convert -> build -> spy -> cli -> cmp.
- `test_smoke_tokens` — M0 acceptance: encoder token export contract.

The CLI takes either `--bin frames.f32.bin` (parity input) or `--images
<dir|files>` — stb-decoded, center-square-cropped and resized entirely in
C++ (the same preprocessing semantics as the official python
`ImageTransform`), so the full chain runs with zero python for real photos.
`--timing` writes `<out-prefix>_timing.json` with per-stage wall clocks
using the same stage names as the official `infer.py`.

## One-click

- repository root: `./run_querysplat.sh quick|python|cpp|compare --help`
  (preflight checks + copy-pasteable fixes for missing checkpoints/models).
- `bash benchmarks/run_benchmark.sh` — collects the parity gate + stage
  timings and renders the report figures (see `benchmarks/`).

## Quantization tiers (measured, S=4, paper, 3 scenes × CUDA/Vulkan/CPU)

| Tier | encoder + decoder | mean cpp-vs-py render | mean cpp-vs-GT (py 26.85) | Verdict |
|---|---|---|---|---|
| **f32** | 4.58 + 4.47 GB | **59.2 dB** | 26.85 | parity reference, default |
| f16 | 2.29 + 2.30 GB | 45.6 dB | 26.84 | half size; f16 file rounding only (weights staged as F32 at load) |
| q8_0 | 1.32 + 1.31 GB | 34.5 dB | 26.89 | smallest; same quality vs GT |

Source: [alignment_matrix.md](benchmarks/charts/querysplat/alignment_matrix.md)
(`benchmarks/alignment_matrix.py`). The three backends agree per tier to
±0.3 dB once the q8_0 activation path is exact: ggml's CPU vec_dot and the
CUDA MMQ kernels quantize the *activations* to q8_0/q8_1 for every q8_0
matmul (a further ~1 dB of cpp-vs-py), so in the default f32-accuracy mode
the decoder and the encoder dequantize q8_0 weights to F32 inside the graph
(`ggml_cast`, a per-call transient) and run an F32 GEMM — the Vulkan
shaders already did. `--cuda-fast` restores the MMQ path.

The 6-bit K-quant tier was removed (converter, GGUFs, docs): against q8_0
its only edge was a 0.27 GB smaller decoder (~0.5 s faster load), while it
lost 0.7-1.9 dB vs GT in every measured scene × backend cell. The q8_0
converter path uses
gguf-py's reference `quants.quantize` (verified byte-identical to ggml's C
quantizer on all 112 matrices).

## Runtime dependency boundary (pure C++)

The inference chain (`querysplat-cli`) is **pure C++**: libmapggml + ggml +
stb only — preprocessing, camera/Plücker math, the decoder graph, activation
heads, PLY export and the rasterizer are all in this tree. No Python is
loaded, spawned or linked at runtime; this is the surface ACloudViewer
integrates against (the same boundary as map-anything-ggml's C API).

Python appears ONLY in developer-facing, run-once tooling, exactly like
map-anything-ggml:

- `scripts/convert_querysplat_to_gguf.py` — one-time checkpoint → GGUF
  conversion (torch/safetensors/gguf).
- `scripts/spy_torch_querysplat.py` / `cmp_querysplat.py` /
  `e2e_querysplat.sh` — parity gates against the official PyTorch graph
  (development-time acceptance, not shipped).

All runtime switches are explicit struct fields (`mapggml::BackendOptions`,
`RuntimeOptions`, `DecoderRuntimeOptions`, the C-API options setters, CLI
argv) — no environment variables are read or written by the querysplat
chain's public interfaces (the vendored ggml transports the resolved
accuracy-profile values to its device-init reads internally; that mechanism
is not part of any interface).

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | submodule pin + upstream expose_tokens extension + skeleton + smoke | done |
| M1 | GGUF converter + decoder graph + host activations + PLY + gate | **done — GATE PASS** (S=2 & S=4, f32 & f16 decoder tiers, 15/15 seams, gaussians corr 0.998) |
| M2 | forward CPU rasterizer + gsplat parity | done — 3 math bugs fixed (Wᵀ transpose, SH +0.5/clamp, Jacobian xz cross-term; single/dual-gaussian bit-exact) + save_png layout bug (tiled-grayscale PNGs; f32 bins were always right); residual audited as uniform fp summation-order noise (top-err pixels at tile edges at base rate 0.446 vs 0.438), not a tile bug; real-scene courtyard recon 25.1 dB vs GT |
| M2.5 | rasterizer gsplat-semantics fix (case1 fog root cause) | done — two divergences from gsplat found and fixed (2026-10-17): ① `persp_proj` lacked the Inria/gsplat **frustum-margin clamp** of the Jacobian's z cross-terms (tx/ty clamped to ±((W−cx)/fx + 0.3·tan_fov)); without it a near-plane gaussian whose center projects outside the frustum gets an unbounded cov2d that covers the screen at ~flat alpha — the "fog" (render alpha ≡ 1.0, depth ≡ near plane); ② per-gaussian alpha floor was 1e-4 instead of gsplat's **1/255** (plus opacity-aware bbox extend and the 0.999 alpha clamp). Proof: the OFFICIAL torch gaussians rendered 11.93 dB through the old cpp rasterizer vs 23.88 dB through gsplat; after the fix the cpp rasterizer reproduces gsplat's per-view PSNR to 0.01 dB on the same bytes |
| M2.6 | full torch alignment: model-noise + preprocessing roots | done — after M2.5 the residual cpp-vs-py gap (29-36 dB) traced to three independent roots, all fixed (2026-10-17): ① **CUDA numerics**: ggml-cuda converts F32 K/V to F16 inside the flash kernels AND runs cuBLAS GEMMs in TF32 (10-bit mantissa; torch defaults allow_tf32=false) — fixed by `RuntimeOptions::f32_accuracy` (exact-F32 softmax/matmul attention, CUDA-only) + `BackendOptions::cuda_f32_accuracy` (the former `GGML_CUDA_DISABLE_TF32` env knob, now an explicit init option), both ON by default, `--cuda-fast` opts out; ② **encoder patch_embed f16 im2col**: ggml_conv_2d hardwires F16 im2col — f32-im2col variant in the submodule's vggt_graph (same fix the decoder already had); ③ **preprocessing**: torchvision `Resize` on tensors is antialiased on downscale (triangle half-width = scale) — implemented in `--images` (verified 2.4e-7 vs torch on the 4135→512 geometry), and JPEG-content files (whatever their extension) now decode through the system libjpeg (PIL's decoder; stb's IDCT differs ±1-2/255) — optional `QS_USE_LIBJPEG` (found via find_package). End state (f32 GGUF tiers, CUDA/Vulkan/CPU): gaussians corr 0.9995+ vs torch, **cpp-vs-py render agreement 46-51 dB** (case0/case1/courtyard), per-view PSNR within 0.05 dB of official | ① `persp_proj` lacked the Inria/gsplat **frustum-margin clamp** of the Jacobian's z cross-terms (tx/ty clamped to ±((W−cx)/fx + 0.3·tan_fov)); without it a near-plane gaussian whose center projects outside the frustum gets an unbounded cov2d that covers the screen at ~flat alpha — the "fog" (render alpha ≡ 1.0, depth ≡ near plane); ② per-gaussian alpha floor was 1e-4 instead of gsplat's **1/255** (plus opacity-aware bbox extend and the 0.999 alpha clamp). Proof: the OFFICIAL torch gaussians rendered 11.93 dB through the old cpp rasterizer vs 23.88 dB through gsplat; after the fix the cpp rasterizer reproduces gsplat's per-view PSNR to 0.01 dB on the same bytes |
| M3 | real-image matrix + launcher + benchmarks | done — `--images` C++ ingest, `--timing`, root `run_querysplat.sh`, `benchmarks/` (11-row latency matrix incl. official py CUDA, real-scene 4-column recon ×2, official DL3DV protocol 10-case table) all shipped |
| M3.5 | official feature-parity audit | done — both official checkpoints supported (`querysplat-8192-f16` sh1 + `querysplat-paper-f16` sh0, per-checkpoint threshold profiles in cmp); DL3DV-Evaluation confirmed as the official dataset (vs VGGT-Omega's MapAnything/ETH3D family); eval-pack zip assembled. Deferred (documented): TTO (needs differentiable rasterizer), VGGT depth/pointcloud & cameras JSON/NPZ side-exports, CUDA rasterizer, C API |
| case1-type inputs (RESOLVED — was a rasterizer bug, not model instability) | the earlier (2026-10-16) "model instability on case1-type inputs" finding was WRONG — the official chain is robust to every input variant (measured: original 480×270 frames 23.87 dB, its own ImageTransform output re-fed as 8-bit PNG 24.04, PIL-resized 512 23.91). The cpp fog (11-13 dB) was the rasterizer's missing persp_proj frustum clamp + too-low alpha floor (see M2.5); it fogged even the official gaussians. After the M2.5 fix, case1 end-to-end (f16 ggufs, CUDA) = 23.82 dB vs official 23.87; CPU f16 = 23.87. A separate, render-invisible numeric note: ggml's conv_2d hardwires f16 im2col, so the f16 encoder's cached-layer tokens carry ~1.3e-3 median noise (torch-simulated and confirmed pointwise); the layer-fusion near-cancellations on case1 amplify it ~200x at a few token dims (gaussians corr 0.926 CUDA / 0.980 CPU vs torch) — renders stay within 0.05 dB of official, and an f32-im2col encoder conv would close the remaining gap if ever needed |
| M4 | CUDA rasterizer, C API, TTO | C API **shipped** (`include/querysplat/capi.h` + `libquerysplat_capi.so`, AICore contract; mapggml C API gained the f32-accuracy option); TTO **shipped** (t1–t7, see `docs/tto_design.md`); CUDA rasterizer deferred |

## Known numeric differences (not bugs)

Same class as map-anything-ggml: fp summation order through 18 decoder
blocks. Measured floors (S=4, f16 encoder + f16 decoder, torch f32 CPU
reference): tokens 9.6e-4 mean, geom_tokens 1.5e-4, deconv 8.6e-4,
color_raw 5.1e-2, **gaussians 5.5e-3 mean / 10% rel-max, corr 0.998**.
The decoder weights contribute almost nothing on top of the encoder f16
tokens (f32 vs f16 decoder tiers differ by <5%). The gate thresholds in
`scripts/cmp_querysplat.py` carry the >= 1.7x headroom rule.

## Hard-won pitfalls (querysplat-specific)

1. **mul_mat output layout**: result ne = {out_channels, rows} — the out
   channel is the FASTEST axis. Host-side consumers of a Linear output must
   index `flat[n * out_features + j]`; a `j * rows + n` read silently
   scrambles the result (this once made the gaussian assembly read raw
   activations transposed — positions blew up to 1e27).
2. **torch stacks TWO LayerNorms** before the geometry kv_proj:
   VGGTEncoder.output_norm (inside _tokens_from_layers) AND
   EncDecBackbone.encoder_norm (inside _encode_to_kv). Skipping the second
   diverges the whole decoder chain (kv corr 0.97, gaussians corr 0.00).
3. **(S,3,H,W) channel planes are per-frame**: the (s,c) segment offset is
   `s*3*H*W + c*H*W`; a "3 contiguous planes" normalize silently
   mis-normalizes every frame past s=0 (S=1 hides the bug).
4. **ggml_conv_2d hardwires im2col to F16** — replaced with a manual
   im2col(F32)+mul_mat in `conv_2d_f32` (numerically irrelevant for the
   gate but kept for exactness).
5. **GGUF tensor names cap at 64 chars** — the verbatim state-dict keys run
   to 71; the converter maps them through `rename_key()` (same keys in
   querysplat_graph.cpp).
6. `layer_weights = softmax(MLP(eye_token))` is constant per checkpoint —
   host-folded at init, the graph branch disappears. The `elm.*` weights
   stay f32 in EVERY quant tier: the host fold reads raw f32/f16 only, and
   at ~2M params quantizing them buys nothing.
7. **save_png must interleave the plane-layout render output** — reading
   the (3,H,W) planes as interleaved RGB produced "3x3-tiled grayscale"
   PNGs while the f32 render bins stayed correct: an image-writer bug that
   looks exactly like a renderer bug. Always audit bins, not PNGs.
