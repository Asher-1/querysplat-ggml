# Model cards — querysplat-ggml GGUF inventory

Every GGUF pairs with exactly one role in the two-graph pipeline:

```
images ──► [encoder GGUF: VGGT-Omega-1B] ──► pose_enc + depth + cached-layer tokens
                                                     │
           [decoder GGUF: QuerySplat] ◄──────────────┘
                     │
                     ▼
        gaussians (524,288 = 8192 tokens × 64) ──► CPU rasterizer ──► renders
```

The two GGUFs of a run MUST come from the same quant tier (`--quant`), and the
encoder must match the decoder's quant tier. The shipped decoder
family is paper (the 8192 GGUF tiers were removed after paper won a
24/24-cell controlled comparison — see the decoder section).

**Tier selection (the one rule that matters):** **f32** is the parity-
reference tier and the default. **f16** is the compact/fast tier — F16-stored
weights are staged as F32 at load, so its renders are bit-identical to f32 on
every backend. **q8_0** is the smallest tier — reconstruction quality within
~0.04 dB of f32 (26.89 vs 26.85 cpp-vs-GT, S=4 means; measured table:
[alignment_matrix.md](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/alignment_matrix.md)).
The f32-accuracy attention + TF32-off modes are default-ON (`--cuda-fast`
opts out).

## Download (Hugging Face)

| GGUF | Size | URL |
|---|---|---|
| `querysplat-paper-f32.gguf` | 4.47 GB | <https://huggingface.co/Asher-1/querysplat_vggto/resolve/main/querysplat-paper-f32.gguf> |
| `querysplat-paper-f16.gguf` | 2.30 GB | <https://huggingface.co/Asher-1/querysplat_vggto/resolve/main/querysplat-paper-f16.gguf> |
| `querysplat-paper-q8_0.gguf` | 1.31 GB | <https://huggingface.co/Asher-1/querysplat_vggto/resolve/main/querysplat-paper-q8_0.gguf> |
| `lpips-vgg-f32.gguf` (eval metric, f32-only) | 56 MB | <https://huggingface.co/Asher-1/querysplat_vggto/resolve/main/lpips-vgg-f32.gguf> |
| `vggt-omega-1b-512-f32.gguf` | 4.26 GB | <https://huggingface.co/Asher-1/map_anything_vggt/resolve/main/vggt-omega-1b-512-f32.gguf> |
| `vggt-omega-1b-512-f16.gguf` | 2.13 GB | <https://huggingface.co/Asher-1/map_anything_vggt/resolve/main/vggt-omega-1b-512-f16.gguf> |
| `vggt-omega-1b-512-q8_0.gguf` | 1.23 GB | <https://huggingface.co/Asher-1/map_anything_vggt/resolve/main/vggt-omega-1b-512-q8_0.gguf> |

Digest note (2026-10-20): the published `vggt-omega-1b-512-f16.gguf` is
byte-identical to the file behind every parity table here (sha256
`e6b074a8…`); the published f32/q8_0 encoder copies hash differently from
the locally measured ones (`e8f36f16…` / `e883d11d…` vs the `e4ea4708…` /
`ba2080ac…` pinned in AICore's asset_digests.h) — a converter-revision skew
that has not been bisected. The converter commands in the encoder section
are the authoritative reproduction path for the measured numbers; the
decoder GGUFs are published from exactly the files measured here
(`e59558d4…` / `e26b2375…` / `aaeaa570…`, sha256 printed by
`scripts/upload_to_hf.py`).

---

## Encoder: VGGT-Omega-1B-512 (frozen, shared by every decoder)

| | |
|---|---|
| Source | `facebook/VGGT-Omega` `vggt_omega_1b_512.pt` (gated; sha256 in repo `SHA256SUMS`) |
| Published GGUFs | [Asher-1/map_anything_vggt](https://huggingface.co/Asher-1/map_anything_vggt) — `vggt-omega-1b-512-{f32,f16,q8_0}.gguf` (digest note above) |
| Converter | `cpp_ggml/third_party/map-anything-ggml/cpp_ggml/scripts/convert_vggt_omega_to_gguf.py` |
| Outputs | pose_enc (S,9) → cameras; depth/conf; cached-layer tokens {layers 4, 11, 17, 23} feeding the QuerySplat decoder |
| Use when | always — it is the only encoder the QuerySplat decoders accept |

### `vggt-omega-1b-512-f32.gguf` (~4.26 GB) — parity tier, default
Full-torch-parity encoder. Required (together with the f32 decoder) for the
46-51 dB cpp-vs-py render agreement documented in the
[alignment matrix](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/alignment_matrix.md).
Produce: `python3 cpp_ggml/third_party/map-anything-ggml/cpp_ggml/scripts/convert_vggt_omega_to_gguf.py checkpoints/vggt_omega_1b_512.pt cpp_ggml/models/gguf/vggt-omega-1b-512-f32.gguf --outtype f32`.

### `vggt-omega-1b-512-f16.gguf` (~2.13 GB) — fast tier
Half the size and faster load; identical graph. Its f16 im2col/weight noise
costs ~0.1-0.4 dB of render agreement on most scenes (documented in
[MODEL_CARDS tier table](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/models/MODEL_CARDS.md)).

---

## Decoder: QuerySplat-paper (shipped decoder family)

| | |
|---|---|
| Source | `inspatio/querysplat` `querysplat_vggto_1B_512_paper.safetensors` (521 tensors, sh_degree=0 — RGB-only color head) + `checkpoints/querysplat_vggto_1B_512_paper.yaml` |
| Published GGUFs | [Asher-1/querysplat_vggto](https://huggingface.co/Asher-1/querysplat_vggto) — decoder tiers + `lpips-vgg-f32.gguf` (publisher: `scripts/upload_to_hf.py`; model card: `models/hf/README.md`) |
| Converter | `cpp_ggml/scripts/convert_querysplat_to_gguf.py` |
| Outputs | 524,288 gaussians (position/opacity/scale/rot + RGB color) |
| Use when | always — the paper checkpoint is the shipped decoder family |

**Why paper is the only decoder family:** in the 24-cell controlled
comparison (2 backends × 4 quant tiers × 3 reference scenes, same encoder,
same inputs —
[alignment_matrix_paper.md](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/alignment_matrix_paper.md)),
the paper checkpoint beat the 8192 checkpoint in **24/24 cells by +0.71 to
+1.91 dB** cpp-vs-GT. The 8192 GGUF tiers were therefore removed from the
inventory on 2026-10-17 (the 8192 safetensors source remains in
`cpp_ggml/models/pytorch/querysplat/` for reproduction).

### `querysplat-paper-f32.gguf` (~4.16 GB) — parity tier, default
Full-torch-parity decoder: 48.4-49.6 dB cpp-vs-py on the three reference
scenes across CUDA/Vulkan/CPU
([alignment_matrix_paper.md](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/alignment_matrix_paper.md)).

### `querysplat-paper-f16.gguf` (~2.14 GB) — compact tier
F16-stored weights are staged as F32 at load (bit-identical to the f32 tier
on every backend — map-anything-ggml a759947). Half the download, same
parity.

### `querysplat-paper-q8_0.gguf` (~1.23 GB) — mid-quality tier
8-bit block quant: cpp-vs-GT within ~0.1 dB of f32 on Vulkan (26.90 vs
26.86), ~1 dB below on CUDA. The re-measured mid tier (the pre-accuracy
measurements had hidden its quality under the CUDA f16-path noise).

---

## Measured performance (RTX 4090 24 GB + i9-14900K, S=4, 512×512)

Everything below is measured by this repo's own scripts —
`benchmarks/alignment_matrix.py` produces the timings and scores,
`benchmarks/plot_benchmarks.py` draws the charts — and referenced by
absolute GitHub URL so the cards stay valid from any clone depth. Full
protocol, per-seam tables and reproduce commands:
[querysplat_report.md](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/querysplat_report.md).

### Speed — latency matrix, C++ backends × tiers vs official pytorch CUDA

![latency matrix](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/latency_matrix.png)

End-to-end = encoder + decoder + render with model load excluded — the one
span where both sides measure identical stage boundaries (the official
chain re-encodes inside `forward_gaussians`, so its per-stage split is not
comparable; e2e is). What the figure says:

- GPU backends land within **1.00-1.13×** of official torch e2e. The
  residual is the encoder stage (0.78-0.89 s vs 0.66 s — the exact-F32
  attention modes cost what the cpp single-encoder-pass saves) plus the
  host-side rasterizer (0.38-0.41 s vs 0.30 s; one CPU rasterizer serves
  every backend, the CUDA port is deferred).
- q8_0 runs at f32 speed on every backend: weights dequantize to F32 GEMMs
  inside the graph, so the tier only changes model-load time.
- CPU is the ~29× fallback floor with the same 0.4 s render.

| Row | enc+dec s | render s | e2e s | vs py |
|---|---|---|---|---|
| official py CUDA f32 | 2.59 | 0.30 | 2.89 | 1.00× |
| cpp Vulkan q8_0 | 2.50 | 0.39 | 2.90 | 1.00× |
| cpp Vulkan f16 | 2.66 | 0.40 | 3.06 | 1.06× |
| cpp Vulkan f32 | 2.67 | 0.41 | 3.08 | 1.06× |
| cpp CUDA f16 | 2.77 | 0.38 | 3.15 | 1.09× |
| cpp CUDA f32 | 2.81 | 0.40 | 3.20 | 1.11× |
| cpp CUDA q8_0 | 2.85 | 0.41 | 3.25 | 1.13× |
| cpp CPU f16 | 82.7 | 0.40 | 83.1 | 28.8× |
| cpp CPU f32 | 84.9 | 0.40 | 85.3 | 29.5× |
| cpp CPU q8_0 | 85.5 | 0.40 | 85.9 | 29.7× |

### Fidelity — quant tiers

![cpp-vs-py alignment](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/alignment_cpp_vs_py_matrix.png)

![quant pareto](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/quant_pareto.png)

| Tier | encoder + decoder | mean cpp-vs-py | mean cpp-vs-GT | Verdict |
|---|---|---|---|---|
| f32 | 4.58 + 4.47 GB | 59.2 dB | 26.85 dB | parity reference |
| f16 | 2.29 + 2.30 GB | 45.6 dB | 26.84 dB | compact — bit-identical render to f32 |
| q8_0 | 1.32 + 1.31 GB | 34.5 dB | 26.89 dB | smallest — within 0.04 dB of f32 vs GT |

The alignment matrix reads as "how close is cpp to torch for this tier"
(fixed 40-53 dB axis, 50 dB 8-bit ceiling marked). cpp-vs-py is
ggml-fidelity (8-bit PNG ceiling ≈ 50 dB), cpp-vs-GT is reconstruction
quality — read the two together: every tier sits within 0.05 dB of the
official torch chain's 26.85 dB vs GT.

### Reconstruction — cpp vs official py vs ground truth

![case1 classroom reconstruction](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/recon_dl3dv_case1_cpp_py_gt.png)

![case0 retail-store reconstruction](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/recon_dl3dv_case0_cpp_py_gt.png)

![courtyard reconstruction](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/recon_courtyard_cpp_py_gt.png)

Panels per figure: input view | GT | official-py render | cpp render | fixed
0-8/255 `|cpp−py|` diff (dark = ≤8 LSB, so panels compare across figures).
case1 (classroom) is the former fog-collapse scene, now aligned at 48.5 dB
cpp-vs-py; case0 (retail store) 49.8 dB; courtyard (ETH3D, 6205×4135 JPEG
inputs exercising the downscale-AA and libjpeg paths) 48.5 dB. Beyond the
3 reference scenes, the 14-case official-split broad validation ran both
chains everywhere: 0 collapses, cpp-vs-py 51-66 dB per case, cpp-vs-GT
within 0.05 dB of py-vs-GT
([broad_validation.md](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/results/querysplat/broad_validation.md),
[charts/broad/](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad)).

### Remaining charts

- [e2e_latency_bar.png](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/e2e_latency_bar.png) —
  per-stage grouped bars (f32 tier + py) for the headline rows.
- [parity_scatter.png](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/parity_scatter.png) —
  per-seam parity-gate scatter (synthetic frames, 15 seams, torch spy vs
  cpp).
- [latency_quant_backend.png](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/latency_quant_backend.png) —
  enc+dec seconds per (quant × backend), written by the measurement pass
  itself.

## Metrics used across the charts

- **cpp-vs-py** — PSNR between the cpp render and the official torch chain's
  render of the same input views: the ggml-fidelity metric. The aligned floor
  is 46-51 dB (fp32 summation-order noise through 42 blocks; ±1-2 LSB flips on
  ~1% of pixels).
- **cpp-vs-GT / py-vs-GT** — reconstruction quality vs the ground-truth input
  frame (square-crop-512). A case1-type collapse reads as <15 dB here.
- **max|diff|, %>2 LSB** — printed per view by
  `benchmarks/compare_reconstruction.py`; the diff panel saturates at 8 LSB.
- **infer / render / total s** — per-stage wall clock from `--timing`
  (infer = encoder + decoder graphs; render = CPU rasterizer).
- **e2e** — encoder + decoder + render seconds, model load excluded: the one
  span both sides measure with identical stage boundaries (the latency
  matrix and the speed table above are e2e-based).
