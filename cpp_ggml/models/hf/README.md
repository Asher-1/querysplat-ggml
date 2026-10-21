---
license: apache-2.0
tags:
- gguf
- ggml
- querysplat
- 3d-gaussian-splatting
- novel-view-synthesis
- pose-free-reconstruction
- vggt
- pure-cpp
language:
- en
---

# QuerySplat GGUF (pure C++ ggml inference)

Pre-converted [QuerySplat](https://github.com/inspatio/querysplat) weights for
the pure C++ (ggml) inference chain in
[querysplat-ggml](https://github.com/Asher-1/querysplat-ggml): a feed-forward
pipeline that turns a sparse multi-view image set into 3D Gaussian splats with
**zero PyTorch at runtime**. The VGGT-Omega encoder GGUFs live in the
companion repo [Asher-1/map_anything_vggt](https://huggingface.co/Asher-1/map_anything_vggt)
(`vggt-omega-1b-512-{f32,f16,q8_0}.gguf` — the only encoder family the
QuerySplat decoders accept).

```
images ─► [encoder GGUF: VGGT-Omega-1B-512] ─► pose + depth + cached-layer tokens
                                                       │
             [decoder GGUF: QuerySplat-paper] ◄────────┘
                       │
              524,288 gaussians ─► rasterizer ─► renders
```

## Files

| File | Size | Role | sha256 |
|---|---|---|---|
| `querysplat-paper-f32.gguf` | 4.47 GB | decoder, parity tier (default) | `e59558d4cda480b743788a56eb5bb27d07fadb42be87e852ff9868002c07e188` |
| `querysplat-paper-f16.gguf` | 2.30 GB | decoder, compact tier (bit-identical render — weights staged as F32) | `e26b237544f0b803df7311131a8ecfa999fed79f4f6c56f06d0748dba6872905` |
| `querysplat-paper-q8_0.gguf` | 1.31 GB | decoder, smallest tier (cpp-vs-GT within 0.04 dB of f32) | `aaeaa570ce59edfa5cd725ea4ac33cc6a20e95966b1daa1d31922eef627f5f38` |
| `lpips-vgg-f32.gguf` | 56 MB | LPIPS-vgg metric weights for `querysplat-eval` (never quantized: the metric is the judge) | `0a2c7475ad0eed64cd40ec7b221b3dc639cc7ed142176554a30cd07aac1c7b7d` |

Encoder (from `Asher-1/map_anything_vggt`): `vggt-omega-1b-512-f32.gguf`
(4.26 GB, parity) · `vggt-omega-1b-512-f16.gguf` (2.13 GB) ·
`vggt-omega-1b-512-q8_0.gguf` (1.23 GB). The encoder and decoder of a run
must use the same quant tier.

## Usage

```bash
# one command (builds if needed, downloads nothing else)
./run_querysplat.sh cpp --input your/images --out outputs/scene

# or directly, official infer.py layout out of the box
cpp_ggml/build-*/bin/querysplat-cli \
    --encoder  vggt-omega-1b-512-f32.gguf \
    --decoder  querysplat-paper-f32.gguf \
    --input_folder your/images --output_dir outputs/scene
```

Backends: CUDA / Vulkan / CPU (compile flags `-DMAP_GGML_CUDA=ON` / `VULKAN`).
The f32-accuracy kernels (exact-F32 attention + cuBLAS, matching torch) are
**default-ON** and are explicit options in the API — no environment variables.

## Measured (RTX 4090 + i9-14900K, S=4, 512×512)

| Row | enc+dec s | render s | e2e s | vs official py CUDA |
|---|---|---|---|---|
| official py (torch f32) CUDA | 2.59 | 0.30 | 2.89 | 1.00× |
| cpp Vulkan q8_0 | 2.50 | 0.39 | 2.90 | 1.00× |
| cpp CUDA f32 | 2.81 | 0.40 | 3.20 | 1.11× |
| cpp CPU f32 | 84.9 | 0.40 | 85.3 | 29.5× |

![latency matrix](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/latency_matrix.png)

Quality (mean over 3 backends × 3 scenes): cpp-vs-py render agreement 59.2 dB
(f32) / 45.6 dB (f16) / 34.5 dB (q8_0); cpp-vs-GT within 0.05 dB of the
official torch chain on every tier (26.84–26.89 dB). Full methodology,
per-seam parity gates and the DL3DV protocol table:
[MODEL_CARDS.md](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/models/MODEL_CARDS.md)
·
[querysplat_report.md](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/querysplat_report.md).

![reconstruction](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/recon_dl3dv_case0_cpp_py_gt.png)

## Provenance

- Decoder: `inspatio/querysplat` `querysplat_vggto_1B_512_paper.safetensors`
  (sh0), converted with `cpp_ggml/scripts/convert_querysplat_to_gguf.py`
  (tensor names shortened to ggml's 64-char key limit; gguf-py shape reversal
  per ggml `ne` convention).
- Encoder: `facebook/VGGT-Omega` `vggt_omega_1b_512.pt`, converted with the
  map-anything-ggml `convert_vggt_omega_to_gguf.py`.
- LPIPS: torchvision VGG16 features + lpips v0.1 linear heads, converted with
  `cpp_ggml/scripts/convert_lpips_to_gguf.py`.

The QuerySplat decoder weights inherit the upstream
[inspatio/querysplat](https://github.com/inspatio/querysplat) release terms;
the conversion tooling and the inference chain are Apache-2.0.

## Citation

```bibtex
@article{querysplat2025,
  title  = {QuerySplat: Query-Based 3D Gaussian Splatting for Sparse-View Reconstruction},
  author = {inspatio},
  note   = {\url{https://github.com/inspatio/querysplat}}
}
```
