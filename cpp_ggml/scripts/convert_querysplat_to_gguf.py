#!/usr/bin/env python3
"""Convert the QuerySplat checkpoint (.safetensors) to GGUF for querysplat-cli.

Usage:
  python3 convert_querysplat_to_gguf.py \
      checkpoints/querysplat_vggto_1B_512_paper.safetensors \
      checkpoints/querysplat_vggto_1B_512_paper.yaml \
      models/gguf/querysplat-paper-f32.gguf --outtype f32   # | f16 | q8_0
  # add --dry-run to enumerate every tensor without writing

The checkpoint holds ONLY the QuerySplat decoder parameters (521 tensors);
the frozen VGGT-Omega encoder travels in its own GGUF produced by the
embedded map-anything-ggml converter (convert_vggt_omega_to_gguf.py).

Converter rules:
  - Only 2-D Linear weights are quantized (q8_0, via gguf-py's reference
    quantizer — byte-identical to ggml_quantize_chunk, verified on all 112
    matrices of the paper checkpoint), skipped when in_features % 32 != 0.
    Under f16, every *.weight with ndim >= 2 (Linear matrices and Conv2d
    kernels) becomes f16.
  - Everything non-Linear stays F32: biases, LayerNorms, LayerScale gammas,
    eye_token, gs_tokens, post_rgb_gs_tokens.

Tensor names are mapped to short keys (ggml caps tensor names at 64
chars; verbatim state-dict keys run up to 71). See rename_key below — the
C++ graph lookups in src/querysplat_graph.cpp use the same keys.

GGUF dimension convention: gguf-py reverses numpy shape when writing, so a
torch Linear weight {out, in} lands as ggml ne {in, out} and a Conv2d kernel
{OC, IC, KH, KW} as ne {KW, KH, IC, OC} — exactly what ggml ops expect.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from gguf import GGUFWriter, GGMLQuantizationType
from gguf import quants
from safetensors.torch import load_file

# Shipped tiers: f32 (parity reference), f16 (compact, bit-identical to f32
# at run time — staged as F32) and q8_0 (smallest). Every other quant was
# removed: measured against q8_0 under the f32-accuracy default it had no
# Pareto position (see benchmarks/charts/querysplat/alignment_matrix.md).
QTYPES = {
    "q8_0": GGMLQuantizationType.Q8_0,
}
Q8_0_BLOCK = 32

FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 8}

VENDOR_PREFIXES = (
    "vggt_encoder.aggregator.",
    "vggt_encoder.camera_head.",
    "vggt_encoder.depth_head.",
)


def _block_key(tail: str) -> str:
    """DecoderBlock-internal keys, verbatim -> short (see rename_key)."""
    t = tail
    t = t.replace("gs_self_attn.gs_self_attn.", "attn.")
    t = t.replace("gs_self_attn.norm.", "self_norm.")
    t = t.replace("gs_self_attn_scale.gamma", "ls_self")
    t = t.replace("gs_cross_attn.gs_token_norm.", "cross_norm.")
    t = t.replace("gs_cross_attn.q_norm.", "cross_qn.")
    t = t.replace("gs_cross_attn.q_proj.", "cross_qp.")
    t = t.replace("gs_cross_attn.out_proj.", "cross_op.")
    t = t.replace("gs_cross_attn_scale.gamma", "ls_cross")
    t = t.replace("mlp.norm.", "mlp_norm.")
    t = t.replace("mlp.mlp.fc1.", "fc1.")
    t = t.replace("mlp.mlp.fc2.", "fc2.")
    t = t.replace("mlp_scale.gamma", "ls_mlp")
    # self-attn qk-norms last (the cross rules above must consume their
    # "gs_cross_attn.q_norm." keys first — "attn.q_norm." is a substring of
    # them)
    t = t.replace("attn.q_norm.", "attn.qn.")
    t = t.replace("attn.k_norm.", "attn.kn.")
    return t


def rename_key(k: str) -> str:
    """Map torch parameter names to short GGUF keys (< 64 chars)."""
    if k == "gs_tokens":
        return "gst"
    if k == "post_rgb_gs_tokens":
        return "prgs"
    if k == "vggt_encoder.eye_token":
        return "eye"
    for prefix, short in (
        ("vggt_encoder.eye_layer_mlp.", "elm."),
        ("vggt_encoder.output_norm.", "onorm."),
        ("patch_embed.", "pe."),
        ("patch_plucker_embed.", "ppe."),
        ("enc_dec_backbone.encoder_norm.", "gen."),
        ("enc_dec_backbone.kv_proj.", "gkv."),
        ("enc_dec_backbone.k_proj_norm.", "gkn."),
        ("original_encoder_norm.", "oen."),
        ("original_kv_proj.", "okv."),
        ("original_k_proj_norm.", "okn."),
        ("non_rgb_activation_head.deconv.", "deconv."),
        ("post_rgb_delta_head.", "dh."),
    ):
        if k.startswith(prefix):
            return short + k[len(prefix):]
    for prefix, short in (
        ("enc_dec_backbone.decoder_blocks.", "dec"),
        ("post_rgb_decoder_blocks.", "pdec"),
    ):
        if k.startswith(prefix):
            idx, tail = k[len(prefix):].split(".", 1)
            return f"{short}{idx}.{_block_key(tail)}"
    raise KeyError(f"unmapped QuerySplat tensor: {k}")


def expected_key_inventory(cfg: dict) -> set:
    """The exact 521-key inventory implied by the config (strict parity)."""
    keys = set()
    keys.update([
        "patch_embed.proj.weight", "patch_embed.proj.bias",
        "patch_embed.norm.weight", "patch_embed.norm.bias",
        "patch_plucker_embed.proj.weight", "patch_plucker_embed.proj.bias",
        "patch_plucker_embed.norm.weight", "patch_plucker_embed.norm.bias",
    ])
    keys.update([
        "vggt_encoder.eye_token",
        "vggt_encoder.eye_layer_mlp.0.weight", "vggt_encoder.eye_layer_mlp.0.bias",
        "vggt_encoder.eye_layer_mlp.1.weight", "vggt_encoder.eye_layer_mlp.1.bias",
        "vggt_encoder.eye_layer_mlp.3.weight", "vggt_encoder.eye_layer_mlp.3.bias",
        "vggt_encoder.output_norm.weight", "vggt_encoder.output_norm.bias",
    ])
    keys.update([
        "enc_dec_backbone.encoder_norm.weight", "enc_dec_backbone.encoder_norm.bias",
        "enc_dec_backbone.kv_proj.weight", "enc_dec_backbone.kv_proj.bias",
        "enc_dec_backbone.k_proj_norm.weight", "enc_dec_backbone.k_proj_norm.bias",
    ])
    for prefix, depth in (
        ("enc_dec_backbone.decoder_blocks", int(cfg["dec_depth"])),
        ("post_rgb_decoder_blocks", int(cfg["vggt_original_encoder_post_decoder_depth"])),
    ):
        for i in range(depth):
            p = f"{prefix}.{i}."
            keys.update([
                p + "gs_self_attn.norm.weight", p + "gs_self_attn.norm.bias",
                p + "gs_self_attn.gs_self_attn.qkv.weight",
                p + "gs_self_attn.gs_self_attn.qkv.bias",
                p + "gs_self_attn.gs_self_attn.q_norm.weight",
                p + "gs_self_attn.gs_self_attn.q_norm.bias",
                p + "gs_self_attn.gs_self_attn.k_norm.weight",
                p + "gs_self_attn.gs_self_attn.k_norm.bias",
                p + "gs_self_attn.gs_self_attn.proj.weight",
                p + "gs_self_attn.gs_self_attn.proj.bias",
                p + "gs_self_attn_scale.gamma",
                p + "gs_cross_attn.gs_token_norm.weight",
                p + "gs_cross_attn.gs_token_norm.bias",
                p + "gs_cross_attn.q_norm.weight", p + "gs_cross_attn.q_norm.bias",
                p + "gs_cross_attn.q_proj.weight", p + "gs_cross_attn.q_proj.bias",
                p + "gs_cross_attn.out_proj.weight", p + "gs_cross_attn.out_proj.bias",
                p + "gs_cross_attn_scale.gamma",
                p + "mlp.norm.weight", p + "mlp.norm.bias",
                p + "mlp.mlp.fc1.weight", p + "mlp.mlp.fc1.bias",
                p + "mlp.mlp.fc2.weight", p + "mlp.mlp.fc2.bias",
                p + "mlp_scale.gamma",
            ])
    keys.update([
        "original_encoder_norm.weight", "original_encoder_norm.bias",
        "original_kv_proj.weight", "original_kv_proj.bias",
        "original_k_proj_norm.weight", "original_k_proj_norm.bias",
        "non_rgb_activation_head.deconv.weight", "non_rgb_activation_head.deconv.bias",
        "post_rgb_gs_tokens", "post_rgb_delta_head.weight", "post_rgb_delta_head.bias",
        "gs_tokens",
    ])
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="QuerySplat .safetensors")
    ap.add_argument("config", help="matching inference yaml (e.g. checkpoints/*.yaml)")
    ap.add_argument("output")
    ap.add_argument("--outtype", default="f16", choices=list(FILE_TYPE.keys()))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the mapping without writing the GGUF")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    print(f"loading {args.checkpoint} ...")
    sd = load_file(args.checkpoint, device="cpu")
    if len(sd) != 521:
        raise ValueError(f"Expected 521 QuerySplat tensors, got {len(sd)}")
    forbidden = [k for k in sd if k.startswith(VENDOR_PREFIXES)]
    if forbidden:
        raise ValueError(f"Checkpoint contains VGGT-Omega tensors: {forbidden[:3]}")

    mapped = {rename_key(k): v for k, v in sd.items()}
    if len(mapped) != len(sd):
        raise ValueError("rename_key collision")
    too_long = [k for k in mapped if len(k) >= 64]
    if too_long:
        raise ValueError(f"GGUF keys exceed the 64-char limit: {too_long[:3]}")

    expected = expected_key_inventory(cfg)
    missing = sorted(expected - set(sd))
    unexpected = sorted(set(sd) - expected)
    if missing or unexpected:
        raise ValueError(
            f"key mismatch vs config inventory: missing={missing[:5]}... "
            f"({len(missing)}), unexpected={unexpected[:5]}... ({len(unexpected)})")
    print(f"521 tensors, key inventory matches the config")

    derived = {
        "color_dim": 3 * (int(cfg["gaussian_sh_degree"]) + 1) ** 2,
        "post_output_dim": 3 * (int(cfg["gaussian_sh_degree"]) + 1) ** 2 + 1,
    }
    print(f"sh_degree={cfg['gaussian_sh_degree']} "
          f"post_output_dim={derived['post_output_dim']} "
          f"num_gs_tokens={cfg['num_gs_tokens']}")

    name = Path(args.checkpoint).stem
    if args.dry_run:
        for k in sorted(mapped):
            v = mapped[k]
            print(f"{k:44s} {tuple(v.shape)} {v.dtype}")
        return

    w = GGUFWriter(args.output, arch="querysplat")
    w.add_string("general.name", name)
    w.add_string("querysplat.dtype", args.outtype)
    w.add_string("querysplat.config_source", str(Path(args.config).resolve()))
    for field in (
        "img_size", "dec_depth", "enc_embed_dim", "enc_num_heads", "mlp_ratio",
        "bg_color", "gaussian_scale_cap", "gaussian_z_offset",
        "gaussian_sh_degree", "gaussian_sh_rest_scale",
        "gaussian_save_opacity_threshold", "num_gs_tokens",
        "num_gaussians_per_token", "token_dim", "gs_token_std",
        "vggt_intermediate_layers", "vggt_original_encoder_patch_size",
        "vggt_original_encoder_post_decoder_depth",
        "vggt_original_encoder_post_use_color_tokens",
        "vggt_original_encoder_post_detach_query", "znear", "zfar",
    ):
        value = cfg[field]
        if isinstance(value, list):
            if value and isinstance(value[0], int):
                w.add_array(f"querysplat.{field}", [int(v) for v in value])
            else:
                w.add_array(f"querysplat.{field}", [float(v) for v in value])
        elif isinstance(value, bool):
            w.add_bool(f"querysplat.{field}", value)
        elif isinstance(value, int):
            w.add_uint32(f"querysplat.{field}", value)
        elif isinstance(value, float):
            w.add_float32(f"querysplat.{field}", value)
        else:
            w.add_string(f"querysplat.{field}", str(value))

    n_q = 0
    for tname in sorted(mapped):
        t = mapped[tname].to(torch.float32).contiguous().numpy()
        is_linear_weight = tname.endswith(".weight") and t.ndim == 2
        # elm.* is the layer-weight MLP (softmax(MLP(eye_token))): the C++
        # side folds it on the host via host_f32(), which reads raw F32/F16
        # only. It is tiny (~2M params), so quantizing it buys nothing —
        # keep every tier's elm.* at f32.
        if tname.startswith("elm."):
            is_linear_weight = False
        if args.outtype in QTYPES and is_linear_weight:
            qt = QTYPES[args.outtype]
            if t.shape[1] % Q8_0_BLOCK != 0:
                print(f"warning: {tname}: K={t.shape[1]} not divisible by "
                      f"{Q8_0_BLOCK}; keeping f32")
                w.add_tensor(tname, t)
                continue
            raw = quants.quantize(t, qt)
            w.add_tensor(tname, raw, raw_shape=raw.shape, raw_dtype=qt)
            n_q += 1
        elif args.outtype == "f16" and tname.endswith(".weight") and t.ndim >= 2:
            # f16 for *.weight tensors (2-D Linear matrices and 4-D conv
            # kernels). Biases, norms, gammas, eye_token, gs_tokens,
            # post_rgb_gs_tokens stay f32 so graph arithmetic never mixes
            # dtypes.
            w.add_tensor(tname, t.astype(np.float16))
        else:
            w.add_tensor(tname, t)

    print(f"quantized matrices: {n_q}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
