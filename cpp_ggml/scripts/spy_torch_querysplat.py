#!/usr/bin/env python3
"""Torch reference dumps for the QuerySplat parity gates.

Hooks the official Python graph (scripts/models/...) at the same seams the
C++ querysplat-cli dumps via QS_DUMP_STAGE, and writes raw f32 bins.

Layout contract (memory identity, ne[0] fastest on the ggml side):
  torch (1, S*Nagg, 2C)          == cpp_tokens.bin           {2C, Nv, 1}
  torch (1, 16, Nv, 128)  keys   == cpp_geom_kv_k.bin        {128, Nv, 16, 1}
  torch (1, 16, Nr, 128)  pkeys  == cpp_post_kv_k.bin        {128, Nr, 16, 1}
  torch (1, 8192, C)  geom toks  == cpp_geom_tokens.bin      {C, 8192, 1}
  torch (1, N, 640)   deconv     == cpp_deconv_raw.bin       {640, N, 1}
  torch (1, N, D*64)  delta      == cpp_color_raw.bin        {D*64, N, 1}
  torch (1, N, 11+cd) gaussians  == cpp_gaussians.bin        (N, 11+cd)
  torch (S, 9) pose_enc          == cpp_pose_enc.bin         {9, S}
  torch (1,S,6,H,W) plucker      == cpp_plucker.bin          {W, H, 6, S}

Usage:
  python3 spy_torch_querysplat.py --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
      --checkpoint checkpoints/querysplat_vggto_1B_512_8192.safetensors \
      --vggt-checkpoint checkpoints/vggt_omega_1b_512.pt \
      --bin frames.f32.bin --S 4 --out-dir /tmp/qs_spy [--device cuda]
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# The spy runs the FORWARD path only. fused_ssim (TTO losses) and gsplat
# (rendering) are imported by the model modules at import time but never
# called here; stub them so the spy can run in torch-only environments.
_fused_stub = types.ModuleType("fused_ssim_cuda")
_fused_stub.fusedssim = None
_fused_stub.fusedssim_backward = None
sys.modules.setdefault("fused_ssim_cuda", _fused_stub)
_gsplat_pkg = types.ModuleType("gsplat")
_gsplat_render = types.ModuleType("gsplat.rendering")
_gsplat_render.rasterization = None
_gsplat_pkg.rendering = _gsplat_render
sys.modules.setdefault("gsplat", _gsplat_pkg)
sys.modules.setdefault("gsplat.rendering", _gsplat_render)

from safetensors.torch import load_file  # noqa: E402

from scripts.models import QuerySplat  # noqa: E402
from scripts.models.vggt_encoder import VGGTEncoder  # noqa: E402
from scripts.options import load_options_yaml  # noqa: E402

VENDOR_PREFIXES = (
    "vggt_encoder.aggregator.",
    "vggt_encoder.camera_head.",
    "vggt_encoder.depth_head.",
)


def save(out_dir: str, name: str, t: torch.Tensor) -> None:
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    arr.tofile(os.path.join(out_dir, name + ".bin"))
    print(f"  {name}: {tuple(arr.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vggt-checkpoint", required=True)
    ap.add_argument("--bin", required=True, help="frames.f32.bin from make_test_frames.py")
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timing", action="store_true",
                    help="write timing.json (same stage boundaries as "
                         "querysplat-cli --timing)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    options = load_options_yaml(args.config)
    options = options.evolve(num_input_views=args.S)

    device = torch.device(args.device)
    model = QuerySplat(options).to(device)
    checkpoint = load_file(args.checkpoint, device="cpu")
    assert len(checkpoint) == 521, len(checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    del checkpoint

    frames = torch.from_numpy(
        np.fromfile(args.bin, dtype=np.float32).copy()
    ).view(args.S, 3, options.img_size[0], options.img_size[1])
    normalize = torch.nn.functional
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = frames.to(device)
    normalized = (images - mean) / std

    # ---- model input (mirrors infer.py build_model_input) ----
    from scripts.models import ModelInput, ModelInputDecoder, ModelInputEncoder

    model_input = ModelInput(
        encoder=ModelInputEncoder(
            images_rgb=normalized.unsqueeze(0),
            images_rgb_unnormalized=images.unsqueeze(0),
        ),
        decoder=ModelInputDecoder(cam_view=torch.empty(0), intrinsics=torch.empty(0)),
    )

    dumps = {}
    stage = {"encode_end": None}  # --timing boundary markers

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---- hook the encoder internals ----
    orig_tokens_from_layers = VGGTEncoder._tokens_from_layers

    def tokens_from_layers(self, output_by_layer, patch_start, batch_size):
        out = orig_tokens_from_layers(self, output_by_layer, patch_start, batch_size)
        dumps["cpp_tokens"] = out.tokens  # (1, S*Nagg, 2C) post output_norm
        save_layer_weights = out.layer_weights.detach().float().cpu().numpy()
        np.savetxt(os.path.join(args.out_dir, "layer_weights.txt"),
                   save_layer_weights.reshape(-1), fmt="%.6f")
        return out

    VGGTEncoder._tokens_from_layers = tokens_from_layers

    orig_encode_kv = type(model.enc_dec_backbone)._encode_to_kv

    model.enc_dec_backbone.kv_proj.register_forward_hook(
        lambda m, inp, out: dumps.__setitem__("cpp_geom_kvproj", out))

    def encode_kv(self, features, run_encoder=False):
        keys, values = orig_encode_kv(self, features, run_encoder)
        # torch keys (1,16,N,128) C-order == the C++ ln-output dump layout
        # {128, N, 16, 1} under the reshape(d,c,b,a) rule — dump as-is
        dumps["cpp_geom_kv_k"] = keys
        dumps["cpp_geom_kv_v"] = values
        return keys, values

    type(model.enc_dec_backbone)._encode_to_kv = encode_kv

    orig_orig_kv = type(model)._original_tokens_to_kv

    def orig_tokens_to_kv(self, tokens):
        dumps["cpp_rgb_tokens"] = tokens  # pre original_encoder_norm
        keys, values = orig_orig_kv(self, tokens)
        dumps["cpp_post_kv_k"] = keys
        dumps["cpp_post_kv_v"] = values
        return keys, values

    type(model)._original_tokens_to_kv = orig_tokens_to_kv

    orig_decode_cameras = VGGTEncoder._decode_cameras

    def decode_cameras(self, final_layer, image_hw):
        _sync()
        stage["encode_end"] = time.perf_counter()
        cam_view, intrinsics, pose_enc = orig_decode_cameras(self, final_layer, image_hw)
        dumps["cpp_pose_enc"] = pose_enc.reshape(args.S, 9)
        dumps["cpp_cam_view"] = cam_view.reshape(-1, 4, 4)  # (S, 4, 4)
        # QuerySplat._decode_cameras already stacks [fx, fy, cx, cy] per view
        dumps["cpp_intrinsics"] = intrinsics.reshape(args.S, 4)
        return cam_view, intrinsics, pose_enc

    VGGTEncoder._decode_cameras = decode_cameras

    orig_plucker = type(model)._plucker_from_vggt_cameras

    def plucker_from_vggt_cameras(self, cam_view, intrinsics, dtype=None,
                                  device=None):
        plucker = orig_plucker(self, cam_view, intrinsics, dtype, device)
        dumps["cpp_plucker"] = plucker[0]  # (S, 6, H, W)
        return plucker

    type(model)._plucker_from_vggt_cameras = plucker_from_vggt_cameras

    orig_decoder_tokens = type(model)._forward_decoder_tokens

    def forward_decoder_tokens(self, latent):
        tokens = orig_decoder_tokens(self, latent)
        dumps["cpp_geom_tokens"] = tokens  # (1, 8192, C)
        return tokens

    type(model)._forward_decoder_tokens = forward_decoder_tokens

    model.non_rgb_activation_head.deconv.register_forward_hook(
        lambda m, inp, out: dumps.__setitem__("cpp_deconv_raw", out))
    model.patch_embed.proj.register_forward_hook(
        lambda m, inp, out: dumps.__setitem__("cpp_pe_conv_pre", out))
    model.patch_plucker_embed.proj.register_forward_hook(
        lambda m, inp, out: dumps.__setitem__("cpp_ppe_conv_pre", out))
    model.post_rgb_delta_head.register_forward_hook(
        lambda m, inp, out: dumps.__setitem__("cpp_color_raw", out))

    # ---- full forward (gaussians) ----
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        gaussians = model.forward_gaussians(model_input)
    _sync()
    t1 = time.perf_counter()
    dumps["cpp_gaussians"] = gaussians[0]  # (N, 11+color_dim)

    if args.timing:
        enc_s = (stage["encode_end"] - t0) if stage["encode_end"] else float("nan")
        timing = {
            "arch": "querysplat-official-py",
            "device": str(device),
            "num_input_views": args.S,
            # boundary = end of the camera-head decode inside forward_gaussians
            "camera_and_depth_seconds": enc_s,
            "image_to_3dgs_seconds": t1 - (stage["encode_end"] or t0),
            "total_seconds": t1 - t0,
        }
        with open(os.path.join(args.out_dir, "timing.json"), "w",
                  encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
            f.write("\n")
        print(json.dumps(timing))

    # torch proj out is (B*V, C, Hp, Wp); flatten(2).transpose(1,2) gives the
    # token-major (P*V?? no — per-view (P, C)) layout the C++ dumps pre-LN
    for key in ("cpp_pe_conv_pre", "cpp_ppe_conv_pre"):
        t = dumps[key]
        if len(t.shape) == 4:
            dumps[key] = t.flatten(2).transpose(1, 2).contiguous()
    print(f"saving {len(dumps)} seams to {args.out_dir}")
    for name, tensor in dumps.items():
        save(args.out_dir, name, tensor)
    print("done")


if __name__ == "__main__":
    main()
