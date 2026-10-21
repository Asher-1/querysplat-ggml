#!/usr/bin/env python3
"""Decoder-only TTO parity gate: cpp KV seams -> official torch decode loop.

Runs the official QuerySplat test-time-optimization loop (decode -> render ->
_tto_losses -> backward -> torch.optim.Adam) on a memory seeded from the C++
chain's dumped KV seams (cpp_geom_kv_k/v, cpp_post_kv_k/v), with cameras from
the eval cameras.npz and ground truth from the case input frames. The encoder
is bypassed (monkeypatched) so its ~1B parameters never occupy GPU memory —
the official full evaluate_jsons --use-tto path needs ~21 GB, this needs ~14.

Dumps (out-dir):
  tto_step_losses.json   per-step L1/SSIM/LPIPS/total
  tto_final_losses.json  final decode losses (official last_tto_losses contract)
  final_gaussians.bin    raw f32 (N, 11+cd) after the last decode
  d_memory_step0/*.bin   first-step memory gradients (raw f32, cpp layout)

Usage:
  python3 benchmarks/tto_decoder_parity.py --kv-dir /tmp/ref/dump \
      --cameras outputs/eval_tto_probe/cpp/dl3dv_val_2views_medium/case_000114_cameras.npz \
      --gt frame_a.png frame_b.png \
      --config checkpoints/querysplat_vggto_1B_512_paper.yaml \
      --checkpoint checkpoints/querysplat_vggto_1B_512_paper.safetensors \
      --n-steps 3 --lr 5e-3 --lpips-weight 0.05 --out-dir /tmp/tto_py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.models import QuerySplat  # noqa: E402
from scripts.models.input_types import (  # noqa: E402
    EncoderLatent,
    ModelInputDecoder,
)
from scripts.options import load_options_yaml  # noqa: E402
from scripts.utils.data import ImageTransform  # noqa: E402


def load_kv_bin(path: Path) -> tuple[torch.Tensor, int]:
    """cpp f32 seam dump {D, N, H, 1} -> torch (1, H, N, D) on device."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % (128 * 16) != 0:
        raise ValueError(f"{path}: size {raw.size} not a multiple of 128*16")
    n_tokens = raw.size // (128 * 16)
    # ggml ne = [128, N, 16, 1] (ne[0] fastest) == numpy reshape(16, N, 128)
    arr = raw.reshape(16, n_tokens, 128)
    return torch.from_numpy(arr).unsqueeze(0).cuda(), n_tokens


def load_gt(paths: list[str], resolution: tuple[int, int], device: str) -> torch.Tensor:
    images = []
    for p in paths:
        with Image.open(p) as im:
            images.append(np.array(im.convert("RGB"), copy=True))
    ims = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
    ims, _, _, _ = ImageTransform(resolution, resolution, max_crop=True).preprocess_images(ims)
    return ims.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-dir", required=True, help="cpp --dump-stage dir with the 4 KV seam bins")
    ap.add_argument("--cameras", help="eval cameras.npz (input_cam_view/intrinsics)")
    ap.add_argument("--w2c-bin", help="cpp_w2c.bin (S*16 row-major) — alternative to --cameras")
    ap.add_argument("--intrinsics-bin", help="cpp_intrinsics.bin (S*4 fx fy cx cy)")
    ap.add_argument("--gt", nargs="+", required=True, help="input-view images (TTO ground truth)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lpips-weight", type=float, default=0.05)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    opt = load_options_yaml(args.config)
    model = QuerySplat(opt).to(device)
    from scripts.infer import load_querysplat_checkpoint

    load_querysplat_checkpoint(model, args.checkpoint)
    model.eval()

    # Freeze everything (official _frozen_parameters) and compute the folded
    # constants BEFORE evicting the encoder to CPU (frees ~4 GB VRAM).
    for p in model.parameters():
        p.requires_grad_(False)
    enc = model.vggt_encoder
    with torch.no_grad():
        eye = enc.eye_token.detach().clone()
        lw = torch.softmax(enc.eye_layer_mlp(eye.float()).to(eye.dtype), dim=-1)
    enc.to("cpu")
    torch.cuda.empty_cache()

    # cpp KV seams -> leaf memories
    kv_dir = Path(args.kv_dir)
    keys, n_geom = load_kv_bin(kv_dir / "cpp_geom_kv_k.bin")
    values, _ = load_kv_bin(kv_dir / "cpp_geom_kv_v.bin")
    pkeys, n_post = load_kv_bin(kv_dir / "cpp_post_kv_k.bin")
    pvalues, _ = load_kv_bin(kv_dir / "cpp_post_kv_v.bin")
    print(f"kv seams: geom N={n_geom} post N={n_post}")

    if args.cameras:
        cam = np.load(args.cameras)
        cam_view = torch.from_numpy(cam["input_cam_view"]).float().unsqueeze(0).to(device)
        intrinsics = torch.from_numpy(cam["intrinsics"]).float().unsqueeze(0).to(device)
    else:
        s = len(args.gt)
        w2c = np.fromfile(args.w2c_bin, dtype=np.float32).reshape(s, 4, 4)
        cam_view = torch.from_numpy(w2c).unsqueeze(0).to(device)  # (1,S,4,4) w2c
        intrinsics = torch.from_numpy(
            np.fromfile(args.intrinsics_bin, dtype=np.float32).reshape(s, 4)
        ).unsqueeze(0).to(device)
    gt = load_gt(args.gt, tuple(opt.img_size), device)
    print(f"cam_view {tuple(cam_view.shape)} gt {tuple(gt.shape)}")

    memory = [
        keys.detach().requires_grad_(True),
        values.detach().requires_grad_(True),
        pkeys.detach().requires_grad_(True),
        pvalues.detach().requires_grad_(True),
    ]

    def decode() -> torch.Tensor:
        latent = EncoderLatent(
            keys=memory[0], values=memory[1], post_rgb_keys=memory[2],
            post_rgb_values=memory[3], eye_token=eye, layer_weights=lw,
        )
        return model.forward_decoder(latent)

    dec_in = ModelInputDecoder(cam_view, intrinsics)
    optimizer = torch.optim.Adam(memory, lr=args.lr)
    step_losses = []
    with torch.set_grad_enabled(True):
        for index in range(args.n_steps):
            optimizer.zero_grad()
            gaussians = decode()
            rendered = model.render_gaussians(gaussians, dec_in)
            losses = model._tto_losses(rendered["images_pred"], gt, args.lpips_weight)
            losses["loss_total"].backward()
            if index == 0:
                d0 = out / "d_memory_step0"
                d0.mkdir(exist_ok=True)
                for name, m in zip(
                    ("d_geom_k", "d_geom_v", "d_post_k", "d_post_v"), memory
                ):
                    m.grad.detach().cpu().numpy().tofile(d0 / f"{name}.bin")
            rec = {
                "step": index + 1,
                "L1": float(losses["loss_l1"]),
                "SSIM": float(losses["loss_ssim"]),
                "LPIPS": None if losses["loss_lpips"] is None else float(losses["loss_lpips"]),
                "total": float(losses["loss_total"]),
            }
            step_losses.append(rec)
            lp = "nan" if rec["LPIPS"] is None else f"{rec['LPIPS']:.6f}"
            print(f"TTO [{rec['step']}/{args.n_steps}] L1={rec['L1']:.6f} "
                  f"SSIM={rec['SSIM']:.6f} LPIPS={lp} total={rec['total']:.6f}")
            optimizer.step()

    with torch.no_grad():
        final = decode()
        rendered = model.render_gaussians(final, dec_in)
        losses = model._tto_losses(rendered["images_pred"], gt, args.lpips_weight)
        final_losses = {
            "L1": float(losses["loss_l1"]),
            "SSIM": float(losses["loss_ssim"]),
            "LPIPS": None if losses["loss_lpips"] is None else float(losses["loss_lpips"]),
        }
        g = final.detach()[0].cpu().numpy().astype(np.float32)
        g.tofile(out / "final_gaussians.bin")

    (out / "tto_step_losses.json").write_text(json.dumps(step_losses, indent=2))
    (out / "tto_final_losses.json").write_text(json.dumps(final_losses, indent=2))
    print(f"final gaussians {g.shape} -> {out/'final_gaussians.bin'}")
    print(f"final losses: {final_losses}")


if __name__ == "__main__":
    main()
