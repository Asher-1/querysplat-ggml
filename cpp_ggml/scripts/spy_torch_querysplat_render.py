#!/usr/bin/env python3
"""gsplat reference rendering for the M2 rasterizer parity gate.

Consumes the querysplat-cli outputs (prefix_gaussians.bin, prefix_cameras.bin)
and renders every view with gsplat.rasterization exactly as
scripts/rendering/gs.py GaussianRenderer.render does, writing float bins:

  <prefix>_render_spy_images.bin  (S,3,H,W)   RGB+ED: [...,:3] RGB, [...,3:] depth
  <prefix>_render_spy_alphas.bin  (S,1,H,W)

Usage:
  python3 spy_torch_querysplat_render.py --bin cli_gaussians.bin \
      --cameras cli_cameras.bin --S 4 --H 512 --W 512 --sh-degree 1 \
      --out-prefix /tmp/qs_render_spy [--device cuda]
"""

import argparse

import numpy as np
import torch
from gsplat.rendering import rasterization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="*_gaussians.bin from querysplat-cli")
    ap.add_argument("--cameras", required=True, help="*_cameras.bin (w2c S*16 + intrinsics S*4)")
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--sh-degree", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    stride = 11 + 3 * (args.sh_degree + 1) ** 2
    ga = torch.from_numpy(
        np.fromfile(args.bin, dtype=np.float32).copy()).view(-1, stride).to(device)
    cam_raw = np.fromfile(args.cameras, dtype=np.float32)
    w2c = torch.from_numpy(cam_raw[: args.S * 16].copy()).view(args.S, 4, 4).to(device)
    k4 = torch.from_numpy(cam_raw[args.S * 16 : args.S * 16 + args.S * 4].copy()).view(args.S, 4).to(device)

    means = ga[:, 0:3].contiguous().float()
    opacity = ga[:, 3].contiguous().float()
    scales = ga[:, 4:7].contiguous().float()
    quats = ga[:, 7:11].contiguous().float()
    colors = ga[:, 11:].contiguous().float()
    if args.sh_degree > 0:
        colors = colors.reshape(*colors.shape[:-1], -1, 3)

    viewmats = w2c.float()  # the cameras.bin already stores w2c (= cam_view^T)
    Ks = torch.zeros(args.S, 3, 3, device=device)
    Ks[:, 0, 0] = k4[:, 0]
    Ks[:, 1, 1] = k4[:, 1]
    Ks[:, 0, 2] = k4[:, 2]
    Ks[:, 1, 2] = k4[:, 3]
    Ks[:, 2, 2] = 1.0
    backgrounds = torch.full((args.S, 3), 0.5, device=device)

    rendered, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacity,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=args.W,
        height=args.H,
        near_plane=0.025,
        far_plane=125.0,
        packed=False,
        backgrounds=backgrounds,
        render_mode="RGB+ED",
        sh_degree=args.sh_degree if args.sh_degree > 0 else None,
    )
    # rendered (S, H, W, 4): RGB + expected depth
    images = rendered[..., :3].permute(0, 3, 1, 2).contiguous()  # (S,3,H,W)
    alphas = alpha.permute(0, 3, 1, 2).contiguous()  # (S,1,H,W)
    images.cpu().numpy().tofile(args.out_prefix + "_images.bin")
    alphas.cpu().numpy().tofile(args.out_prefix + "_alphas.bin")
    print(f"wrote {args.out_prefix}_images.bin / _alphas.bin "
          f"(S={args.S} N={ga.shape[0]})")


if __name__ == "__main__":
    main()
