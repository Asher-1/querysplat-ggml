#!/usr/bin/env python3
"""render_backward parity: cpp_ggml's CPU rasterizer backward vs gsplat autograd.

Runs the C++ harness (test_render_bwd) on a real gaussian/cameras bin pair,
recomputes the same loss seed's gradient with gsplat.rasterization +
torch.autograd on CUDA, and reports per-block max_abs / max_rel / corr.

Usage:
  python3 render_backward_parity.py --gaussians PREFIX_gaussians.bin \
      --cameras PREFIX_cameras.bin --S 2 --bin build-cpu/bin/test_render_bwd \
      [--work /tmp/qs_rbwd] [--H 512] [--W 512]
"""

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from gsplat.rendering import rasterization


def lcg_stream(n: int, seed: int = 0x9E3779B97F4A7C15) -> np.ndarray:
    # identical to tests/render_bwd_dump.cpp lcg_next(): float32 in [0,1)
    state = seed
    out = np.empty(n, dtype=np.float32)
    mask = (1 << 64) - 1
    for i in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        out[i] = np.float32((state >> 40) & 0xFFFFFF) / np.float32(0x1000000)
    return out


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def stats(name: str, cpp: np.ndarray, ref: np.ndarray) -> dict:
    d = np.abs(cpp.astype(np.float64) - ref.astype(np.float64))
    scale = np.max(np.abs(ref.astype(np.float64))) + 1e-30
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-8)
    rel = d / denom
    a = cpp.astype(np.float64).ravel()
    b = ref.astype(np.float64).ravel()
    if a.std() < 1e-30 or b.std() < 1e-30:
        corr = 1.0 if a.std() < 1e-30 and b.std() < 1e-30 else 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    row = {
        "name": name,
        "max_abs": float(d.max()),
        "max_rel": float(rel.max()),
        "corr": corr,
        "ref_scale": float(scale),
    }
    print(
        f"{name:>8s}  max_abs={row['max_abs']:.3e}  max_rel={row['max_rel']:.3e}"
        f"  corr={row['corr']:+.6f}  (|ref|max={row['ref_scale']:.3e})"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussians", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--bin", default="build-cpu/bin/test_render_bwd")
    ap.add_argument("--work", default="/tmp/qs_rbwd")
    ap.add_argument("--near", type=float, default=0.025)
    ap.add_argument("--far", type=float, default=125.0)
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    prefix = str(work / "rbwd")

    # ---- C++ side: forward render + backward dump ----
    subprocess.run(
        [args.bin, args.gaussians, args.cameras, str(args.S), str(args.H),
         str(args.W), prefix],
        check=True,
    )
    meta = prefix + "_meta.txt"
    n_str, s_str, h_str, w_str, cd_str = Path(meta).read_text().split()
    N, S, H, W, color_dim = (int(x) for x in (n_str, s_str, h_str, w_str, cd_str))
    stride = 11 + color_dim
    cpp_imgs = read_f32(Path(prefix + "_images.bin")).reshape(S, 3, H, W)
    dimages = read_f32(Path(prefix + "_dimages.bin")).reshape(S, 3, H, W)
    cpp_grads = torch.from_numpy(
        read_f32(Path(prefix + "_grads.bin")).reshape(N, stride).copy()
    )

    # ---- gsplat reference (CUDA autograd) ----
    g = torch.from_numpy(read_f32(Path(args.gaussians)).reshape(N, stride).copy())
    cams = read_f32(Path(args.cameras))
    w2c = torch.from_numpy(cams[: S * 16].reshape(S, 4, 4).copy())
    intr = torch.from_numpy(cams[S * 16 : S * 16 + S * 4].reshape(S, 4).copy())
    means = g[:, 0:3].contiguous().float().cuda()[None].requires_grad_(True)
    opac = g[:, 3].contiguous().float().cuda()[None].requires_grad_(True)
    scales = g[:, 4:7].contiguous().float().cuda()[None].requires_grad_(True)
    quats = g[:, 7:11].contiguous().float().cuda()[None].requires_grad_(True)
    colors = g[:, 11:].contiguous().float().cuda()[None].requires_grad_(True)
    viewmats = w2c.float().cuda()[None]
    Ks = torch.zeros(1, S, 3, 3, device="cuda")
    for v in range(S):
        Ks[0, v, 0, 0] = intr[v, 0]
        Ks[0, v, 1, 1] = intr[v, 1]
        Ks[0, v, 0, 2] = intr[v, 2]
        Ks[0, v, 1, 2] = intr[v, 3]
        Ks[0, v, 2, 2] = 1.0
    bg = torch.full((1, S, 3), 0.5, device="cuda")
    din = torch.from_numpy(dimages.copy()).float().cuda()

    rendered, alphas, meta_out = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opac,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        near_plane=args.near,
        far_plane=args.far,
        packed=False,
        backgrounds=bg,
        render_mode="RGB+ED",
        sh_degree=None,  # post-activation colors (gs.py: sh0 passes None)
    )
    # gsplat returns (1,S,H,W,4) RGB+ED; our images are (S,3,H,W)
    py_imgs = rendered[0, ..., :3].permute(0, 3, 1, 2)
    print(f"N={N} S={S} H={H} W={W} color_dim={color_dim}")
    stats("fwd_img", cpp_imgs, py_imgs.detach().cpu().numpy())

    loss = (din * py_imgs).sum()
    loss.backward()
    py_grads = torch.cat(
        [means.grad[0], opac.grad[0, :, None], scales.grad[0], quats.grad[0],
         colors.grad[0]],
        dim=-1,
    ).cpu()
    rows = [
        stats("d_pos", cpp_grads[:, 0:3].numpy(), py_grads[:, 0:3].numpy()),
        stats("d_op", cpp_grads[:, 3].numpy(), py_grads[:, 3].numpy()),
        stats("d_scale", cpp_grads[:, 4:7].numpy(), py_grads[:, 4:7].numpy()),
        stats("d_rot", cpp_grads[:, 7:11].numpy(), py_grads[:, 7:11].numpy()),
        stats("d_color", cpp_grads[:, 11:].numpy(), py_grads[:, 11:].numpy()),
    ]
    ok = all(r["corr"] > 0.99 for r in rows) and rows[0]["corr"] > 0.999
    print("RENDER_BACKWARD PARITY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
