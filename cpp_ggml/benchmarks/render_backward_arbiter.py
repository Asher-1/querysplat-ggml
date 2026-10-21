#!/usr/bin/env python3
"""render_backward arbitration: cpp vs gsplat-autograd vs fp64 reference.

For a gaussian/cameras bin pair (S,H,W) runs all three backward paths on the
same deterministic loss seed and reports, per gaussian block, the max distance
of each fp32 implementation to the fp64 mathematical reference. fp32 noise is
expected to dominate in the tails; a *formula* bug shows as one
implementation being uniformly much farther from fp64 than the other.

Usage:
  python3 render_backward_arbiter.py --gaussians G.bin --cameras C.bin \
      --S 2 --H 128 --W 128 --bin build-cpu/bin/test_render_bwd \
      [--work /tmp/qs_arb --seed-seed 7 --n 200 --random-gaussians]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from render_backward_parity import lcg_stream, read_f32  # noqa: E402
from render_backward_fp64ref import render as fp64_render  # noqa: E402


def gsplat_grads(g_np, w2c, intr, S, H, W, din_np):
    from gsplat.rendering import rasterization

    N = g_np.shape[0]
    gt = torch.from_numpy(g_np)
    means = gt[:, 0:3].float().cuda()[None].requires_grad_(True)
    opac = gt[:, 3].float().cuda()[None].requires_grad_(True)
    scales = gt[:, 4:7].float().cuda()[None].requires_grad_(True)
    quats = gt[:, 7:11].float().cuda()[None].requires_grad_(True)
    colors = gt[:, 11:].float().cuda()[None].requires_grad_(True)
    vm = torch.from_numpy(np.ascontiguousarray(w2c)).float().cuda()[None]
    Ks = torch.zeros(1, S, 3, 3, device="cuda")
    for v in range(S):
        Ks[0, v, 0, 0] = float(intr[v, 0])
        Ks[0, v, 1, 1] = float(intr[v, 1])
        Ks[0, v, 0, 2] = float(intr[v, 2])
        Ks[0, v, 1, 2] = float(intr[v, 3])
        Ks[0, v, 2, 2] = 1.0
    bg = torch.full((1, S, 3), 0.5, device="cuda")
    din = torch.from_numpy(din_np.copy()).float().cuda()
    r, _, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac,
        colors=colors, viewmats=vm, Ks=Ks, width=W, height=H,
        near_plane=0.025, far_plane=125.0, packed=False, backgrounds=bg,
        render_mode="RGB+ED", sh_degree=None)
    py_i = r[0, ..., :3].permute(0, 3, 1, 2)
    (din * py_i).sum().backward()
    g_out = torch.cat([means.grad[0], opac.grad[0, :, None], scales.grad[0],
                       quats.grad[0], colors.grad[0]], dim=-1).cpu().numpy()
    return g_out, py_i.detach().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build-cpu/bin/test_render_bwd")
    ap.add_argument("--work", default="/tmp/qs_arb")
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--random-gaussians", action="store_true")
    ap.add_argument("--gaussians", default=None)
    ap.add_argument("--cameras", default=None)
    args = ap.parse_args()

    S, H, W = args.S, args.H, args.W
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.gaussians:
        g_np = read_f32(Path(args.gaussians)).reshape(-1, 14)
        cams = read_f32(Path(args.cameras))
    else:
        rng = np.random.default_rng(7)
        g_np = np.zeros((args.n, 14), dtype=np.float32)
        g_np[:, 0:3] = np.c_[rng.uniform(-0.5, 0.5, args.n),
                             rng.uniform(-0.5, 0.5, args.n),
                             rng.uniform(1.0, 4.0, args.n)]
        g_np[:, 3] = rng.uniform(0.1, 0.9, args.n)
        g_np[:, 4:7] = np.exp(rng.uniform(np.log(0.02), np.log(0.08),
                                          (args.n, 3)))
        q = rng.normal(size=(args.n, 4))
        g_np[:, 7:11] = q / np.linalg.norm(q, axis=1, keepdims=True)
        g_np[:, 11:14] = rng.uniform(0, 1, (args.n, 3))
        cams = np.zeros(S * 20, dtype=np.float32)
        w2c0 = np.eye(4, dtype=np.float32)
        cams[: S * 16] = np.tile(w2c0, (S, 1, 1)).ravel()
        cams[S * 16:] = np.tile(np.array([300.0, 300.0, W / 2, H / 2],
                                         dtype=np.float32), (S, 1)).ravel()
    stride = g_np.shape[1]
    N = g_np.shape[0]
    (work / "g.bin").write_bytes(g_np.tobytes())
    (work / "c.bin").write_bytes(cams.astype(np.float32).tobytes())

    subprocess.run([args.bin, str(work / "g.bin"), str(work / "c.bin"),
                    str(S), str(H), str(W), str(work / "a")], check=True,
                   capture_output=True)
    cpp = np.fromfile(work / "a_grads.bin", dtype=np.float32) \
        .reshape(N, stride)
    cpp_i = np.fromfile(work / "a_images.bin", dtype=np.float32) \
        .reshape(S, 3, H, W)
    din_np = np.fromfile(work / "a_dimages.bin", dtype=np.float32) \
        .reshape(S, 3, H, W)

    py, py_i = gsplat_grads(g_np, cams[: S * 16].reshape(S, 4, 4),
                            cams[S * 16: S * 16 + S * 4].reshape(S, 4),
                            S, H, W, din_np)

    w2c64 = torch.from_numpy(cams[: S * 16].reshape(S, 4, 4)
                             .astype(np.float64))
    intr64 = torch.from_numpy(cams[S * 16: S * 16 + S * 4].reshape(S, 4)
                              .astype(np.float64))
    g64 = torch.from_numpy(g_np.astype(np.float64)).requires_grad_(True)
    i64 = fp64_render(g64, w2c64, intr64, S, H, W, 0.5)
    (torch.from_numpy(din_np.astype(np.float64)) * i64).sum().backward()
    ref = g64.grad.numpy().astype(np.float32)
    i64n = i64.detach().numpy()

    print(f"N={N} S={S} H={H} W={W}")
    print(f"fwd |cpp-py|={np.abs(cpp_i - py_i).max():.3e} "
          f"|cpp-fp64|={np.abs(cpp_i - i64n).max():.3e} "
          f"|py-fp64|={np.abs(py_i - i64n).max():.3e}")
    for name, sl in [("d_pos", slice(0, 3)), ("d_op", slice(3, 4)),
                     ("d_scale", slice(4, 7)), ("d_rot", slice(7, 11)),
                     ("d_color", slice(11, 14))]:
        r64 = ref[:, sl].astype(np.float64)
        dc = np.abs(cpp[:, sl].astype(np.float64) - r64)
        dp = np.abs(py[:, sl].astype(np.float64) - r64)
        rel = np.abs(r64) > 1e-3 * (np.abs(r64).max() + 1e-30)
        # median relative-to-max error over significant entries: robust to
        # the fp32 tails
        med_c = np.median(dc[rel]) if rel.any() else 0.0
        med_p = np.median(dp[rel]) if rel.any() else 0.0
        print(f"{name:>8s}  |cpp-fp64| max={dc.max():.3e} med={med_c:.3e}   "
              f"|py-fp64| max={dp.max():.3e} med={med_p:.3e}   "
              f"|fp64|max={np.abs(r64).max():.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
