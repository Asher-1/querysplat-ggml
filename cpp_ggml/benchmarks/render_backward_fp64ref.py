#!/usr/bin/env python3
"""fp64 reference rasterizer (pure torch) for render_backward arbitration.

Independent re-implementation of the gsplat RGB+ED blend in float64 with
autograd: EWA projection (frustum clamp + 0.3 blur), global per-view depth
sort, front-to-back alpha blend with the min(0.999) clamp, the 1/255 alpha
floor, the next_T <= 1e-4 exclusive stop and the T_final background tail.
The global (per-view) depth sort differs from gsplat's per-tile radix order
only in fp rounding, not in the mathematical value.

Usage:
  python3 render_backward_fp64ref.py <gaussians.bin> <cameras.bin> S H W \
      <prefix_with_dimages_and_images>
Writes <prefix>_grads_fp64.bin (N,14) and prints per-block magnitudes plus
the forward distance to the cpp render.
"""

import sys

import numpy as np
import torch

FLOOR = 1.0 / 255.0
T_STOP = 1e-4
MAX_ALPHA = 0.999


def render(g: torch.Tensor, w2c: torch.Tensor, intr: torch.Tensor, S: int,
           H: int, W: int, bg: float):
    N = g.shape[0]
    means, opac = g[:, 0:3], g[:, 3]
    scales, quats, colors = g[:, 4:7], g[:, 7:11], g[:, 11:]
    imgs = []
    for v in range(S):
        R, t = w2c[v, :3, :3], w2c[v, :3, 3]
        mc = means @ R.T + t
        x, y, z = mc[:, 0], mc[:, 1], mc[:, 2]
        w_, x_, y_, z_ = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        inv = 1.0 / torch.sqrt(w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_)
        w_, x_, y_, z_ = w_ * inv, x_ * inv, y_ * inv, z_ * inv
        Rm = torch.stack([
            1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_),
            2 * (x_ * z_ + w_ * y_),
            2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_),
            2 * (y_ * z_ - w_ * x_),
            2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_),
            1 - 2 * (x_ * x_ + y_ * y_),
        ], dim=-1).reshape(N, 3, 3)
        M = Rm * scales[:, None, :]
        cov_c = R @ (M @ M.transpose(1, 2)) @ R.T
        fx, fy, cx, cy = intr[v]
        lxp = (W - cx) / fx + 0.15 * W / fx
        lxn = cx / fx + 0.15 * W / fx
        lyp = (H - cy) / fy + 0.15 * H / fy
        lyn = cy / fy + 0.15 * H / fy
        rz = 1.0 / z
        tx = z * torch.clamp(x * rz, -lxn, lxp)
        ty = z * torch.clamp(y * rz, -lyn, lyp)
        j00, j02 = fx * rz, -fx * tx * rz * rz
        j11, j12 = fy * rz, -fy * ty * rz * rz
        cxx, cxy, cxz = cov_c[:, 0, 0], cov_c[:, 0, 1], cov_c[:, 0, 2]
        cyy, cyz, czz = cov_c[:, 1, 1], cov_c[:, 1, 2], cov_c[:, 2, 2]
        a2 = j00 * j00 * cxx + 2 * j00 * j02 * cxz + j02 * j02 * czz + 0.3
        b2 = (j00 * j11 * cxy + j00 * j12 * cxz + j02 * j11 * cyz +
              j02 * j12 * czz)
        c2 = j11 * j11 * cyy + 2 * j11 * j12 * cyz + j12 * j12 * czz + 0.3
        det = a2 * c2 - b2 * b2
        mx, my, depth = x * fx * rz + cx, y * fy * rz + cy, z
        order = torch.argsort(depth)
        ca, cb, cc = (c2 / det)[order], (-b2 / det)[order], (a2 / det)[order]
        mxo, myo, opo = mx[order], my[order], opac[order]
        colo, z_ord = colors[order], z[order]
        xs = torch.arange(W, dtype=torch.float64)[None, :] + 0.5
        ys = torch.arange(H, dtype=torch.float64)[:, None] + 0.5
        dx = xs[None] - mxo[:, None, None]  # (N, H, W)
        dy = ys[None] - myo[:, None, None]
        sig = ca[:, None, None] * dx * dx + 2 * cb[:, None, None] * dx * dy \
            + cc[:, None, None] * dy * dy
        op_s = opo[:, None, None]
        sigma_max = 2.0 * torch.log(op_s / FLOOR) + 1e-3
        valid = (sig >= 0) & (sig <= sigma_max)
        a_raw = op_s * torch.exp(-0.5 * sig)
        valid = valid & (a_raw >= FLOOR)
        a = torch.clamp(a_raw, max=MAX_ALPHA)
        keep = torch.where(valid, 1.0 - a, torch.ones_like(a))
        nextT = torch.cumprod(keep, dim=0)          # T after splat k
        T_ext = torch.cat([torch.ones_like(nextT[:1]), nextT], dim=0)  # (N+1)
        over = (nextT <= T_STOP) & valid
        any_over = over.any(dim=0)
        first_done = torch.where(
            any_over, over.float().argmax(dim=0),
            torch.full((H, W), float(N), dtype=torch.float64)).long()
        kk = torch.arange(N, dtype=torch.int64)[:, None, None]
        active = valid & (kk < first_done[None])
        wk = torch.where(active, a * T_ext[:-1], torch.zeros_like(a))
        rend = torch.einsum("khw,kc->chw", wk, colo)
        T_end = T_ext[first_done, torch.arange(H)[:, None],
                      torch.arange(W)[None, :]]
        imgs.append(rend + T_end * bg)
    return torch.stack(imgs)


def main() -> int:
    gpath, cpath, S, H, W = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                             int(sys.argv[4]), int(sys.argv[5]))
    prefix = sys.argv[6]
    g_np = np.fromfile(gpath, dtype=np.float32).reshape(-1, 14)
    cams = np.fromfile(cpath, dtype=np.float32)
    w2c = torch.from_numpy(cams[: S * 16].reshape(S, 4, 4).astype(np.float64))
    intr = torch.from_numpy(cams[S * 16: S * 16 + S * 4].reshape(S, 4).astype(np.float64))
    dim = np.fromfile(prefix + "_dimages.bin", dtype=np.float32) \
        .reshape(S, 3, H, W)
    din = torch.from_numpy(dim.astype(np.float64))

    g = torch.from_numpy(g_np.astype(np.float64)).requires_grad_(True)
    images = render(g, w2c, intr, S, H, W, 0.5)
    cpp_i = np.fromfile(prefix + "_images.bin", dtype=np.float32) \
        .reshape(S, 3, H, W)
    fwd = np.abs(images.detach().numpy() - cpp_i.astype(np.float64))
    py_i64 = images.detach().numpy()
    print(f"fp64 |cpp-fp64| fwd max={fwd.max():.3e}  "
          f"mean={fwd.mean():.3e}")
    (din * images).sum().backward()
    grad = g.grad.numpy().astype(np.float32)
    grad.tofile(prefix + "_grads_fp64.bin")
    # distance of each implementation to the fp64 truth, per block
    cpp = np.fromfile(prefix + "_grads.bin", dtype=np.float32) \
        .reshape(g_np.shape[0], 14)
    for name, sl in [("d_pos", slice(0, 3)), ("d_op", slice(3, 4)),
                     ("d_scale", slice(4, 7)), ("d_rot", slice(7, 11)),
                     ("d_color", slice(11, 14))]:
        ref = grad[:, sl].astype(np.float64)
        dc = np.abs(cpp[:, sl].astype(np.float64) - ref).max()
        dr = np.abs(grad[:, sl].astype(np.float64) - ref).max()
        print(f"{name:>8s}  max|cpp-fp64|={dc:.3e}  max|py-fp64|={dr:.3e}  "
              f"max|fp64|={np.abs(ref).max():.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
