#!/usr/bin/env python3
"""TTO component parity: host fused_ssim/L1 + activation backward vs torch.

The torch side re-implements fused-ssim's math in pure torch (the
fused_ssim_cuda extension is not installed in this environment) and runs
torch.autograd through a numpy-exact re-implementation of
activate_gaussians. Comparisons are fp32-exact up to summation order.

Usage: tto_components_parity.py [--bin build-cpu/bin/test_tto_components]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

FLOOR = 1.0 / 255.0


def gauss_kernel_1d(k=11, sigma=1.5):
    x = torch.arange(k, dtype=torch.float64) - k // 2
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def conv_same(x, k2d):
    # zero-padded same 11x11 convolution, channels-first (C,H,W)
    pad = k2d.shape[-1] // 2
    xc = torch.nn.functional.conv2d(
        x[None, None].double(), k2d.flip(-1, -2)[None, None].double(),
        padding=pad)
    return xc[0, 0]


def fused_ssim_torch(img1, img2, need_grad=True):
    # pure-torch fused-ssim (train=True): returns (S.mean(), grads-img1)
    k1 = gauss_kernel_1d()
    k2d = (k1[:, None] * k1[None, :])
    C1, C2 = 1e-4, 9e-4
    img1 = img1.double().requires_grad_(need_grad)
    img2d = img2.double()
    x = img1.reshape(-1, *img1.shape[-2:])      # (V*3, H, W)
    y = img2d.reshape(-1, *img1.shape[-2:])
    conv = lambda t: torch.stack([conv_same(t[c], k2d) for c in range(t.shape[0])])
    mu1 = conv(x)
    mu2 = conv(y)
    s1 = conv(x * x) - mu1 * mu1
    s2 = conv(y * y) - mu2 * mu2
    s12 = conv(x * y) - mu1 * mu2
    A1 = 2 * mu1 * mu2 + C1
    A2 = 2 * s12 + C2
    B1 = mu1 * mu1 + mu2 * mu2 + C1
    B2 = s1 + s2 + C2
    S = A1 * A2 / (B1 * B2)
    return S.mean(), img1, (mu1, mu2, s1, s2, A1, A2, B1, B2)


def activate_torch(geom_raw, color_raw, n_tokens, gpt, post_out, cap, zoff,
                   sh, rest):
    """numpy-order-exact re-implementation of activate_gaussians."""
    d = torch.float64
    geom_raw = geom_raw.double()
    color_raw = color_raw.double()
    scale_shift = 1.0 - torch.log(torch.tensor(cap, dtype=d))
    log_cap = torch.log(torch.tensor(cap, dtype=d))
    ga = geom_raw.reshape(n_tokens, gpt, 10)
    ca = color_raw.reshape(n_tokens, gpt, post_out)
    pos = torch.sign(ga[..., 0:3]) * torch.expm1(torch.abs(ga[..., 0:3]))
    pos[..., 2] = pos[..., 2] + zoff
    scale = torch.exp(torch.clamp(ga[..., 3:6] - scale_shift, max=log_cap))
    rot = torch.nn.functional.normalize(ga[..., 6:10], dim=-1, eps=1e-12)
    out_geom = torch.cat([pos, scale, rot], dim=-1)
    color_dim = 3 * (sh + 1) * (sh + 1)
    if sh == 0:
        color = 0.5 * torch.tanh(ca[..., 0:3]) + 0.5
    else:
        raise SystemExit("sh>0 not exercised here")
    # QuerySplat._activate_post_output: opacity sits at column color_dim,
    # NOT at sh*3 (which would alias the color columns at sh=0)
    op = torch.sigmoid(ca[..., color_dim: color_dim + 1] - 2.0)
    return torch.cat([out_geom[..., 0:3], op, out_geom[..., 3:6], rot * 0 + out_geom[..., 6:10],
                      color], dim=-1).reshape(-1, 11 + 3)


def stats(name, cpp, ref):
    d = np.abs(cpp.astype(np.float64) - ref.astype(np.float64))
    a = cpp.astype(np.float64).ravel()
    b = ref.astype(np.float64).ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
    print(f"{name:>12s}  max_abs={d.max():.3e}  corr={corr:+.7f}")
    return d.max(), corr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build-cpu/bin/test_tto_components")
    ap.add_argument("--work", default="/tmp/qs_tto_comp")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ok = True

    # ---- fused_ssim / L1 loss ----
    rng = np.random.default_rng(3)
    V, H, W = 2, 96, 96
    pred = rng.uniform(0, 1, (V, 3, H, W)).astype(np.float32)
    gt = np.clip(pred + rng.normal(0, 0.08, pred.shape), 0, 1) \
        .astype(np.float32)
    (work / "pred.bin").write_bytes(pred.tobytes())
    (work / "gt.bin").write_bytes(gt.tobytes())
    subprocess.run([args.bin, "ssim", str(work / "pred.bin"),
                    str(work / "gt.bin"), str(V), str(H), str(W),
                    str(work / "s")], check=True)
    l1_s, ssim_s = map(float, (work / "s_ssim_val.txt").read_text().split())
    cpp_grad = np.fromfile(work / "s_ssim_grad.bin", dtype=np.float32) \
        .reshape(V, 3, H, W)
    pt = torch.from_numpy(pred)
    gt_t = torch.from_numpy(gt)
    ssim_mean, img1, _ = fused_ssim_torch(pt, gt_t)
    # the L1 term MUST flow through the same img1 node the SSIM backward
    # differentiates, or its gradient is silently dropped from the reference
    loss = torch.abs(img1 - gt_t.double()).mean() \
        + 0.2 * (1 - ssim_mean) / 2
    loss.backward()
    ref_grad = img1.grad.numpy()
    l1_ref = torch.abs(pt.double() - gt_t.double()).mean().item()
    print(f"l1   cpp={l1_s:.8f} torch={l1_ref:.8f}")
    print(f"ssim cpp={ssim_s:.8f} torch={ssim_mean.item():.8f}")
    ok &= abs(l1_s - l1_ref) < 1e-6 and abs(ssim_s - ssim_mean.item()) < 1e-6
    m, c = stats("d_pred", cpp_grad, ref_grad)
    ok &= c > 0.999999 and m < 1e-4

    # ---- activation backward (paper: sh0, post_out=4, gpt=64) ----
    n_tokens, gpt, post_out, cap, zoff, sh, rest = 96, 64, 4, 0.075, 1.0, 0, 0.1
    geom_raw = rng.normal(0, 0.5, (n_tokens, 10 * gpt)).astype(np.float32)
    color_raw = rng.normal(0, 0.5, (n_tokens, post_out * gpt)).astype(np.float32)
    # d_gaussians: LCG in [-1,1)
    state = np.uint64(12345)
    dg = np.empty((n_tokens * gpt, 14), dtype=np.float32)
    for i in range(dg.size):
        state = np.uint64(state * np.uint64(6364136223846793005) +
                          np.uint64(1442695040888963407))
        dg.flat[i] = ((int(state) >> 40) & 0xFFFFFF) / 0x1000000 - 0.5
    (work / "geom.bin").write_bytes(geom_raw.tobytes())
    (work / "color.bin").write_bytes(color_raw.tobytes())
    (work / "dg.bin").write_bytes(dg.tobytes())
    subprocess.run([args.bin, "act", str(work / "geom.bin"),
                    str(work / "color.bin"), str(work / "dg.bin"),
                    str(n_tokens), str(gpt), str(post_out), str(cap),
                    str(zoff), str(sh), str(rest), str(work / "a")],
                   check=True)
    cpp_dgeom = np.fromfile(work / "a_act_dgeom.bin", dtype=np.float32)
    cpp_dcolor = np.fromfile(work / "a_act_dcolor.bin", dtype=np.float32)
    g_geom = torch.from_numpy(geom_raw.copy()).double().requires_grad_(True)
    g_Color = torch.from_numpy(color_raw.copy()).double().requires_grad_(True)
    out = activate_torch(g_geom, g_Color, n_tokens, gpt, post_out, cap, zoff,
                         sh, rest)
    (torch.from_numpy(dg.astype(np.float64)) * out).sum().backward()
    m, c = stats("d_geom_raw", cpp_dgeom, g_geom.grad.numpy().ravel())
    ok &= c > 0.999999
    m, c = stats("d_color_raw", cpp_dcolor, g_Color.grad.numpy().ravel())
    ok &= c > 0.999999

    # ---- LPIPS value_and_grad (official lpips package as reference) ----
    import lpips as lpips_pkg
    loss_fn = lpips_pkg.LPIPS(net="vgg", verbose=False).eval()
    loss_fn.requires_grad_(False)
    V2, H2, W2 = 1, 96, 96
    pred2 = rng.uniform(0, 1, (V2, 3, H2, W2)).astype(np.float32)
    gt2 = np.clip(pred2 + rng.normal(0, 0.1, pred2.shape), 0, 1) \
        .astype(np.float32)
    (work / "pred2.bin").write_bytes(pred2.tobytes())
    (work / "gt2.bin").write_bytes(gt2.tobytes())
    subprocess.run([args.bin, "lpips", "models/gguf/lpips-vgg-f32.gguf",
                    str(work / "pred2.bin"), str(work / "gt2.bin"), str(V2),
                    str(H2), str(W2), str(work / "l")], check=True)
    cpp_val = np.fromfile(work / "l_lpips_val.bin", dtype=np.float32)
    cpp_lgrad = np.fromfile(work / "l_lpips_grad.bin", dtype=np.float32) \
        .reshape(V2, 3, H2, W2)
    # the official package runs fp32; keep the reference in fp32 too
    t_pred = torch.from_numpy(pred2.copy()).float().requires_grad_(True)
    t_gt = torch.from_numpy(gt2.copy()).float()
    lp_val = loss_fn.forward(t_gt, t_pred, normalize=False)
    lp_val.sum().backward()
    ref_lgrad = t_pred.grad.numpy()
    print(f"lpips val cpp={cpp_val[0]:.8f} torch={lp_val.item():.8f}")
    ok &= abs(cpp_val[0] - lp_val.item()) < 1e-5
    m, c = stats("d_lpips", cpp_lgrad, ref_lgrad)
    ok &= c > 0.9999

    print("TTO COMPONENTS " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
