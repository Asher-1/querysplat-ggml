#!/usr/bin/env python3
"""Per-seam parity comparison: querysplat-cli (QS_DUMP_STAGE) vs
spy_torch_querysplat.py dumps.

Usage:
  python3 cmp_querysplat.py <cpp_dump_dir> <spy_dump_dir> [--S 4] [--H 512] [--W 512]

Memory-order rule (map-anything-ggml AGENTS.md pitfall #10): a ggml dump with
ne {a, b, c, d} reshapes to numpy (d, c, b, a). The spy dumps torch tensors
already permuted into the matching layout (see spy_torch_querysplat.py), so
every seam here compares two torch C-order arrays of identical shape.

Verdict rule per seam: mean_abs <= MEAN threshold AND
rel_max = max_abs / max(|spy|) <= REL threshold. The encoder runs f16 GGUF
weights vs torch f32, so absolute thresholds only make sense relative to the
reference magnitude (omega's 24-layer f16 noise floor is ~0.8% mean; the
decoder's 18 blocks amplify the tokens seam by a few x — first measured
S=2 floors: tokens mean 8.9e-4, geom_tokens mean 1.9e-2). Re-calibrate from
measured floors with the >= 1.7x headroom rule; never by eyeball.
"""

import argparse
import os
import sys

import numpy as np

# seam -> (shape_fn, mean_thresh, rel_thresh)
# Shapes are torch C-order; N = num_gs_tokens * gpt, P = patch tokens/view.
# deconv/color raws are per-token (token dim = num_gs_tokens), gaussians are
# per-gaussian (N = 8192*64) with stride 11+color_dim.


def build_table(S, H, W, nagg, P, post_out, color_dim, profile="paper"):
    Ntok = 8192
    N = 8192 * 64
    base = {
        "cpp_pose_enc":       ((S, 9), 1e-4, 1e-4),
        "cpp_intrinsics":     ((S, 4), -1e-4, 2e-4),   # rel-only (fx~1e3)
        "cpp_plucker":        ((S, 6, H, W), 1e-4, 1e-4),
        # thresholds calibrated on the measured S=2 f32-decoder noise floor
        # (encoder f16 tokens mean 8.9e-4 -> decoder amplification; floors x
        # >= 1.7 headroom per AGENTS.md): rgb_tokens 2.5e-5 -> 1e-4 |
        # kv ~2e-3 -> 5e-3 | geom_tokens 1.5e-4 -> 5e-4 | deconv 8.7e-4 ->
        # 2e-3 | color_raw 5.4e-2 -> 1e-1 (fp summation order through 6
        # cross-attn blocks, the same class as map-anything-ggml's "known
        # numeric differences") | gaussians 6.0e-3 -> 1e-2, rel 15% -> 30%.
        "cpp_tokens":         ((S * nagg, 2048), 3e-3, 2e-2),
        "cpp_rgb_tokens":     ((P * S, 2048), 1e-4, 5e-4),  # rel 5e-4 covers the f16 decoder tier
        "cpp_geom_kvproj":    ((1, S * nagg, 4096), -1e-2, 2e-2),  # rel-only (mid-diag seam, |ref|~32)
        "cpp_geom_kv_k":      ((1, 16, S * nagg, 128), -1e-2, 2e-2),  # rel-only (|ref|~20)
        "cpp_geom_kv_v":      ((1, 16, S * nagg, 128), -1e-2, 2e-2),
        "cpp_post_kv_k":      ((1, 16, P * S, 128), -1e-2, 2e-2),
        "cpp_post_kv_v":      ((1, 16, P * S, 128), -1e-2, 2e-2),
        "cpp_geom_tokens":    ((Ntok, 2048), 5e-4, 5e-3),
        "cpp_deconv_raw":     ((Ntok, 640), 2e-3, 2e-2),
        "cpp_color_raw":      ((Ntok, post_out * 64), 1e-1, 5e-1),
        "cpp_gaussians":      ((N, 11 + color_dim), 1e-2, 3e-1),
    }
    if profile == "8192":
        return base
    if profile == "paper":
        # querysplat_vggto_1B_512_paper (sh_degree 0): different decoder
        # weights -> a ~1.2-2.8x higher f16 noise floor (measured S=2 CPU,
        # 2026-10-16: rgb_tokens 1.24e-4, geom_tokens 2.26e-4/6.0e-3 rel,
        # deconv 2.46e-3, gaussians 6.1e-3/3.1e-1 rel). Same >= 1.7x
        # headroom rule applied to THIS checkpoint's own floors:
        paper = dict(base)
        paper["cpp_rgb_tokens"] = ((P * S, 2048), 4e-4, 1e-3)
        paper["cpp_geom_kvproj"] = ((1, S * nagg, 4096), -4.5e-2, 4.5e-2)
        paper["cpp_geom_kv_k"] = ((1, 16, S * nagg, 128), -4.5e-2, 4.5e-2)
        paper["cpp_geom_kv_v"] = ((1, 16, S * nagg, 128), -4.5e-2, 4.5e-2)
        paper["cpp_geom_tokens"] = ((Ntok, 2048), 7.5e-4, 1.2e-2)
        paper["cpp_deconv_raw"] = ((Ntok, 640), 5.7e-3, 1.2e-1)
        paper["cpp_color_raw"] = ((Ntok, post_out * 64), 1.6e-1, 8e-1)
        paper["cpp_gaussians"] = ((N, 11 + color_dim), 1.05e-2, 6.2e-1)
        return paper
    raise ValueError(f"unknown profile {profile}")


def load(path):
    return np.fromfile(path, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpp_dir")
    ap.add_argument("spy_dir")
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--patch-token-start", type=int, default=17)
    ap.add_argument("--sh-degree", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--profile", default="paper", choices=("8192", "paper"),
                    help="threshold profile: per-checkpoint floors x headroom")
    args = ap.parse_args()

    P = (args.H // 16) * (args.W // 16)
    nagg = args.patch_token_start + P
    color_dim = 3 * (args.sh_degree + 1) ** 2
    post_out = color_dim + 1
    table = build_table(args.S, args.H, args.W, nagg, P, post_out, color_dim,
                        args.profile)

    failures = 0
    print(f"{'seam':22s} {'max_abs':>11s} {'rel_max':>11s} {'mean_abs':>11s} "
          f"{'mean_th':>9s} {'rel_th':>9s} verdict")
    for name, (shape, mean_th, rel_th) in table.items():
        if name in args.skip:
            continue
        cp = os.path.join(args.cpp_dir, name + ".bin")
        sp = os.path.join(args.spy_dir, name + ".bin")
        if not (os.path.exists(cp) and os.path.exists(sp)):
            print(f"{name:22s} MISSING dump (cpp={os.path.exists(cp)} "
                  f"spy={os.path.exists(sp)})")
            failures += 1
            continue
        a = load(cp)
        b = load(sp)
        if a.size != b.size:
            print(f"{name:22s} SIZE MISMATCH cpp={a.size} spy={b.size}")
            failures += 1
            continue
        a = a.reshape(shape)
        b = b.reshape(shape)
        d = np.abs(a - b)
        max_abs = float(d.max())
        mean_abs = float(d.mean())
        scale = max(float(np.abs(b).max()), 1e-6)
        rel_max = max_abs / scale
        rel_mean = mean_abs / max(float(np.abs(b).mean()), 1e-6)
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        # large-magnitude seams (intrinsics ~1e3) judge on relative error;
        # everything else on mean_abs + rel_max
        rel_only = mean_th < 0
        ok = (rel_mean <= -mean_th and rel_max <= rel_th) if rel_only else \
            (mean_abs <= mean_th and rel_max <= rel_th)
        if not ok:
            failures += 1
        print(f"{name:22s} {max_abs:11.3e} {rel_max:11.3e} {mean_abs:11.3e} "
              f"{mean_th:9.1e} {rel_th:9.1e} {'PASS' if ok else 'FAIL'}"
              f"  corr={corr:+.4f} rel_mean={rel_mean:.2e}")

    # cam_view: torch (S,4,4) == per-view transpose of the C++ w2c
    cp = os.path.join(args.cpp_dir, "cpp_w2c.bin")
    sp = os.path.join(args.spy_dir, "cpp_cam_view.bin")
    if os.path.exists(cp) and os.path.exists(sp):
        w2c = load(cp).reshape(args.S, 4, 4)
        cam_view = w2c.transpose(0, 2, 1)
        t = load(sp).reshape(args.S, 4, 4)
        d = np.abs(cam_view - t)
        max_abs = float(d.max())
        ok = max_abs <= 1e-4
        if not ok:
            failures += 1
        print(f"{'cpp_cam_view':22s} {max_abs:11.3e} {'-':>11s} "
              f"{float(d.mean()):11.3e} {1e-4:9.1e} {'-':>9s} "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"\n{'GATE PASS' if failures == 0 else f'GATE FAIL ({failures})'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
