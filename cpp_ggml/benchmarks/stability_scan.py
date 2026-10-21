#!/usr/bin/env python3
"""Input-perturbation stability metric per case.

The case1 finding: a 0.002-0.004/px input difference (8-bit quantization
level, from a different resize kernel) flips the feed-forward output between
a clean reconstruction and a fogged collapse. This script quantifies that
fragility per case: render the SAME case through cpp_ggml with several
input-preprocessing variants and report the PSNR spread against GT.

Variants (all fed to the identical cpp chain):
  stb       — original frames through the cpp stb upsample (--images)
  official  — frames pre-upsampled by the OFFICIAL torchvision ImageTransform
              (pipeline: python preprocess -> frames.bin -> cpp --bin)

Interpretation: spread = max PSNR − min PSNR across variants. A robust case
has spread < 1 dB; a fragile case (case1-type) can exceed 10 dB. The official
python chain exhibits the same fragility (its own re-sampled variants
collapse identically — see querysplat_report.md §6.1).

Usage:
  python3 stability_scan.py --case 0 --scene-dir <images_8 dir> \
      --frames f0.png f1.png f2.png f3.png --work /tmp/stability \
      --official-frames-bin /tmp/case0_frames_official.bin
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QS = os.path.join(ROOT, "cpp_ggml")
CLI_CUDA = os.path.join(QS, "build-cuda", "bin", "querysplat-cli")
ENCODER = os.path.join(QS, "models", "gguf", "vggt-omega-1b-512-f32.gguf")
DECODER = os.path.join(QS, "models", "gguf", "querysplat-paper-f32.gguf")


def square_crop_512(img, size=512):
    """The chain's GT preprocessing: center square crop of min(h,w), resize.
    NOTE: the GT must be cropped the same way as the model input — comparing
    against the UNCROPPED frame stretched to a square reads as ~12 dB even
    for a perfect render (measured 2026-10-16).
    """
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w + side) // 2, (h + side) // 2))
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float64)


def psnr(a, b):
    mse = ((a.astype(np.float64) - b) ** 2).mean()
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def run_cpp(frames_bin, out_prefix, device):
    cmd = [CLI_CUDA if device != "cpu" else os.path.join(QS, "build-cpu", "bin", "querysplat-cli"),
           "--encoder", ENCODER, "--decoder", DECODER, "--bin", frames_bin,
           "--S", "4", "--out-prefix", out_prefix, "--device", device, "--render"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--scene-dir", required=True, help="images_8 dir")
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--official-frames-bin", default=None,
                    help="frames.bin produced by the OFFICIAL ImageTransform")
    ap.add_argument("--gt-size", type=int, default=512)
    args = ap.parse_args()

    variants = {}
    # stb variant: original frames through cpp's own upsample
    stb_inputs = [os.path.join(args.scene_dir, f) for f in args.frames]
    rc = run_cpp(None, None, None) if False else None
    prefix = os.path.join(args.work, f"case{args.case}_stb")
    cmd = [CLI_CUDA, "--encoder", ENCODER, "--decoder", DECODER,
           "--images", ",".join(stb_inputs), "--out-prefix", prefix,
           "--device", "cuda", "--render"]
    if subprocess.run(cmd, capture_output=True).returncode == 0:
        a = np.fromfile(prefix + "_render_images.bin", dtype=np.float32).reshape(4, 3, 512, 512)
        variants["stb"] = a

    # official variant: frames preprocessed by the official ImageTransform
    if args.official_frames_bin and os.path.isfile(args.official_frames_bin):
        prefix = os.path.join(args.work, f"case{args.case}_official")
        cmd = [CLI_CUDA, "--encoder", ENCODER, "--decoder", DECODER,
               "--bin", args.official_frames_bin, "--S", "4",
               "--out-prefix", prefix, "--device", "cuda", "--render"]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            a = np.fromfile(prefix + "_render_images.bin", dtype=np.float32).reshape(4, 3, 512, 512)
            variants["official"] = a

    if not variants:
        print("no renders produced")
        return 1

    table = []
    for name, a in variants.items():
        psnrs = []
        for i, f in enumerate(args.frames):
            gt = square_crop_512(Image.open(os.path.join(args.scene_dir, f)).convert("RGB"))
            r = np.clip(a[i].transpose(1, 2, 0) * 255, 0, 255)
            psnrs.append(psnr(r, gt))
        table.append((name, psnrs))
        print(f"{name:9s} per-view PSNR:", [round(p, 2) for p in psnrs],
              " mean:", round(float(np.mean(psnrs)), 2))

    if len(table) >= 2:
        means = [float(np.mean(p)) for _, p in table]
        spread = max(means) - min(means)
        print(f"case {args.case} fragility (PSNR spread across preprocessing "
              f"variants): {spread:.2f} dB -> "
              f"{'ROBUST' if spread < 1.5 else 'FRAGILE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
