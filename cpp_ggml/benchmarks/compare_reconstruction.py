#!/usr/bin/env python3
"""End-to-end reconstruction comparison: rendered views vs ground truth.

Builds the sam3d/map-anything-style side-by-side figure:
  rows = input views, cols = [cpp_ggml render | official py render (opt) | GT | |cpp-GT| diff]
and prints per-view PSNR (GT preprocessed with the same square-crop-512 the
chain applies, so pixel grids align).

Developer-only tool (matplotlib/PIL/numpy); the runtime chain stays pure C++.

Usage:
  python3 compare_reconstruction.py --cpp-prefix OUT/cpp/querysplat \
      --gt-dir /path/to/DL3DV-Evaluation/<scene>/nerfstudio/images_8 \
      --frames f0.png f1.png f2.png f3.png \
      [--py-dir OUT/python/rendered] --out recon_case.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def square_crop_512(img: Image.Image, size: int = 512) -> np.ndarray:
    """The chain's preprocessing: center square crop of side min(h,w), resize."""
    w, h = img.size
    side = min(w, h)
    x0, y0 = (w - side) // 2, (h - side) // 2
    img = img.crop((x0, y0, x0 + side, y0 + side))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-prefix", required=True,
                    help="cpp render prefix (.../cpp/querysplat)")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--frames", nargs="+", required=True,
                    help="GT frame filenames, sorted like --images")
    ap.add_argument("--py-dir", default=None,
                    help="official python rendered dir (optional)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    views = len(args.frames)
    cpp_imgs, gt_imgs, py_imgs = [], [], []
    for f in args.frames:
        cpp_p = os.path.join(args.cpp_prefix + f"_render_view{len(cpp_imgs)}.png")
        cpp_imgs.append(np.asarray(Image.open(cpp_p), dtype=np.float64))
        gt_imgs.append(square_crop_512(Image.open(os.path.join(args.gt_dir, f))))
        if args.py_dir:
            py_imgs.append(np.asarray(Image.open(
                os.path.join(args.py_dir, f"render_view{len(py_imgs)}.png")),
                dtype=np.float64))

    rows, cols, ext = views, 4 if py_imgs else 3, []
    table = []
    # FIXED diff scale for cross-figure comparability: |cpp-py| maps to
    # [0, 8/255] (8-bit rounding boundary flips saturate at 8 LSB). Without a
    # fixed scale, imshow stretches each figure's own max to the colormap and
    # a 48 dB diff (max ~24 LSB) LOOKS worse than a 48 dB diff with max 59.
    DIFF_VMAX = 8.0  # 8 LSB in 8-bit code-value units
    for i in range(views):
        p_cpp = psnr(cpp_imgs[i], gt_imgs[i])
        row = {"view": i, "psnr_cpp_db": round(p_cpp, 2)}
        entry = [cpp_imgs[i] / 255.0, gt_imgs[i] / 255.0]
        if py_imgs:
            p_py = psnr(py_imgs[i], gt_imgs[i])
            row["psnr_py_db"] = round(p_py, 2)
            # the interesting column: the two CHAINS against each other,
            # not each against GT — this is the ggml-fidelity visualization
            row["psnr_cpp_vs_py_db"] = round(psnr(cpp_imgs[i], py_imgs[i]), 2)
            entry.insert(1, py_imgs[i] / 255.0)
        diff = (np.abs(cpp_imgs[i] - py_imgs[i]) if py_imgs
                else np.abs(cpp_imgs[i] - gt_imgs[i])).mean(-1)
        row["max_abs_diff"] = round(float(diff.max()), 2)
        row["pct_gt_2lsb"] = round(float((diff > 2).mean() * 100.0), 3)
        entry.append(np.clip(diff, 0.0, DIFF_VMAX))
        ext.append(entry)
        table.append(row)
        print(row)

    if py_imgs:
        titles = ("cpp_ggml (C++)", "official py (torch)", "ground truth",
                  "|cpp − py|")
    else:
        titles = ("cpp_ggml (C++)", "ground truth", "|cpp − GT|")
    fig, axes = plt.subplots(rows, cols,
                             figsize=(3.0 * cols, 3.05 * rows))
    if rows == 1:
        axes = axes[None, :]
    for r in range(rows):
        for c in range(len(titles)):
            ax = axes[r][c]
            img = ext[r][c]
            # fixed [0, 8 LSB] scale on the diff column (last column)
            if c == len(titles) - 1:
                ax.imshow(img, cmap="inferno", vmin=0.0, vmax=DIFF_VMAX)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0:
                p = table[r]["psnr_cpp_db"]
                ax.set_ylabel(f"view {r}\n{p:.1f} dB", fontsize=9)
    means = (f"mean PSNR cpp {np.mean([t['psnr_cpp_db'] for t in table]):.2f} dB"
             + (f" | official py {np.mean([t['psnr_py_db'] for t in table]):.2f} dB"
                if py_imgs else "")
             + (f" | cpp-vs-py {np.mean([t['psnr_cpp_vs_py_db'] for t in table]):.2f} dB"
                if py_imgs else ""))
    fig.suptitle(f"End-to-end reconstruction vs ground truth ({means}; "
                 f"|diff| panel fixed scale 0-8/255, saturated there)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
