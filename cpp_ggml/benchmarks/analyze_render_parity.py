#!/usr/bin/env python3
"""Diagnose the cpp-vs-gsplat render residual: tile-boundary clustering?

Hypotheses for the residual (scene corr 0.90 on noise frames):
  A. tile-boundary artifact — high-error pixels hug the 16x16 tile edges
     (tile assignment / bbox clipping / sort ties at tile seams);
  B. uniform accumulation noise — fp summation order through per-pixel
     alpha blending (same class as the known numeric differences).

Method: per-pixel |cpp-gsplat| on all S views; rank pixels by error; compare
the fraction of top-k% error pixels that lie within EDGE=2 px of a tile
boundary against the geometric base rate; plus per-tile mean-error heatmap.

Usage:
  python3 analyze_render_parity.py --cpp cpp_render_images.bin \
      --py torch_render_images.bin --S 4 --H 512 --W 512 \
      [--top 5] [--edge 2] --out render_parity_tiles.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--py", required=True)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--top", type=float, default=5.0)
    ap.add_argument("--edge", type=int, default=2)
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = np.fromfile(args.cpp, dtype=np.float32).reshape(args.S, 3, args.H, args.W)
    b = np.fromfile(args.py, dtype=np.float32).reshape(args.S, 3, args.H, args.W)
    err = np.abs(a - b).mean(1)  # (S, H, W), per-pixel mean abs over RGB

    H, W, T, E = args.H, args.W, args.tile, args.edge
    yy, xx = np.mgrid[0:H, 0:W]
    near_edge = ((yy % T < E) | (yy % T >= T - E) |
                 (xx % T < E) | (xx % T >= T - E))
    base_rate = near_edge.mean()

    top = np.quantile(err, 1 - args.top / 100.0)
    hot = err >= top
    hot_edge = hot & near_edge[None]
    frac = hot_edge.sum() / max(hot.sum(), 1)
    print(f"per-pixel err: mean={err.mean():.4f} p99={np.quantile(err, 0.99):.4f}")
    print(f"tile-edge px share of top {args.top}% errors: {frac:.3f} "
          f"(geometric base rate {base_rate:.3f})")
    verdict = ("A: tile-boundary clustering — fix tile assignment/sort ties"
               if frac > base_rate * 1.5 else
               "B: uniform accumulation noise — fp summation-order class")
    print(f"verdict: {verdict}")

    tile_err = err.reshape(args.S, H // T, T, W // T, T).mean((2, 4))
    fig, axes = plt.subplots(2, args.S, figsize=(3.2 * args.S, 6.4))
    for s in range(args.S):
        im0 = axes[0][s].imshow(err[s], cmap="inferno")
        axes[0][s].set_title(f"view {s}: |cpp-gsplat|", fontsize=9)
        axes[0][s].set_xticks([]); axes[0][s].set_yticks([])
        im1 = axes[1][s].imshow(tile_err[s], cmap="inferno")
        axes[1][s].set_title(f"view {s}: tile mean (16px)", fontsize=9)
        axes[1][s].set_xticks([]); axes[1][s].set_yticks([])
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.suptitle(f"render parity residual — {verdict}", fontsize=10)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
