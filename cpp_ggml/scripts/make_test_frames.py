#!/usr/bin/env python3
"""Write the canonical parity input: frames.f32.bin = (S,3,H,W) f32 in [0,1].

Deterministic across runs/numpy versions (PCG64, seeded). Both
spy_torch_querysplat.py and querysplat-cli consume this exact file so the
two sides share one bit-identical input.

Usage: python3 make_test_frames.py --out /tmp/qs_frames.bin [--S 4] [--H 512] [--W 512]
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()

    rng = np.random.Generator(np.random.PCG64(args.seed))
    frames = rng.random((args.S, 3, args.H, args.W), dtype=np.float32)
    frames.tofile(args.out)
    print(f"wrote {args.out}: S={args.S} H={args.H} W={args.W} "
          f"({frames.nbytes} bytes)")


if __name__ == "__main__":
    main()
