#!/usr/bin/env python3
"""Component parity for querysplat-eval: feeds identical inputs to the official
evaluations/{metrics,cameras}.py and to the C++ ports (test_eval_parity) and
reports the max per-view difference.

  metrics: real 256x256 render/GT PNG pairs from an official --save-images run
           (default: outputs/eval_protocol_paper/python), plus their swapped
           and noise-perturbed variants so SSIM/LPIPS cover a range of values.
  align:   the predicted (S=N+T) and input-only cameras an official run would
           align, drawn as random similarity-transformed rigs (incl. a
           reflection-correction case and a degenerate one-centre reference).

    python cpp_ggml/benchmarks/eval_parity_unit.py [--device cuda]
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from evaluations.cameras import align_cameras_to_reference  # noqa: E402
from evaluations.metrics import ImageMetrics  # noqa: E402

BIN = os.path.join(ROOT, "cpp_ggml", "build-cuda", "bin", "test_eval_parity")
LPIPS_GGUF = os.path.join(ROOT, "cpp_ggml", "models", "gguf", "lpips-vgg-f32.gguf")


def load_pairs(run_dir, limit):
    preds, gts = [], []
    for r in sorted(glob.glob(os.path.join(run_dir, "**", "renders", "*.png"), recursive=True)):
        g = r.replace(os.sep + "renders" + os.sep, os.sep + "gt" + os.sep)
        if not os.path.exists(g):
            continue
        preds.append(np.asarray(Image.open(r).convert("RGB"), np.float32) / 255.0)
        gts.append(np.asarray(Image.open(g).convert("RGB"), np.float32) / 255.0)
        if len(preds) >= limit:
            break
    if not preds:
        raise SystemExit(f"no renders/gt pairs under {run_dir} (run evaluate_jsons.py --save-images)")
    p = np.stack(preds).transpose(0, 3, 1, 2)
    g = np.stack(gts).transpose(0, 3, 1, 2)
    rng = np.random.default_rng(0)
    noisy = np.clip(p + rng.normal(0, 0.05, p.shape).astype(np.float32), 0, 1)
    # swapped order exercises the SSIM/LPIPS asymmetry (official passes (gt, pred))
    return (np.ascontiguousarray(np.concatenate([p, g, noisy])),
            np.ascontiguousarray(np.concatenate([g, p, g])))


def run_metrics(args):
    pred, gt = load_pairs(args.run_dir, args.limit)
    V, _, H, W = pred.shape
    ref = ImageMetrics(args.device)(torch.from_numpy(pred).to(args.device),
                                    torch.from_numpy(gt).to(args.device))
    with tempfile.TemporaryDirectory() as d:
        pred.tofile(os.path.join(d, "pred.f32"))
        gt.tofile(os.path.join(d, "gt.f32"))
        subprocess.run([BIN, "metrics", d, str(V), str(H), str(W), LPIPS_GGUF, args.device],
                       check=True, stdout=subprocess.DEVNULL)
        cpp = np.fromfile(os.path.join(d, "cpp_metrics.f32"), np.float32).reshape(3, V)
    print(f"metrics: {V} view pairs at {H}x{W}")
    worst = {}
    for k, row in zip(("psnr", "ssim", "lpips"), cpp):
        diff = np.abs(np.asarray(ref[k], np.float64) - row.astype(np.float64))
        worst[k] = float(diff.max())
        print(f"  {k:5s} max|cpp-py| = {diff.max():.3e}  mean = {diff.mean():.3e}  "
              f"(py range {min(ref[k]):.4f}..{max(ref[k]):.4f})")
    return worst


def random_rig(rng, n):
    """n row-major w2c [R|t] with spread centres and random orientations."""
    out = []
    for _ in range(n):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        a, b, c, d = q
        R = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                      [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                      [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = rng.normal(scale=2.0, size=3)
        out.append(np.linalg.inv(c2w))
    return np.stack(out).astype(np.float32)


def run_align(args):
    rng = np.random.default_rng(1)
    worst = 0.0
    cases = [(4, 2), (6, 4), (16, 12), (5, 4), (3, 2)]
    for ci, (n, m) in enumerate(cases):
        w2c = random_rig(rng, n)
        ref = random_rig(rng, m)
        if ci == 3:   # reference rig mirrored: forces the det<0 correction branch
            ref[:, :3, :3] = ref[:, :3, :3] @ np.diag([1, 1, -1]).astype(np.float32)
        if ci == 4:   # degenerate reference (coincident centres): scale -> 1
            c2w = np.linalg.inv(ref)
            c2w[:, :3, 3] = c2w[0, :3, 3]
            ref = np.linalg.inv(c2w).astype(np.float32)
        cv = torch.from_numpy(w2c).transpose(-2, -1)[None].contiguous()
        rv = torch.from_numpy(ref).transpose(-2, -1)[None].contiguous()
        py = align_cameras_to_reference(cv, rv, m)[0].transpose(-2, -1).numpy()
        with tempfile.TemporaryDirectory() as d:
            w2c.tofile(os.path.join(d, "w2c.f32"))
            ref.tofile(os.path.join(d, "ref_w2c.f32"))
            subprocess.run([BIN, "align", d, str(n), str(m)], check=True)
            cpp = np.fromfile(os.path.join(d, "cpp_aligned.f32"), np.float32).reshape(n, 4, 4)
        diff = float(np.abs(cpp - py).max())
        worst = max(worst, diff)
        print(f"  align n={n:2d} ref={m:2d}  max|cpp-py| = {diff:.3e}  "
              f"(|w2c| max {np.abs(py).max():.2f})")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "outputs", "eval_protocol_paper", "python"))
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    m = run_metrics(args)
    print("align: random rigs (row-major w2c in, aligned w2c out)")
    a = run_align(args)
    ok = m["psnr"] < 1e-3 and m["ssim"] < 1e-4 and m["lpips"] < 1e-3 and a < 1e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
