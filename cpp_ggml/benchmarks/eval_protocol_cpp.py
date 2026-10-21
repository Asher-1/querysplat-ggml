#!/usr/bin/env python3
"""Protocol-level parity: diff a querysplat-eval output tree against an
official evaluations/evaluate_jsons.py tree run on the same splits/cases.

Both sides run the full protocol independently (C++: own encoder, decoder,
camera alignment, rasterizer, GT resize, PSNR/SSIM/LPIPS; nothing is borrowed
from the official run). This script only reads the two trees and reports:

  * structure: identical relative file sets; identical JSON key order of
    metrics.json / timing.json / summary.json;
  * per case x group: |delta| of the case-mean PSNR/SSIM/LPIPS;
  * cameras.npz: max |delta| of input_cam_view / target_cam_view / intrinsics;
  * gt/*.png: max |delta| (the resized GT must agree to within 1 LSB);
  * renders/*.png: cpp-vs-official PSNR (uint8 renders);
  * split summaries side by side, and the timing medians.

    python cpp_ggml/benchmarks/eval_protocol_cpp.py \
        --official outputs/eval_protocol_paper/python \
        --cpp outputs/eval_protocol_paper/cpp_cuda \
        --out cpp_ggml/benchmarks/results/querysplat/dl3dv_protocol_cpp.md
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

GROUPS = ("input", "interp", "extrap")
METRICS = ("psnr", "ssim", "lpips")


def rel_files(root):
    return sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, "**", "*"),
                                                             recursive=True) if os.path.isfile(p))


def key_paths(x, p=""):
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            out.append(p + "/" + k)
            out += key_paths(v, p + "/" + k)
    return out


def png(path):
    return np.asarray(Image.open(path).convert("RGB"), np.int16)


def psnr_u8(a, b):
    mse = ((a.astype(np.float64) - b) ** 2).mean()
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A, B = args.official, args.cpp

    fa, fb = rel_files(A), rel_files(B)
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    key_mismatch = []
    for rel in fa:
        if rel.endswith(".json") and rel != "run_config.json" and rel in fb:
            if key_paths(json.load(open(os.path.join(A, rel)))) != \
                    key_paths(json.load(open(os.path.join(B, rel)))):
                key_mismatch.append(rel)

    rows, worst = [], {"cam": 0.0, "gt": 0, "metric": {m: 0.0 for m in METRICS}}
    render_psnr, timing = [], {"official": [], "cpp": []}
    for mpath in sorted(glob.glob(os.path.join(A, "*", "*", "case_*", "metrics.json"))):
        case_dir = os.path.dirname(mpath)
        rel = os.path.relpath(case_dir, A)
        other = os.path.join(B, rel)
        if not os.path.isdir(other):
            continue
        ma = json.load(open(mpath))
        mb = json.load(open(os.path.join(other, "metrics.json")))
        row = {"split": rel.split(os.sep)[0], "case": ma["case_index"], "group": ma["target_group"]}
        for m in METRICS:
            row[m] = (ma[m], mb[m])
            worst["metric"][m] = max(worst["metric"][m], abs(ma[m] - mb[m]))
        ca = np.load(os.path.join(case_dir, "cameras.npz"))
        cb = np.load(os.path.join(other, "cameras.npz"))
        # relative to each array's magnitude (intrinsics are ~200 px, cam_view
        # entries ~1), so one number is comparable across the three arrays
        row["cam"] = max(float(np.abs(ca[k].astype(np.float64) - cb[k]).max() / np.abs(ca[k]).max())
                         for k in ca.files)
        worst["cam"] = max(worst["cam"], row["cam"])
        rp = []
        for r in sorted(glob.glob(os.path.join(case_dir, "renders", "*.png"))):
            name = os.path.basename(r)
            rp.append(psnr_u8(png(r), png(os.path.join(other, "renders", name))))
            g = np.abs(png(os.path.join(case_dir, "gt", name)) - png(os.path.join(other, "gt", name)))
            worst["gt"] = max(worst["gt"], int(g.max()))
        row["render_psnr"] = float(np.mean(rp)) if rp else float("nan")
        render_psnr += rp
        for side, d in (("official", case_dir), ("cpp", other)):
            timing[side].append(json.load(open(os.path.join(d, "timing.json"))))
        rows.append(row)
    if not rows:
        print("no common cases")
        return 1

    L = ["# DL3DV protocol: querysplat-eval (C++) vs evaluate_jsons.py (official)", ""]
    ra, rb = json.load(open(os.path.join(A, "run_config.json"))), json.load(open(os.path.join(B, "run_config.json")))
    L += [f"- official: `{A}` (torch {ra.get('torch_version', '?')}, checkpoint "
          f"`{os.path.basename(str(ra.get('checkpoint')))}`)",
          f"- cpp: `{B}` (device `{rb.get('device')}`, encoder `{os.path.basename(str(rb.get('encoder')))}`, "
          f"decoder `{os.path.basename(str(rb.get('decoder')))}`)",
          f"- split_sha256 equal: **{ra.get('split_sha256') == rb.get('split_sha256')}**; "
          f"protocol strings equal: **{all(ra.get(k) == rb.get(k) for k in ('camera_protocol', 'metric_protocol'))}**", ""]
    L += ["## Structure", "",
          f"- files: {len(fa)} official / {len(fb)} cpp; only-official {len(only_a)}, only-cpp {len(only_b)}",
          f"- JSON key-order mismatches (metrics/timing/summary): {len(key_mismatch)}"]
    L += [f"  - `{x}`" for x in (only_a + only_b + key_mismatch)[:20]]
    L += ["", "## Split summaries", "", "| split | target | official PSNR / SSIM / LPIPS | cpp PSNR / SSIM / LPIPS | ΔPSNR |",
          "| --- | --- | --- | --- | ---: |"]
    for sa in sorted(glob.glob(os.path.join(A, "*", "summary.json"))):
        sb = os.path.join(B, os.path.relpath(sa, A))
        if not os.path.exists(sb):
            continue
        ja, jb = json.load(open(sa)), json.load(open(sb))
        for g in GROUPS:
            if g not in ja.get("groups", {}):
                continue
            a = [ja["groups"][g]["metrics"][m]["mean"] for m in METRICS]
            b = [jb["groups"][g]["metrics"][m]["mean"] for m in METRICS]
            L.append(f"| {os.path.basename(os.path.dirname(sa))} | {g} | "
                     f"{a[0]:.4f} / {a[1]:.4f} / {a[2]:.4f} | {b[0]:.4f} / {b[1]:.4f} / {b[2]:.4f} | "
                     f"{b[0] - a[0]:+.4f} |")
    L += ["", "## Per case", "",
          "| split | case | group | PSNR off / cpp | SSIM off / cpp | LPIPS off / cpp | cams max rel\\|Δ\\| | render PSNR cpp↔off |",
          "| --- | ---: | --- | --- | --- | --- | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: (r["split"], r["case"], GROUPS.index(r["group"]))):
        L.append(f"| {r['split']} | {r['case']} | {r['group']} | "
                 + " | ".join(f"{r[m][0]:.4f} / {r[m][1]:.4f}" for m in METRICS)
                 + f" | {r['cam']:.2e} | {r['render_psnr']:.2f} |")
    med = lambda side, k: float(np.median([t[k] for t in timing[side]]))  # noqa: E731
    L += ["", "## Worst case", "",
          f"- case-mean |ΔPSNR| {worst['metric']['psnr']:.4f} dB, |ΔSSIM| {worst['metric']['ssim']:.4f}, "
          f"|ΔLPIPS| {worst['metric']['lpips']:.4f}",
          f"- cameras.npz max relative |Δ| (per array, / max|official|) {worst['cam']:.3e}",
          f"- gt/*.png max |Δ| {worst['gt']} LSB",
          f"- renders cpp↔official PSNR: mean {np.mean(render_psnr):.2f} dB, min {np.min(render_psnr):.2f} dB "
          f"({len(render_psnr)} views)", "",
          "## Timing (median s per case-group)", "",
          "| | reconstruction_seconds | target_camera_render_metrics_seconds |", "| --- | ---: | ---: |"]
    for side in ("official", "cpp"):
        L.append(f"| {side} | {med(side, 'reconstruction_seconds'):.3f} | "
                 f"{med(side, 'target_camera_render_metrics_seconds'):.3f} |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    ok = not only_a and not only_b and not key_mismatch and worst["gt"] <= 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
