#!/usr/bin/env python3
"""Broad official-split validation: run BOTH chains (cpp_ggml and the official
python infer) on every locally-available case of the official DL3DV-Evaluation
split jsons, and flag any case whose cpp reconstruction collapses or whose
cpp-vs-py agreement degrades — the regression net for case1-type failures.

A case = (scene_id, input frame tuple) from the official split jsons
(4/2/12 views x small/medium/large target samplings share input sets; the
input-view reconstruction is the alignment surface, so cases are deduped by
their input frame tuple).

Per case, reported:
  - cpp-vs-GT PSNR (reconstruction quality; case1-type collapse reads <15 dB)
  - py-vs-GT PSNR (official reference on the same inputs)
  - cpp-vs-py PSNR (chain alignment; the aligned floor is ~46-50 dB)
  - verdict: OK / COLLAPSE (cpp-vs-GT < 15) / GAP (cpp-vs-py < 40)

Usage:
  python3 broad_validation.py --work /tmp/qs_broad [--max-cases 24] \
      [--out benchmarks/results/querysplat/broad_validation.md]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QS = os.path.join(ROOT, "cpp_ggml")
CLI = os.path.join(QS, "build-cuda", "bin", "querysplat-cli")
ENCODER = os.path.join(QS, "models/gguf/vggt-omega-1b-512-f32.gguf")
DECODER = os.path.join(QS, "models/gguf/querysplat-paper-f32.gguf")
JSONS = sorted(glob.glob(os.path.join(ROOT, "evaluations/jsons_dl3dv/dl3dv_val_*.json")))
DATASET_ROOT = os.path.join(ROOT, "data/DL3DV-Evaluation")
PY = os.environ.get("PY", "/home/asher/anaconda3/envs/python3.13/bin/python")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def gt_crop(img, size=512):
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w + side) // 2, (h + side) // 2))
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/qs_broad")
    ap.add_argument("--max-cases", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(
        QS, "benchmarks/results/querysplat/broad_validation.md"))
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    local_scenes = {d for d in os.listdir(DATASET_ROOT)
                    if os.path.isdir(os.path.join(DATASET_ROOT, d, "nerfstudio",
                                                  "images_8"))}
    avail_frames = {d: {os.path.basename(p) for p in glob.glob(
        os.path.join(DATASET_ROOT, d, "nerfstudio", "images_8", "*.png"))}
        for d in local_scenes}
    print(f"local scenes: {sorted(local_scenes)}")

    # collect unique cases across all official split jsons
    cases = {}
    for jpath in JSONS:
        j = json.load(open(jpath, encoding="utf-8"))
        if "samples" not in j:
            continue  # sampling.json is a split definition, not cases
        for si, s in enumerate(j["samples"]):
            if s["scene_id"] not in local_scenes:
                continue
            frames = [e["path"].split("/")[-1] for e in s["input_views"]]
            if not all(f in avail_frames.get(s["scene_id"], set())
                       for f in frames):
                continue  # frames not in the local subset of the gated dataset
            key = (s["scene_id"], tuple(frames))
            cases.setdefault(key, {
                "scene_id": s["scene_id"],
                "frames": frames,
                "jsons": [],
                "view_mode": j["view_mode"],
            })["jsons"].append(f"{os.path.basename(jpath)}#{si}")
    print(f"unique locally-runnable cases: {len(cases)}")
    cases = list(cases.values())[:args.max_cases]

    rows = []
    for ci, case in enumerate(cases):
        scene_dir = os.path.join(DATASET_ROOT, case["scene_id"], "nerfstudio",
                                 "images_8")
        paths = [os.path.join(scene_dir, f) for f in case["frames"]]
        if not all(os.path.isfile(p) for p in paths):
            print(f"case {ci}: missing frames, skip")
            continue

        # ---- cpp chain (f32 + f32-accuracy, default mode) ----
        cprefix = os.path.join(args.work, f"case_{ci:03d}_cpp")
        r = sh([CLI, "--encoder", ENCODER, "--decoder", DECODER,
                "--images", ",".join(paths), "--out-prefix", cprefix,
                "--device", "cuda", "--render"])
        if r.returncode != 0:
            print(f"case {ci}: cpp FAILED\n{r.stdout[-600:]}", file=sys.stderr)
            rows.append({"case": ci, "frames": len(case["frames"]),
                         "view_mode": case["view_mode"], "verdict": "CPP_FAIL"})
            continue

        # ---- official python chain on the same inputs (symlinked folder) ----
        indir = os.path.join(args.work, f"case_{ci:03d}_in")
        poutdir = os.path.join(args.work, f"case_{ci:03d}_py")
        if not os.path.isdir(poutdir):
            os.makedirs(indir, exist_ok=True)
            for i, p in enumerate(paths):
                dst = os.path.join(indir, f"frame_{i:04d}.png")
                if not os.path.islink(dst):
                    os.symlink(os.path.abspath(p), dst)
            # fused_ssim_cuda is import-time-required by scripts.models but
            # only used by TTO — stub it (same trick as the parity spy) so
            # the inference-only env works.
            stub = (
                "import sys, types; m = types.ModuleType('fused_ssim_cuda'); "
                "m.fusedssim = None; m.fusedssim_backward = None; "
                "sys.modules['fused_ssim_cuda'] = m; "
                "import runpy; runpy.run_module('scripts.infer', run_name='__main__')"
            )
            r2 = sh([PY, "-c", stub, "--config",
                     os.path.join(ROOT, "checkpoints",
                                  "querysplat_vggto_1B_512_paper.yaml"),
                     "--checkpoint", os.path.join(
                         ROOT, "checkpoints",
                         "querysplat_vggto_1B_512_paper.safetensors"),
                     "--input_folder", indir, "--output_dir", poutdir],
                    cwd=ROOT)
            if r2.returncode != 0:
                print(f"case {ci}: py FAILED\n{r2.stdout[-600:]}", file=sys.stderr)
                rows.append({"case": ci, "frames": len(case["frames"]),
                             "view_mode": case["view_mode"], "verdict": "PY_FAIL"})
                continue

        n = len(case["frames"])
        cpp_gt, py_gt, cpp_py = [], [], []
        for i in range(n):
            a = np.asarray(Image.open(f"{cprefix}_render_view{i}.png"),
                           dtype=np.float64)
            g = gt_crop(Image.open(paths[i]).convert("RGB"))
            cpp_gt.append(psnr(a, g))
            b = np.asarray(Image.open(os.path.join(poutdir, "rendered",
                                                   f"render_view{i}.png")),
                           dtype=np.float64)
            py_gt.append(psnr(b, g))
            cpp_py.append(psnr(a, b))
        cpp_gt_m, py_gt_m, cpp_py_m = (float(np.mean(cpp_gt)),
                                       float(np.mean(py_gt)),
                                       float(np.mean(cpp_py)))
        verdict = "OK"
        if cpp_gt_m < 15.0 or min(cpp_gt) < 12.0:
            verdict = "COLLAPSE"
        elif cpp_py_m < 40.0:
            verdict = "GAP"
        # per-case comparison figure into the charts tree (fixed diff scale)
        try:
            charts = os.path.join(QS, "benchmarks/charts/querysplat/broad")
            os.makedirs(charts, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(n, 4, figsize=(11.0, 2.6 * n), squeeze=False)
            for i in range(n):
                a = np.asarray(Image.open(f"{cprefix}_render_view{i}.png"),
                               dtype=np.float64)
                b = np.asarray(Image.open(os.path.join(
                    poutdir, "rendered", f"render_view{i}.png")), dtype=np.float64)
                g = gt_crop(Image.open(paths[i]).convert("RGB"))
                # per-pixel max over RGB: imshow ignores cmap/vmax for HxWx3
                # input, so a 3-channel diff would be clipped to [0,1] instead
                diff = np.clip(np.abs(a - b).max(axis=-1), 0, 8.0)
                for col, (img, title) in enumerate((
                        (a, "cpp"), (b, "official py"), (g, "gt"),
                        (diff, "|cpp-py| 0-8/255"))):
                    ax = axes[i][col]
                    if col == 3:
                        ax.imshow(img, cmap="inferno", vmin=0, vmax=8.0)
                    else:
                        # render/GT panels are 0-255 float64 — imshow would
                        # clip [0,1]-scaled floats to white; display as uint8
                        ax.imshow(np.clip(img, 0, 255).astype(np.uint8))
                    ax.set_xticks([]); ax.set_yticks([])
                    if col == 0:
                        ax.set_ylabel(f"v{i}\n{cpp_gt[i]:.1f} dB", fontsize=7)
                    if i == 0:
                        ax.set_title(title, fontsize=8)
            fig.suptitle(f"case {ci} ({case['view_mode']}, {n} views) — "
                         f"cpp-vs-GT {cpp_gt_m:.2f} dB, cpp-vs-py {cpp_py_m:.2f} dB",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(charts, f"case_{ci:03d}.png"), dpi=110)
            plt.close(fig)
        except Exception as e:  # figures must never fail the validation
            print(f"case {ci}: figure skipped ({e})")
        rows.append({"case": ci, "frames": n, "view_mode": case["view_mode"],
                     "cpp_vs_gt": round(cpp_gt_m, 2),
                     "min_cpp_vs_gt": round(min(cpp_gt), 2),
                     "py_vs_gt": round(py_gt_m, 2),
                     "cpp_vs_py": round(cpp_py_m, 2),
                     "verdict": verdict})
        print(rows[-1])

    # report
    ok = [r for r in rows if "cpp_vs_gt" in r]
    n_collapse = sum(1 for r in rows if r.get("verdict") == "COLLAPSE")
    n_gap = sum(1 for r in rows if r.get("verdict") == "GAP")
    lines = ["# Broad official-split validation (cpp f32 + f32-accuracy vs official py)",
             "",
             f"cases: {len(rows)} run / {len(rows)} attempted locally "
             f"(2 of 55 official scenes on disk); collapses: {n_collapse}; "
             f"alignment gaps (cpp-vs-py < 40 dB): {n_gap}",
             "",
             "Per-case comparison figures: "
             "[charts/querysplat/broad/](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad)",
             "",
             "| case | views | mode | cpp-vs-GT | min cpp-vs-GT | py-vs-GT | cpp-vs-py | verdict | figure |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "cpp_vs_gt" in r:
            lines.append(f"| {r['case']} | {r['frames']} | {r['view_mode']} "
                         f"| {r['cpp_vs_gt']} | {r['min_cpp_vs_gt']} "
                         f"| {r['py_vs_gt']} | {r['cpp_vs_py']} | {r['verdict']} | "
                         f"[fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_{r['case']:03d}.png) |")
        else:
            lines.append(f"| {r['case']} | {r.get('frames','')} "
                         f"| {r.get('view_mode','')} | — | — | — | — | {r['verdict']} | — |")
    if ok:
        lines.append("")
        lines.append(f"**means**: cpp-vs-GT "
                     f"{round(float(np.mean([x['cpp_vs_gt'] for x in ok])), 2)} dB, "
                     f"py-vs-GT {round(float(np.mean([x['py_vs_gt'] for x in ok])), 2)} dB, "
                     f"cpp-vs-py {round(float(np.mean([x['cpp_vs_py'] for x in ok])), 2)} dB")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
