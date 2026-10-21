#!/usr/bin/env python3
"""Quant x backend alignment matrix: cpp-vs-py render agreement and per-stage
latency for every shipped GGUF tier on every compiled backend — the single
measurement pass behind every quality/latency table and chart in the docs.

For each scene, the official python chain (scripts/infer.py, paper
checkpoint, CUDA) is run once on the same input folder into
<work>/py_<scene>/ (cached), giving the reference renders AND the official
stage timing. Then for each (backend, quant) x scene:
  1. run the cpp chain on the same input views (f32-accuracy default mode);
  2. score cpp renders vs the official renders (cpp-vs-py, the ggml-fidelity
     metric) and vs the ground truth;
  3. collect the per-stage timing json.

Outputs:
  <out-dir>/alignment_matrix.{md,json}
  <out-dir>/alignment_cpp_vs_py_matrix.png    quality per scene
  <out-dir>/latency_quant_backend.png         enc+dec seconds per row
  <results>/timing_cpp_s<S>_<backend>_<quant>.json   scene-mean timing
  <results>/timing_py_s<S>_cuda.json                 scene-mean timing
  <results>/quant_tiers.json                         size + quality per tier

Usage:
  python3 alignment_matrix.py [--work /tmp/qs_align_matrix] \
      [--backends cuda vulkan cpu] [--quants f32 f16 q8_0] \
      [--scenes case0 case1 courtyard] [--merge]

--merge keeps the rows of the existing alignment_matrix.json whose
(backend, quant) pair is not re-run this time, so one tier/backend can be
re-measured without repeating the whole matrix.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QS = os.path.join(ROOT, "cpp_ggml")
RESULTS = os.path.join(QS, "benchmarks/results/querysplat")
PY = os.environ.get("PY", "/home/asher/anaconda3/envs/python3.13/bin/python")
CFG = os.path.join(ROOT, "checkpoints/querysplat_vggto_1B_512_paper.yaml")
CKPT = os.path.join(ROOT, "checkpoints/querysplat_vggto_1B_512_paper.safetensors")

# input folders hold exactly the listed frames (symlinks), so the official
# infer (sorted folder listing) and cpp --images see the same view order
SCENES = {
    "case0": ("outputs/dl3dv_case0/python_inputs",
              ["frame_00027.png", "frame_00038.png", "frame_00043.png", "frame_00052.png"]),
    "case1": ("outputs/dl3dv_case1/python_inputs",
              ["frame_00153.png", "frame_00155.png", "frame_00187.png", "frame_00190.png"]),
    "courtyard": ("outputs/courtyard/python_inputs",
                  ["DSC_0286.png", "DSC_0287.png", "DSC_0288.png", "DSC_0289.png"]),
}
STAGES = ("camera_and_depth_seconds", "image_to_3dgs_seconds", "render_seconds")
COLORS = {"f32": "#2a72d4", "f16": "#e08214", "q8_0": "#9467bd"}

# fused_ssim_cuda is import-time-required by scripts.models but only used by
# TTO — stub it so the inference-only env runs the official script unchanged
_INFER = (
    "import sys, types; m = types.ModuleType('fused_ssim_cuda'); "
    "m.fusedssim = None; m.fusedssim_backward = None; "
    "sys.modules['fused_ssim_cuda'] = m; "
    "import runpy; runpy.run_module('scripts.infer', run_name='__main__')"
)


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


def load_png(path):
    return np.asarray(Image.open(path), dtype=np.float64)


def py_reference(work, scene):
    """Official infer on the scene folder; cached by output dir."""
    src, _ = SCENES[scene]
    out = os.path.join(work, f"py_{scene}")
    if not os.path.isfile(os.path.join(out, "inference_timing.json")):
        r = sh([PY, "-c", _INFER, "--config", CFG, "--checkpoint", CKPT,
                "--input_folder", os.path.join(ROOT, src), "--output_dir", out],
               cwd=ROOT)
        if r.returncode != 0:
            print(f"py/{scene}: FAILED\n{(r.stdout + r.stderr)[-800:]}", file=sys.stderr)
            return None
    return out


def mean_timing(items):
    keys = [k for k in items[0] if k.endswith("_seconds")]
    return {k: round(float(np.mean([t[k] for t in items])), 4) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(QS, "benchmarks/charts/querysplat"))
    ap.add_argument("--work", default="/tmp/qs_align_matrix")
    ap.add_argument("--backends", nargs="+", default=["cuda", "vulkan", "cpu"])
    ap.add_argument("--quants", nargs="+", default=["f32", "f16", "q8_0"])
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--merge", action="store_true",
                    help="keep existing rows of (backend, quant) pairs not re-run")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    n_views = {len(SCENES[s][1]) for s in args.scenes}
    assert len(n_views) == 1, "timing files are keyed by one view count"
    S = n_views.pop()

    # ---- official python reference (renders + stage timing) ----
    py = {}
    for scene in args.scenes:
        out = py_reference(args.work, scene)
        if out is None:
            continue
        src, frames = SCENES[scene]
        vs_gt = [psnr(load_png(os.path.join(out, "rendered", f"render_view{i}.png")),
                      gt_crop(Image.open(os.path.join(ROOT, src, f)).convert("RGB")))
                 for i, f in enumerate(frames)]
        t = json.load(open(os.path.join(out, "inference_timing.json")))
        py[scene] = {"dir": out, "py_vs_gt": round(float(np.mean(vs_gt)), 2),
                     "timing": {k: t[k] for k in STAGES}}
        print(f"py/{scene}: {py[scene]['py_vs_gt']} dB vs GT, {py[scene]['timing']}")
    if py:
        with open(os.path.join(RESULTS, f"timing_py_s{S}_cuda.json"), "w") as f:
            json.dump({"side": "official py (torch)", "device": "cuda",
                       "num_input_views": S, "scenes": sorted(py),
                       **mean_timing([v["timing"] for v in py.values()])}, f, indent=2)

    # ---- cpp rows ----
    rows = []
    for backend in args.backends:
        cli = os.path.join(QS, f"build-{backend}", "bin", "querysplat-cli")
        if not os.path.isfile(cli):
            print(f"skip backend {backend}: no cli at {cli}")
            continue
        for quant in args.quants:
            enc = os.path.join(QS, "models/gguf", f"vggt-omega-1b-512-{quant}.gguf")
            dec = os.path.join(QS, "models/gguf", f"querysplat-paper-{quant}.gguf")
            if not (os.path.isfile(enc) and os.path.isfile(dec)):
                print(f"skip {backend}/{quant}: gguf tier missing")
                continue
            timings = []
            for scene in args.scenes:
                if scene not in py:
                    continue
                src, frames = SCENES[scene]
                prefix = os.path.join(args.work, f"{backend}_{quant}_{scene}")
                r = sh([cli, "--encoder", enc, "--decoder", dec,
                        "--images", ",".join(os.path.join(ROOT, src, f) for f in frames),
                        "--out-prefix", prefix, "--device", backend, "--render",
                        "--timing"])
                if r.returncode != 0:
                    print(f"{backend}/{quant}/{scene}: FAILED\n{r.stdout[-800:]}",
                          file=sys.stderr)
                    rows.append({"backend": backend, "quant": quant, "scene": scene,
                                 "status": "failed"})
                    continue
                cpp_vs_py, cpp_vs_gt = [], []
                for i, f in enumerate(frames):
                    a = load_png(f"{prefix}_render_view{i}.png")
                    cpp_vs_py.append(psnr(a, load_png(os.path.join(
                        py[scene]["dir"], "rendered", f"render_view{i}.png"))))
                    cpp_vs_gt.append(psnr(a, gt_crop(Image.open(
                        os.path.join(ROOT, src, f)).convert("RGB"))))
                timing = json.load(open(prefix + "_timing.json"))
                timings.append(timing)
                rows.append({
                    "backend": backend, "quant": quant, "scene": scene,
                    "status": "ok",
                    "cpp_vs_py": round(float(np.mean(cpp_vs_py)), 2),
                    "cpp_vs_gt": round(float(np.mean(cpp_vs_gt)), 2),
                    "min_cpp_vs_gt": round(float(np.min(cpp_vs_gt)), 2),
                    "infer_s": round(timing["camera_and_depth_seconds"]
                                     + timing["image_to_3dgs_seconds"], 2),
                    "render_s": round(timing["render_seconds"], 2),
                    "total_s": round(timing["total_seconds"], 2),
                })
                print(rows[-1])
            if timings:
                with open(os.path.join(RESULTS, f"timing_cpp_s{S}_{backend}_{quant}.json"),
                          "w") as f:
                    json.dump({"side": "cpp_ggml", "device": backend, "quant": quant,
                               "num_input_views": S, "scenes": args.scenes,
                               **mean_timing(timings)}, f, indent=2)

    if args.merge:
        old_json = os.path.join(args.out_dir, "alignment_matrix.json")
        if os.path.isfile(old_json):
            rerun = {(r["backend"], r["quant"]) for r in rows}
            kept = [r for r in json.load(open(old_json))["rows"]
                    if (r["backend"], r["quant"]) not in rerun]
            rows = kept + rows
    order_b = ["cuda", "vulkan", "cpu"]
    order_q = list(COLORS)
    rows.sort(key=lambda r: (order_b.index(r["backend"]) if r["backend"] in order_b else 9,
                             order_q.index(r["quant"]) if r["quant"] in order_q else 9,
                             args.scenes.index(r["scene"]) if r["scene"] in args.scenes else 9))
    backends = [b for b in order_b if any(r["backend"] == b for r in rows)]
    quants = [q for q in order_q if any(r["quant"] == q for r in rows)]

    ok = [r for r in rows if r.get("status") == "ok"]
    agg = {}
    for r in ok:
        agg.setdefault((r["backend"], r["quant"]), []).append(r)

    def m(items, key):
        return round(float(np.mean([x[key] for x in items])), 2)

    # ---- quant tier table (file size + measured quality, all backends) ----
    tiers = []
    for quant in quants:
        enc = os.path.join(QS, "models/gguf", f"vggt-omega-1b-512-{quant}.gguf")
        dec = os.path.join(QS, "models/gguf", f"querysplat-paper-{quant}.gguf")
        items = [r for r in ok if r["quant"] == quant]
        if not items or not os.path.isfile(dec):
            continue
        tiers.append({"tier": quant,
                      "encoder_gb": round(os.path.getsize(enc) / 1e9, 2),
                      "decoder_gb": round(os.path.getsize(dec) / 1e9, 2),
                      "mean_cpp_vs_py_db": m(items, "cpp_vs_py"),
                      "mean_cpp_vs_gt_db": m(items, "cpp_vs_gt")})
    with open(os.path.join(RESULTS, "quant_tiers.json"), "w") as f:
        json.dump({"comment": f"paper decoder + VGGT-Omega encoder, S={S}, 512x512; "
                   "means over every backend x scene of alignment_matrix.py",
                   "py_vs_gt_db": round(float(np.mean([v["py_vs_gt"] for v in py.values()])), 2)
                   if py else None,
                   "tiers": tiers}, f, indent=2)

    # ---- markdown ----
    lines = ["# Quant x backend alignment matrix (f32-accuracy default mode)",
             "",
             f"S={S} views, 512x512, paper decoder. cpp-vs-py = render agreement",
             "with the official torch chain run on the same inputs (higher is better;",
             "identical 8-bit PNGs read 99). GT = vs ground truth. Seconds are one",
             "fresh process per row (n=1, warm-up included, model load excluded).",
             "",
             "## Official python reference (CUDA)",
             "",
             "| scene | py-vs-GT | enc s | dec s | render s |",
             "|---|---|---|---|---|"]
    for scene, v in py.items():
        t = v["timing"]
        lines.append(f"| {scene} | {v['py_vs_gt']} | {t['camera_and_depth_seconds']:.2f} "
                     f"| {t['image_to_3dgs_seconds']:.2f} | {t['render_seconds']:.2f} |")
    lines += ["",
              "## cpp rows",
              "",
              "| backend | quant | scene | cpp-vs-py | cpp-vs-GT | min cpp-vs-GT | enc+dec s | render s | total s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") == "ok":
            lines.append(f"| {r['backend']} | {r['quant']} | {r['scene']} "
                         f"| {r['cpp_vs_py']} | {r['cpp_vs_gt']} | {r['min_cpp_vs_gt']} "
                         f"| {r['infer_s']} | {r['render_s']} | {r['total_s']} |")
        else:
            lines.append(f"| {r['backend']} | {r['quant']} | {r['scene']} "
                         f"| FAILED | — | — | — | — | — |")
    lines += ["",
              "## Means per (backend, quant)",
              "",
              "| backend | quant | mean cpp-vs-py | mean cpp-vs-GT | mean enc+dec s | mean render s |",
              "|---|---|---|---|---|---|"]
    for (b, q), items in sorted(agg.items()):
        lines.append(f"| {b} | {q} | {m(items, 'cpp_vs_py')} | {m(items, 'cpp_vs_gt')} "
                     f"| {m(items, 'infer_s')} | {m(items, 'render_s')} |")
    if py:
        pt = [v["timing"] for v in py.values()]
        lines.append(f"| official py cuda | f32 | — "
                     f"| {round(float(np.mean([v['py_vs_gt'] for v in py.values()])), 2)} "
                     f"| {round(float(np.mean([t['camera_and_depth_seconds'] + t['image_to_3dgs_seconds'] for t in pt])), 2)} "
                     f"| {round(float(np.mean([t['render_seconds'] for t in pt])), 2)} |")
    md = os.path.join(args.out_dir, "alignment_matrix.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out_dir, "alignment_matrix.json"), "w") as f:
        json.dump({"py": {k: {kk: vv for kk, vv in v.items() if kk != "dir"}
                          for k, v in py.items()}, "rows": rows}, f, indent=1)
    print(f"wrote {md}")

    # ---- quality chart per scene: py-vs-GT (red) then every cpp bar ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenes = [s for s in args.scenes if s in py]
    fig, axes = plt.subplots(1, len(scenes), figsize=(5.4 * len(scenes), 4.0),
                             squeeze=False)
    for si, scene in enumerate(scenes):
        ax = axes[0][si]
        labels, vals, cols, agrees = ["py\n(torch)"], [py[scene]["py_vs_gt"]], ["#d62728"], [None]
        for backend in backends:
            for quant in quants:
                hit = [r for r in ok if r["backend"] == backend
                       and r["quant"] == quant and r["scene"] == scene]
                if hit:
                    labels.append(f"{backend[:2]}\n{quant}")
                    vals.append(hit[0]["cpp_vs_gt"])
                    cols.append(COLORS.get(quant, "#999"))
                    agrees.append(hit[0]["cpp_vs_py"])
        bars = ax.bar(range(len(vals)), vals, color=cols)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylim(min(vals) - 2.0, max(vals) + 2.0)
        for bar, v, ag in zip(bars, vals, agrees):
            txt = f"{v:.1f}" if ag is None else f"{v:.1f}\n({ag:.1f})"
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.12, txt,
                    ha="center", fontsize=5.6)
        ax.set_title(scene, fontsize=10)
        ax.set_ylabel("vs ground truth PSNR (dB) — (cpp-vs-py) annotated", fontsize=7)
    fig.suptitle("render quality per quant tier and backend; red = official torch "
                 "reference; (cpp-vs-py agreement) annotated on cpp bars", fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "alignment_cpp_vs_py_matrix.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")

    # ---- latency chart: enc+dec seconds per row, scene mean ----
    bars_ = [(f"{b}\n{q}", m(items, "infer_s"), COLORS.get(q, "#999"))
             for (b, q), items in sorted(agg.items())]
    if py:
        pt = [v["timing"] for v in py.values()]
        bars_.append(("official py\ncuda", round(float(np.mean(
            [t["camera_and_depth_seconds"] + t["image_to_3dgs_seconds"] for t in pt])), 2),
            "#808080"))
    if bars_:
        fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(bars_), 3.6))
        bars = ax.bar(range(len(bars_)), [b[1] for b in bars_], color=[b[2] for b in bars_])
        ax.set_xticks(range(len(bars_)))
        ax.set_xticklabels([b[0] for b in bars_], fontsize=7)
        ax.set_yscale("log")
        for bar, (_, v, _) in zip(bars, bars_):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.05, f"{v:.2f}",
                    ha="center", fontsize=6)
        ax.set_ylabel(f"encoder + decoder seconds (S={S}, log)", fontsize=8)
        ax.set_title(f"stage latency, mean of {len(scenes)} scenes (n=1 each)", fontsize=10)
        fig.tight_layout()
        out_png2 = os.path.join(args.out_dir, "latency_quant_backend.png")
        fig.savefig(out_png2, dpi=140)
        plt.close(fig)
        print(f"wrote {out_png2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
