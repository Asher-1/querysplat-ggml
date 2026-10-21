#!/usr/bin/env python3
"""Render the cpp_ggml benchmark figures for the QuerySplat report.

Developer-only tool (matplotlib): the runtime chain stays pure C++ — this
script only post-processes artifacts produced by alignment_matrix.py and
e2e_querysplat.sh.

Inputs (all optional; missing pieces are skipped with a warning):
  <results>/timing_cpp_s<S>_<backend>_f32.json  stage timing, C++ (scene mean)
  <results>/timing_py_s<S>_cuda.json            stage timing, official infer
  <results>/quant_tiers.json        measured quant tier table (size/agreement)
  <work>/cpp_gaussians.bin          predicted gaussians, C++ (parity gate)
  <work>/spy/cpp_gaussians.bin      predicted gaussians, torch reference

Usage:
  python3 plot_benchmarks.py --out charts/querysplat [--work /tmp/qs_bench_s2]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CPP_LABEL = "cpp_ggml (C++)"
PY_LABEL = "official py (torch)"


def load_json(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    print(f"  [skip] missing {path}")
    return None


def fig_latency(results, out_dir, views):
    """Grouped bars per stage for the f32 (parity) tier on every backend plus
    the official python CUDA reference — same inputs, same stage boundaries.

    Sources: results/querysplat/timing_cpp_s<S>_<backend>_f32.json and
    timing_py_s<S>_cuda.json, written by alignment_matrix.py (scene means).
    """
    sources = []
    for name, fname, color in (
            ("cpp_ggml\nCUDA", f"timing_cpp_s{views}_cuda_f32.json", "#7a5fb5"),
            ("cpp_ggml\nVulkan", f"timing_cpp_s{views}_vulkan_f32.json", "#3f8f5f"),
            ("cpp_ggml\nCPU", f"timing_cpp_s{views}_cpu_f32.json", "#2b7bba"),
            ("official py\nCUDA", f"timing_py_s{views}_cuda.json", "#c9a227")):
        t = load_json(os.path.join(results, fname))
        if t:
            sources.append((name, t, color))
    if not sources:
        print("  [skip] no timing jsons")
        return
    stages = [
        ("camera_and_depth_seconds", "encoder (camera/depth/tokens)"),
        ("image_to_3dgs_seconds", "decoder + activations (3DGS)"),
        ("render_seconds", "render (input views)"),
    ]
    fig, axes = plt.subplots(1, len(stages), figsize=(4.0 * len(stages), 3.8))
    for ax, (key, title) in zip(axes, stages):
        vals = [s[1].get(key) or 0.0 for s in sources]
        bars = ax.bar([s[0] for s in sources], vals, color=[s[2] for s in sources])
        for b, v in zip(bars, vals):
            if v > 0:
                ax.annotate(f"{v:.2f}s" if v < 100 else f"{v:.0f}s",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("seconds (lower is better)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"QuerySplat per-stage latency, f32 tier, S={views}, 512x512 "
                 "(scene mean, n=1 each)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "e2e_latency_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_latency_matrix(results, out_dir, views):
    """Latency matrix: every cpp (backend x tier) row plus the official py CUDA
    reference — the direct "is the C++ chain faster than pytorch, and where
    does the time go" figure.

    Left   : end-to-end seconds (encoder + decoder + render; model load and
             IO excluded — the identical stage boundaries on both sides) with
             the "N x py" ratio annotated and the py reference dashed.
    Middle : per-stage breakdown heatmap for the cpp rows.
    Right  : enc+dec throughput (scenes/s) — the generative-stage analog of
             an LLM decode tok/s panel.

    Log x-axes (a deliberate deviation from the linear layout this figure is
    modeled on): the CPU rows sit ~29x past the GPU cluster, and linear bars
    would hide the 6-13% GPU-vs-py gap this figure exists to show.

    Sources: results/querysplat/timing_cpp_s<S>_<backend>_<tier>.json and
    timing_py_s<S>_cuda.json, written by alignment_matrix.py (scene means).
    """
    backends = (("cuda", "CUDA", "#7a5fb5"), ("vulkan", "Vulkan", "#3f8f5f"),
                ("cpu", "CPU", "#2b7bba"))
    tiers = ("f32", "f16", "q8_0")
    rows = []  # (label, color, encoder, decoder, render) — device then tier order
    for key, name, color in backends:
        for tier in tiers:
            t = load_json(os.path.join(
                results, f"timing_cpp_s{views}_{key}_{tier}.json"))
            if t:
                rows.append((f"{name} {tier}", color,
                             t["camera_and_depth_seconds"],
                             t["image_to_3dgs_seconds"], t["render_seconds"]))
    if not rows:
        print("  [skip] no cpp timing jsons for the latency matrix")
        return
    py = load_json(os.path.join(results, f"timing_py_s{views}_cuda.json"))
    py_row = None
    if py:
        py_row = ("official py CUDA", "#c9a227", py["camera_and_depth_seconds"],
                  py["image_to_3dgs_seconds"], py["render_seconds"])
    py_e2e = sum(py_row[2:]) if py_row else 0.0

    fig, (ax, axm, ax2) = plt.subplots(
        1, 3, figsize=(13.5, 0.42 * len(rows) + 2.4),
        gridspec_kw={"width_ratios": [1.25, 1.0, 0.9]})

    # ---- panel 1: end-to-end, every config + the python reference ----
    # sorted by latency; the color marks the py row (no cpp row uses gold)
    e2e_rows = sorted(rows + ([py_row] if py_row else []),
                      key=lambda r: r[2] + r[3] + r[4])
    ys = range(len(e2e_rows))
    e2e_vals = [r[2] + r[3] + r[4] for r in e2e_rows]
    ax.barh(list(ys), e2e_vals, color=[r[1] for r in e2e_rows])
    ax.set_yticks(list(ys), [r[0] for r in e2e_rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    for y, (_label, color, *stages) in zip(ys, e2e_rows):
        v = sum(stages)
        mult = f"  ({v / py_e2e:.2f}x py)" if py_e2e and color != py_row[1] else ""
        ax.text(v * 1.06, y, f"{v:.2f}s{mult}", va="center", fontsize=8)
    if py_e2e:
        ax.axvline(py_e2e, color="#777777", ls="--", lw=1.2)
    ax.set_xlim(1.0, max(e2e_vals) * 4)  # log: leave label room past the CPU bar
    ax.set_xlabel("encoder + decoder + render seconds (log, model load excluded)")
    ax.set_title("end-to-end latency\n(S=4, 512x512, scene mean, lower is better)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=9)

    # ---- panel 2: per-stage heatmap (cpp rows, seconds) ----
    data = np.array([[r[2], r[3], r[4], r[2] + r[3] + r[4]] for r in rows])
    im = axm.imshow(data, cmap="YlOrRd", aspect="auto")
    axm.set_xticks(range(4), ("encoder\n(cam/depth)", "decoder\n(3DGS)",
                              "render", "e2e"), fontsize=8)
    axm.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    for i in range(len(rows)):
        for j in range(4):
            v = data[i, j]
            axm.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.1f}",
                     ha="center", va="center", fontsize=8,
                     color="black" if v < data.max() * 0.55 else "white")
    axm.set_title("stage breakdown (s, scene mean)", fontsize=10)
    fig.colorbar(im, ax=axm, shrink=0.75, label="s")

    # ---- panel 3: enc+dec throughput (scenes/s) ----
    tp_rows = [(r[0], r[1], 1.0 / (r[2] + r[3])) for r in rows]
    if py_row:
        tp_rows.append((py_row[0], py_row[1], 1.0 / (py_row[2] + py_row[3])))
    tp_rows.sort(key=lambda r: r[2])
    ys = range(len(tp_rows))
    tp_vals = [r[2] for r in tp_rows]
    ax2.barh(list(ys), tp_vals, color=[r[1] for r in tp_rows])
    ax2.set_yticks(list(ys), [r[0] for r in tp_rows], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    for y, (_l, _c, v) in zip(ys, tp_rows):
        ax2.text(v * 1.06, y, f"{v:.3f}", va="center", fontsize=8)
    ax2.set_xlim(0.004, max(tp_vals) * 3)
    ax2.set_xlabel("enc+dec throughput (scenes/s, log)")
    ax2.set_title("throughput\n(higher is better)", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(f"QuerySplat latency matrix — C++ GGML backends x quant tiers vs "
                 f"official pytorch (RTX 4090 + i9-14900K, S={views}, 512x512, "
                 f"scene mean n=1)", fontsize=11, y=1.0)
    fig.tight_layout()
    path = os.path.join(out_dir, "latency_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_parity(work, out_dir):
    """Gaussian-level parity scatter: torch reference (x) vs C++ (y)."""
    cpp_p = os.path.join(work, "cpp_gaussians.bin")
    spy_p = os.path.join(work, "spy", "cpp_gaussians.bin")
    if not (os.path.isfile(cpp_p) and os.path.isfile(spy_p)):
        print("  [skip] parity scatter needs both gaussians bins")
        return
    a = np.fromfile(cpp_p, dtype=np.float32)
    b = np.fromfile(spy_p, dtype=np.float32)
    if a.size != b.size:
        print(f"  [skip] gaussians size mismatch ({a.size} vs {b.size})")
        return
    # stride = 11 + color_dim = 14 for the paper ckpt (sh degree 0, RGB);
    # channels 0..3 are x, y, z, opacity
    stride = 14
    a = a.reshape(-1, stride)
    b = b.reshape(-1, stride)
    rng = np.random.default_rng(0)
    idx = rng.choice(a.shape[0], size=min(40000, a.shape[0]), replace=False)
    channels = [(0, "x"), (1, "y"), (2, "z"), (3, "opacity")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (c, name) in zip(axes, channels):
        x, y = b[idx, c], a[idx, c]
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.scatter(x, y, s=1.5, alpha=0.25, color="#2b7bba")
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlabel("official py (torch)")
        ax.set_ylabel("cpp_ggml (C++)")
        ax.set_title(f"gaussian {name}  corr={corr:+.4f}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Gaussian parameter parity (C++ ggml vs official torch, "
                 "same input)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "parity_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_quant(results, out_dir):
    """Quant tiers: GGUF size vs measured render agreement (quant_tiers.json,
    written by alignment_matrix.py)."""
    data = load_json(os.path.join(results, "quant_tiers.json"))
    if not data or not data.get("tiers"):
        return
    tiers = data["tiers"]
    names = [t["tier"] for t in tiers]
    sizes = [t["encoder_gb"] + t["decoder_gb"] for t in tiers]
    agree = [t["mean_cpp_vs_py_db"] for t in tiers]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    b1 = ax1.bar(names, sizes, color="#6a9f58")
    for b, v in zip(b1, sizes):
        ax1.annotate(f"{v:.2f} GB", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=9)
    ax1.set_title("encoder + decoder GGUF size")
    ax1.set_ylabel("GB")
    ax1.grid(axis="y", alpha=0.3)
    b2 = ax2.bar(names, agree, color="#2a72d4")
    for b, v in zip(b2, agree):
        ax2.annotate(f"{v:.1f} dB", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=9)
    ax2.set_title("mean cpp-vs-py render PSNR (all backends x scenes)")
    ax2.set_ylabel("dB (higher is better)")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Quantization tiers shipped", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "quant_pareto.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None,
                    help="gate work dir (e2e_querysplat.sh OUT) for the parity scatter")
    ap.add_argument("--results", default=None,
                    help="benchmarks/results/querysplat dir (quant_tiers.json)")
    ap.add_argument("--out", required=True, help="charts output dir")
    ap.add_argument("--views", type=int, default=4,
                    help="view count S for the timing glob")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = args.results or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "querysplat")

    print("figures:")
    fig_latency(results, args.out, args.views)
    fig_latency_matrix(results, args.out, args.views)
    if args.work:
        fig_parity(args.work, args.out)
    fig_quant(results, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
