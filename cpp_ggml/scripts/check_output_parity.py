#!/usr/bin/env python3
"""Writer-parity gate for querysplat-cli --output_dir (dev tooling only).

Feeds the *cpp* tensors (querysplat-cli --dump-stage + --out-prefix --render
dumps) through the *official* scripts/output_utils.py writers / infer.py
output code and compares the resulting tree with the cpp --output_dir tree
file by file. Model numerics are excluded by construction (same tensors on
both sides), so any difference is a writer difference.

Comparison per file type:
  json/txt/ply  byte-identical
  png           decoded pixels identical (PIL vs stb zlib streams differ)
  npz           same members, dtypes, shapes, values (stored vs deflate ok)
  *_distribution.png  reported only (C++ re-plot, not matplotlib's raster)

usage: check_output_parity.py --cpp-tree DIR --dump DIR --prefix PREFIX
                              --input-folder DIR --config YAML --S N
                              [--thresholds 0.05 ...] [--target-points N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Only the writers run here: fused_ssim (TTO) and gsplat (rasterization) are
# imported at module load by the official code but never called, so stub them
# for torch-only environments (renders come from the cpp --render dump).
_fused = types.ModuleType("fused_ssim_cuda")
_fused.fusedssim = _fused.fusedssim_backward = None
sys.modules.setdefault("fused_ssim_cuda", _fused)
_gs_pkg, _gs_render = types.ModuleType("gsplat"), types.ModuleType("gsplat.rendering")
_gs_render.rasterization = None
_gs_pkg.rendering = _gs_render
sys.modules.setdefault("gsplat", _gs_pkg)
sys.modules.setdefault("gsplat.rendering", _gs_render)

from scripts.infer import list_input_images  # noqa: E402
from scripts.options import load_options_yaml  # noqa: E402
from scripts.output_utils import (  # noqa: E402
    opacity_threshold_suffix,
    preprocessed_input_frame_path,
    save_colored_pointcloud,
    save_gaussian_alpha_distribution,
    save_gaussian_scale_distribution,
    save_inference_timing,
    save_predicted_input_cameras,
    save_rgb_tensor_image,
    save_vggt_depth_pointcloud,
    save_vggt_input_depths,
)
from scripts.rendering.gs import GaussianRenderer  # noqa: E402


def load_f32(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def build_python_tree(a, out):
    dev = torch.device("cuda")
    S, H, W = a.S, 512, 512
    options = load_options_yaml(a.config).evolve(num_input_views=S)
    paths = list_input_images(a.input_folder)
    sources = [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]

    frames = load_f32(os.path.join(a.dump, "cpp_input_frames.f32"), (S, 3, H, W))
    w2c = load_f32(os.path.join(a.dump, "cpp_w2c.bin"), (S, 4, 4))
    intr = load_f32(os.path.join(a.dump, "cpp_intrinsics.bin"), (S, 4))
    pose = load_f32(os.path.join(a.dump, "cpp_pose_enc.bin"), (S, 9))
    depth = np.fromfile(os.path.join(a.dump, "cpp_depth.bin"), dtype=np.float32)
    conf = np.fromfile(os.path.join(a.dump, "cpp_depth_conf.bin"), dtype=np.float32)
    dh = dw = int(round((depth.size // S) ** 0.5))
    gauss = np.fromfile(os.path.join(a.dump, "cpp_gaussians.bin"), dtype=np.float32)
    stride = 11 + (3 if options.gaussian_sh_degree == 0 else 12)
    gauss = gauss.reshape(1, -1, stride)
    renders = load_f32(a.prefix + "_render_images.bin", (S, 3, H, W))

    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
    prediction = {
        "cam_view": t(np.transpose(w2c, (0, 2, 1))[None]),
        "intrinsics": t(intr[None]),
        "pose_enc": t(pose[None]),
        "depth": t(depth.reshape(1, S, 1, dh, dw)),
        "depth_conf": t(conf.reshape(1, S, 1, dh, dw)),
    }
    images = t(frames[None])
    gaussians = t(gauss)

    os.makedirs(os.path.join(out, "input_frames"), exist_ok=True)
    os.makedirs(os.path.join(out, "rendered"), exist_ok=True)
    for index, (_, source) in enumerate(sources):
        save_rgb_tensor_image(images[0, index],
                              os.path.join(out, preprocessed_input_frame_path(index, source)))
    for index, image in enumerate(torch.from_numpy(renders)):
        pixels = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(pixels).save(os.path.join(out, "rendered", f"render_view{index}.png"))
    save_predicted_input_cameras(prediction, sources, out)
    save_vggt_input_depths(prediction, sources, out)
    save_vggt_depth_pointcloud(prediction, images, sources, out, a.target_points)
    save_colored_pointcloud(gaussians, os.path.join(out, "pointcloud.ply"),
                            options.gaussian_save_opacity_threshold, options)
    save_gaussian_alpha_distribution(gaussians, out)
    save_gaussian_scale_distribution(gaussians, out, max_scale=options.gaussian_scale_cap)
    thresholds = sorted(set(a.thresholds))
    renderer = GaussianRenderer(options)
    for thr in thresholds:
        name = (f"gaussians{opacity_threshold_suffix(thr)}.ply"
                if len(thresholds) > 1 else "gaussians.ply")
        renderer.save_ply(gaussians, os.path.join(out, name), thr)
    cpp_timing = json.load(open(os.path.join(a.cpp_tree, "inference_timing.json")))
    timing = {
        "device": "cuda",
        "num_input_views": S,
        "camera_and_depth_seconds": cpp_timing["camera_and_depth_seconds"],
        "image_to_3dgs_seconds": cpp_timing["image_to_3dgs_seconds"],
        "render_seconds": cpp_timing["render_seconds"],
        "use_tto": False,
        "tto_n_steps": 0,
        "tto_lr": None,
        "tto_lpips_weight": None,
        "tto_save_steps": [],
        "gaussian_save_opacity_thresholds": thresholds,
    }
    save_inference_timing(timing, out)
    with open(os.path.join(out, "camera_info.txt"), "w", encoding="utf-8") as stream:
        stream.write("Input and render cameras are predicted by the frozen VGGT-Omega camera head.\n")
        stream.write(f"num_input_views: {S}\n")


def files(root):
    return sorted(os.path.relpath(os.path.join(d, f), root)
                  for d, _, fs in os.walk(root) for f in fs)


# The only values allowed to differ: torch.linalg.inv's f32 LU (c2w) and the
# cuBLAS einsum that consumes it (depth point cloud xyz) are library/ISA
# dependent; the cpp side rounds the exact inverse once. Everything derived
# from them is bounded in ulp (f32) or relative error (f64 statistics).
ULP_KEYS = {"c2w"}
REL_KEYS = {"mean_nearest_neighbor_distance": 1e-5}
MAX_ULP = 64


def ulp_f32(x, y):
    a = np.asarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(y, dtype=np.float32).view(np.int32).astype(np.int64)
    return int(np.abs(a - b).max()) if a.size else 0


def json_residual(x, y, key=None):
    """(ok, worst) — structural equality except whitelisted numeric keys."""
    if isinstance(x, dict) and isinstance(y, dict):
        if list(x) != list(y):
            return False, "keys"
        worst = {}
        for k in x:
            ok, w = json_residual(x[k], y[k], k)
            if not ok:
                return False, f"{k}: {w}"
            for wk, wv in w.items():
                worst[wk] = max(worst.get(wk, 0), wv)
        return True, worst
    if isinstance(x, list) and isinstance(y, list):
        if key in ULP_KEYS:
            u = ulp_f32(x, y)
            return u <= MAX_ULP, {key: u}
        if len(x) != len(y):
            return False, "len"
        worst = {}
        for a, b in zip(x, y):
            ok, w = json_residual(a, b, key)
            if not ok:
                return False, w
            for wk, wv in w.items():
                worst[wk] = max(worst.get(wk, 0), wv)
        return True, worst
    if x == y:
        return True, {}
    if key in REL_KEYS and isinstance(x, float) and isinstance(y, float):
        rel = abs(x - y) / max(abs(y), 1e-30)
        return rel <= REL_KEYS[key], {key: rel}
    return False, f"{key}: {x!r} != {y!r}"


def ply_residual(a, b):
    from plyfile import PlyData

    va, vb = PlyData.read(a)["vertex"].data, PlyData.read(b)["vertex"].data
    if va.dtype != vb.dtype or len(va) != len(vb):
        return False, "layout"
    worst = 0.0
    for n in va.dtype.names:
        if n in ("x", "y", "z"):
            # ulp at the scene scale (per-point ulps blow up near 0)
            ref = vb[n].astype(np.float64)
            d = np.abs(va[n].astype(np.float64) - ref).max() if len(ref) else 0.0
            scale = np.abs(ref).max() if len(ref) else 1.0
            worst = max(worst, float(d / (np.finfo(np.float32).eps * max(scale, 1e-30))))
        elif not np.array_equal(va[n], vb[n]):
            return False, f"{n} differs"
    return worst <= MAX_ULP, {"xyz_scene_ulp": round(worst, 2)}


def compare(cpp, py):
    fc, fp = files(cpp), files(py)
    ok = True
    for rel in sorted(set(fc) | set(fp)):
        if rel not in fc or rel not in fp:
            print(f"MISSING  {rel}  (cpp={rel in fc}, py={rel in fp})")
            ok = False
            continue
        a, b = os.path.join(cpp, rel), os.path.join(py, rel)
        ba, bb = open(a, "rb").read(), open(b, "rb").read()
        if ba == bb:
            print(f"BYTES    {rel}")
            continue
        if rel.endswith(".png"):
            ia = np.asarray(Image.open(a)).astype(np.int16)
            ib = np.asarray(Image.open(b)).astype(np.int16)
            if ia.shape == ib.shape and (ia == ib).all():
                print(f"PIXELS   {rel}")
                continue
            if "distribution" in rel:
                d = np.abs(ia - ib) if ia.shape == ib.shape else None
                print(f"REPLOT   {rel}  shape cpp={ia.shape} py={ib.shape}"
                      + ("" if d is None else
                         f" mean|d|={d.mean():.2f} px>32: {(d.max(-1) > 32).mean() * 100:.2f}%"))
                continue
            d = np.abs(ia - ib)
            print(f"DIFF     {rel}  max={d.max()} n={(d > 0).sum()}")
            ok = False
        elif rel.endswith(".npz"):
            za, zb = np.load(a), np.load(b)
            same = list(za.keys()) == list(zb.keys())
            resid = {}
            for k in zb.keys():
                if k not in za:
                    continue
                x, y = za[k], zb[k]
                if x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y):
                    continue
                if k in ULP_KEYS and x.dtype == y.dtype and x.shape == y.shape:
                    resid[k] = ulp_f32(x, y)
                    same &= resid[k] <= MAX_ULP
                    continue
                same = False
                print(f"  npz {k}: dtype {x.dtype}/{y.dtype} shape {x.shape}/{y.shape}")
            tag = "ARRAYS  " if not resid else "ULP     "
            print(f"{tag if same else 'DIFF    '} {rel}" + (f"  {resid}" if resid else ""))
            ok &= same
        elif rel.endswith(".json") and rel != "inference_timing.json":
            good, worst = json_residual(json.loads(ba), json.loads(bb))
            if good and not worst:
                good, worst = False, "same values, different formatting"
            print(f"{'ULP     ' if good else 'DIFF    '} {rel}  {worst}")
            ok &= good
        elif rel == "vggt_depth_pointcloud.ply":
            good, worst = ply_residual(a, b)
            print(f"{'ULP     ' if good else 'DIFF    '} {rel}  {worst}")
            ok &= good
        else:
            ta, tb = ba.decode("latin1").splitlines(), bb.decode("latin1").splitlines()
            first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
                         min(len(ta), len(tb)))
            nd = sum(x != y for x, y in zip(ta, tb)) + abs(len(ta) - len(tb))
            print(f"DIFF     {rel}  ({nd} lines differ, sizes {len(ba)}/{len(bb)})")
            if first < max(len(ta), len(tb)) and not rel.endswith(".ply"):
                print(f"    cpp: {ta[first][:160] if first < len(ta) else '<eof>'}")
                print(f"    py : {tb[first][:160] if first < len(tb) else '<eof>'}")
            ok = False
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpp-tree", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--input-folder", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--S", type=int, required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.05])
    p.add_argument("--target-points", type=int, default=200000)
    p.add_argument("--py-tree", default=None)
    a = p.parse_args()
    py = a.py_tree or a.cpp_tree.rstrip("/") + "_pywriters"
    build_python_tree(a, py)
    ok = compare(a.cpp_tree, py)
    print("WRITER PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
