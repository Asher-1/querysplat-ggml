# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the released QuerySplat models on the fixed DL3DV paper splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.cameras import align_cameras_to_reference
from evaluations.metrics import ImageMetrics
from scripts.infer import build_model_input, load_querysplat_checkpoint
from scripts.models import ModelInput, ModelInputDecoder, QuerySplat
from scripts.options import load_options_yaml
from scripts.utils.data import ImageTransform

GROUPS = ("input", "interp", "extrap")
METRICS = ("psnr", "ssim", "lpips")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolution(value):
    try:
        height, width = (int(part) for part in value.lower().split("x"))
        if min(height, width) < 11:
            raise ValueError
        return height, width
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected HEIGHTxWIDTH, with both dimensions >= 11") from error


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/DL3DV-Evaluation")
    parser.add_argument("--eval-jsons", type=Path, nargs="+",
                        default=sorted((ROOT / "evaluations/jsons_dl3dv").glob("*views*.json")))
    parser.add_argument("--output-dir", type=Path, required=True, help="A new or empty output directory")
    parser.add_argument("--gpus", help="Comma-separated visible GPU ids; otherwise use the first visible GPU")
    parser.add_argument("--batch-size", type=int, choices=[1], default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="First N cases of each selected JSON")
    selection.add_argument("--case-indices", type=int, nargs="+", help="Zero-based case indices in each JSON")
    parser.add_argument("--input-resolution", type=resolution, default=(256, 256),
                        help="Render and metric resolution; reconstruction uses the model resolution (default: 256x256)")
    parser.add_argument("--use-tto", action="store_true")
    parser.add_argument("--tto-n-steps", type=int, default=20)
    parser.add_argument("--tto-optimization-target", choices=["kv", "features"], default="kv")
    parser.add_argument("--tto-lr", type=float, default=5e-3)
    parser.add_argument("--tto-lpips-weight", type=float, default=0.05)
    parser.add_argument("--save-images", action="store_true", help="Save metric-resolution renders and GT")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.tto_n_steps < 0 or not math.isfinite(args.tto_lr) or args.tto_lr <= 0:
        parser.error("TTO steps must be non-negative and learning rate finite and positive")
    if not math.isfinite(args.tto_lpips_weight) or args.tto_lpips_weight < 0:
        parser.error("LPIPS weight must be finite and non-negative")
    if args.gpus and (not all(x.isdigit() for x in args.gpus.split(","))
                      or len(set(args.gpus.split(","))) != len(args.gpus.split(","))):
        parser.error("--gpus must list distinct non-negative GPU ids")
    return args


def selected_indices(config, args):
    samples = config.get("samples", [])
    if not samples or config.get("total_samples") != len(samples):
        raise ValueError("Evaluation JSON must contain its declared, nonempty sample list")
    indices = args.case_indices if args.case_indices is not None else list(range(len(samples)))[:args.limit]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(samples) for i in indices):
        raise ValueError("Case indices must be distinct and within the JSON sample list")
    expected = int(config["num_input_views"])
    if expected not in (2, 4, 12):
        raise ValueError("Supported evaluation input view counts are 2, 4 and 12")
    for index in indices:
        for group, count in (("input", expected), ("interp", 4), ("extrap", 4)):
            if len(samples[index][f"{group}_views"]) != count:
                raise ValueError(f"Case {index}: expected {count} {group} views")
    return indices


def image_path(entry, config, dataset_root):
    # Published JSONs are relative to the parent of DL3DV-Evaluation. The
    # override is the dataset root itself, which may have any local name.
    relative = Path(entry["path"])
    prefix = Path(config["dataset_path"]).name
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != prefix:
        raise ValueError(f"Unexpected dataset-relative image path: {relative}")
    return dataset_root.joinpath(*relative.parts[1:])


def preprocess_images(entries, config, dataset_root, model_resolution):
    images = []
    for entry in entries:
        path = image_path(entry, config, dataset_root)
        with Image.open(path) as image:
            images.append(np.array(image.convert("RGB"), copy=True))
    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
    images, _, _, _ = ImageTransform(model_resolution, model_resolution, max_crop=True).preprocess_images(images)
    return images


def metric_images(images, size):
    images = images.clamp(0, 1)
    if tuple(images.shape[-2:]) != tuple(size):
        images = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
    return images


def render_at_resolution(model, gaussians, decoder, size):
    """Rasterize at the scoring resolution with consistently scaled intrinsics."""
    original = model.gs.opt
    source_h, source_w = original.img_size
    target_h, target_w = size
    scale = decoder.intrinsics.new_tensor(
        [target_w / source_w, target_h / source_h, target_w / source_w, target_h / source_h]
    )
    decoder = ModelInputDecoder(decoder.cam_view, decoder.intrinsics * scale)
    try:
        model.gs.opt = original.evolve(img_size=size)
        images = model.render_gaussians(gaussians, decoder)["images_pred"][0]
    finally:
        model.gs.opt = original
    return images.clamp(0, 1), decoder


def case_dir(output, split, group, index, sample):
    return output / split / group / f"case_{index:06d}_{sample['scene_id'][:12]}"


def save_images(images, directory, entries):
    directory.mkdir(parents=True, exist_ok=True)
    for index, (tensor, entry) in enumerate(zip(images, entries)):
        pixels = (tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(pixels).save(directory / f"{index:02d}_frame_{entry['frame_index']:06d}.png")


def evaluate_case(model, metrics, options, args, config, split, index, sample, device):
    start = time.perf_counter()
    inputs = sample["input_views"]
    input_images = preprocess_images(inputs, config, args.dataset_root, options.img_size)
    encoder = build_model_input(input_images.to(device)).encoder
    with torch.no_grad():
        latent, input_cameras, input_intrinsics = model.forward_encoder_with_cameras(encoder)
    if args.use_tto:
        del latent
        gaussians = model.forward_tto_memory(
            ModelInput(encoder, ModelInputDecoder(input_cameras, input_intrinsics)),
            n_steps=args.tto_n_steps, lr=args.tto_lr, lpips_weight=args.tto_lpips_weight,
            save_steps=[], optimization_target=args.tto_optimization_target,
        )
    else:
        with torch.inference_mode():
            gaussians = model.forward_decoder(latent)
        del latent
    if not torch.isfinite(gaussians).all():
        raise ValueError(f"{split} case {index}: non-finite Gaussians")
    torch.cuda.synchronize(device)
    reconstruction_seconds = time.perf_counter() - start
    for group in GROUPS:
        group_start = time.perf_counter()
        targets = sample[f"{group}_views"]
        # Preserve the historical input+input pass when evaluating input views.
        all_entries = inputs + targets
        images = preprocess_images(all_entries, config, args.dataset_root, options.img_size).to(device)
        with torch.inference_mode():
            cameras, intrinsics, _ = model.vggt_encoder.predict_cameras(images[None], options.img_size)
            cameras = align_cameras_to_reference(cameras, input_cameras, len(inputs))
            decoder = ModelInputDecoder(cameras[:, len(inputs):], intrinsics[:, len(inputs):])
            prediction, decoder = render_at_resolution(model, gaussians, decoder, args.input_resolution)
            target = metric_images(images[len(inputs):], args.input_resolution)
            per_view = metrics(prediction, target)
        values = {name: float(np.mean(per_view[name])) for name in METRICS}
        if not all(math.isfinite(v) for values_list in per_view.values() for v in values_list):
            raise ValueError(f"{split} case {index}/{group}: non-finite metrics")
        destination = case_dir(args.output_dir, split, group, index, sample)
        write_json(destination / "metrics.json", {
            **values, "case_index": index, "scene_id": sample["scene_id"], "target_group": group,
            "per_view": per_view, "per_view_frames": [entry["frame_index"] for entry in targets],
            "input_frames": [entry["frame_index"] for entry in inputs],
            "metric_image_hw": list(args.input_resolution), "model_image_hw": list(options.img_size),
        })
        torch.cuda.synchronize(device)
        write_json(destination / "timing.json", {
            "reconstruction_seconds": reconstruction_seconds,
            "target_camera_render_metrics_seconds": time.perf_counter() - group_start,
        })
        np.savez(destination / "cameras.npz", input_cam_view=input_cameras.cpu().numpy(),
                 target_cam_view=decoder.cam_view.cpu().numpy(), intrinsics=decoder.intrinsics.cpu().numpy())
        if args.save_images:
            save_images(prediction, destination / "renders", targets)
            save_images(target, destination / "gt", targets)
    if args.use_tto:
        write_json(args.output_dir / split / f"case_{index:06d}_tto_losses.json", model.last_tto_losses)


def worker(rank, world_size, args):
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    options = load_options_yaml(args.config)
    model = QuerySplat(options).to(device)
    load_querysplat_checkpoint(model, str(args.checkpoint))
    model.eval()
    metrics = ImageMetrics(device)
    for path in args.eval_jsons:
        config = load_json(path)
        indices = selected_indices(config, args)[rank::world_size]
        current = options.evolve(num_input_views=config["num_input_views"])
        model.opt = model.gs.opt = model.vggt_encoder.opt = current
        for index in tqdm(indices, desc=f"GPU {rank} {path.stem}", position=rank):
            try:
                evaluate_case(model, metrics, current, args, config, path.stem, index,
                              config["samples"][index], device)
            except Exception as error:
                raise RuntimeError(f"Failed {path.name}, case {index}, scene {config['samples'][index]['scene_id']}") from error


def stats(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": len(values)}


def summarize(args):
    summaries = []
    for path in args.eval_jsons:
        config = load_json(path)
        indices = selected_indices(config, args)
        summary = {"split": path.stem, "num_input_views": config["num_input_views"],
                   "scale": config["scale"], "num_cases": len(indices), "groups": {}}
        for group in GROUPS:
            rows = [load_json(case_dir(args.output_dir, path.stem, group, index,
                                      config["samples"][index]) / "metrics.json") for index in indices]
            summary["groups"][group] = {
                "num_cases": len(rows),
                "metrics": {name: stats([v for row in rows for v in row["per_view"][name]]) for name in METRICS},
                "case_metrics": {name: stats([row[name] for row in rows]) for name in METRICS},
            }
        write_json(args.output_dir / path.stem / "summary.json", summary)
        summaries.append(summary)
    averages = []
    for views in (2, 4, 12):
        splits = [s for s in summaries if s["num_input_views"] == views]
        if {s["scale"] for s in splits} != {"small", "medium", "large"}:
            continue
        averages.append({"num_input_views": views, "num_cases": sum(s["num_cases"] for s in splits),
                         "groups": {group: {name: float(np.mean([
                             s["groups"][group]["metrics"][name]["mean"] for s in splits
                         ])) for name in METRICS} for group in GROUPS}})
    write_json(args.output_dir / "summary.json", {"splits": summaries, "scale_averages": averages})
    lines = ["# Evaluation results", "", "PSNR/SSIM: higher is better; LPIPS: lower is better.", "",
             "| Split | Cases | Target | PSNR | SSIM | LPIPS |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for s in summaries:
        for group in GROUPS:
            numbers = " | ".join(f"{s['groups'][group]['metrics'][name]['mean']:.4f}" for name in METRICS)
            lines.append(f"| {s['split']} | {s['num_cases']} | {group} | {numbers} |")
    for s in averages:
        for group in GROUPS:
            numbers = " | ".join(f"{s['groups'][group][name]:.4f}" for name in METRICS)
            lines.append(f"| {s['num_input_views']}views / scale average | {s['num_cases']} | {group} | {numbers} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    if not args.eval_jsons or len({p.stem for p in args.eval_jsons}) != len(args.eval_jsons):
        raise ValueError("Provide evaluation JSONs with unique split names")
    for path in args.eval_jsons:
        selected_indices(load_json(path), args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use a new output directory to avoid mixing runs: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    world_size = len(args.gpus.split(",")) if args.gpus else 1
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration["eval_jsons"] = [str(path.resolve()) for path in args.eval_jsons]
    configuration["split_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.eval_jsons}
    configuration["model_config"] = vars(load_options_yaml(args.config))
    configuration["camera_protocol"] = "input-only reconstruction; input+target prediction per group; similarity alignment to input cameras; no dataset poses"
    configuration["metric_protocol"] = "model-resolution preprocessing/reconstruction/cameras/TTO; direct metric-resolution rendering with scaled intrinsics; GT bilinear antialias resize; PSNR, skimage Gaussian SSIM, VGG LPIPS"
    configuration["reconstruct_at_model_resolution"] = True
    configuration["input_resolution_behavior"] = "metric_render_resolution"
    configuration["checkpoint_bytes"] = args.checkpoint.stat().st_size
    configuration["checkpoint_sha256"] = file_sha256(args.checkpoint)
    configuration["torch_version"] = torch.__version__
    write_json(args.output_dir / "run_config.json", configuration)
    if world_size > 1:
        # Prime pretrained LPIPS files once before starting independent workers.
        metric = ImageMetrics("cpu")
        del metric
        mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        worker(0, 1, args)
    summarize(args)


if __name__ == "__main__":
    main()
