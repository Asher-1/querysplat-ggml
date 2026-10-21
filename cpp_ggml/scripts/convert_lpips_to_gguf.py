#!/usr/bin/env python3
"""Convert the LPIPS-vgg (v0.1) metric weights to GGUF for querysplat-eval.

The official evaluator (evaluations/metrics.py) scores with
`lpips.LPIPS(net="vgg")`: torchvision's ImageNet VGG16 feature stack
(vgg16-397923af.pth, conv layers features.{0,2,5,7,10,12,14,17,19,21,24,26,28})
plus the five learned 1x1 "lin" heads shipped inside the lpips package
(weights/v0.1/vgg.pth, lin{0..4}.model.1.weight). Both are read here and
written as F32 (metric weights are not quantized: the metric is the
reference).

Usage:
  python3 convert_lpips_to_gguf.py models/gguf/lpips-vgg-f32.gguf \
      [--vgg ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth] \
      [--lin <site-packages>/lpips/weights/v0.1/vgg.pth]

Tensor keys: conv{i}.weight / conv{i}.bias (i = 0..12 in network order),
lin{l}.weight (l = 0..4). gguf-py reverses numpy shapes, so a Conv2d kernel
{OC, IC, 3, 3} lands as ggml ne {3, 3, IC, OC} and a lin kernel {1, C, 1, 1}
as ne {1, 1, C, 1}.
"""

import argparse
import os
from pathlib import Path

import torch
from gguf import GGUFWriter

VGG_CONV_INDICES = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
VGG_URL_NAME = "vgg16-397923af.pth"


def default_lin_path():
    try:
        import lpips
    except ImportError:
        return None
    return Path(lpips.__file__).parent / "weights" / "v0.1" / "vgg.pth"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output")
    p.add_argument("--vgg", default=str(Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
                                        / "hub/checkpoints" / VGG_URL_NAME))
    p.add_argument("--lin", default=None)
    args = p.parse_args()
    lin_path = Path(args.lin) if args.lin else default_lin_path()
    if lin_path is None or not lin_path.is_file():
        raise SystemExit("LPIPS lin weights not found: pass --lin <lpips>/weights/v0.1/vgg.pth")

    vgg = torch.load(args.vgg, map_location="cpu", weights_only=True)
    lin = torch.load(lin_path, map_location="cpu", weights_only=True)

    w = GGUFWriter(args.output, arch="lpips")
    w.add_string("general.name", "lpips-vgg-v0.1")
    w.add_string("lpips.vgg_source", VGG_URL_NAME)
    w.add_string("lpips.lin_source", "lpips/weights/v0.1/vgg.pth")
    for i, idx in enumerate(VGG_CONV_INDICES):
        w.add_tensor(f"conv{i}.weight", vgg[f"features.{idx}.weight"].float().numpy())
        w.add_tensor(f"conv{i}.bias", vgg[f"features.{idx}.bias"].float().numpy())
    for l in range(5):
        w.add_tensor(f"lin{l}.weight", lin[f"lin{l}.model.1.weight"].float().numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    n = len(VGG_CONV_INDICES) * 2 + 5
    print(f"wrote {args.output}: {n} tensors")


if __name__ == "__main__":
    main()
