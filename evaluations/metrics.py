# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""The paper's per-view PSNR, Gaussian-window SSIM and VGG LPIPS."""

import torch
from lpips import LPIPS
from skimage.metrics import structural_similarity


class ImageMetrics:
    def __init__(self, device):
        self.lpips = LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prediction, target):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("Metrics require matching [views, 3, height, width] images")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("Non-finite evaluation images")
        prediction, target = prediction.clamp(0, 1), target.clamp(0, 1)
        psnr = -10 * (prediction - target).square().mean((1, 2, 3)).log10()
        lpips = self.lpips(target, prediction, normalize=True)[:, 0, 0, 0]
        ssim = [
            float(structural_similarity(
                gt.cpu().numpy(), pred.cpu().numpy(), win_size=11,
                gaussian_weights=True, channel_axis=0, data_range=1.0,
            ))
            for gt, pred in zip(target, prediction)
        ]
        return {"psnr": psnr.cpu().tolist(), "ssim": ssim, "lpips": lpips.cpu().tolist()}
