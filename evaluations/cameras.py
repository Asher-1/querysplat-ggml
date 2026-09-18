# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Align independently predicted cameras to the input-only reconstruction frame."""

import torch


def align_cameras_to_reference(cam_view, reference_cam_view, num_reference_views):
    """Match the paper evaluator's orientation average and center-spread scale.

    Cameras are transposed world-to-camera matrices. Only the overlapping input
    cameras determine the similarity transform; dataset poses are never used.
    """
    if num_reference_views < 2 or reference_cam_view.shape[1] != num_reference_views:
        raise ValueError("Camera alignment requires at least two matching input cameras")
    c2w = torch.linalg.inv(cam_view.transpose(-2, -1))
    reference = torch.linalg.inv(reference_cam_view.transpose(-2, -1))
    source_ref = c2w[:, :num_reference_views]
    source_rotation = source_ref[..., :3, :3].float()
    target_rotation = reference[..., :3, :3].float()
    covariance = torch.einsum("bvij,bvkj->bik", target_rotation, source_rotation)
    u, _, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    determinant = torch.det(rotation)
    if torch.any(determinant < 0):
        correction = torch.ones((*rotation.shape[:-2], 3), device=rotation.device)
        correction[..., -1] = torch.where(determinant < 0, -1.0, 1.0)
        rotation = u @ torch.diag_embed(correction) @ vh
    rotation = rotation.to(c2w.dtype)
    source_centers, target_centers = source_ref[..., :3, 3], reference[..., :3, 3]
    source_mean, target_mean = source_centers.mean(1), target_centers.mean(1)
    source_var = (source_centers - source_mean[:, None]).square().sum((-1, -2)).clamp_min(1e-8)
    target_var = (target_centers - target_mean[:, None]).square().sum((-1, -2))
    scale = torch.sqrt(target_var / source_var)
    scale = torch.where(torch.isfinite(scale) & (target_var > 1e-8), scale, torch.ones_like(scale))
    translation = target_mean - scale[:, None] * torch.einsum("bij,bj->bi", rotation, source_mean)
    aligned = c2w.clone()
    aligned[..., :3, :3] = torch.einsum("bij,bvjk->bvik", rotation, c2w[..., :3, :3])
    aligned[..., :3, 3] = (
        scale[:, None, None] * torch.einsum("bij,bvj->bvi", rotation, c2w[..., :3, 3])
        + translation[:, None]
    )
    return torch.linalg.inv(aligned).transpose(-2, -1).to(cam_view.dtype)
