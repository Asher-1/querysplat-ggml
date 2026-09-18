<h1 align="center">QuerySplat: Decoupling Geometry and Appearance Representations in 3DGS Prediction</h1>

<p align="center">Official implementation of <strong>QuerySplat</strong>.</p>

<p align="center"><a href="https://inspatio.github.io/querysplat/">Project Page</a> &nbsp;|&nbsp; <a href="https://huggingface.co/inspatio/querysplat">Hugging Face</a> &nbsp;|&nbsp; <a href="https://arxiv.org/abs/2608.01186">arXiv</a></p>

<p align="center"><img src="assets/teaser.png" alt="QuerySplat teaser" width="100%"></p>

This repository contains the official inference implementation of QuerySplat. The release includes custom-image preprocessing, 3D Gaussian prediction and rendering, VGGT-Omega camera/depth prediction, and optional test-time optimization (TTO).

## Updates

- **2026-09-18**: Added the paper checkpoint, DL3DV evaluation code and splits, and evaluation results and protocol documentation.
- **2026-08-04**: Initial release of QuerySplat inference code and pretrained weights.

## Installation

QuerySplat requires Linux, a CUDA-capable NVIDIA GPU, and CUDA-enabled PyTorch. The release has been tested with Python 3.12, PyTorch 2.11, and CUDA 12.8.

```bash
git clone https://github.com/inspatio/QuerySplat.git
cd QuerySplat

conda create -n querysplat python=3.12 -y
conda activate querysplat

# Tested configuration: PyTorch 2.11.0 + CUDA 12.8.
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install -U huggingface_hub
```

`fused-ssim` is built from a pinned upstream source revision and uses the PyTorch/CUDA installation from the preceding step. The CUDA extensions used by `gsplat` and `fused-ssim` must be compatible with your PyTorch and CUDA installation. LPIPS may download its pretrained VGG16 weights on first use.

## Checkpoints

QuerySplat and VGGT-Omega weights are distributed separately. QuerySplat loads its geometry/appearance parameters from the QuerySplat checkpoint and loads the frozen VGGT-Omega aggregator, camera head, and depth head from the original VGGT-Omega checkpoint.

| Component | Download | Required path |
| --- | --- | --- |
| QuerySplat | [inspatio/querysplat](https://huggingface.co/inspatio/querysplat) | `checkpoints/querysplat_vggto_1B_512_8192.safetensors` |
| VGGT-Omega 1B/512 | [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) | `checkpoints/vggt_omega_1b_512.pt` |
| Inference config | Included in this repository | `checkpoints/querysplat_vggto_1B_512_8192.yaml` |

```bash
mkdir -p checkpoints

hf download inspatio/querysplat \
  querysplat_vggto_1B_512_8192.safetensors \
  --local-dir checkpoints

hf download facebook/VGGT-Omega \
  vggt_omega_1b_512.pt \
  --local-dir checkpoints

sha256sum -c SHA256SUMS
```

The checkpoint directory must contain:

```text
checkpoints/
├── querysplat_vggto_1B_512_8192.safetensors
├── querysplat_vggto_1B_512_8192.yaml
└── vggt_omega_1b_512.pt
```

## Inference

Place any number of images from one scene in `--input_folder`. Run inference with TTO:

```bash
python -m scripts.infer \
  --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --checkpoint checkpoints/querysplat_vggto_1B_512_8192.safetensors \
  --input_folder data/my_scene \
  --output_dir outputs/my_scene \
  --use_tto
```

Omit `--use_tto` to run the feed-forward model without test-time optimization.

### Important Options

- `--tto_n_steps`: Number of TTO optimization steps. Default: `20`.
- `--tto_lr`: TTO learning rate. Default: `5e-3`.
- `--tto_lpips_weight`: LPIPS weight in the TTO reconstruction objective. Default: `0.05`.
- `--tto_save_step STEP [STEP ...]`: Save additional Gaussian PLY files at the requested TTO steps.
- `--gaussian_save_opacity_threshold VALUE [VALUE ...]`: Opacity thresholds for Gaussian PLY export. Multiple values produce one PLY per threshold. Default: `0.05`.
- `--save_gaussian_alpha_distribution`: Save Gaussian opacity distribution statistics and plots.
- `--save_gaussian_scale_distribution`: Save Gaussian scale distribution statistics and plots.
- `--save_predicted_input_cameras`: Export predicted input cameras as JSON and NPZ files.
- `--save_vggt_input_depths`: Export per-view VGGT-Omega depth and confidence products.
- `--save_vggt_depth_pointcloud`: Export a colored point cloud reconstructed from VGGT-Omega depth predictions.
- `--vggt_depth_pointcloud_target_points N`: Target number of depth point-cloud samples; required with `--save_vggt_depth_pointcloud`.

## Evaluations

We evaluate on [DL3DV-Evaluation](https://huggingface.co/datasets/DL3DV/DL3DV-Evaluation/tree/main) using 2, 4, and 12 input views across small, medium, and large temporal windows. The nine fixed evaluation splits in `evaluations/jsons_dl3dv/` each contain 300 cases, with four interpolation and four extrapolation target views per case.

We provide two checkpoints with different strengths. The paper uses `querysplat_vggto_1B_512_paper` ([download](https://huggingface.co/inspatio/querysplat/resolve/main/querysplat_vggto_1B_512_paper.safetensors)), which uses SH degree 0 and achieves better evaluation metrics. The previously released `querysplat_vggto_1B_512_8192` ([download](https://huggingface.co/inspatio/querysplat/resolve/main/querysplat_vggto_1B_512_8192.safetensors)) uses SH degree 1 and produces visually better scene reconstructions. Use the paper checkpoint to reproduce the reported results, or the previous release when visual reconstruction quality is the priority.

The table reports interpolation results without TTO, averaged over the small, medium, and large splits. Higher PSNR/SSIM and lower LPIPS are better. **Bold** and <u>underlined</u> values indicate the best and second-best results between the two checkpoints.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Pose-free</th>
      <th colspan="3">2 views</th>
      <th colspan="3">4 views</th>
      <th colspan="3">12 views</th>
    </tr>
    <tr>
      <th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th>
      <th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th>
      <th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>querysplat_vggto_1B_512_8192</code></td><td align="center">✓</td>
      <td><u>21.2392</u></td><td><u>0.6973</u></td><td><u>0.2757</u></td>
      <td><u>24.3436</u></td><td><u>0.7937</u></td><td><u>0.2013</u></td>
      <td><u>23.6938</u></td><td><u>0.7660</u></td><td><u>0.2441</u></td>
    </tr>
    <tr>
      <td><code>querysplat_vggto_1B_512_paper</code></td><td align="center">✓</td>
      <td><strong>21.3888</strong></td><td><strong>0.6990</strong></td><td><strong>0.2585</strong></td>
      <td><strong>24.5765</strong></td><td><strong>0.8002</strong></td><td><strong>0.1843</strong></td>
      <td><strong>23.7005</strong></td><td><strong>0.7685</strong></td><td><strong>0.2297</strong></td>
    </tr>
  </tbody>
</table>

> [!IMPORTANT]
> In all paper evaluations, **pose-free methods** use their own pose estimator in two forward passes. The first uses only input views to reconstruct the scene and establish reference cameras. The second uses input and target views to predict target-camera parameters, aligned to the reconstruction through the shared input cameras. Target images are used only for camera estimation and scoring; reconstruction and TTO use only input views. **Pose-required methods** directly use the dataset-provided target-camera parameters.

To evaluate all nine splits with the paper checkpoint, run the following from the repository root. Set `--dataset-root` to the extracted DL3DV-Evaluation directory containing the scene folders. Reconstruction uses 512×512 images; rendering and scoring use 256×256.

```bash
python -m evaluations.evaluate_jsons \
  --config checkpoints/querysplat_vggto_1B_512_paper.yaml \
  --checkpoint checkpoints/querysplat_vggto_1B_512_paper.safetensors \
  --dataset-root /path/to/DL3DV-Evaluation \
  --input-resolution 256x256 \
  --gpus 0,1,2,3 \
  --output-dir outputs/evaluations/paper
```

For the previous checkpoint, replace `paper` with `8192` in the config/checkpoint paths and use a separate output directory. For TTO20 or TTO50, append `--use-tto --tto-n-steps 20 --tto-optimization-target kv` or change the step count to `50`.

## Acknowledgements

QuerySplat builds on and benefits from [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) for image encoding, camera prediction, and depth prediction, and [TokenGS](https://github.com/nv-tlabs/TokenGS) for important implementation foundations and references.

## Citation

```bibtex
@article{li2026querysplat,
  title={QuerySplat: Decoupling Geometry and Appearance Representations in 3DGS Prediction},
  author={Li, Yinglong and Shen, Donghui and Zhang, Xiaoyu and Ye, Zhichao and Wu, Hongyu and Hao, Aimin and Zhang, Guofeng and Liu, Haomin},
  journal={arXiv preprint arXiv:2608.01186},
  year={2026},
  url={https://arxiv.org/abs/2608.01186}
}
```

## License

Copyright (c) 2026 Inspatio. All rights reserved.

The QuerySplat-authored portions of this release are provided under the Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details. The vendored VGGT-Omega/DINOv3 source is provided under the [FAIR Noncommercial Research License](third_party/vggt_omega/LICENSE) and retains its original upstream notices.
