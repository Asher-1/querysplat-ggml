# DL3DV protocol: querysplat-eval (C++) vs evaluate_jsons.py (official)

- official: `outputs/eval_protocol_paper/python` (torch 2.7.1+cu126, checkpoint `querysplat_vggto_1B_512_paper.safetensors`)
- cpp: `outputs/eval_protocol_paper/cpp_cuda` (device `cuda`, encoder `vggt-omega-1b-512-f32.gguf`, decoder `querysplat-paper-f32.gguf`)
- split_sha256 equal: **True**; protocol strings equal: **True**

## Structure

- files: 334 official / 334 cpp; only-official 0, only-cpp 0
- JSON key-order mismatches (metrics/timing/summary): 0

## Split summaries

| split | target | official PSNR / SSIM / LPIPS | cpp PSNR / SSIM / LPIPS | ΔPSNR |
| --- | --- | --- | --- | ---: |
| dl3dv_val_4views_medium | input | 27.0936 / 0.9109 / 0.0922 | 27.1010 / 0.9108 / 0.0922 | +0.0074 |
| dl3dv_val_4views_medium | interp | 23.9562 / 0.8528 / 0.1503 | 23.9597 / 0.8524 / 0.1505 | +0.0035 |
| dl3dv_val_4views_medium | extrap | 20.6126 / 0.7648 / 0.2289 | 20.6028 / 0.7643 / 0.2289 | -0.0098 |

## Per case

| split | case | group | PSNR off / cpp | SSIM off / cpp | LPIPS off / cpp | cams max rel\|Δ\| | render PSNR cpp↔off |
| --- | ---: | --- | --- | --- | --- | ---: | ---: |
| dl3dv_val_4views_medium | 0 | input | 27.7281 / 27.7463 | 0.9240 / 0.9239 | 0.0901 / 0.0906 | 3.78e-04 | 43.50 |
| dl3dv_val_4views_medium | 0 | interp | 27.3906 / 27.4385 | 0.9202 / 0.9201 | 0.0905 / 0.0914 | 3.78e-04 | 43.28 |
| dl3dv_val_4views_medium | 0 | extrap | 19.6395 / 19.6325 | 0.7340 / 0.7340 | 0.2521 / 0.2513 | 4.30e-04 | 38.24 |
| dl3dv_val_4views_medium | 1 | input | 21.8665 / 21.7754 | 0.8044 / 0.8009 | 0.1563 / 0.1571 | 1.18e-03 | 39.27 |
| dl3dv_val_4views_medium | 1 | interp | 18.6872 / 18.6149 | 0.7103 / 0.7084 | 0.2714 / 0.2725 | 1.18e-03 | 36.55 |
| dl3dv_val_4views_medium | 1 | extrap | 16.7571 / 16.7254 | 0.6609 / 0.6596 | 0.3033 / 0.3044 | 1.29e-03 | 36.74 |
| dl3dv_val_4views_medium | 83 | input | 28.8566 / 28.9006 | 0.9528 / 0.9532 | 0.0645 / 0.0640 | 5.99e-04 | 43.27 |
| dl3dv_val_4views_medium | 83 | interp | 23.7003 / 23.7065 | 0.8793 / 0.8794 | 0.1180 / 0.1184 | 5.99e-04 | 42.87 |
| dl3dv_val_4views_medium | 83 | extrap | 20.4524 / 20.4707 | 0.8264 / 0.8267 | 0.2125 / 0.2118 | 5.99e-04 | 37.01 |
| dl3dv_val_4views_medium | 87 | input | 28.6850 / 28.7198 | 0.9432 / 0.9437 | 0.0639 / 0.0635 | 4.10e-04 | 43.64 |
| dl3dv_val_4views_medium | 87 | interp | 28.1424 / 28.1275 | 0.9390 / 0.9391 | 0.0681 / 0.0679 | 4.10e-04 | 43.63 |
| dl3dv_val_4views_medium | 87 | extrap | 22.4100 / 22.4160 | 0.8194 / 0.8178 | 0.1664 / 0.1665 | 4.10e-04 | 40.20 |
| dl3dv_val_4views_medium | 131 | input | 23.6174 / 23.5871 | 0.8374 / 0.8373 | 0.1183 / 0.1193 | 6.19e-04 | 41.57 |
| dl3dv_val_4views_medium | 131 | interp | 22.0411 / 22.0731 | 0.8381 / 0.8391 | 0.1754 / 0.1742 | 5.82e-04 | 41.09 |
| dl3dv_val_4views_medium | 131 | extrap | 20.5782 / 20.5704 | 0.7446 / 0.7454 | 0.2539 / 0.2540 | 7.20e-04 | 39.38 |
| dl3dv_val_4views_medium | 157 | input | 29.3379 / 29.3363 | 0.9448 / 0.9446 | 0.0686 / 0.0681 | 4.92e-04 | 44.02 |
| dl3dv_val_4views_medium | 157 | interp | 22.6522 / 22.5881 | 0.8346 / 0.8326 | 0.1631 / 0.1644 | 5.02e-04 | 42.61 |
| dl3dv_val_4views_medium | 157 | extrap | 21.9153 / 21.9257 | 0.8073 / 0.8066 | 0.1716 / 0.1715 | 6.56e-04 | 39.96 |
| dl3dv_val_4views_medium | 179 | input | 28.4291 / 28.3797 | 0.9441 / 0.9438 | 0.0843 / 0.0840 | 4.10e-04 | 43.50 |
| dl3dv_val_4views_medium | 179 | interp | 26.6385 / 26.6215 | 0.9229 / 0.9226 | 0.0980 / 0.0973 | 3.95e-04 | 42.89 |
| dl3dv_val_4views_medium | 179 | extrap | 19.8470 / 19.8808 | 0.7279 / 0.7300 | 0.2847 / 0.2838 | 1.03e-03 | 37.62 |
| dl3dv_val_4views_medium | 195 | input | 26.4248 / 26.4395 | 0.9041 / 0.9044 | 0.1040 / 0.1042 | 4.29e-04 | 40.89 |
| dl3dv_val_4views_medium | 195 | interp | 22.5037 / 22.5177 | 0.8265 / 0.8255 | 0.1721 / 0.1708 | 3.69e-04 | 40.70 |
| dl3dv_val_4views_medium | 195 | extrap | 22.4332 / 22.3532 | 0.7837 / 0.7808 | 0.1842 / 0.1857 | 6.09e-04 | 38.98 |
| dl3dv_val_4views_medium | 237 | input | 28.0226 / 28.1475 | 0.9292 / 0.9304 | 0.0873 / 0.0860 | 8.02e-04 | 41.62 |
| dl3dv_val_4views_medium | 237 | interp | 20.2271 / 20.3270 | 0.7426 / 0.7429 | 0.2564 / 0.2567 | 7.10e-04 | 38.66 |
| dl3dv_val_4views_medium | 237 | extrap | 19.5080 / 19.4804 | 0.7490 / 0.7473 | 0.2580 / 0.2593 | 8.94e-04 | 36.70 |
| dl3dv_val_4views_medium | 244 | input | 27.9676 / 27.9779 | 0.9252 / 0.9254 | 0.0851 / 0.0850 | 4.10e-04 | 43.43 |
| dl3dv_val_4views_medium | 244 | interp | 27.5787 / 27.5820 | 0.9147 / 0.9145 | 0.0904 / 0.0910 | 3.66e-04 | 43.17 |
| dl3dv_val_4views_medium | 244 | extrap | 22.5857 / 22.5731 | 0.7952 / 0.7945 | 0.2019 / 0.2007 | 4.24e-04 | 39.98 |

## Worst case

- case-mean |ΔPSNR| 0.1249 dB, |ΔSSIM| 0.0034, |ΔLPIPS| 0.0015
- cameras.npz max relative |Δ| (per array, / max|official|) 1.286e-03
- gt/*.png max |Δ| 1 LSB
- renders cpp↔official PSNR: mean 40.83 dB, min 32.55 dB (120 views)

## Timing (median s per case-group)

| | reconstruction_seconds | target_camera_render_metrics_seconds |
| --- | ---: | ---: |
| official | 1.778 | 1.021 |
| cpp | 2.085 | 1.664 |
