#!/usr/bin/env python3
"""Dump the FULL gaussian tensor exactly as scripts.infer produces it
(same preprocessing, same model, same forward path) so it can be diffed
against the spy harness's cpp_gaussians.bin.

Run with the python3.13 env (gsplat/torch present). Official code path only
— no stubs needed beyond fused_ssim (TTO-only).
"""

import sys, types
m = types.ModuleType('fused_ssim_cuda')
m.fusedssim = None; m.fusedssim_backward = None
sys.modules['fused_ssim_cuda'] = m
sys.path.insert(0, '.')
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from scripts.models import ModelInput, ModelInputDecoder, ModelInputEncoder, QuerySplat
from scripts.options import load_options_yaml
from scripts.utils.data import ImageTransform
import torchvision.transforms as transforms

cfg = 'checkpoints/querysplat_vggto_1B_512_paper.yaml'
ckpt = 'checkpoints/querysplat_vggto_1B_512_paper.safetensors'
frames = sys.argv[1:-1]  # 480x270 original frames, sorted; last arg = out bin

options = load_options_yaml(cfg)
tr = ImageTransform(crop_size=options.img_size, sample_size=options.img_size, max_crop=True)
imgs = []
for p in frames:
    im = np.array(Image.open(p).convert('RGB'), copy=True)
    imgs.append(torch.from_numpy(im).permute(2, 0, 1).float() / 255.0)
images = torch.stack(imgs)
options = options.evolve(num_input_views=len(frames))
images_t = tr.preprocess_images(images)[0]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model_input = ModelInput(
    encoder=ModelInputEncoder(images_rgb=normalize(images_t.clone()).unsqueeze(0),
                              images_rgb_unnormalized=images_t.unsqueeze(0)),
    decoder=ModelInputDecoder(cam_view=torch.empty(0), intrinsics=torch.empty(0)),
)

device = torch.device('cuda')
model = QuerySplat(options).to(device)
sd = load_file(ckpt, device='cpu')
model.load_state_dict(sd, strict=False)
model.eval()
del sd
model_input = ModelInput(
    encoder=ModelInputEncoder(images_rgb=model_input.encoder.images_rgb.to(device),
                              images_rgb_unnormalized=model_input.encoder.images_rgb_unnormalized.to(device)),
    decoder=model_input.decoder,
)
# ---- spy-equivalent seam hooks (write to /tmp/G_infer_seams) ----
import os
seam_dir = '/tmp/G_infer_seams'
os.makedirs(seam_dir, exist_ok=True)
dumps = {}
def _save(name, t):
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    arr.tofile(os.path.join(seam_dir, name + '.bin'))
    print('  seam', name, arr.shape)

from scripts.models.vggt_encoder import VGGTEncoder
orig_tokens_from_layers = VGGTEncoder._tokens_from_layers
def tokens_from_layers(self, output_by_layer, patch_start, batch_size):
    out = orig_tokens_from_layers(self, output_by_layer, patch_start, batch_size)
    dumps['cpp_tokens'] = out.tokens
    return out
VGGTEncoder._tokens_from_layers = tokens_from_layers

orig_decode_cameras = VGGTEncoder._decode_cameras
def decode_cameras(self, final_layer, image_hw):
    return orig_decode_cameras(self, final_layer, image_hw)
VGGTEncoder._decode_cameras = decode_cameras

model.non_rgb_activation_head.deconv.register_forward_hook(
    lambda m, inp, out: dumps.__setitem__('cpp_deconv_raw', out))
model.patch_embed.proj.register_forward_hook(
    lambda m, inp, out: dumps.__setitem__('cpp_pe_conv_pre', out))
model.post_rgb_delta_head.register_forward_hook(
    lambda m, inp, out: dumps.__setitem__('cpp_color_raw', out))

with torch.inference_mode():
    gaussians = model.forward_gaussians(model_input)
for key in ('cpp_pe_conv_pre',):
    t = dumps.get(key)
    if t is not None and len(t.shape) == 4:
        dumps[key] = t.flatten(2).transpose(1, 2).contiguous()
for name, tensor in dumps.items():
    _save(name, tensor)
dumps['cpp_gaussians'] = gaussians[0]
_save('cpp_gaussians', gaussians[0])

g = gaussians.detach().to(torch.float32).contiguous().cpu().numpy()
print('gaussians shape:', g.shape)
out = sys.argv[-1] if sys.argv[-1].endswith('.bin') else '/tmp/G_infer_full.bin'
g.tofile(out)
print('wrote', out)
