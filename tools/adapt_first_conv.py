#!/usr/bin/env python3
"""Adapt an RGB-pretrained checkpoint to an N-band input layer.

The first convolution of an ImageNet model has weight shape (C_out, 3, k, k).
For multispectral inputs the layer needs (C_out, N, k, k).  This tool rewrites
that tensor so that the rest of the pretrained network can still be loaded:

* ``--mode mean``  (default) every new band receives the mean of the RGB filters;
* ``--mode map``   bands listed in ``--rgb-bands`` (0-based positions of the R, G, B
  bands in the new stack) copy the R, G, B filters, the others receive the mean.

Examples::

    # ResNet-50 V1c stem for a 10-band Sentinel-2 stack whose R,G,B are bands 2,1,0
    python tools/adapt_first_conv.py resnet50_v1c-2cccc1ad.pth resnet50_v1c_10band.pth \\
        --key stem.0.weight --bands 10 --mode map --rgb-bands 2 1 0
    # MiT-B2 patch embedding
    python tools/adapt_first_conv.py mit_b2_20220624-66e8bf70.pth mit_b2_10band.pth \\
        --key layers.0.0.projection.weight --bands 10

Use ``--list`` to print candidate keys (4-D tensors with 3 input channels).
"""
import argparse

import torch

import _bootstrap  # noqa: F401


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('src'); p.add_argument('dst', nargs='?')
    p.add_argument('--key', help='state-dict key of the first convolution weight')
    p.add_argument('--bands', type=int, help='number of input bands of the new model')
    p.add_argument('--mode', choices=['mean', 'map'], default='mean')
    p.add_argument('--rgb-bands', type=int, nargs=3, metavar='I', help='positions of R, G, B in the new stack (mode map)')
    p.add_argument('--list', action='store_true', help='list candidate keys and exit')
    a = p.parse_args()

    ckpt = torch.load(a.src, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    cands = [k for k, v in sd.items() if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 3]
    if a.list or not a.key:
        print('candidate first-conv keys:'); [print('  ', k, tuple(sd[k].shape)) for k in cands]
        return
    if a.key not in sd:
        raise SystemExit(f'{a.key} not in checkpoint; candidates: {cands}')
    w = sd[a.key].float()
    n = a.bands
    new = w.mean(dim=1, keepdim=True).repeat(1, n, 1, 1)
    if a.mode == 'map':
        if not a.rgb_bands:
            raise SystemExit('--rgb-bands required for --mode map')
        for src_ch, dst_ch in enumerate(a.rgb_bands):
            new[:, dst_ch] = w[:, src_ch]
    # keep the response magnitude comparable to the 3-band original
    new = new * (3.0 / n) if a.mode == 'mean' else new
    sd[a.key] = new.to(sd[a.key].dtype)
    out = a.dst or a.src.replace('.pth', f'_{n}band.pth')
    torch.save({'state_dict': sd} if 'state_dict' in (ckpt if isinstance(ckpt, dict) else {}) else sd, out)
    print(f'{a.key}: {tuple(w.shape)} -> {tuple(new.shape)}  saved to {out}')


if __name__ == '__main__':
    main()
