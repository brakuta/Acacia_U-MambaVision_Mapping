#!/usr/bin/env python3
"""Download the ImageNet-pretrained encoder weights referenced by the tutorial configs
into ``work_dirs/pretrained`` so that training can run offline.

    python tools/download_zoo_weights.py                 # all
    python tools/download_zoo_weights.py resnet50_v1c mit_b2
"""
import argparse
import os
import urllib.request

import _bootstrap  # noqa: F401

WEIGHTS = {
    'resnet50_v1c': 'https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth',
    'resnet101_v1c': 'https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
    'mit_b1': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth',
    'mit_b2': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth',
    'swin_tiny': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth',
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('names', nargs='*', default=list(WEIGHTS), choices=list(WEIGHTS))
    p.add_argument('--dest', default='work_dirs/pretrained')
    a = p.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    for n in a.names:
        url = WEIGHTS[n]; out = os.path.join(a.dest, os.path.basename(url))
        if os.path.exists(out):
            print(f'[{n}] exists: {out}'); continue
        print(f'[{n}] {url} -> {out}')
        urllib.request.urlretrieve(url, out)
    print('Use them in configs with init_cfg=dict(type="Pretrained", checkpoint="<path>") or '
          'model.pretrained="<path>"; set HF/openmmlab URLs only when online.')


if __name__ == '__main__':
    main()
