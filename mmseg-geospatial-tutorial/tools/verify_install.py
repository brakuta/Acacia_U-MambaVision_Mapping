#!/usr/bin/env python3
"""Check that the MMSegmentation stack is functional.

    python tools/verify_install.py                       # imports, versions, CUDA, mmcv.ops
    python tools/verify_install.py --config templates/project/configs/segformer_b2.py   # + build and forward pass
"""
import argparse
import importlib
import sys
import time

import _bootstrap  # noqa: F401

REQUIRED = ['torch', 'torchvision', 'mmengine', 'mmcv', 'mmseg', 'numpy', 'cv2', 'PIL', 'geoseg']
GEO = ['rasterio', 'geopandas', 'shapely', 'pyproj', 'osgeo']


def _ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, '__version__', 'ok')
    except Exception as e:  # noqa: BLE001
        return f'MISSING ({type(e).__name__}: {e})'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=None, help='build this config and run a forward pass')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--device', default=None)
    a = p.parse_args()
    ok = True
    print(f'python {sys.version.split()[0]}')
    for m in REQUIRED:
        v = _ver(m); ok &= not v.startswith('MISSING'); print(f'  {m:16s} {v}')
    print('geospatial (needed for tiling and inference tools):')
    for m in GEO:
        print(f'  {m:16s} {_ver(m)}')
    import torch
    cuda = torch.cuda.is_available()
    gpu = f' | {torch.cuda.get_device_name(0)} | torch cuda {torch.version.cuda}' if cuda else ''
    print(f'CUDA available: {cuda}{gpu}')
    try:
        import mmcv.ops  # noqa: F401
        print('  mmcv.ops (compiled extension): ok')
    except Exception as e:  # noqa: BLE001
        print(f'  mmcv.ops: not importable ({e}); MMSegmentation requires a full mmcv build'); ok = False
    if a.config:
        device = a.device or ('cuda:0' if cuda else 'cpu')
        from mmengine.config import Config
        from mmseg.registry import MODELS
        from mmseg.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(a.config)
        cfg.model.pretrained = None
        if isinstance(cfg.model.get('backbone'), dict):
            cfg.model.backbone.pop('init_cfg', None)
        t0 = time.time()
        model = MODELS.build(cfg.model).to(device).eval()
        in_ch = cfg.model.backbone.get('in_channels', 3) if isinstance(cfg.model.get('backbone'), dict) else 3
        n = sum(q.numel() for q in model.parameters()) / 1e6
        print(f'{a.config}: {n:.2f} M parameters, built in {time.time() - t0:.1f} s on {device}')
        x = torch.randn(1, in_ch, a.size, a.size, device=device)
        meta = dict(img_shape=(a.size, a.size), ori_shape=(a.size, a.size), pad_shape=(a.size, a.size))
        if cuda and device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            t0 = time.time(); y = model.encode_decode(x, [meta])
            if cuda and device.startswith('cuda'):
                torch.cuda.synchronize()
        msg = f'forward {tuple(x.shape)} -> {tuple(y.shape)} in {time.time() - t0:.2f} s'
        if cuda and device.startswith('cuda'):
            msg += f', peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB'
        print(msg)
    print('RESULT:', 'OK' if ok else 'PROBLEMS FOUND (see above)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
