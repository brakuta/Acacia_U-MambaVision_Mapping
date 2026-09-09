#!/usr/bin/env python3
"""Check that the software stack required by U-MV is functional.

    python tools/verify_install.py                 # imports, versions, CUDA, mamba_ssm
    python tools/verify_install.py --variant small # + build U-MV-small and run a forward pass
    python tools/verify_install.py --variant small --checkpoint Pretrained_Weights/U-MV-small_latest.pth
"""
import argparse
import importlib
import sys
import time

import _bootstrap  # noqa: F401

REQUIRED = ['torch', 'torchvision', 'mmengine', 'mmcv', 'mmseg', 'transformers', 'timm',
            'huggingface_hub', 'einops', 'numpy', 'cv2', 'PIL']
GEO = ['rasterio', 'geopandas', 'shapely', 'pyproj', 'osgeo']


def _ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, '__version__', 'ok')
    except Exception as e:  # noqa: BLE001
        return f'MISSING ({type(e).__name__}: {e})'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['tiny', 'small', 'base'], default=None)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--config', default=None, help='config to build (default: repo config of --variant)')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--device', default=None)
    a = p.parse_args()

    ok = True
    print(f'python {sys.version.split()[0]}')
    for m in REQUIRED:
        v = _ver(m)
        ok &= not v.startswith('MISSING')
        print(f'  {m:16s} {v}')
    print('geospatial (needed for tools/*geospatial_inference.py):')
    for m in GEO:
        print(f'  {m:16s} {_ver(m)}')

    import torch
    cuda = torch.cuda.is_available()
    gpu = f' | {torch.cuda.get_device_name(0)} | torch cuda {torch.version.cuda}' if cuda else ''
    print(f'CUDA available: {cuda}{gpu}')
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
        print('  mamba_ssm selective_scan_fn: ok')
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f'  mamba_ssm: MISSING or broken ({e}) -> MambaVision backbones cannot run')

    try:
        import mmcv.ops  # noqa: F401
        print('  mmcv.ops (compiled extension): ok')
    except Exception as e:  # noqa: BLE001
        print(f'  mmcv.ops: not importable ({e}); mmseg requires a full mmcv build')
        ok = False

    if a.variant:
        device = a.device or ('cuda:0' if cuda else 'cpu')
        from mmseg.registry import MODELS
        from mmseg.utils import register_all_modules
        from umv.compat import load_config
        register_all_modules(init_default_scope=True)
        cfg = load_config(a.config or f'configs/mambavision/U-MV-{a.variant}.py')
        if a.checkpoint:
            cfg.model.backbone['pretrained'] = False
        t0 = time.time()
        model = MODELS.build(cfg.model).to(device).eval()
        if a.checkpoint:
            from mmengine.runner import load_checkpoint
            load_checkpoint(model, a.checkpoint, map_location='cpu')
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'U-MV-{a.variant}: {n:.2f} M parameters, built in {time.time() - t0:.1f} s on {device}')
        x = torch.randn(1, 3, a.size, a.size, device=device)
        if cuda and device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            t0 = time.time()
            meta = dict(img_shape=(a.size, a.size), ori_shape=(a.size, a.size), pad_shape=(a.size, a.size))
            y = model.encode_decode(x, [meta])
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
