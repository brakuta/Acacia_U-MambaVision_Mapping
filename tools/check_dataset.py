#!/usr/bin/env python3
"""Validate a U-MV dataset folder before training or evaluation.

Checks, per split: image/mask basename correspondence, band count, dtype,
tile size, mask values (expects {0, 1} plus optional 255), and reports
ArcGIS side-car files (*.aux.xml, *.ovr, *.xml) that are ignored by the
loader but inflate copies.

    python tools/check_dataset.py /data                           # default splits
    python tools/check_dataset.py /data --splits train val test2 Generalizability --sample 200
"""
import argparse
import collections
import random
from pathlib import Path

import _bootstrap  # noqa: F401

SIDECAR_SUFFIXES = ('.aux.xml', '.ovr', '.xml', '.tfw', '.cpg', '.dbf')


def check_split(root: Path, split: str, suffix: str, sample: int, seed: int = 0) -> bool:
    import numpy as np
    import rasterio

    img_dir, ann_dir = root / 'img_dir' / split, root / 'ann_dir' / split
    ok = True
    if not img_dir.is_dir() or not ann_dir.is_dir():
        print(f'[{split}] MISSING: {img_dir if not img_dir.is_dir() else ann_dir}')
        return False
    imgs = {p.name: p for p in img_dir.iterdir() if p.suffix.lower() == suffix and p.is_file()}
    anns = {p.name: p for p in ann_dir.iterdir() if p.suffix.lower() == suffix and p.is_file()}
    side = [p for d in (img_dir, ann_dir) for p in d.iterdir() if p.name.lower().endswith(SIDECAR_SUFFIXES)]
    only_img, only_ann = sorted(set(imgs) - set(anns)), sorted(set(anns) - set(imgs))
    pairs = sorted(set(imgs) & set(anns))
    print(f'[{split}] images={len(imgs)} masks={len(anns)} pairs={len(pairs)} sidecars={len(side)}')
    if only_img or only_ann:
        ok = False
        print(f'   !! {len(only_img)} images without mask (e.g. {only_img[:3]}), '
              f'{len(only_ann)} masks without image (e.g. {only_ann[:3]})')
    if not pairs:
        return False

    rng = random.Random(seed)
    subset = pairs if len(pairs) <= sample else rng.sample(pairs, sample)
    shapes, img_dtypes, bands, mask_vals, mask_dtypes, mask_bands, mismatch = (
        collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(),
        collections.Counter(), collections.Counter(), [])
    for name in subset:
        with rasterio.open(imgs[name]) as im, rasterio.open(anns[name]) as an:
            shapes[(im.width, im.height)] += 1
            img_dtypes[im.dtypes[0]] += 1
            bands[im.count] += 1
            mask_dtypes[an.dtypes[0]] += 1
            mask_bands[an.count] += 1
            if (im.width, im.height) != (an.width, an.height):
                mismatch.append(name)
            for v in np.unique(an.read(1)):
                mask_vals[int(v)] += 1
    print(f'   sampled {len(subset)}: sizes={dict(shapes)} img_dtype={dict(img_dtypes)} bands={dict(bands)}')
    print(f'   mask dtype={dict(mask_dtypes)} mask bands={dict(mask_bands)} mask values={sorted(mask_vals)}')
    if mismatch:
        ok = False
        print(f'   !! image/mask size mismatch in {len(mismatch)} pairs (e.g. {mismatch[:3]})')
    if any(b != 3 for b in bands) and not all(b == 4 for b in bands):
        print('   !! expected 3-band RGB images (4-band RGBA is tolerated; alpha is dropped)')
    if any(b != 1 for b in mask_bands):
        ok = False
        print('   !! masks must be single-band index rasters')
    if not set(mask_vals) <= {0, 1, 255}:
        ok = False
        print('   !! mask values outside {0, 1, 255}: re-encode masks (0 = background, 1 = acacia)')
    if 1 not in mask_vals:
        print('   !! no acacia pixels (value 1) found in the sample')
    if side:
        print(f'   note: {len(side)} side-car files (e.g. {[p.name for p in side[:2]]}) are ignored by the loader')
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data_root')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test', 'Generalizability'])
    p.add_argument('--suffix', default='.tif')
    p.add_argument('--sample', type=int, default=100, help='pairs inspected per split')
    a = p.parse_args()
    root = Path(a.data_root)
    results = [check_split(root, s, a.suffix.lower(), a.sample) for s in a.splits]
    print('RESULT:', 'OK' if all(results) else 'PROBLEMS FOUND (see above)')
    raise SystemExit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
