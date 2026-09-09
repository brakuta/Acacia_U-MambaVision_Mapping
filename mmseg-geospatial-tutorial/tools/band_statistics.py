#!/usr/bin/env python3
"""Per-band mean and standard deviation of a folder of raster tiles, for the
``data_preprocessor`` of multispectral configs.

    python tools/band_statistics.py /data/lulc_s2/img_dir/train --bands 1 2 3 4 5 6 7 8 9 10 --scale 1e-4 --sample 500
"""
import argparse
import glob
import os
import random

import numpy as np

import _bootstrap  # noqa: F401


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder'); p.add_argument('--suffix', default='.tif')
    p.add_argument('--bands', type=int, nargs='*', default=None)
    p.add_argument('--scale', type=float, default=1.0, help='applied before statistics (e.g. 1e-4 for reflectance x 10000)')
    p.add_argument('--sample', type=int, default=300, help='number of tiles to read')
    a = p.parse_args()
    import rasterio
    files = sorted(glob.glob(os.path.join(a.folder, f'*{a.suffix}')))
    random.Random(0).shuffle(files)
    files = files[:a.sample]
    s = None; s2 = None; n = 0
    for f in files:
        with rasterio.open(f) as src:
            arr = src.read(a.bands).astype(np.float64) * a.scale   # (C,H,W)
        c = arr.reshape(arr.shape[0], -1)
        s = c.sum(1) if s is None else s + c.sum(1)
        s2 = (c ** 2).sum(1) if s2 is None else s2 + (c ** 2).sum(1)
        n += c.shape[1]
    mean = s / n; std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    print(f'{len(files)} tiles, {n} pixels per band')
    print('mean =', [round(float(v), 5) for v in mean])
    print('std  =', [round(float(v), 5) for v in std])


if __name__ == '__main__':
    main()
