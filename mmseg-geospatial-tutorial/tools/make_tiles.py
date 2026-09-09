#!/usr/bin/env python3
"""Cut an orthomosaic and its rasterised label layer into aligned training tiles.

    python tools/make_tiles.py --image ortho.tif --mask labels.tif --out /data/my_project \\
        --tile 512 --overlap 0 --split-blocks 4 --val-frac 0.15 --test-frac 0.15 --min-labeled 0.0

* image and mask must share grid and CRS (same width/height/transform); use
  ``gdalwarp -tr ... -te ...`` or ArcGIS *Polygon to Raster* with the ortho as
  snap raster to guarantee this;
* tiles are written as GeoTIFF (georeferenced), 8-bit images / 8-bit masks;
* **split by spatial blocks**, not by random tile: the raster is divided into
  ``split_blocks x split_blocks`` blocks and whole blocks are assigned to
  train/val/test, which prevents neighbouring, near-duplicate tiles from
  leaking between splits;
* tiles whose mask is entirely NoData (255) or that contain less than
  ``--min-labeled`` fraction of labeled pixels are skipped; tiles with fewer
  than ``--min-target`` target pixels are kept with probability ``--keep-empty``
  (down-sampling of background-only tiles);
* an index CSV lists every tile with its split, block and class pixel counts.
"""
import argparse
import csv
import os
import random

import numpy as np

import _bootstrap  # noqa: F401


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image', required=True); p.add_argument('--mask', required=True)
    p.add_argument('--out', required=True, help='dataset root (img_dir/ann_dir are created)')
    p.add_argument('--tile', type=int, default=512); p.add_argument('--overlap', type=int, default=0)
    p.add_argument('--bands', type=int, nargs='*', default=None, help='image bands to keep (default all)')
    p.add_argument('--split-blocks', type=int, default=4, help='grid of spatial blocks for the split')
    p.add_argument('--val-frac', type=float, default=0.15); p.add_argument('--test-frac', type=float, default=0.15)
    p.add_argument('--nodata-label', type=int, default=255)
    p.add_argument('--min-labeled', type=float, default=0.0, help='min fraction of labeled (non-255) pixels')
    p.add_argument('--min-target', type=int, default=0, help='pixels of any class > 0 below which a tile counts as empty')
    p.add_argument('--keep-empty', type=float, default=1.0, help='probability of keeping an empty tile')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--prefix', default='')
    a = p.parse_args()

    import rasterio
    from rasterio.windows import Window
    rng = random.Random(a.seed)
    with rasterio.open(a.image) as im, rasterio.open(a.mask) as mk:
        assert (im.width, im.height) == (mk.width, mk.height), 'image and mask grids differ'
        if im.transform != mk.transform:
            print('warning: image and mask transforms differ; check alignment')
        W, H = im.width, im.height
        bands = a.bands or list(range(1, im.count + 1))
        stride = a.tile - a.overlap
        # block assignment
        nb = a.split_blocks
        blocks = [(bx, by) for by in range(nb) for bx in range(nb)]
        rng.shuffle(blocks)
        n_test = max(1, round(len(blocks) * a.test_frac)) if a.test_frac > 0 else 0
        n_val = max(1, round(len(blocks) * a.val_frac)) if a.val_frac > 0 else 0
        split_of = {b: 'test' for b in blocks[:n_test]}
        split_of.update({b: 'val' for b in blocks[n_test:n_test + n_val]})
        split_of.update({b: 'train' for b in blocks[n_test + n_val:]})
        for s in ('train', 'val', 'test'):
            os.makedirs(os.path.join(a.out, 'img_dir', s), exist_ok=True)
            os.makedirs(os.path.join(a.out, 'ann_dir', s), exist_ok=True)
        rows, counts = [], {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}
        idx = 0
        for y in range(0, H - a.tile + 1, stride):
            for x in range(0, W - a.tile + 1, stride):
                win = Window(x, y, a.tile, a.tile)
                m = mk.read(1, window=win)
                labeled = (m != a.nodata_label)
                if labeled.mean() < max(a.min_labeled, 1e-9):
                    counts['skipped'] += 1; continue
                n_target = int(((m > 0) & labeled).sum())
                if n_target <= a.min_target and rng.random() > a.keep_empty:
                    counts['skipped'] += 1; continue
                bx, by = min(nb - 1, (x + a.tile // 2) * nb // W), min(nb - 1, (y + a.tile // 2) * nb // H)
                split = split_of[(bx, by)]
                name = f'{a.prefix}{idx:012d}.tif'; idx += 1
                img = im.read(bands, window=win)
                tr = im.window_transform(win)
                prof = dict(driver='GTiff', width=a.tile, height=a.tile, crs=im.crs, transform=tr, compress='deflate')
                with rasterio.open(os.path.join(a.out, 'img_dir', split, name), 'w', count=len(bands), dtype=img.dtype, **prof) as d:
                    d.write(img)
                with rasterio.open(os.path.join(a.out, 'ann_dir', split, name), 'w', count=1, dtype='uint8', nodata=None, **prof) as d:
                    d.write(m.astype(np.uint8)[None])
                cls, cnt = np.unique(m[labeled], return_counts=True)
                rows.append(dict(tile=name, split=split, block=f'{bx},{by}', x=x, y=y,
                                 labeled_frac=round(float(labeled.mean()), 4),
                                 **{f'class_{int(c)}': int(n) for c, n in zip(cls, cnt)}))
                counts[split] += 1
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith('class_'), k))
    with open(os.path.join(a.out, 'tiles_index.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('tiles written:', counts, '-> index:', os.path.join(a.out, 'tiles_index.csv'))


if __name__ == '__main__':
    main()
