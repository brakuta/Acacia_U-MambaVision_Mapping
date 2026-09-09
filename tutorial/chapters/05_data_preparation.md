# 5. From GIS layers to training tiles

## 5.1 What the model needs

| Item | Requirement | Why |
|---|---|---|
| Images | georeferenced rasters of fixed tile size; 8-bit 3-band for RGB workflows, any dtype/band count with the rasterio loader | fixed size gives regular batches; georeferencing lets predictions return to GIS |
| Masks | single-band 8-bit rasters on the **same grid** as the images; pixel value = class index (0 = background); 255 = ignore | the loss compares pixel by pixel |
| Names | identical base names in `img_dir/<split>` and `ann_dir/<split>` | this is how pairs are matched |
| Splits | `train`, `val` (model selection), `test` (final report), optionally an out-of-distribution set | the test set must never influence choices |

Tile size is a design decision, not a default. The tile must contain the
object with its context at the working GSD: a 5 m crown at 2.5 cm is 200 px,
and its shadow and neighbours matter, hence 1024 px for the Acacia project;
buildings at 10 cm fit in 512 px; a 10 m Sentinel-2 field pattern fits in
256 px. Larger tiles cost memory quadratically.

## 5.2 The reference workflow (ArcGIS Pro)

1. **Reference layer.** Verified polygons with an integer class field
   (`class_id`: 1 = target, 2 = second class, ...). Areas not interpreted are
   left outside the polygons and outside an *interpretation extent* polygon.
2. **Rasterise.** *Polygon to Raster*: value field `class_id`, cell size and
   snap raster = the orthomosaic, `NoData` → 0 with *Reclassify*, and 255
   outside the interpretation extent (mask with *Extract by Mask* then
   *Con(IsNull(x), 255, x)*). Export as 8-bit unsigned GeoTIFF.
3. **Tile.** Either *Export Training Data For Deep Learning* (Classified
   Tiles, tile size, stride, no rotation) or the team tool below.
4. **Check.** `python tools/check_dataset.py /data/<project>`.

The same is achieved with GDAL: `gdal_rasterize -a class_id -tr <res> <res>
-te <extent> -ot Byte -a_nodata 255 polygons.gpkg labels.tif`.

## 5.3 Tiling with `tools/make_tiles.py`

```bash
python tools/make_tiles.py --image /data/raw/ortho_block12.tif --mask /data/raw/labels_block12.tif \
    --out /data/buildings --tile 512 --overlap 0 --split-blocks 4 --val-frac 0.15 --test-frac 0.15 \
    --min-labeled 0.5 --min-target 50 --keep-empty 0.3 --prefix b12_
```

What it does, and why:

* **Spatial-block split.** The raster is divided into 4 × 4 blocks and whole
  blocks are assigned to train/val/test. Randomly assigning tiles would put
  near-identical neighbours into train and test and inflate the test score;
  the Acacia study used spatially distinct regions for the same reason.
* **Empty-tile control.** Tiles without any target pixel are kept with
  probability 0.3, which keeps enough pure background to learn the negatives
  without drowning the positives.
* **Ignore handling.** Tiles with less than 50 % interpreted pixels are
  skipped; 255 pixels inside kept tiles remain 255 and are ignored by the loss.
* **Provenance.** `tiles_index.csv` records split, block, offsets and class
  pixel counts per tile; it is the file from which class frequencies and
  weights are computed.

Repeat per orthomosaic with a different `--prefix`; the folder structure is
shared. Multiple regions: keep at least one whole region out of training as
the generalisability set, exactly as in the paper.

## 5.4 Checking the dataset

```bash
python tools/check_dataset.py /data/buildings --splits train val test
```

reports pair counts, sizes, dtypes, band counts and mask values per split, and
side-car files (`*.aux.xml`, `*.ovr`) that ArcGIS leaves behind. Then render a
few augmented samples to see what the network will see:

```bash
python tools/analysis_tools/browse_dataset.py projects/buildings/configs/upernet_r50.py --output-dir work_dirs/browse --not-show
```

**Checkpoint.** `RESULT: OK`; mask values are exactly the class indices plus
255; the overlays in `work_dirs/browse` show the mask on the correct objects
(a shifted mask means a grid mismatch between image and label rasters).

## 5.5 Class statistics and imbalance

From `tiles_index.csv` (pandas):

```python
import pandas as pd
t = pd.read_csv('/data/buildings/tiles_index.csv')
cols = [c for c in t.columns if c.startswith('class_')]
freq = t.loc[t.split == 'train', cols].sum(); print((freq / freq.sum()).round(4))
```

A target class below ~5 % of pixels calls for a region-based loss (Dice,
Lovasz) in addition to cross-entropy, background-tile down-sampling at tiling
time, and `cat_max_ratio` in `RandomCrop` (rejects crops dominated by one
class). See chapter 7.4 for the losses.

## 5.6 Multispectral and 16-bit data

Keep the native radiometry in the tiles (uint16 reflectance × 10 000 for
Sentinel-2, 12-bit for many UAV multispectral sensors); scaling to reflectance
is done at load time (chapter 6.6). Record per-band statistics once:

```bash
python tools/band_statistics.py /data/lulc_s2/img_dir/train --bands 1 2 3 4 5 6 7 8 9 10 --scale 1e-4
```

and copy the printed `mean`/`std` into the config's `data_preprocessor`.

## Pitfalls

* Masks saved as RGB or as 0/255 instead of 0/1: `check_dataset.py` reports
  `mask values=[0, 255]` or `mask bands=3`; re-export as single-band index.
* Compressed masks with a colour table are fine; masks with a NoData value of
  0 are not (0 is background, not NoData) — set NoData to none or 255.
* JPEG-compressed image tiles lose fine texture; use deflate/LZW.
* Tiles that straddle the orthomosaic edge contain black (0) fill; either skip
  them (`--min-labeled`) or mark the fill as 255 in the mask.

## Exercise 4

Tile a small orthomosaic with `--split-blocks 2` and `--split-blocks 8`. How
does the test set differ, and which one gives an estimate closer to
performance on a new region?
