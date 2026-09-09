# 6. Datasets, pipelines and augmentation

## 6.1 The dataset object

`BaseSegDataset` lists image–mask pairs and applies a pipeline to each. The
generic class is sufficient for most projects; the class names and colours are
passed through `metainfo`:

```python
dataset = dict(
    type='BaseSegDataset',
    data_root='/data/buildings',
    data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
    img_suffix='.tif', seg_map_suffix='.tif',
    metainfo=dict(classes=('background', 'building'), palette=[[0, 0, 0], [220, 60, 30]]),
    reduce_zero_label=False,
    pipeline=train_pipeline)
```

Arguments that matter:

| Argument | Meaning |
|---|---|
| `data_root`, `data_prefix` | folders; pairs are matched by base name: `img_dir/train/x.tif` ↔ `ann_dir/train/x.tif` |
| `img_suffix`, `seg_map_suffix` | file endings; MMSegmentation's defaults are `.jpg`/`.png`, hence the explicit `.tif` |
| `metainfo` | `classes` (index order) and `palette` (display colours); written into checkpoints, used by metrics and visualisation |
| `reduce_zero_label` | if `True`, label 0 becomes 255 (ignored) and every other label is decremented; used by datasets whose 0 means "unlabeled" (ADE20K, Potsdam). Keep `False` when 0 is a real background class |
| `ignore_index` | label excluded from loss and metrics (255) |
| `ann_file` | optional text file listing base names, to define a split without moving files |
| `indices` | use only the first N (or a list of) samples, for quick debugging runs |

A permanent dataset deserves a class (`umv/datasets/uav_acacia.py`): it fixes
suffixes and classes so that every config is shorter and consistent. The class
is 15 lines: subclass `BaseSegDataset`, set `METAINFO`, set the suffix defaults
in `__init__`, decorate with `@DATASETS.register_module()`.

## 6.2 The pipeline: loading

A pipeline is a list of transforms applied in order to a `results` dictionary
that starts as `{'img_path': ..., 'seg_map_path': ...}`.

| Transform | What it loads | Use |
|---|---|---|
| `LoadImageFromFile` | 8-bit 1- or 3-band image via OpenCV, **BGR** order, uint8 | RGB tiles (jpg, png, tif) |
| `LoadAnnotations(reduce_zero_label=False)` | single-band mask via Pillow, uint8 | always, after the image loader (in the test pipeline after `Resize`, so that the mask is not resized) |
| `LoadSingleRSImageFromFile(to_float32=True)` | any GeoTIFF via GDAL, all bands, file order, float32 | multispectral, when GDAL Python bindings exist |
| `LoadMultipleRSImageFromFile` | two rasters (`img_path`, `img_path2`) | change detection |
| `LoadRasterioImage(bands=..., scale=..., nodata_fill=..., clip=...)` (team) | selected bands via rasterio, float32, scaled | RGB+NIR, multispectral, 16-bit, SAR |
| `LoadImageFromNDArray` | an in-memory array | inference from Python |

The BGR order of `LoadImageFromFile` is the reason for `bgr_to_rgb=True` in
every RGB config: the preprocessor converts to RGB before normalising with the
ImageNet mean `[123.675, 116.28, 103.53]`, which is in RGB order. Loaders that
return bands in the requested order (`LoadRasterioImage`) must be paired with
`bgr_to_rgb=False`.

## 6.3 The pipeline: augmentation

Verified signatures (MMSegmentation 1.2.2):

| Transform | Arguments | Effect |
|---|---|---|
| `RandomResize` | `scale=(W, H)`, `ratio_range=(0.5, 2.0)`, `keep_ratio=True` | scale the tile by a random factor: the network sees objects at several sizes |
| `RandomCrop` | `crop_size=(H, W)`, `cat_max_ratio=0.75`, `ignore_index=255` | take a crop; reject crops where one class exceeds 75 % of pixels (up to 10 tries) |
| `RandomFlip` | `prob=0.5`, `direction='horizontal'` or `'vertical'` | for nadir imagery both directions are valid |
| `RandomRotate` | `prob`, `degree=30`, `pad_val=0`, `seg_pad_val=255` | rotation with padding; objects without preferred orientation (roads, fields) |
| `PhotoMetricDistortion` | `brightness_delta=32`, `contrast_range=(0.5,1.5)`, `saturation_range=(0.5,1.5)`, `hue_delta=18` | radiometric jitter; **8-bit 3-band only** |
| `RandomRotFlip` | `rotate_prob`, `flip_prob`, `degree=(-20, 20)` | combined |
| `RandomMosaic` | `prob`, `img_scale=(640, 640)` | four tiles in one; needs `MultiImageMixDataset` |
| `RandomCutOut` | `prob`, `n_holes`, `cutout_shape` or `cutout_ratio` | erases patches (regularisation) |
| `Rerange` | `min_value=0`, `max_value=255` | linear rescale to a range |
| `CLAHE` | `clip_limit=40.0`, `tile_grid_size=(8, 8)` | contrast equalisation (8-bit) |
| `AdjustGamma` | `gamma=1.0` | gamma (8-bit) |
| `Resize` | `scale=(W, H)`, `keep_ratio=True` | deterministic resize (test pipeline) |
| `ResizeToMultiple` | `size_divisor=32` | pad-free sizing for transformers |
| `RGB2Gray` | `out_channels`, `weights` | grayscale |
| `Albu` | `transforms=[...]` (Albumentations) | any Albumentations transform on 8-bit RGB |
| `GenerateEdge` | `edge_width=3` | boundary maps for edge-aware losses |
| `PackSegInputs` | `meta_keys=(...)` | final step: `inputs` (C×H×W tensor) + `data_samples` (`SegDataSample` with `gt_sem_seg` and metadata) |

Reference training pipeline for 8-bit RGB (the Acacia recipe):

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]
```

Choosing augmentations is a statement about invariances. Nadir aerial imagery
is invariant to flips in both axes and to rotation; oblique imagery is not.
Scale jitter should cover the GSD range at which the model will be applied
(0.5–2.0 covers 1.25–5 cm around a 2.5 cm training GSD). Radiometric jitter
stands in for illumination and sensor differences; it does not replace
training data from other seasons.

## 6.4 The data preprocessor

`SegDataPreProcessor` runs on the GPU on each batch: `bgr_to_rgb`, then
`(x - mean) / std` per channel, then padding to `size` (or to a multiple of
`size_divisor`) with `pad_val` for images and `seg_pad_val=255` for masks, and
stacking. `mean` and `std` have one entry per input channel; the ImageNet
values are correct for 8-bit RGB with an ImageNet-pretrained encoder, and
dataset statistics (`tools/band_statistics.py`) for anything else.

## 6.5 Dataloaders

```python
train_dataloader = dict(
    batch_size=4, num_workers=8, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(...))
val_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False), dataset=dict(...))
```

`InfiniteSampler` draws batches forever (iteration-based training);
`DefaultSampler` makes one pass (evaluation). `num_workers` are CPU processes
running the pipeline; start at 8 and raise it while `data_time` in the log is
larger than ~10 % of `time`. `batch_size` is per GPU. Oversampling a small
dataset: wrap it as `dict(type='RepeatDataset', times=5, dataset=dict(...))`;
combining datasets: `dict(type='ConcatDataset', datasets=[...])`.

## 6.6 Multi-band and non-8-bit inputs

Three changes turn the RGB recipe into a multispectral one:

```python
bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                      # 1-based, output order
train_pipeline = [
    dict(type='LoadRasterioImage', bands=bands, scale=1 / 10000.0, clip=(0.0, 1.5)),   # 1. loader
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(256, 256), ratio_range=(0.75, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.9),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # 2. no PhotoMetricDistortion / CLAHE / AdjustGamma (8-bit BGR only)
    dict(type='PackSegInputs'),
]
data_preprocessor = dict(type='SegDataPreProcessor', size=(256, 256),
                         mean=[...10 values...], std=[...10 values...], bgr_to_rgb=False)   # 3. per band
model = dict(backbone=dict(in_channels=10), ...)
```

Variants:

| Input | Loader settings | `mean`/`std` |
|---|---|---|
| RGB 8-bit (default) | `LoadImageFromFile`, `bgr_to_rgb=True` | ImageNet |
| RGB+NIR 8-bit (4 bands) | `LoadRasterioImage(bands=[1,2,3,4])`, `bgr_to_rgb=False` | ImageNet values + a NIR mean/std from `band_statistics.py`, or all four from statistics |
| Sentinel-2 L2A uint16 | `LoadRasterioImage(bands=..., scale=1e-4, clip=(0, 1.5))` | statistics in reflectance |
| UAV multispectral 12-bit | `scale=1/4095` | statistics |
| SAR dB float32 | `clip=(-30, 5)`, optional `offset` | statistics |
| GDAL loader | `LoadSingleRSImageFromFile` reads all bands in file order as float32, no scaling: add `scale` via the preprocessor `std` (e.g. `std=[10000*s ...]`) or pre-scale the tiles |

Pretrained encoders expect 3 channels. Options for `in_channels != 3`:
(a) adapt the first convolution of an ImageNet checkpoint with
`tools/adapt_first_conv.py` and load it with `init_cfg=dict(type='Pretrained',
checkpoint=...)`; (b) train the stem from scratch (`pretrained=None`); (c) keep
the RGB bands in the pretrained stem and feed extra bands to a second small
stem (custom backbone; beyond this tutorial). Option (a) is the default choice
and preserves most of the pretraining benefit.

**Pitfall.** `SegVisualizationHook` and `browse_dataset.py` render three
channels as RGB; with 10-band inputs set `default_hooks.visualization.draw=False`
or expect meaningless images.

## 6.7 Verifying a pipeline in Python

```python
from mmengine.config import Config
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)
cfg = Config.fromfile('projects/lulc/configs/upernet_r50_10band.py')
ds = DATASETS.build(cfg.train_dataloader.dataset)
item = ds[0]
print(item['inputs'].shape, item['inputs'].dtype)            # torch.Size([10, 256, 256]) torch.float32
print(item['data_samples'].gt_sem_seg.data.unique())         # class indices present (+255)
print(item['data_samples'].metainfo)                         # img_path, ori_shape, flip, ...
```

Ten lines that answer most "why does my model not learn" questions: wrong
band order, wrong scaling, masks with unexpected values, or a crop that does
not contain the target.

## Exercise 5

Write the training pipeline and preprocessor for a 4-band (R, G, B, NIR) 8-bit
UAV dataset stored as B, G, R, NIR in the file, keeping the ImageNet
normalisation for the three visible bands. Which `bands` order and which
`bgr_to_rgb` value are correct?
