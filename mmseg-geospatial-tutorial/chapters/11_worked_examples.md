# 11. Worked examples

Four projects, one pattern: copy the template, edit the dataset base, pick
model configs, train, compare, map. Every config below is in `examples/` and
is loaded and built in the test suite.

## 11.1 Tree crowns from UAV RGB (binary, 2.5 cm, 1024² tiles)

The running example: *Acacia tortilis* crown tiles (Gibril et al., 2026),
binary masks, splits `train`, `val`, `test2` and a spatially separate
`Generalizability` region. `examples/acacia_rgb/configs/_base_/dataset.py`
points the generic template at `/data/acacia` (1024² crops, batch 2) and four
zoo model configs share it, so a model comparison is one command:

```bash
bash templates/project/scripts/run_experiments.sh examples/acacia_rgb
python tools/compare_runs.py work_dirs/acacia_rgb
```

Design choices to notice: 1024² crops because a crown needs its shadow and
neighbours; scale jitter 0.5–2.0; CE + Dice; sliding-window testing with a
512 stride; the evaluation on a separate region (`Generalizability`).
Reference values for orientation (test mIoU, from the publication):
SegFormer-B2 85.2, U-Net-R50 84.2, PSPNet-R50 81.9, DeepLabV3+-R50 81.1.

## 11.2 Building footprints from aerial RGB (binary, 5–10 cm, 512² tiles)

`examples/buildings_rgb/`. Differences: 512² crops (buildings are
compact); CE + 2·Dice against the background majority; 8-connected
polygonisation and `--min-area 4` at export; footprint regularisation in GIS
afterwards. Suggested baselines: DeepLabV3+-R50 (sharp edges from the
low-level skip) and SegFormer-B2. Typical pitfalls: shadows of tall buildings
labeled as background create boundary errors — include them in the reference
polygons consistently (roof outline, not shadow); flat roofs versus paved
lots need context, so do not go below 512².

## 11.3 Roads (binary, thin connected structures, 768² tiles)

`examples/roads_rgb/`. Differences: larger crops for continuity;
vertical flips and ±30° rotation (no preferred orientation); CE + Lovasz
(IoU surrogate, rewards connectedness); no minimum-area filter at export, and
a thinning step to obtain centrelines. Evaluate the road IoU and, if
connectivity matters, a topological measure computed in GIS (number of
connected components versus the reference). SegFormer-B2 and UPerNet-R50 are
the provided configs.

## 11.4 Land cover from 10-band Sentinel-2 (7 classes, uint16, 256² tiles)

`examples/lulc_multispectral/`. This is the example to study for any
non-RGB sensor. The dataset base uses `LoadRasterioImage` with reflectance
scaling and clipping, drops `PhotoMetricDistortion`, sets per-band `mean`/`std`
and `bgr_to_rgb=False`; the model configs set `in_channels=10`, disable the
RGB visualisation hook and use CE + Dice. To keep ImageNet pretraining:

```bash
python tools/download_zoo_weights.py resnet50_v1c
python tools/adapt_first_conv.py work_dirs/pretrained/resnet50_v1c-2cccc1ad.pth work_dirs/pretrained/resnet50_v1c_10band.pth \
    --key stem.0.weight --bands 10 --mode map --rgb-bands 2 1 0      # B4=R, B3=G, B2=B at stack positions 2,1,0
```

and set `backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='work_dirs/pretrained/resnet50_v1c_10band.pth'))`.
Compute the statistics with `tools/band_statistics.py` before training; the
values in the example config are placeholders.

## 11.5 Reusing a trained model for a new tree species

Fine-tune rather than retrain: `load_from` the crown model's best checkpoint,
a new dataset base with the new species, learning rate 2e-5, 10k iterations
(chapter 8.6). If the imagery is a different sensor or GSD, extend
`ratio_range` so that the training scale distribution covers the new GSD.

## 11.6 A checklist for any new task

1. Define classes and the ignore policy; write the class table into the
   dataset base.
2. Choose the tile size from object size × context at the working GSD.
3. Rasterise, tile with spatial-block splits, hold out one region.
4. `check_dataset.py`, `browse_dataset.py`.
5. Loader and normalisation for the sensor (chapter 6.6).
6. Two baselines from different families; smoke test; full runs.
7. `compare_runs.py`; error analysis; one round of targeted data or loss changes.
8. Test-split and held-out-region evaluation, reported once.
9. Map with the geospatial pipeline; post-process in GIS.
10. Record the experiment (chapter 12).
