# Appendix A. Cheat-sheets

## A.1 Commands

```
# environment
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm mmseg
python tools/verify_install.py [--variant small]
python tools/download_zoo_weights.py          # ResNet/MiT/Swin encoder weights for offline use
# data
python tools/make_tiles.py --image O.tif --mask L.tif --out /data/P --tile 512 --split-blocks 4 --val-frac .15 --test-frac .15
python tools/check_dataset.py /data/P --splits train val test
python tools/band_statistics.py /data/P/img_dir/train --bands 1 2 3 4 --scale 1
python tools/analysis_tools/browse_dataset.py CFG --output-dir work_dirs/browse --not-show
# configs
python tools/misc/print_config.py CFG [--cfg-options k=v]
# training
python tools/train.py CFG --work-dir work_dirs/P/M [--amp] [--resume] [--cfg-options k=v ...]
bash templates/project/scripts/run_experiments.sh projects/P [M1 M2 ...]
# evaluation
python tools/test.py CFG CKPT_OR_WORKDIR [--test-split S] [--tta] [--out DIR] [--show-dir DIR]
python tools/compare_runs.py work_dirs/P --out summary.md
python tools/analysis_tools/get_flops.py CFG --shape 512 512
python tools/analysis_tools/confusion_matrix.py CFG PRED_DIR OUT_DIR
python tools/analysis_tools/analyze_logs.py work_dirs/P/M/<ts>/vis_data/<ts>.json --keys mIoU loss
# prediction
python tools/geospatial_inference.py --config CFG --checkpoint CKPT --input ortho.tif --output out.gpkg [--save-prob]
python tools/batch_geospatial_inference.py --config CFG --checkpoint CKPT --input-dir D --output-dir O --skip-existing
python tools/inspect_checkpoint.py CKPT_OR_WORKDIR
```

## A.2 Config keys most often changed

| Key | Typical values |
|---|---|
| `train_dataloader.dataset.data_root` | `/data/<project>` |
| `train_dataloader.batch_size` / `num_workers` | 2–8 / 4–16 |
| `train_dataloader.dataset.pipeline[i].crop_size` | 256–1024 |
| `model.data_preprocessor.size` | = crop size |
| `model.data_preprocessor.mean/std/bgr_to_rgb` | ImageNet + True (RGB via OpenCV); statistics + False (rasterio) |
| `model.backbone.in_channels` | 3, 4, 10 |
| `model.decode_head.num_classes` (+ `auxiliary_head`) | classes incl. background |
| `model.decode_head.loss_decode` | CE; CE+Dice; CE+Lovasz |
| `model.test_cfg` | `dict(mode='slide', crop_size=..., stride=...)` or `dict(mode='whole')` |
| `optim_wrapper.optimizer.lr` | 6e-5–1e-4 (AdamW), 1e-2 (SGD) |
| `train_cfg.max_iters` / `val_interval` | 20k–160k / 5 % |
| `default_hooks.checkpoint.save_best` | `'mIoU'` |
| `load_from` / `resume` | checkpoint path / True |
| `randomness.seed` | 0, 1, 2 |

## A.3 Transform quick list

Loading: `LoadImageFromFile`, `LoadAnnotations`, `LoadSingleRSImageFromFile`,
`LoadRasterioImage`, `LoadImageFromNDArray`. Geometry: `RandomResize`,
`Resize`, `ResizeToMultiple`, `RandomCrop`, `RandomFlip`, `RandomRotate`,
`RandomRotFlip`, `RandomMosaic`, `RandomCutOut`. Radiometry (8-bit):
`PhotoMetricDistortion`, `CLAHE`, `AdjustGamma`, `Rerange`, `RGB2Gray`,
`Albu`. Formatting: `PackSegInputs`, `GenerateEdge`.

## A.4 Registered names used in this tutorial

Backbones: `ResNetV1c`, `UNet`, `MixVisionTransformer`, `SwinTransformer`,
`TIMMBackbone`. Heads: `FCNHead`, `PSPHead`, `DepthwiseSeparableASPPHead`,
`UPerHead`, `SegformerHead`. Losses: `CrossEntropyLoss`, `DiceLoss`,
`LovaszLoss`, `FocalLoss`, `TverskyLoss`. Datasets: `BaseSegDataset`,
`RepeatDataset`, `ConcatDataset`.
Metrics: `IoUMetric`. Hooks: `CheckpointHook`, `LoggerHook`,
`SegVisualizationHook`, `EarlyStoppingHook`. Schedulers: `LinearLR`, `PolyLR`,
`CosineAnnealingLR`. Optimisers: `AdamW`, `SGD`.
