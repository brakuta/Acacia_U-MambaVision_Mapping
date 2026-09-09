# Appendix B. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'X is not in the mmseg::model registry'` | custom module not imported, or typo | `custom_imports = dict(imports=['geoseg'])` (or your package); check spelling and scope prefix |
| `Duplicate key is not allowed among bases` | two `_base_` files define the same variable | keep helper variables in one base |
| `TypeError: string indices must be integers` / `'str' and 'int'` in a config | arithmetic on a `{{_base_.x}}` placeholder | compute in the base file, reference whole values |
| `RuntimeError: SyncBatchNorm ... single process` | `SyncBN` on one GPU | `norm_cfg = dict(type='BN', requires_grad=True)` |
| `size mismatch for decode_head.conv_seg.weight` | `num_classes` differs from the checkpoint | expected when transferring to new classes (warning); an error means the config is wrong |
| `IndexError: index 255 is out of bounds` in `cross_entropy` | `class_weight` with ignored pixels | remove `class_weight`; use Dice/Lovasz/Focal |
| `AssertionError: ... in_channels` / shape error at the first conv | `in_channels` ≠ bands produced by the loader | align `bands`, `in_channels`, `mean/std` lengths |
| all-background predictions, mIoU ≈ 50 % (binary) | masks 0/255 or RGB; wrong suffix; `reduce_zero_label=True` on a 0-based mask | `check_dataset.py`; set `reduce_zero_label=False` |
| loss NaN after a few hundred iterations | lr too high; no `clip_grad`; NaN/inf pixels in 16-bit input | lower lr, `clip_grad`, `nodata_fill`, `clip` |
| `CUDA out of memory` | batch/crop too large | `--amp`; smaller batch; smaller crop; `mode='slide'` at test |
| GPU utilisation low, `data_time` high | CPU workers or disk | `num_workers` up; SSD; COG; smaller tiles |
| `DataLoader worker ... Bus error` | container shared memory | run with `--ipc=host` (compose does) |
| WSL2 freezes when memory is full | sysmem fallback | NVIDIA Control Panel → Prefer No Sysmem Fallback |
| `OSError: cannot connect to huggingface.co` (timm) | offline | download the timm weights once online; `HF_HUB_OFFLINE=1` |
| `open-mmlab://resnet50_v1c` download fails | offline | `tools/download_zoo_weights.py`; `pretrained='work_dirs/pretrained/resnet50_v1c-2cccc1ad.pth'` |
| overlays in `browse_dataset` look wrong for multispectral | 3-band assumption | ignore or make RGB copies; `draw=False` |
| `PhotoMetricDistortion` error with float images | 8-bit assumption | remove it for non-8-bit inputs |
| predictions shifted relative to the image in GIS | tiles written without georeferencing, or image/mask grid mismatch | use `make_tiles.py`; check `gdalinfo` of image and mask |
| test mIoU far below validation | leakage-free split reveals over-fitting, or different region | expected; report both; more diverse training data |
| `tools/test.py` finds no images | wrong `data_prefix` or split name | `--test-split`, `check_dataset.py` |
