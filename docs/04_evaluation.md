# 4. Evaluation

## 4.1 Metrics

`IoUMetric` with `iou_metrics=['mIoU', 'mFscore']` reports, per class and averaged
over background and acacia, the intersection-over-union, F-score, precision and
recall (Equations 2–7 of the paper). Averages are over the two classes
(`mIoU`, `mFscore`); the acacia-class values (`IoU.acacia`, `Fscore.acacia`)
are the quantities of ecological interest.

## 4.2 Commands

```bash
# in-distribution test split (img_dir/test)
python tools/test.py configs/mambavision/U-MV-small.py work_dirs/U-MV-small/best_mIoU_iter_95000.pth

# out-of-distribution split (img_dir/Generalizability)
python tools/test.py configs/mambavision/U-MV-small.py work_dirs/U-MV-small/best_mIoU_iter_95000.pth \
    --test-split Generalizability

# released checkpoints (after git lfs pull)
python tools/test.py configs/mambavision/U-MV-tiny.py  Pretrained_Weights/U-MV-tiny_latest.pth
python tools/test.py configs/mambavision/U-MV-small.py Pretrained_Weights/U-MV-small_latest.pth
python tools/test.py configs/mambavision/U-MV-base.py  Pretrained_Weights/U-MV-base_latest.pth
```

Options:

| Option | Effect |
|---|---|
| `--test-split NAME` | Evaluate on `img_dir/NAME` + `ann_dir/NAME` (added in this repository). |
| `--work-dir DIR` | Where the JSON metrics and log are written. |
| `--out DIR` | Save predicted index masks for offline analysis. |
| `--show-dir DIR` | Save colour overlays (input, ground truth, prediction). |
| `--tta` | Multi-scale (0.5–1.75) + horizontal-flip test-time augmentation (slower; not used in the paper). |

Inference during evaluation uses the sliding-window mode of the config
(1024 × 1024 window, 512 stride), which matches training resolution.

## 4.3 Reference results (paper, Table 1)

| Model | Validation mIoU | Validation mF-score | Test mIoU | Test mF-score |
|---|---:|---:|---:|---:|
| U-MV-t | 87.91 | 93.20 | 85.44 | 91.61 |
| U-MV-s | 88.02 | 93.27 | 85.38 | 91.58 |
| U-MV-b | 88.11 | 93.32 | 85.30 | 91.52 |

Validation covers ~6.5 km² (2 400 tiles) and the independent test set
~6.7 km² (3 125 tiles). On the generalisability set (~11 km², 2 165 tiles)
U-MV-s reached 89.48 % mIoU and 94.17 % mF-score (§4.4 of the paper).
A freshly evaluated released checkpoint should reproduce the published values
within about ±0.1 percentage points; larger deviations indicate a data or
preprocessing mismatch (band order, mask encoding, image suffix) rather than
model drift.

## 4.4 Confirming the checkpoint ↔ config pairing

```bash
python tools/inspect_checkpoint.py work_dirs/U-MV-small/best_mIoU_iter_95000.pth
```

prints the variant inferred from the decoder shapes, iteration, MMEngine
version and dataset metadata stored in the checkpoint.
