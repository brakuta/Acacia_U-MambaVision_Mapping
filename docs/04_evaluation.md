# 4. Evaluation

## 4.1 Metrics

`IoUMetric` with `iou_metrics=['mIoU', 'mFscore']` reports, per class and averaged
over background and acacia, the intersection-over-union, F-score, precision and
recall (Equations 2–7 of the paper). Averages are over the two classes
(`mIoU`, `mFscore`); the acacia-class values (`IoU.acacia`, `Fscore.acacia`)
are the quantities of ecological interest.

## 4.2 Commands

```bash
CKPT="/weights/mambavision-s_generic-unet_acacia-88"   # folder -> its best_mIoU_iter_*.pth
python tools/test.py configs/mambavision/U-MV-small.py "$CKPT" --test-split test2            # in-distribution
python tools/test.py configs/mambavision/U-MV-small.py "$CKPT" --test-split Generalizability # out-of-distribution
python tools/test.py configs/mambavision/U-MV-small.py work_dirs/U-MV-small --test-split test2   # own run
# released checkpoints (identical to the best_mIoU files; after git lfs pull)
for v in tiny small base; do
  python tools/test.py configs/mambavision/U-MV-$v.py Pretrained_Weights/U-MV-${v}_latest.pth --test-split test2
done
```

Without `--test-split` the config's default split `img_dir/test` is used
(the archive names it `test2`).

Options:

| Option | Effect |
|---|---|
| `--test-split NAME` | Evaluate on `img_dir/NAME` + `ann_dir/NAME` (added in this repository). |
| `--work-dir DIR` | Where the JSON metrics and log are written. |
| `--out DIR` | Save predicted index masks for offline analysis. |
| `--show-dir DIR` | Save colour overlays (input, ground truth, prediction). |
| `--tta` | Multi-scale (0.5–1.75) + horizontal-flip test-time augmentation (slower; not used in the paper). |

Evaluation uses the config's sliding-window mode (1024 × 1024 window, stride 512).

## 4.3 Reference results (paper, Table 1)

| Model | Validation mIoU | Validation mF-score | Test mIoU | Test mF-score |
|---|---:|---:|---:|---:|
| U-MV-t | 87.91 | 93.20 | 85.44 | 91.61 |
| U-MV-s | 88.02 | 93.27 | 85.38 | 91.58 |
| U-MV-b | 88.11 | 93.32 | 85.30 | 91.52 |

Validation covers ~6.5 km² (2 407 archived tiles) and the independent test
set ~6.7 km² (3 123 tiles, split `test2`). On the generalisability set
(~11 km², 2 162 tiles)
U-MV-s reached 89.48 % mIoU and 94.17 % mF-score (§4.4 of the paper).
A freshly evaluated released checkpoint should reproduce the published values
within about ±0.1 percentage points; larger deviations indicate a data or
preprocessing mismatch (band order, mask encoding, image suffix) rather than
model drift.

To confirm that a checkpoint matches a config,
`python tools/inspect_checkpoint.py <file or work dir>` prints the variant
inferred from the decoder shapes together with the iteration and the
MMEngine version stored in the checkpoint.
