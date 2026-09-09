# 9. Evaluation

## 9.1 Metrics

`IoUMetric` computes, per class and averaged over classes:

| Key | Definition |
|---|---|
| `IoU` / `mIoU` | TP / (TP + FP + FN) |
| `Fscore` / `mFscore` | 2·P·R / (P + R) (Dice) |
| `Precision`, `Recall` | TP / (TP + FP), TP / (TP + FN) |
| `Acc` / `mAcc` | per-class pixel accuracy (= recall) |
| `aAcc` | overall pixel accuracy (dominated by background; not a model-selection metric) |

`iou_metrics=['mIoU', 'mFscore']` requests both families. Pixels labeled 255 are
excluded. For a two-class problem report the target-class IoU and F-score in
addition to the means, as in the paper (the background IoU is always high).

## 9.2 Running the test

```bash
python tools/test.py projects/buildings/configs/upernet_r50.py work_dirs/buildings/upernet_r50 \
    --work-dir work_dirs/buildings/upernet_r50/test                       # folder -> best_mIoU_iter_*.pth
python tools/test.py <config> <ckpt> --test-split Generalizability          # other split
python tools/test.py <config> <ckpt> --out work_dirs/.../preds              # save index masks
python tools/test.py <config> <ckpt> --show-dir work_dirs/.../vis           # overlays
python tools/test.py <config> <ckpt> --tta                                  # flips + scales
```

Test-time augmentation averages predictions over the `tta_pipeline` scales and
flips; expect +0.3–1.0 mIoU at 6–12× the cost. Report with and without, and
never compare a TTA result with a non-TTA one.

`mode='slide'` (window and stride in `test_cfg`) evaluates each tile in
overlapping windows and averages logits; it matters when test tiles are larger
than the training crop. `mode='whole'` runs the full tile at once.

## 9.3 Comparing runs

```bash
python tools/compare_runs.py work_dirs/buildings --out work_dirs/buildings/summary.md --csv summary.csv
```

| run | best_iter | best_mIoU | best_mFscore | last_iter | last_mIoU | test_mIoU | test_mFscore |
|---|---|---|---|---|---|---|---|
| buildings/segformer_b2 | 36000 | 81.2 | 89.5 | 40000 | 80.9 | 80.1 | 88.7 |
| buildings/deeplabv3plus_r50 | 30000 | 79.8 | 88.6 | 40000 | 79.5 | 78.6 | 87.9 |

Model selection uses `best_mIoU` on validation; the report uses the test
split, evaluated once, with the selected checkpoint. Add parameters and FLOPs:

```bash
python tools/analysis_tools/get_flops.py projects/buildings/configs/segformer_b2.py --shape 512 512
```

## 9.4 Error analysis

Confusion matrix from saved predictions:

```bash
python tools/test.py <config> <ckpt> --out work_dirs/x/preds        # writes one index PNG per test image
python tools/analysis_tools/confusion_matrix.py <config> work_dirs/x/preds work_dirs/x/cm
```

The confusion-matrix tool pairs every file in the prediction folder with the
test dataset in order, so the prediction folder must contain the test split
only, produced by a single `test.py --out` run.

Visual review: `--show-dir` overlays; sort tiles by per-tile IoU (a short
script over `--out` masks and the ground truth) and inspect the worst 20. The
paper's Fig. 13 pattern applies to most projects: errors concentrate at
shadows, mixed crowns/adjacent objects, low-contrast backgrounds and boundary
pixels. Those categories decide the next action: more training examples of
that condition, a different loss, or a post-processing rule.

## 9.5 Generalisability

A test set from the same regions as training measures interpolation. A held-out
region (different date, terrain, sensor settings) measures what the model does
on new surveys, which is the operational question. Keep one such region from
the start (`make_tiles.py` per orthomosaic, hold one out), report both, and
expect the gap to be larger for CNNs than for transformers/hybrids, as in the
Acacia study.

## Exercise 8

For a binary model with mIoU = 85 %, background IoU = 97 % and target IoU =
73 %: which number goes into the abstract, and why would `aAcc = 98 %` be
misleading?
