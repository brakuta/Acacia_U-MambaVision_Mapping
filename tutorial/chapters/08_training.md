# 8. Training

## 8.1 Starting a run

```bash
python tools/train.py projects/buildings/configs/upernet_r50.py --work-dir work_dirs/buildings/upernet_r50
```

Options: `--resume` (continue from the last checkpoint in the work directory),
`--amp` (mixed precision), `--cfg-options key=value ...`. Without `--work-dir`
the run goes to `work_dirs/<config name>/`. Each run creates a timestamped
sub-folder with the log, `vis_data/scalars.json` (all logged numbers) and the
resolved config.

Before a long run, a two-minute smoke test catches most configuration errors:

```bash
python tools/train.py <config> --work-dir work_dirs/smoke \
    --cfg-options train_cfg.max_iters=20 train_cfg.val_interval=10 \
                  train_dataloader.dataset.indices=16 val_dataloader.dataset.indices=8 \
                  default_hooks.logger.interval=1
```

**Checkpoint.** The log shows 20 iterations with decreasing `loss`, one
validation with an mIoU table over your class names, and a checkpoint file.

## 8.2 The schedule

```python
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05, betas=(0.9, 0.999))
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer,
                     clip_grad=dict(max_norm=1.0, norm_type=2),
                     paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                     'norm': dict(decay_mult=0.0)}))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0, power=0.9, begin=500, end=40000, by_epoch=False)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
```

| Element | Meaning | Rules of thumb |
|---|---|---|
| optimiser | AdamW for transformers and hybrids (lr 6e-5–1e-4); SGD momentum 0.9 (lr 0.01) is the zoo default for ResNet models and also works | keep one optimiser across the models you compare |
| `lr_mult` for backbone | pretrained encoder learns 10× slower than the new head | the Acacia runs: encoder 1e-5, decoder 1e-4 |
| `decay_mult=0` for norm | no weight decay on normalisation parameters | standard |
| warm-up (`LinearLR`) | avoids early divergence of AdamW | 500–1500 iterations |
| `PolyLR` power 0.9 | smooth decay to 0 | the segmentation default |
| `max_iters` | budget; iterations × batch / tiles = epochs | 40k for 5–30k tiles; 100k for the paper |
| `val_interval` | how often the validation set is scored | 5 % of `max_iters` |
| `clip_grad` | caps the gradient norm; stabilises small batches | 1.0 (0.01 in the Acacia runs) |

Batch size is limited by memory; 2–4 at 1024², 8 at 512². Since BatchNorm
statistics degrade with batch 1–2, prefer `--amp` over a smaller batch, or
use GroupNorm-based models (SegFormer uses LayerNorm and is insensitive).

## 8.3 Hooks

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, save_best='mIoU', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=200))
custom_hooks = [dict(type='EarlyStoppingHook', monitor='mIoU', patience=5, min_delta=0.1)]
```

`save_best='mIoU'` keeps `best_mIoU_iter_<n>.pth`, the checkpoint of record;
`max_keep_ckpts` limits disk use (each checkpoint stores the optimiser state,
2–3× the model size). `SegVisualizationHook(draw=True)` writes validation
overlays to `vis_data/vis_image` (RGB only). `EarlyStoppingHook` stops when
validation mIoU has not improved by 0.1 for 5 validations.

## 8.4 Reading the log

```
Iter(train) [ 2050/40000]  base_lr: 9.6e-05 lr: 9.6e-05  eta: 3:12:41  time: 0.30  data_time: 0.01
                            memory: 9124  loss: 0.41  decode.loss_ce: 0.21  decode.loss_dice: 0.15  aux.loss_ce: 0.05  decode.acc_seg: 96.1
Iter(val) [500/500]  aAcc: 97.2  mIoU: 78.4  mAcc: 85.1  mFscore: 87.3  mPrecision: 89.0  mRecall: 85.1
```

| Field | Diagnosis |
|---|---|
| `data_time` ≫ 0.1 × `time` | data loading is the bottleneck: more `num_workers`, local SSD, smaller tiles or COG |
| `memory` close to the GPU limit | reduce batch or crop, use `--amp` |
| `loss` flat from the start | learning rate too low, wrong labels (check masks), or `num_classes` mismatch |
| `loss` NaN | learning rate too high, missing `clip_grad`, or NaN in 16-bit inputs (check `nodata_fill`) |
| `decode.acc_seg` high but `mIoU` low | class imbalance: accuracy is dominated by background |
| validation mIoU drops while train loss falls | over-fitting: more augmentation, earlier stop, smaller model |

Curves: `tools/analysis_tools/analyze_logs.py <run>/vis_data/<timestamp>.json --keys mIoU loss`,
or TensorBoard (`dict(type='TensorboardVisBackend')` in `visualizer.vis_backends`).

## 8.5 Training several models

```bash
bash tutorial/templates/project/scripts/run_experiments.sh projects/buildings deeplabv3plus_r50 segformer_b2 umv_small
```

trains each config, evaluates it on the test split and writes
`work_dirs/buildings/summary.md`. Run it inside `tmux`. Keep everything that
is compared identical except the model (chapter 12 gives the record template).

## 8.6 Fine-tuning a trained model on new data

```python
load_from = '/weights/mambavision-s_generic-unet_acacia-88/best_mIoU_iter_95000.pth'
optim_wrapper = dict(optimizer=dict(lr=2e-5))
param_scheduler = [dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=10000, by_epoch=False)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
```

Same classes: everything loads. New class count: the head's final layer has a
different shape; MMEngine reports a size mismatch warning for those tensors and
initialises them randomly, which is the intended behaviour for transfer.

## 8.7 Reproducibility

`randomness = dict(seed=0)` fixes data order and initialisation;
`deterministic=True` also fixes cuDNN algorithms (slower). Two seeds bound the
run-to-run variation, needed before claiming a 0.3 mIoU difference is real.

## Exercise 7

Run the smoke test on two configs and, from the logged `time`, estimate the
duration of a full 40k-iteration run and the epochs it represents.
