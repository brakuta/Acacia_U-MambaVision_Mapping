# U-MV-tiny: MambaVision-T-1K encoder (80/160/320/640) + U-Net decoder.
#
# NOTE ON HYPERPARAMETERS. The configuration originally released for the
# tiny variant differs from the small/base ones in three settings:
#   * base learning rate 1e-5 (small/base: 1e-4; the paper reports 1e-5),
#   * 150,000 iterations (small/base and paper: 100,000),
#   * data_preprocessor.size = (512, 512) (small/base: (1024, 1024); this
#     only affects padding of tiles smaller than the size and is inert for
#     1024 x 1024 tiles).
# These values are preserved verbatim below so that the released
# U-MV-tiny checkpoint is documented by the configuration that produced it.
# To train the tiny variant under the harmonised 100k schedule instead,
# remove the overrides in this file (see docs/07_reproducibility.md).
_base_ = [
    '../_base_/models/umv_unet.py',
    '../_base_/datasets/uav_acacia_dataset.py',
    '../_base_/schedules/schedule_100k_adamw.py',
    '../_base_/default_runtime.py',
]

model = dict(
    data_preprocessor=dict(size=(512, 512)),
    backbone=dict(variant='tiny'),
    decode_head=dict(encoder_channels=[80, 160, 320, 640]))

optimizer = dict(type='AdamW', lr=1e-5, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
optim_wrapper = dict(optimizer=optimizer)
param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=150000, by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=150000, val_interval=5000)
