# Configuration for the U-MV-Tiny Model


_base_ = [
    '../_base_/datasets/uav_acacia_dataset.py',
    '../_base_/default_runtime.py'
]
# 1. Custom Imports
custom_imports = dict(
    imports=[
        'mmseg.custom_models.mamba_vision',
        'mmseg.custom_models.generic_unet_head',
    ],
    allow_failed_imports=False)

# 2. Model Definition
crop_size = (512, 512)
num_classes = 2
norm_cfg = dict(type='BN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mamba_tiny_vision_timm'
    ),
    decode_head=dict(
        type='GenericUNetHead',
        # Channel numbers are preserved exactly from the working configuration.
        encoder_channels=[80, 160, 320, 640],
        decoder_channels=(256, 128, 64, 32),
        in_index=[0, 1, 2, 3],
        num_classes=num_classes,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=3.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(512, 512)))

# Note: All dataset settings are inherited from the _base_ dataset file.

# 3. Training Schedule & Optimizer
optimizer = dict(type='AdamW', lr=1e-5, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'norm': dict(decay_mult=0.0)
        }))

param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=150000, by_epoch=False)
]

# 4. Training, Validation, and Test Loops (Preserved)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=150000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 5. Hooks (Preserved)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))