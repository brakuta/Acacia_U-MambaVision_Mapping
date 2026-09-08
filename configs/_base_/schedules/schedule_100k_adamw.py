# Optimisation schedule used for the released U-MV-small and U-MV-base
# checkpoints: AdamW, 100k iterations, polynomial decay (power 0.9),
# backbone learning-rate multiplier 0.1, gradient clipping.
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
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
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=100000, by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
