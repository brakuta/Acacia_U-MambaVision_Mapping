# Training schedule shared by every model config of this project.
# 40k iterations is a reasonable first budget for 5k-30k tiles.  Validation every 2k iterations; best checkpoint by mIoU.
max_iters = 40000
val_interval = 2000

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper', optimizer=optimizer,
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1), 'norm': dict(decay_mult=0.0)}))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),   # warm-up
    dict(type='PolyLR', eta_min=0, power=0.9, begin=500, end=max_iters, by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval,
                    save_best='mIoU', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=200))
randomness = dict(seed=0)
