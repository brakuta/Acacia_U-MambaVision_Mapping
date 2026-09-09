# UPerNet R50 with a 10-band input.  Per-band mean/std are dataset statistics in reflectance units
# (compute them with tools/band_statistics.py); bgr_to_rgb is False because LoadRasterioImage
# returns bands in the requested order.
_base_ = [
    'mmseg::_base_/models/upernet_r50.py',
    './_base_/dataset.py',
    './_base_/schedule.py',
    'mmseg::_base_/default_runtime.py',
]
crop_size = {{_base_.crop_size}}
num_classes = {{_base_.num_classes}}
norm_cfg = dict(type='BN', requires_grad=True)
n_bands = 10
band_mean = [0.08, 0.10, 0.12, 0.15, 0.20, 0.23, 0.25, 0.26, 0.22, 0.15]   # placeholder statistics
band_std = [0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09, 0.09, 0.08, 0.07]

model = dict(
    data_preprocessor=dict(size=crop_size, mean=band_mean, std=band_std, bgr_to_rgb=False),
    # ImageNet weights for the RGB stem cannot be loaded directly into a 10-band stem;
    # either adapt them with tools/adapt_first_conv.py and set init_cfg below, or train the stem from scratch.
    pretrained=None,
    backbone=dict(in_channels=n_bands, norm_cfg=norm_cfg,
                  init_cfg=None),  # e.g. dict(type='Pretrained', checkpoint='work_dirs/pretrained/resnet50_v1c_10band.pth')
    decode_head=dict(num_classes=num_classes, norm_cfg=norm_cfg,
                     # imbalance handled by the Dice term; class_weight would fail on the 255 (unlabeled) pixels
                     loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                                  dict(type='DiceLoss', loss_weight=1.0)]),
    auxiliary_head=dict(num_classes=num_classes, norm_cfg=norm_cfg),
    test_cfg=dict(mode='whole'))

# the whole network is new at the stem: use the full learning rate everywhere
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.0)}))
# visualisation assumes 3-band images; disable for multispectral inputs
default_hooks = dict(visualization=dict(type='SegVisualizationHook', draw=False))
