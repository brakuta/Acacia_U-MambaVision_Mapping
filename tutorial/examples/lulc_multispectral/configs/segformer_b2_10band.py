# SegFormer MiT-B2 with a 10-band input (patch embedding adapted or trained from scratch).
_base_ = [
    'mmseg::_base_/models/segformer_mit-b0.py',
    './_base_/dataset.py',
    './_base_/schedule.py',
    'mmseg::_base_/default_runtime.py',
]
crop_size = {{_base_.crop_size}}
num_classes = {{_base_.num_classes}}
band_mean = [0.08, 0.10, 0.12, 0.15, 0.20, 0.23, 0.25, 0.26, 0.22, 0.15]
band_std = [0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09, 0.09, 0.08, 0.07]

model = dict(
    data_preprocessor=dict(size=crop_size, mean=band_mean, std=band_std, bgr_to_rgb=False),
    backbone=dict(in_channels=10, embed_dims=64, num_layers=[3, 4, 6, 3],
                  init_cfg=None),  # or Pretrained checkpoint adapted with tools/adapt_first_conv.py
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=num_classes),
    test_cfg=dict(mode='whole'))
default_hooks = dict(visualization=dict(type='SegVisualizationHook', draw=False))
