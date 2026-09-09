# UPerNet with ResNet-50: multi-scale feature-pyramid decoder.
_base_ = [
    'mmseg::_base_/models/upernet_r50.py',
    './_base_/dataset.py',
    './_base_/schedule.py',
    'mmseg::_base_/default_runtime.py',
]
crop_size = {{_base_.crop_size}}
num_classes = {{_base_.num_classes}}
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=num_classes, norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=num_classes, norm_cfg=norm_cfg),
    test_cfg={{_base_.slide_test_cfg}})
