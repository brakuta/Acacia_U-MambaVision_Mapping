# DeepLabV3+ R50 for building footprints: CE + Dice (weight 2) against the background majority.
# NOTE: CrossEntropyLoss(class_weight=...) cannot be combined with ignored pixels (255, also produced by
# padding) in MMSegmentation 1.2.2 - it indexes the weight vector by every label value and crashes.
_base_ = [
    'mmseg::_base_/models/deeplabv3plus_r50-d8.py',
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
    decode_head=dict(num_classes=num_classes, norm_cfg=norm_cfg,
                     loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                                  dict(type='DiceLoss', loss_weight=2.0)]),
    auxiliary_head=dict(num_classes=num_classes, norm_cfg=norm_cfg),
    test_cfg={{_base_.slide_test_cfg}})
