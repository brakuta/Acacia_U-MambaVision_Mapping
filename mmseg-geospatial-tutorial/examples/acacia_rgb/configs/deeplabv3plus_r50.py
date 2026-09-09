# DeepLabV3+ with ResNet-50 (dilated, output stride 8): the classic CNN baseline.
_base_ = [
    'mmseg::_base_/models/deeplabv3plus_r50-d8.py',   # from the installed MMSegmentation
    './_base_/dataset.py',
    './_base_/schedule.py',
    'mmseg::_base_/default_runtime.py',
]
crop_size = {{_base_.crop_size}}
num_classes = {{_base_.num_classes}}
norm_cfg = dict(type='BN', requires_grad=True)         # single GPU: BN instead of SyncBN

model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=num_classes, norm_cfg=norm_cfg,
                     loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                                  dict(type='DiceLoss', loss_weight=1.0)]),
    auxiliary_head=dict(num_classes=num_classes, norm_cfg=norm_cfg),
    test_cfg={{_base_.slide_test_cfg}})
