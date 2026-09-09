# U-MV-small (MambaVision-S encoder + U-Net decoder), the team's own architecture.
_base_ = [
    './_base_/dataset.py',
    './_base_/schedule.py',
    'mmseg::_base_/default_runtime.py',
]
crop_size = {{_base_.crop_size}}
num_classes = {{_base_.num_classes}}
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(type='SegDataPreProcessor', size=crop_size,
                           mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                           bgr_to_rgb=True, pad_val=0, seg_pad_val=255),
    backbone=dict(type='MambaVisionBackbone', variant='small', pretrained=True),
    decode_head=dict(type='GenericUNetHead', encoder_channels=[96, 192, 384, 768],
                     decoder_channels=(256, 128, 64, 32), in_index=[0, 1, 2, 3],
                     num_classes=num_classes, dropout_ratio=0.1, norm_cfg=norm_cfg, align_corners=False,
                     loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                                  dict(type='DiceLoss', loss_weight=3.0)]),
    train_cfg=dict(),
    test_cfg={{_base_.slide_test_cfg}})
