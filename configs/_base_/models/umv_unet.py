# U-MV: MambaVision encoder + lightweight U-Net decoder.
# Variant-specific fields (backbone.variant, decode_head.encoder_channels)
# are set in configs/mambavision/U-MV-*.py.
num_classes = 2
norm_cfg = dict(type='BN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(1024, 1024),
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
        type='MambaVisionBackbone',
        variant='small',
        pretrained=True,        # ImageNet-1K initialisation from the HF Hub
        local_files_only=None,  # None -> follow HF_HUB_OFFLINE
    ),
    decode_head=dict(
        type='GenericUNetHead',
        encoder_channels=[96, 192, 384, 768],
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
