# U-MV-small: MambaVision-S-1K encoder (96/192/384/768) + U-Net decoder.
_base_ = [
    '../_base_/models/umv_unet.py',
    '../_base_/datasets/uav_acacia_dataset.py',
    '../_base_/schedules/schedule_100k_adamw.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(variant='small'),
    decode_head=dict(encoder_channels=[96, 192, 384, 768]))
