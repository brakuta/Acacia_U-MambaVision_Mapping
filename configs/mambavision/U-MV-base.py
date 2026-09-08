# U-MV-base: MambaVision-B-1K encoder (128/256/512/1024) + U-Net decoder.
_base_ = [
    '../_base_/models/umv_unet.py',
    '../_base_/datasets/uav_acacia_dataset.py',
    '../_base_/schedules/schedule_100k_adamw.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(variant='base'),
    decode_head=dict(encoder_channels=[128, 256, 512, 1024]))
