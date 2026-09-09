# U-MV-tiny: MambaVision-T-1K encoder (80/160/320/640) + U-Net decoder.
#
# Schedule verified against the training log of the released checkpoint
# (work directory mambavision-t_generic-unet_acacia, 20250820_065211.log):
# base lr 1e-4 with backbone multiplier 0.1, 100 000 iterations, validation
# every 5 000; best validation mIoU 87.91 % at iteration 100 000
# (= best_mIoU_iter_100000.pth = Pretrained_Weights/U-MV-tiny_latest.pth).
# An earlier revision of this file carried lr 1e-5 and 150 000 iterations,
# which did not correspond to the run; see docs/07_reproducibility.md.
_base_ = [
    '../_base_/models/umv_unet.py',
    '../_base_/datasets/uav_acacia_dataset.py',
    '../_base_/schedules/schedule_100k_adamw.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(variant='tiny'),
    decode_head=dict(encoder_channels=[80, 160, 320, 640]))
