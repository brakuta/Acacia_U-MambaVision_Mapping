# SegFormer with MiT-B2: a lightweight transformer baseline (ImageNet-pretrained encoder).
_base_ = [
    'mmseg::_base_/models/segformer_mit-b0.py',
    './_base_/dataset.py',
    './_base_/schedule.py',
    'mmseg::_base_/default_runtime.py',
]
crop_size = {{_base_.crop_size}}
num_classes = {{_base_.num_classes}}
# ImageNet-1K weights converted for MMSegmentation (download once with tools/download_zoo_weights.py)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'

model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
                  embed_dims=64, num_layers=[3, 4, 6, 3]),        # B0 -> B2
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=num_classes),
    test_cfg={{_base_.slide_test_cfg}})

# SegFormer trains its head 10x faster than the encoder and without weight decay on norms
optim_wrapper = dict(
    optimizer=dict(lr=6e-5, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.), 'norm': dict(decay_mult=0.),
                                    'head': dict(lr_mult=10.)}))
