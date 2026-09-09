# 10-band Sentinel-2 L2A tiles (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12), uint16 reflectance x 10000,
# masks 0..6 with 255 = unlabeled (ignored).  Values are scaled to reflectance in [0, ~1].
custom_imports = dict(imports=['geoseg'], allow_failed_imports=False)

data_root = '/data/lulc_s2'
metainfo = dict(
    classes=('background', 'built-up', 'road', 'water', 'bare', 'vegetation', 'agriculture'),
    palette=[[0, 0, 0], [220, 60, 30], [255, 200, 0], [30, 90, 220], [200, 170, 120], [40, 150, 60], [150, 220, 80]])
num_classes = len(metainfo['classes'])
img_suffix = '.tif'
seg_map_suffix = '.tif'
crop_size = (256, 256)
bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# sliding-window inference over whole tiles, half-window stride
slide_test_cfg = dict(mode='slide', crop_size=crop_size, stride=(crop_size[0] // 2, crop_size[1] // 2))
dataset_type = 'BaseSegDataset'

train_pipeline = [
    dict(type='LoadRasterioImage', bands=bands, scale=1 / 10000.0, clip=(0.0, 1.5)),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=crop_size, ratio_range=(0.75, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # no PhotoMetricDistortion: it assumes 8-bit BGR images
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadRasterioImage', bands=bands, scale=1 / 10000.0, clip=(0.0, 1.5)),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

_common = dict(type=dataset_type, data_root=data_root, metainfo=metainfo,
               img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=False)
train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(**_common, data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(**_common, data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                 pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(**_common, data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
                 pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
