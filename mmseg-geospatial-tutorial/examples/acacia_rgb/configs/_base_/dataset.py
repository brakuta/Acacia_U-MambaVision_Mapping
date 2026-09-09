# Dataset definition shared by every model config of this project.
# Edit the four blocks marked EDIT.  Nothing else needs changing for a new
# 8-bit RGB project.  For multispectral data see examples/lulc_multispectral.
custom_imports = dict(imports=['geoseg'], allow_failed_imports=False)

# ---- EDIT 1: where the tiles are (inside the container: /data/<project>)
data_root = '/data/acacia'

# ---- EDIT 2: classes and display colours (index 0 first)
metainfo = dict(
    classes=('background', 'acacia'),
    palette=[[0, 0, 0], [255, 0, 37]])

# ---- EDIT 3: file suffixes of image and mask tiles
num_classes = len(metainfo['classes'])
img_suffix = '.tif'
seg_map_suffix = '.tif'

# ---- EDIT 4: tile size used for training crops
crop_size = (1024, 1024)   # a 5 m crown at 2.5 cm GSD needs its shadow and neighbours as context

# sliding-window inference over whole tiles, half-window stride
slide_test_cfg = dict(mode='slide', crop_size=crop_size, stride=(crop_size[0] // 2, crop_size[1] // 2))
dataset_type = 'BaseSegDataset'   # generic dataset; classes come from `metainfo`

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=crop_size, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]
img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='TestTimeAug', transforms=[
        [dict(type='Resize', scale_factor=r, keep_ratio=True) for r in img_ratios],
        [dict(type='RandomFlip', prob=0., direction='horizontal'),
         dict(type='RandomFlip', prob=1., direction='horizontal')],
        [dict(type='LoadAnnotations')],
        [dict(type='PackSegInputs')],
    ])
]

_common = dict(type=dataset_type, data_root=data_root, metainfo=metainfo,
               img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=False)

train_dataloader = dict(
    batch_size=2, num_workers=8, persistent_workers=True,   # 1024 x 1024 tiles: batch 2 on a 24 GB GPU
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(**_common, data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(**_common, data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                 pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(**_common, data_prefix=dict(img_path='img_dir/test2', seg_map_path='ann_dir/test2'),
                 pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
