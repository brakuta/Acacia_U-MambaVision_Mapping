# UAV Acacia tortilis crown dataset (binary: background / acacia).
#
# Expected layout under `data_root`:
#   img_dir/{train,val,test,Generalizability}/*.tif
#   ann_dir/{train,val,test,Generalizability}/*.tif   (single-band index masks, 0/1)
#
# Override `data_root` from the command line without editing this file:
#   python tools/train.py <config> --cfg-options data_root=/data/UAV_Acacia_Data \
#       train_dataloader.dataset.data_root=/data/UAV_Acacia_Data ...
# or simply mount the dataset at /data inside the Docker container.
custom_imports = dict(imports=['umv'], allow_failed_imports=False)

dataset_type = 'UAVAcaciaDataset'
data_root = '/data'

crop_size = (1024, 1024)
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')],
        ])
]

# num_workers: 8 is a portable default; the original experiments used a
# 42-worker loader on a 64 GB workstation. Raise it if CPU/RAM permit.
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

# In-distribution test split. For the out-of-distribution split use
#   python tools/test.py <config> <ckpt> --test-split Generalizability
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
