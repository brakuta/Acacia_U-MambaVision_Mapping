# dataset settings
dataset_type = 'ADE20KDataset'

data_root = r'/data'
#data_root = r'H:\UAV_Data'   


crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'), #for RGB
    #dict(type='LoadSingleRSImageFromFile'), #for Multichannel
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'), #for RGB
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'), #for RGB
    #dict(type='LoadSingleRSImageFromFile'), #for Multichannel
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None), #for RGB
    #dict(type='LoadSingleRSImageFromFile', backend_args=None), #for Multichannel
    #dict(type='LoadImageFromFile')
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=42,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False, 
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
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
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))


test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False, 
        data_prefix=dict(
            img_path='img_dir/Generalizability',
            seg_map_path='ann_dir/Generalizability'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mFscore'])
test_evaluator = val_evaluator
