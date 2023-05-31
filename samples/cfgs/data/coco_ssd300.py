__dataset_type = "CocoDataset"
__data_root = "/mnt/data/dataset/coco/"

__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)

__train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=__img_norm_cfg['mean'],
        to_rgb=__img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **__img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
__test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **__img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_train2017.json",
        img_prefix=__data_root + "train2017/",
        pipeline=__train_pipeline,
    ),
    val=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_val2017.json",
        img_prefix=__data_root + "val2017/",
        test_mode=True,
        pipeline=__test_pipeline,
    ),
    test=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_val2017.json",
        img_prefix=__data_root + "val2017/",
        test_mode=True,
        pipeline=__test_pipeline,
    ),
)
