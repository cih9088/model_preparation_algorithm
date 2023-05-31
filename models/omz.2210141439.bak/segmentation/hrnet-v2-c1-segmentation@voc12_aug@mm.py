# https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr18_512x512_20k_voc12aug.py

task_adapt = None

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(_delete_=True, interval=2000, metric=['mIoU', 'mDice'], pre_eval=True)
custom_hooks = []

# optimizer
optimizer = dict(_delete_=True, type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

# learning policy
lr_config = dict(_delete_=True, policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

dataset_type = 'PascalVOCDataset'
dataset_root = "/mnt/data/dataset/voc/VOC2012/"
img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1024, 320)
crop_size = (320, 320)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        img_dir='JPEGImages',
        ann_dir=['SegmentationClass', 'SegmentationClassAug'],
        split=[
            'ImageSets/Segmentation/train.txt',
            'ImageSets/Segmentation/trainaug.txt'
        ],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        test_mode=True,
        pipeline=test_pipeline))

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144)))),
    decode_head=dict(
        type='FCNHead',
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        channels=sum([18, 36, 72, 144]),
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

cudnn_benchmark = True
