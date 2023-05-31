# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3

task_adapt = None

dataset_type = 'PascalVOCDataset'
dataset_root = "/mnt/data/dataset/voc/VOC2012/"
norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
crop_size = (513, 513)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 513), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
#  test_pipeline = [
#      dict(type='LoadImageFromFile'),
#      dict(
#          type='MultiScaleFlipAug',
#          img_scale=(513, 513),
#          flip=False,
#          transforms=[
#              dict(type='Resize', keep_ratio=False),
#              dict(type='RandomFlip'),
#              dict(type='Normalize', **norm_cfg),
#              dict(type='ImageToTensor', keys=['img']),
#              dict(type='Collect', keys=['img']),
#          ])
#  ]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(513, 513),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Pad',
                size=crop_size,
                pad_val=(127.5, 127.5, 127.5),
                seg_pad_val=255
            ),
            dict(type='RandomFlip'),
            dict(type='Normalize', **norm_cfg),
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
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/train.txt',
        #  ann_dir=['SegmentationClass', 'SegmentationClassAug'],
        #  split=[
        #      'ImageSets/Segmentation/train.txt',
        #      'ImageSets/Segmentation/aug.txt'
        #  ])),
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

runner = dict(
    max_epochs=10
)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MMOVBackbone',
        outputs="MobilenetV2/expanded_conv_16/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_",
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    decode_head=dict(
        type='MMOVDecodeHead',
        verify_shape=False,
        inputs=dict(
            extractor=["AvgPool2D/AvgPool", "aspp0/Conv2D"],
            cls_seg="logits/semantic/Conv2D",
        ),
        outputs=dict(
            extractor="concat_projection/Relu",
            cls_seg="logits/semantic/BiasAdd/Add",
        ),
        in_channels=320,
        in_index=0,
        input_transform=None,
        num_classes=21,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    #  auxiliary_head=dict(
    #      type='FCNHead',
    #      in_channels=1024,
    #      in_index=2,
    #      channels=256,
    #      num_convs=1,
    #      concat_input=False,
    #      dropout_ratio=0.1,
    #      num_classes=19,
    #      norm_cfg=norm_cfg,
    #      align_corners=False,
    #      loss_decode=dict(
    #          type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
