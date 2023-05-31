# https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ocrnet/ocrnet_hr18_512x512_20k_voc12aug.py

task_adapt = None

# runtime settings
runner = dict(type="IterBasedRunner", max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(_delete_=True, interval=2000, metric=["mIoU", "mDice"], pre_eval=True)
custom_hooks = []

# optimizer
optimizer = dict(_delete_=True, type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

# learning policy
lr_config = dict(_delete_=True, policy="poly", power=0.9, min_lr=1e-4, by_epoch=False)

dataset_type = "PascalVOCDataset"
dataset_root = "/mnt/data/dataset/voc/VOC2012/"
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_scale = (1024, 2048)
img_size = (1024, 2048)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=img_scale[::-1], ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=img_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=img_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        img_dir="JPEGImages",
        ann_dir=["SegmentationClass", "SegmentationClassAug"],
        split=[
            "ImageSets/Segmentation/train.txt",
            "ImageSets/Segmentation/trainaug.txt",
        ],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        split="ImageSets/Segmentation/val.txt",
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        split="ImageSets/Segmentation/val.txt",
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    backbone=dict(
        type='MMOVBackbone',
        outputs=[
            "relu_260.tmp_0",
            "relu_261.tmp_0",
            "relu_263.tmp_0",
            "relu_267.tmp_0",
        ],
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    #  decode_head=dict(
    #      type='MMOVDecodeHead',
    #      verify_shape=False,
    #      inputs=dict(
    #          extractor=[
    #              "Multiply_61545",
    #              "Multiply_61575",
    #          ],
    #          cls_seg="Multiply_61625",
    #      ),
    #      outputs=dict(
    #          extractor="concat_1.tmp_0",
    #          cls_seg="conv2d_631.tmp_1",
    #      ),
    #      in_channels=[48, 96, 192, 384],
    #      in_index=(0, 1, 2, 3),
    #      input_transform='resize_concat',
    #      num_classes=19,
    #      dropout_ratio=-1,
    #      align_corners=False,
    #      loss_decode=dict(
    #          type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    #
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=21,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=21,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

#  # model settings
#  norm_cfg = dict(type='BN', requires_grad=True)
#  model = dict(
#      type='CascadeEncoderDecoder',
#      num_stages=2,
#      pretrained='open-mmlab://msra/hrnetv2_w18',
#      backbone=dict(
#          type='HRNet',
#          norm_cfg=norm_cfg,
#          norm_eval=False,
#          extra=dict(
#              stage1=dict(
#                  num_modules=1,
#                  num_branches=1,
#                  block='BOTTLENECK',
#                  num_blocks=(4, ),
#                  num_channels=(64, )),
#              stage2=dict(
#                  num_modules=1,
#                  num_branches=2,
#                  block='BASIC',
#                  num_blocks=(4, 4),
#                  num_channels=(18, 36)),
#              stage3=dict(
#                  num_modules=4,
#                  num_branches=3,
#                  block='BASIC',
#                  num_blocks=(4, 4, 4),
#                  num_channels=(18, 36, 72)),
#              stage4=dict(
#                  num_modules=3,
#                  num_branches=4,
#                  block='BASIC',
#                  num_blocks=(4, 4, 4, 4),
#                  num_channels=(18, 36, 72, 144)))),
#      decode_head=[
#          dict(
#              type='FCNHead',
#              in_channels=[18, 36, 72, 144],
#              channels=sum([18, 36, 72, 144]),
#              in_index=(0, 1, 2, 3),
#              input_transform='resize_concat',
#              kernel_size=1,
#              num_convs=1,
#              concat_input=False,
#              dropout_ratio=-1,
#              num_classes=19,
#              norm_cfg=norm_cfg,
#              align_corners=False,
#              loss_decode=dict(
#                  type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
#          dict(
#              type='OCRHead',
#              in_channels=[18, 36, 72, 144],
#              in_index=(0, 1, 2, 3),
#              input_transform='resize_concat',
#              channels=512,
#              ocr_channels=256,
#              dropout_ratio=-1,
#              num_classes=19,
#              norm_cfg=norm_cfg,
#              align_corners=False,
#              loss_decode=dict(
#                  type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#      ],
#      # model training and testing settings
#      train_cfg=dict(),
#      test_cfg=dict(mode='whole'))
