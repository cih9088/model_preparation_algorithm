# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd_mobilenet_v1_coco

task_adapt = None

runner = dict(max_epochs=1024)

evaluation = dict(
    _delete_=True,
    interval=2,
    metric="mAP",
    save_best="mAP",
)

custom_hooks = []

optimizer = dict(type="SGD", lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)

lr_config = dict(
    _delete_=True,
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[682, 938],
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

dataset_type = "VOCDataset"
dataset_root = "/mnt/data/dataset/voc/VOC2007/"
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (300, 300)

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Expand",
        mean=img_norm_cfg["mean"],
        to_rgb=img_norm_cfg["to_rgb"],
        ratio_range=(1, 4),
    ),
    dict(
        type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3
    ),
    dict(type="Resize", img_scale=img_size, keep_ratio=False),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size,
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
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix=dataset_root,
        ann_file=dataset_root + "ImageSets/Layout/train.txt",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        img_prefix=dataset_root,
        ann_file=dataset_root + "ImageSets/Layout/val.txt",
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        img_prefix=dataset_root,
        ann_file=dataset_root + "ImageSets/Layout/test.txt",
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

model = dict(
    super_type=None,
    type="SingleStageDetector",
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.xml",
        outputs=["FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6"],
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    neck=dict(
        type="MMOVSSDNeck",
        verify_shape=False,
        inputs=dict(
            extra_layers=[
                "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise",
                "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/batchnorm/mul_1",
                "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/batchnorm/mul_1",
                "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/batchnorm/mul_1",
                "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/BatchNorm/batchnorm/mul_1",
            ],
        ),
        outputs=dict(
            extra_layers=[
                "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6",
                "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Relu6",
                "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Relu6",
                "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Relu6",
                "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6",
            ],
        ),
    ),
    bbox_head=dict(
        type="SSDHead",
        in_channels=(512, 1024, 512, 256, 256, 128),
        num_classes=20,
        anchor_generator=dict(
            type="SSDAnchorGeneratorClustered",
            strides=[16, 30, 60, 100, 150, 300],
            heights=[
                [30, 42.4264, 84.8528],
                [105, 74.2462, 148.492, 60.6218, 181.874, 125.499],
                [150, 106.066, 212.132, 86.6025, 259.821, 171.026],
                [195, 137.886, 275.772, 112.583, 337.767, 216.333],
                [240, 169.706, 339.411, 138.564, 415.713, 261.534],
                [285, 201.525, 403.051, 164.545, 493.659, 292.404],
            ],
            widths=[
                [30, 84.8528, 42.4264],
                [105, 148.492, 74.2462, 181.865, 60.6187, 125.499],
                [150, 212.132, 106.066, 259.808, 86.5982, 171.026],
                [195, 275.772, 137.886, 337.75, 112.578, 216.333],
                [240, 339.411, 169.706, 415.692, 138.557, 261.534],
                [285, 403.051, 201.525, 493.634, 164.537, 292.404],
            ],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False,
        ),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type="nms", iou_threshold=0.6),
        min_bbox_size=0,
        score_thr=0.30,
        max_per_img=100,
    ),
)
