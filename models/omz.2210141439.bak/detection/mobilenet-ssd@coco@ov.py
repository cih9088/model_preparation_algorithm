# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-ssd

task_adapt = None

runner = dict(max_epochs=24)

evaluation = dict(
    _delete_=True,
    metric="bbox",
    save_best="bbox_mAP",
    classwise=True,
)

optimizer = dict(type="SGD", lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)
custom_hooks = []

lr_config = dict(
    _delete_=True,
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22],
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

dataset_type = "CocoDataset"
dataset_root = "/mnt/data/dataset/coco/"
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
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    _delete_=True,
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type="RepeatDataset",
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=dataset_root,
            ann_file="annotations/instances_train2017.json",
            img_prefix="images/train2017/",
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file="annotations/instances_val2017.json",
        img_prefix="images/val2017/",
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file="annotations/instances_val2017.json",
        img_prefix="images/val2017/",
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

model = dict(
    super_type=None,
    type="SingleStageDetector",
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/mobilenet-ssd/FP32/mobilenet-ssd.xml",
        outputs=["conv11/relu"],
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
                "conv12/dw/WithoutBiases",
                "conv14_1/WithoutBiases",
                "conv15_1/WithoutBiases",
                "conv16_1/WithoutBiases",
                "conv17_1/WithoutBiases",
            ],
        ),
        outputs=dict(
            extra_layers=[
                "conv13/relu",
                "conv14_2/relu",
                "conv15_2/relu",
                "conv16_2/relu",
                "conv17_2/relu",
            ],
        ),
    ),
    bbox_head=dict(
        type="SSDHead",
        in_channels=(512, 1024, 512, 256, 256, 128),
        num_classes=80,
        anchor_generator=dict(
            type="SSDAnchorGenerator",
            scale_major=False,
            input_size=img_size[0],
            basesize_ratio_range=(0.35, 0.95),
            strides=[16, 32, 64, 100, 150, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
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
        nms=dict(type="nms", iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.25,
        max_per_img=100,
    ),
)
