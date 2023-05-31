# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tf

task_adapt = None

runner = dict(max_epochs=273)

evaluation = dict(
    metric="bbox",
    save_best="bbox_mAP",
    classwise=True,
)

custom_hooks = [
    dict(
        type="LazyEarlyStoppingHook",
        start=3,
        patience=10,
        iteration_patience=0,
        metric="bbox_mAP",
        interval=1,
        priority=75,
    ),
]

optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy="step",
    warmup="linear",
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246],
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

# dataset settings
dataset_type = "CocoDataset"
dataset_root = "/mnt/data/dataset/coco/"
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (416, 416)

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Expand",
        mean=img_norm_cfg["mean"],
        to_rgb=img_norm_cfg["to_rgb"],
        ratio_range=(1, 2),
    ),
    dict(
        type="MinIoURandomCrop",
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3,
    ),
    dict(type="Resize", img_scale=[(320, 320), img_size], keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
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
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file="annotations/instances_train2017.json",
        img_prefix="images/train2017/",
        pipeline=train_pipeline,
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
    type="YOLOV3",
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/yolo-v3-tf/FP32/yolo-v3-tf.xml",
        outputs=[
            "add_10/add||concatenate_1/concat",
            "add_18/add||concatenate/concat",
            "add_22/add",
        ],
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    neck=dict(
        type="MMOVYOLOV3Neck",
        verify_shape=False,
        inputs=dict(
            detect1="conv2d_52/Conv2D",
            detect2="conv2d_60/Conv2D",
            detect3="conv2d_68/Conv2D",
            conv1="conv2d_59/Conv2D",
            conv2="conv2d_67/Conv2D",
        ),
        outputs=dict(
            detect1="leaky_re_lu_56/LeakyRelu",
            detect2="leaky_re_lu_63/LeakyRelu",
            detect3="leaky_re_lu_70/LeakyRelu",
            conv1="leaky_re_lu_58/LeakyRelu",
            conv2="leaky_re_lu_65/LeakyRelu",
        ),
    ),
    bbox_head=dict(
        type="YOLOV3Head",
        num_classes=80,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type="YOLOAnchorGenerator",
            base_sizes=[
                [(116, 90), (156, 198), (373, 326)],
                [(30, 61), (62, 45), (59, 119)],
                [(10, 13), (16, 30), (33, 23)],
            ],
            strides=[32, 16, 8],
        ),
        bbox_coder=dict(type="YOLOBBoxCoder"),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0, reduction="sum"
        ),
        loss_conf=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0, reduction="sum"
        ),
        loss_xy=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=2.0, reduction="sum"
        ),
        loss_wh=dict(type="MSELoss", loss_weight=2.0, reduction="sum"),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="GridAssigner", pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0
        )
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type="nms", iou_threshold=0.45),
        max_per_img=100,
    ),
)

cudnn_benchmark = True
