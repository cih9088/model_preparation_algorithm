# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tf

task_adapt = None

evaluation = dict(
    metric='bbox',
    classwise=True,
)

dataset_type = "CocoDataset"
dataset_root = "/mnt/data/dataset/coco/"
norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (416, 416)

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type="Normalize", **norm_cfg),
            dict(type="Pad", size=img_size),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    _delete_=True,
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(),
    val=dict(),
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
    type="YOLOV3",
    backbone=dict(
        type="MMOVBackbone",
        model_path='public/yolo-v3-tf/FP32/yolo-v3-tf.xml',
        outputs=[
            "add_10/add||concatenate_1/concat",
            "add_18/add||concatenate/concat",
            "add_22/add"
        ],
        remove_normalize=False,
        merge_bn=False,
        paired_bn=False,
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
            conv2="leaky_re_lu_65/LeakyRelu"
        ),
    ),
    bbox_head=dict(
        type='MMOVYOLOV3Head',
        verify_shape=False,
        inputs=dict(
            convs_bridge=[
                "conv2d_57/Conv2D",
                "conv2d_65/Conv2D",
                "conv2d_73/Conv2D"
            ],
            convs_pred=[
                "conv2d_58/Conv2D",
                "conv2d_66/Conv2D",
                "conv2d_74/Conv2D"
            ]
        ),
        outputs=dict(
            convs_bridge=[
                "leaky_re_lu_57/LeakyRelu",
                "leaky_re_lu_64/LeakyRelu",
                "leaky_re_lu_71/LeakyRelu"
            ],
            convs_pred=[
                "conv2d_58/Conv2D",
                "conv2d_66/Conv2D",
                "conv2d_74/Conv2D"
            ]
        ),
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        num_classes=80,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(
            type='MSELoss',
            loss_weight=2.0,
            reduction='sum'
        )
    ),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0
        )
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(
            type='nms',
            iou_threshold=0.45
        ),
        max_per_img=100
    )
)
