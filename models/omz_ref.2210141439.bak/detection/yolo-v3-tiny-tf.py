# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tiny-tf

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
    samples_per_gpu=8,
    workers_per_gpu=4,
    test=dict(
        type="CocoDataset",
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
        model_path='public/yolo-v3-tiny-tf/FP32/yolo-v3-tiny-tf.xml',
        outputs=[
            "leaky_re_lu_4/LeakyRelu||concatenate/concat",
            "leaky_re_lu_4/LeakyRelu||max_pooling2d_4/MaxPool"
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
            detect1="max_pooling2d_4/MaxPool",
            detect2="",
            conv1="conv2d_10/Conv2D",
        ),
        outputs=dict(
            detect1="leaky_re_lu_7/LeakyRelu",
            detect2="",
            conv1="leaky_re_lu_9/LeakyRelu",
        ),
    ),
    bbox_head=dict(
        type='MMOVYOLOV3Head',
        verify_shape=False,
        inputs=dict(
            convs_bridge=[
                "conv2d_8/Conv2D",
                "conv2d_11/Conv2D",
            ],
            convs_pred=[
                "conv2d_9/Conv2D",
                "conv2d_12/Conv2D",
            ],
        ),
        outputs=dict(
            convs_bridge=[
                "leaky_re_lu_8/LeakyRelu",
                "leaky_re_lu_10/LeakyRelu",
            ],
            convs_pred=[
                "conv2d_9/Conv2D",
                "conv2d_12/Conv2D",
            ],
        ),
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(81, 82), (135, 169), (344, 319)],
                        [(23, 27), (37, 58), (81, 82)]],
            strides=[32, 16]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16],
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
