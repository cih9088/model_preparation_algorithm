# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd-resnet34-1200-onnx

task_adapt = None
cudnn_benchmark = True

evaluation = dict(
    _delete_=True,
    interval=1,
    metric="bbox",
    classwise=True,
)

norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (1200, 1200)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    test=dict(
        pipeline=test_pipeline,
    ),
)

model = dict(
    super_type=None,
    type="SingleStageDetector",
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.xml",
        outputs="Relu_317",
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
                "Convolution_347",
                "Convolution_445",
                "Convolution_543",
                "Convolution_641",
                "Convolution_739",
            ],
        ),
        outputs=dict(
            extra_layers=[
                "Relu_321",
                "Relu_325",
                "Relu_329",
                "Relu_333",
                "Relu_337",
            ],
        ),
    ),
    bbox_head=dict(
        type="MMOVSSDHead",
        verify_shape=False,
        transpose_reg=True,
        transpose_cls=True,
        background_index=0,
        inputs=dict(
            reg_convs=[
                "Convolution_837",
                "Convolution_1043",
                "Convolution_1249",
                "Convolution_1455",
                "Convolution_1661",
                "Convolution_1867",
            ],
            cls_convs=[
                "Convolution_940",
                "Convolution_1146",
                "Convolution_1352",
                "Convolution_1558",
                "Convolution_1764",
                "Convolution_1970",
            ],
        ),
        outputs=dict(
            reg_convs=[
                "Conv_338",
                "Conv_360",
                "Conv_382",
                "Conv_404",
                "Conv_426",
                "Conv_448",
            ],
            cls_convs=[
                "Conv_349",
                "Conv_371",
                "Conv_393",
                "Conv_415",
                "Conv_437",
                "Conv_459",
            ],
        ),
        num_classes=80,
        anchor_generator=dict(
            type="SSDAnchorGenerator",
            scale_major=False,
            input_size=img_size[0],
            basesize_ratio_range=(0.15, 0.9),
            strides=[24, 48, 92, 171, 400, 400],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
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
        nms=dict(type="nms", iou_threshold=0.5),
        min_bbox_size=0,
        score_thr=0.50,
        max_per_img=200,
    ),
)
