# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssdlite_mobilenet_v2

task_adapt = None
cudnn_benchmark = True

evaluation = dict(
    _delete_=True,
    interval=1,
    metric="bbox",
    classwise=True,
)

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (300, 300)

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
        outputs=[
            "FeatureExtractor/MobilenetV2/expanded_conv_13/expand/Relu6||BoxPredictor_0/BoxEncodingPredictor_depthwise/depthwise",
            "FeatureExtractor/MobilenetV2/expanded_conv_13/expand/Relu6||BoxPredictor_0/ClassPredictor_depthwise/depthwise",
            "FeatureExtractor/MobilenetV2/Conv_1/Relu6"],
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
                "FeatureExtractor/MobilenetV2/layer_19_1_Conv2d_2_1x1_256/BatchNorm/batchnorm/mul_1",
                "FeatureExtractor/MobilenetV2/layer_19_1_Conv2d_3_1x1_128/BatchNorm/batchnorm/mul_1",
                "FeatureExtractor/MobilenetV2/layer_19_1_Conv2d_4_1x1_128/BatchNorm/batchnorm/mul_1",
                "FeatureExtractor/MobilenetV2/layer_19_1_Conv2d_5_1x1_64/BatchNorm/batchnorm/mul_1",
            ],
        ),
        outputs=dict(
            extra_layers=[
                "FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_2_3x3_s2_512/Relu6",
                "FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_3_3x3_s2_256/Relu6",
                "FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_4_3x3_s2_256/Relu6",
                "FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_5_3x3_s2_128/Relu6",
            ],
        ),
    ),
    bbox_head=dict(
        type="MMOVSSDHead",
        verify_shape=False,
        transpose_reg=False,
        transpose_cls=False,
        background_index=0,
        inputs=dict(
            reg_convs=[
                "BoxPredictor_0/BoxEncodingPredictor_depthwise/depthwise",
                "BoxPredictor_1/BoxEncodingPredictor_depthwise/depthwise",
                "BoxPredictor_2/BoxEncodingPredictor_depthwise/depthwise",
                "BoxPredictor_3/BoxEncodingPredictor_depthwise/depthwise",
                "BoxPredictor_4/BoxEncodingPredictor_depthwise/depthwise",
                "BoxPredictor_5/BoxEncodingPredictor_depthwise/depthwise",
            ],
            cls_convs=[
                "BoxPredictor_0/ClassPredictor_depthwise/depthwise",
                "BoxPredictor_1/ClassPredictor_depthwise/depthwise",
                "BoxPredictor_2/ClassPredictor_depthwise/depthwise",
                "BoxPredictor_3/ClassPredictor_depthwise/depthwise",
                "BoxPredictor_4/ClassPredictor_depthwise/depthwise",
                "BoxPredictor_5/ClassPredictor_depthwise/depthwise",
            ],
        ),
        outputs=dict(
            reg_convs=[
                "BoxPredictor_0/BoxEncodingPredictor/BiasAdd/Add",
                "BoxPredictor_1/BoxEncodingPredictor/BiasAdd/Add",
                "BoxPredictor_2/BoxEncodingPredictor/BiasAdd/Add",
                "BoxPredictor_3/BoxEncodingPredictor/BiasAdd/Add",
                "BoxPredictor_4/BoxEncodingPredictor/BiasAdd/Add",
                "BoxPredictor_5/BoxEncodingPredictor/BiasAdd/Add",
            ],
            cls_convs=[
                "BoxPredictor_0/ClassPredictor/BiasAdd/Add",
                "BoxPredictor_1/ClassPredictor/BiasAdd/Add",
                "BoxPredictor_2/ClassPredictor/BiasAdd/Add",
                "BoxPredictor_3/ClassPredictor/BiasAdd/Add",
                "BoxPredictor_4/ClassPredictor/BiasAdd/Add",
                "BoxPredictor_5/ClassPredictor/BiasAdd/Add",
            ],
        ),
        num_classes=90,
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
