# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd_mobilenet_v1_fpn_coco

task_adapt = None

evaluation = dict(
    _delete_=True,
    interval=1,
    metric="bbox",
    classwise=True,
)

dataset_type = "CocoDataset"
dataset_root = "/mnt/data/dataset/coco/"
norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (640, 640)

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
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(),
    val=dict(),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file="annotations/instances_val2017.json",
        img_prefix="images/val2017/",
        with_classes_from_paper=True,
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
        outputs=[
            "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6||FeatureExtractor/MobilenetV1/fpn/top_down/projection_1/Conv2D",
            "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6||FeatureExtractor/MobilenetV1/fpn/top_down/projection_2/Conv2D",
            "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6",
        ],
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    neck=dict(
        type="MMOVFPN",
        verify_shape=False,
        inputs=dict(
            laterals=[
                "FeatureExtractor/MobilenetV1/fpn/top_down/projection_1/Conv2D",
                "FeatureExtractor/MobilenetV1/fpn/top_down/projection_2/Conv2D",
                "FeatureExtractor/MobilenetV1/fpn/top_down/projection_3/Conv2D",
            ],
            fpn=[
                "FeatureExtractor/MobilenetV1/fpn/top_down/smoothing_1/Conv2D",
                "FeatureExtractor/MobilenetV1/fpn/top_down/smoothing_2/Conv2D",
                "",
                "FeatureExtractor/MobilenetV1/fpn/bottom_up_Conv2d_14/Conv2D",
                "FeatureExtractor/MobilenetV1/fpn/bottom_up_Conv2d_15/Conv2D",
            ],
        ),
        outputs=dict(
            laterals=[
                "FeatureExtractor/MobilenetV1/fpn/top_down/projection_1/BiasAdd/Add",
                "FeatureExtractor/MobilenetV1/fpn/top_down/projection_2/BiasAdd/Add",
                "FeatureExtractor/MobilenetV1/fpn/top_down/projection_3/BiasAdd/Add",
            ],
            fpn=[
                "FeatureExtractor/MobilenetV1/fpn/top_down/smoothing_1/Relu6",
                "FeatureExtractor/MobilenetV1/fpn/top_down/smoothing_2/Relu6",
                "",
                "FeatureExtractor/MobilenetV1/fpn/bottom_up_Conv2d_14/Relu6",
                "FeatureExtractor/MobilenetV1/fpn/bottom_up_Conv2d_15/Relu6",
            ],
        ),
        add_extra_convs='on_lateral',
        num_outs=5,
    ),

    #  neck=dict(
    #      type="MMOVSSDNeck",
    #      verify_shape=False,
    #      inputs=dict(
    #          extra_layers=[[
    #              "FeatureExtractor/MobilenetV1/fpn/top_down/projection_1/Conv2D",
    #              "FeatureExtractor/MobilenetV1/fpn/top_down/projection_2/Conv2D",
    #              "FeatureExtractor/MobilenetV1/fpn/top_down/projection_3/Conv2D",
    #          ]],
    #      ),
    #      outputs=dict(
    #          extra_layers=[[
    #              "FeatureExtractor/MobilenetV1/fpn/top_down/smoothing_1/Relu6",
    #              "FeatureExtractor/MobilenetV1/fpn/top_down/smoothing_2/Relu6",
    #              "FeatureExtractor/MobilenetV1/fpn/top_down/projection_3/BiasAdd/Add||WeightSharedConvolutionalBoxPredictor_2/BoxPredictionTower/conv2d_0/Conv2D",
    #              "FeatureExtractor/MobilenetV1/fpn/top_down/projection_3/BiasAdd/Add||WeightSharedConvolutionalBoxPredictor_2/ClassPredictionTower/conv2d_0/Conv2D",
    #              "FeatureExtractor/MobilenetV1/fpn/bottom_up_Conv2d_14/Relu6||WeightSharedConvolutionalBoxPredictor_3/ClassPredictionTower/conv2d_0/Conv2D",
    #              "FeatureExtractor/MobilenetV1/fpn/bottom_up_Conv2d_14/Relu6||WeightSharedConvolutionalBoxPredictor_3/BoxPredictionTower/conv2d_0/Conv2D",
    #              "FeatureExtractor/MobilenetV1/fpn/bottom_up_Conv2d_15/Relu6",
    #          ]],
    #      ),
    #  ),
    bbox_head=dict(
        type="MMOVSSDHead",
        verify_shape=False,
        transpose_reg=False,
        transpose_cls=False,
        background_index=0,
        inputs=dict(
            reg_convs=[
                "WeightSharedConvolutionalBoxPredictor/BoxPredictionTower/conv2d_0/Conv2D",
                "WeightSharedConvolutionalBoxPredictor_1/BoxPredictionTower/conv2d_0/Conv2D",
                "WeightSharedConvolutionalBoxPredictor_2/BoxPredictionTower/conv2d_0/Conv2D",
                "WeightSharedConvolutionalBoxPredictor_3/BoxPredictionTower/conv2d_0/Conv2D",
                "WeightSharedConvolutionalBoxPredictor_4/BoxPredictionTower/conv2d_0/Conv2D",
            ],
            cls_convs=[
                "WeightSharedConvolutionalBoxPredictor/ClassPredictionTower/conv2d_0/Conv2D",
                "WeightSharedConvolutionalBoxPredictor_1/ClassPredictionTower/conv2d_0/Conv2D",
                "WeightSharedConvolutionalBoxPredictor_2/ClassPredictionTower/conv2d_0/Conv2D",
                "WeightSharedConvolutionalBoxPredictor_3/ClassPredictionTower/conv2d_0/Conv2D",
                "WeightSharedConvolutionalBoxPredictor_4/ClassPredictionTower/conv2d_0/Conv2D",
            ],
        ),
        outputs=dict(
            reg_convs=[
                "WeightSharedConvolutionalBoxPredictor/BoxPredictor/BiasAdd/Add",
                "WeightSharedConvolutionalBoxPredictor_1/BoxPredictor/BiasAdd/Add",
                "WeightSharedConvolutionalBoxPredictor_2/BoxPredictor/BiasAdd/Add",
                "WeightSharedConvolutionalBoxPredictor_3/BoxPredictor/BiasAdd/Add",
                "WeightSharedConvolutionalBoxPredictor_4/BoxPredictor/BiasAdd/Add",
            ],
            cls_convs=[
                "WeightSharedConvolutionalBoxPredictor/ClassPredictor/BiasAdd/Add",
                "WeightSharedConvolutionalBoxPredictor_1/ClassPredictor/BiasAdd/Add",
                "WeightSharedConvolutionalBoxPredictor_2/ClassPredictor/BiasAdd/Add",
                "WeightSharedConvolutionalBoxPredictor_3/ClassPredictor/BiasAdd/Add",
                "WeightSharedConvolutionalBoxPredictor_4/ClassPredictor/BiasAdd/Add",
            ],
        ),
        num_classes=90,
        anchor_generator=dict(
            type="SSDAnchorGeneratorClustered",
            strides=[8, 16, 32, 64, 128],
            heights=[
                [32, 45.2548, 22.6274, 32, 45.2548, 64],
                [64, 90.5097, 45.2548, 64, 90.5097, 128],
                [128, 181.019, 90.5097, 128, 181.019, 256],
                [256, 362.039, 181.019, 256, 362.039, 512],
                [512, 724.077, 362.039, 512, 724.077, 1024],
            ],
            widths=[
                [32, 45.2548, 45.2548, 64, 22.6274, 32],
                [64, 90.5097, 90.5097, 128, 45.2548, 64],
                [128, 181.019, 181.019, 256, 90.5097, 128],
                [256, 362.039, 362.039, 512, 181.019, 256],
                [512, 724.077, 724.077, 1024, 362.039, 512],
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
