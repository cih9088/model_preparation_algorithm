# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-ssd

task_adapt = None
cudnn_benchmark = True

evaluation = dict(_delete_=True, start=0, metric="mAP", save_best="mAP")

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (300, 300)

test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    test=dict(
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
        type="MMOVSSDHead",
        verify_shape=False,
        transpose_reg=False,
        transpose_cls=False,
        background_index=0,
        inputs=dict(
            reg_convs=[
                "conv11_mbox_loc/WithoutBiases",
                "conv13_mbox_loc/WithoutBiases",
                "conv14_2_mbox_loc/WithoutBiases",
                "conv15_2_mbox_loc/WithoutBiases",
                "conv16_2_mbox_loc/WithoutBiases",
                "conv17_2_mbox_loc/WithoutBiases",
            ],
            cls_convs=[
                "conv11_mbox_conf/WithoutBiases",
                "conv13_mbox_conf/WithoutBiases",
                "conv14_2_mbox_conf/WithoutBiases",
                "conv15_2_mbox_conf/WithoutBiases",
                "conv16_2_mbox_conf/WithoutBiases",
                "conv17_2_mbox_conf/WithoutBiases",
            ],
        ),
        outputs=dict(
            reg_convs=[
                "conv11_mbox_loc",
                "conv13_mbox_loc",
                "conv14_2_mbox_loc",
                "conv15_2_mbox_loc",
                "conv16_2_mbox_loc",
                "conv17_2_mbox_loc",
            ],
            cls_convs=[
                "conv11_mbox_conf",
                "conv13_mbox_conf",
                "conv14_2_mbox_conf",
                "conv15_2_mbox_conf",
                "conv16_2_mbox_conf",
                "conv17_2_mbox_conf",
            ],
        ),
        num_classes=20,
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
