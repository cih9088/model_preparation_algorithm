# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd512

task_adapt = None
cudnn_benchmark = True

evaluation = dict(_delete_=True, start=0, metric="mAP", save_best="mAP")

norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (512, 512)

test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
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
        model_path="public/ssd512/FP32/ssd512.xml",
        outputs=["relu4_3||7439", "relu7"],
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
                "conv6_1/WithoutBiases",
                "conv7_1/WithoutBiases",
                "conv8_1/WithoutBiases",
                "conv9_1/WithoutBiases",
                "conv10_1/WithoutBiases",
            ],
            l2_norm=["7439"],
        ),
        outputs=dict(
            extra_layers=[
                "conv6_2_relu",
                "conv7_2_relu",
                "conv8_2_relu",
                "conv9_2_relu",
                "conv10_2_relu",
            ],
            l2_norm=["conv4_3_norm"],
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
                "conv4_3_norm_mbox_loc/WithoutBiases",
                "fc7_mbox_loc/WithoutBiases",
                "conv6_2_mbox_loc/WithoutBiases",
                "conv7_2_mbox_loc/WithoutBiases",
                "conv8_2_mbox_loc/WithoutBiases",
                "conv9_2_mbox_loc/WithoutBiases",
                "conv10_2_mbox_loc/WithoutBiases",
            ],
            cls_convs=[
                "conv4_3_norm_mbox_conf/WithoutBiases",
                "fc7_mbox_conf/WithoutBiases",
                "conv6_2_mbox_conf/WithoutBiases",
                "conv7_2_mbox_conf/WithoutBiases",
                "conv8_2_mbox_conf/WithoutBiases",
                "conv9_2_mbox_conf/WithoutBiases",
                "conv10_2_mbox_conf/WithoutBiases",
            ],
        ),
        outputs=dict(
            reg_convs=[
                "conv4_3_norm_mbox_loc",
                "fc7_mbox_loc",
                "conv6_2_mbox_loc",
                "conv7_2_mbox_loc",
                "conv8_2_mbox_loc",
                "conv9_2_mbox_loc",
                "conv10_2_mbox_loc",
            ],
            cls_convs=[
                "conv4_3_norm_mbox_conf",
                "fc7_mbox_conf",
                "conv6_2_mbox_conf",
                "conv7_2_mbox_conf",
                "conv8_2_mbox_conf",
                "conv9_2_mbox_conf",
                "conv10_2_mbox_conf",
            ],
        ),
        num_classes=20,
        anchor_generator=dict(
            type="SSDAnchorGenerator",
            scale_major=False,
            input_size=512,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
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
        score_thr=0.02,
        max_per_img=200,
    ),
)
