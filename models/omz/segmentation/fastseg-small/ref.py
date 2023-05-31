# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/fastseg-small

task_adapt = None
cudnn_benchmark = True

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_scale = (2048, 1024)
crop_size = (1024, 2048)  # H, W

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=crop_size[::-1],
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
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
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=test_pipeline,
    ),
    test=dict(
        pipeline=test_pipeline,
    ),
)

model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="MMOVBackbone",
        outputs=[
            "Mul_353",
            "Conv_52||Shape_382",
            "Conv_52||Conv_401",
            "Mul_22||Shape_405",
            "Mul_22||Conv_424",
        ],
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
    ),
    decode_head=dict(
        type="MMOVDecodeHead",
        inputs=dict(
            extractor=[
                "Conv_354/WithoutBiases",
                "AveragePool_358",
                "Shape_361",
                "Shape_382",
                "Conv_401",
                "Shape_405",
                "Conv_424",
            ],
            cls_seg="Conv_428/WithoutBiases",
        ),
        outputs=dict(
            extractor="Relu_427",
            cls_seg="Conv_428",
        ),
        in_channels=(576, 16, 16),
        in_index=(0, 1, 2),
        input_transform="multiple_select",
        num_classes=19,
        dropout_ratio=-1,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
