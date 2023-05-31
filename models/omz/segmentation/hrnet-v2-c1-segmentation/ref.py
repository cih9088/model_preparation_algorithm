# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/hrnet-v2-c1-segmentation

task_adapt = None
cudnn_benchmark = True

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_scale = (1024, 320)
crop_size = (320, 320)  # H, W

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
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
    samples_per_gpu=4,
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
            "onnx::Shape_4286",
            "onnx::Shape_4334",
            "onnx::Shape_4366",
            "onnx::Shape_4385",
        ],
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
    ),
    decode_head=dict(
        type="MMOVDecodeHead",
        inputs=dict(
            extractor="Convolution_19783",
            cls_seg="Convolution_19832",
        ),
        outputs=dict(
            extractor="onnx::Conv_4434",
            cls_seg="x",
        ),
        in_channels=(48, 96, 192, 384),
        in_index=(0, 1, 2, 3),
        input_transform="resize_concat",
        num_classes=150,
        dropout_ratio=-1,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
