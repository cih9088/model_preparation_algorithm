# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3
# https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug.py

task_adapt = None
cudnn_benchmark = True

# optimizer
optimizer = dict(_delete_=True, type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

# learning policy
lr_config = dict(_delete_=True, policy="poly", power=0.9, min_lr=1e-4, by_epoch=False)

# runtime settings
runner = dict(type="IterBasedRunner", max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
custom_hooks = []
evaluation = dict(_delete_=True, interval=8000, metric=["mIoU", "mDice"], pre_eval=True)

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_scale = (2048, 513)
crop_size = (513, 513)  # H, W

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
    dict(type="LoadImageFromFile", to_float32=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=crop_size[::-1],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(
                type="Pad",
                size=crop_size,
                pad_val=(127.5, 127.5, 127.5),
                seg_pad_val=255,
            ),
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
        outputs="MobilenetV2/expanded_conv_16/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_",
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
    ),
    decode_head=dict(
        type="MMOVDecodeHead",
        inputs=dict(
            extractor=["AvgPool2D/AvgPool", "aspp0/Conv2D"],
            cls_seg="logits/semantic/Conv2D",
        ),
        outputs=dict(
            extractor="concat_projection/Relu",
            cls_seg="logits/semantic/BiasAdd/Add",
        ),
        in_channels=320,
        in_index=0,
        input_transform=None,
        num_classes=21,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
