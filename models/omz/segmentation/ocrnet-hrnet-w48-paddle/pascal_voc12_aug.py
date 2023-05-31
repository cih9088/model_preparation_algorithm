# https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ocrnet/ocrnet_hr18_512x512_40k_voc12aug.py

task_adapt = None
cudnn_benchmark = True

# runtime settings
runner = dict(type="IterBasedRunner", max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(_delete_=True, interval=8000, metric=["mIoU", "mDice"], pre_eval=True)
custom_hooks = []

# optimizer
optimizer = dict(_delete_=True, type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

# learning policy
lr_config = dict(_delete_=True, policy="poly", power=0.9, min_lr=1e-4, by_epoch=False)

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_scale = (2048, 1024)
crop_size = (512, 512)  # H, W

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
        img_scale=img_scale,
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

# model settings
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="CascadeEncoderDecoder",
    num_stages=2,
    backbone=dict(
        type="MMOVBackbone",
        outputs=[
            "relu_260.tmp_0",
            "relu_261.tmp_0",
            "relu_263.tmp_0",
            "relu_267.tmp_0",
        ],
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    decode_head=[
        dict(
            type="FCNHead",
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=21,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
        dict(
            type="OCRHead",
            in_channels=[48, 96, 192, 384],
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=21,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        ),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
