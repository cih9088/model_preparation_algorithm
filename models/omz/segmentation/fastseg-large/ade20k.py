# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/fastseg-large
# https://github.com/open-mmlab/mmsegmentation/blob/6c746fad9cbc96ad6ef1f155b0dc327884796fc3/configs/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes.py

task_adapt = None
cudnn_benchmark = True

# runtime settings
runner = dict(type="IterBasedRunner", max_iters=320000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(_delete_=True, interval=4000, metric=["mIoU", "mDice"], pre_eval=True)
custom_hooks = []

# optimizer
optimizer = dict(_delete_=True, type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

# learning policy
lr_config = dict(_delete_=True, policy="poly", power=0.9, min_lr=1e-4, by_epoch=False)

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_scale = (2048, 1024)
crop_size = (1024, 2048)  # H, W

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

norm_cfg = dict(type="BN", eps=0.001, requires_grad=True)
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="MMOVBackbone",
        outputs=[
            "Add_26||Conv_448",
            "Add_52||Conv_425",
            "Mul_377",
        ],
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    decode_head=dict(
        type="LRASPPHead",
        in_channels=(16, 24, 960),
        in_index=(0, 1, 2),
        channels=128,
        input_transform="multiple_select",
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="ReLU"),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
