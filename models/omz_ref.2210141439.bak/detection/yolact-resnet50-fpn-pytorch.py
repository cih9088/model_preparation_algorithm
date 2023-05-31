# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolact-resnet50-fpn-pytorch
# https://github.com/open-mmlab/mmdetection/blob/master/configs/yolact/yolact_r50_1x8_coco.py

task_adapt = None

runner = dict(max_epochs=55)

optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
custom_hooks = []

# learning policy
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[20, 42, 49, 52])

cudnn_benchmark = True

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (1 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

evaluation = dict(
    _delete_=True,
    metric=["bbox", "segm"],
    classwise=True,
)

dataset_type = "CocoDataset"
dataset_root = "/mnt/data/dataset/coco/"
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (550, 550)

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(4.0, 4.0)),
    dict(
        type="Expand",
        mean=img_norm_cfg["mean"],
        to_rgb=img_norm_cfg["to_rgb"],
        ratio_range=(1, 4),
    ),
    dict(
        type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3
    ),
    dict(type="Resize", img_scale=img_size[::-1], keep_ratio=False),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size[::-1],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    _delete_=True,
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file="annotations/instances_train2017.json",
        img_prefix="images/train2017/",
        with_classes_from_paper=True,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file="annotations/instances_val2017.json",
        img_prefix="images/val2017/",
        with_classes_from_paper=True,
        test_mode=True,
        pipeline=test_pipeline,
    ),
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
    type="YOLACT",
    pretrained=None,
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml",
        outputs=[
            "input.184||Convolution_3008",
            "input.332||Convolution_2892",
            "input.408",
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
                "Convolution_3008",
                "Convolution_2892",
                "Convolution_2775",
            ],
            fpn=[
                "Convolution_3155",
                "Convolution_3106",
                "Convolution_3057",
                "Convolution_3204",
                "Convolution_3252",
            ],
        ),
        outputs=dict(
            laterals=[
                "onnx::Add_562",
                "onnx::Add_545",
                "onnx::Add_527",
            ],
            fpn=[
                "onnx::Conv_569",
                "onnx::Conv_567",
                "onnx::Conv_565",
                "input.416",
                "input.420",
            ],
        ),
        num_outs=5,
        add_extra_convs="on_output",
        upsample_cfg=dict(mode="bilinear"),
    ),
    bbox_head=dict(
        type="MMOVYOLACTHead",
        verify_shape=False,
        transpose_reg=False,
        transpose_cls=False,
        transpose_coeff=False,
        background_index=0,
        inputs=dict(
            head_convs=["Convolution_3555"],
            conv_cls="Convolution_3691",
            conv_reg="Convolution_3604",
            conv_coeff="Convolution_3778",
        ),
        outputs=dict(
            head_convs=["onnx::Conv_590"],
            conv_cls="onnx::Transpose_603",
            conv_reg="onnx::Transpose_591",
            conv_coeff="onnx::Transpose_615",
        ),
        num_classes=80,
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=3,
            scales_per_octave=1,
            base_sizes=[8, 16, 32, 64, 128],
            ratios=[0.5, 1.0, 2.0],
            strides=[550.0 / x for x in [69, 35, 18, 9, 5]],
            centers=[(550 * 0.5 / x, 550 * 0.5 / x) for x in [69, 35, 18, 9, 5]],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            reduction="none",
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.5),
        num_protos=32,
        use_ohem=True,
    ),
    mask_head=dict(
        type="MMOVYOLACTProtonet",
        inputs="Convolution_3300",
        outputs="onnx::Transpose_587",
        num_protos=32,
        num_classes=80,
        max_masks_to_train=100,
        loss_mask_weight=6.125,
    ),
    segm_head=dict(
        type="YOLACTSegmHead",
        num_classes=80,
        in_channels=256,
        loss_segm=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False,
        ),
        # smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        iou_thr=0.5,
        top_k=200,
        max_per_img=100,
    ),
)
