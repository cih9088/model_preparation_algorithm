# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/faster_rcnn_inception_resnet_v2_atrous_coco

task_adapt = None

runner = dict(max_epochs=1024)

evaluation = dict(
    _delete_=True,
    interval=2,
    metric="mAP",
    save_best="mAP",
)

custom_hooks = []

optimizer = dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    _delete_=True,
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[500, 1000],
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

dataset_type = "VOCDataset"
dataset_root = "/mnt/data/dataset/voc/VOC2007/"
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (600, 1024)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=img_size[::-1], keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    #  dict(type="Pad", size=img_size),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size[::-1],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            #  dict(type="Pad", size=img_size),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix=dataset_root,
        ann_file=dataset_root + "ImageSets/Layout/train.txt",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        img_prefix=dataset_root,
        ann_file=dataset_root + "ImageSets/Layout/val.txt",
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        img_prefix=dataset_root,
        ann_file=dataset_root + "ImageSets/Layout/test.txt",
        test_mode=True,
        pipeline=test_pipeline,
    ),
)


# model settings
model = dict(
    super_type=None,
    type="FasterRCNN",
    pretrained=None,
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/faster_rcnn_inception_resnet_v2_atrous_coco/FP32/faster_rcnn_inception_resnet_v2_atrous_coco.xml",
        inputs="image_tensor",
        outputs="FirstStageFeatureExtractor/InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Relu",
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    neck=None,
    rpn_head=dict(
        type="RPNHead",
        in_channels=1088,
        feat_channels=512,
        anchor_generator=dict(
            type="AnchorGenerator",
            base_sizes=[256],
            scales=[0.25, 0.5, 1, 2],
            ratios=[0.5, 1.0, 2.0],
            strides=[8],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=17, sampling_ratio=0),
            out_channels=1088,
            featmap_strides=[8],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=1088,
            fc_out_channels=1024,
            roi_feat_size=17,
            num_classes=20,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            nms_post=1000,
            nms_thr=0.7,
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            nms_post=1000,
            nms_thr=0.7,
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100
        )
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ),
)

cudnn_benchmark = True
