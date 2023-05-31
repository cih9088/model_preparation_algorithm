# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mask_rcnn_resnet50_atrous_coco
# https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py

task_adapt = None
cudnn_benchmark = True

runner = dict(type='EpochBasedRunner', max_epochs=12)

# optimizer
optimizer = dict(_delete_=True, type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
custom_hooks = []

# learning policy
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (1 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

evaluation = dict(
    _delete_=True,
    metric=['bbox', 'segm'],
    classwise=True,
)

norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (800, 1365)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=img_size[::-1], keep_ratio=True, fit_to_window=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **norm_cfg),
    dict(type="Pad", size=img_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size[::-1],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True, fit_to_window=True),
            dict(type='RandomFlip'),
            dict(type="Normalize", **norm_cfg),
            dict(type="Pad", size=img_size),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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

train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2147483647,
        nms_post=100,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2147483647,
        nms_post=100,
        max_num=1000,
        nms_thr=0.69,
        min_bbox_size=1),
    rcnn=dict(
        score_thr=0.3,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5)
)


# model settings
model = dict(
    super_type=None,
    type='MaskRCNN',
    pretrained=None,
    backbone=dict(
        type='MMOVBackbone',
        model_path='public/faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml',
        inputs="image_tensor",
        outputs='FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu',
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    neck=None,
    rpn_head=dict(
        type='MMOVRPNHead',
        model_path='public/faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml',
        transpose_reg=True,
        inputs="Conv/Conv2D",
        outputs=[
            "FirstStageBoxPredictor/ClassPredictor/BiasAdd/Add",
            "FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd/Add"
        ],
        verify_shape=False,
        anchor_generator=dict(
            type='AnchorGenerator',
            base_sizes=[256],
            scales=[0.25, 0.5, 1, 2],
            ratios=[0.5, 1.0, 2.0],
            strides=[8],
            ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIInterpolationPool', output_size=14, mode='bilinear'),
            out_channels=1024,
            featmap_strides=[8]
        ),
        bbox_head=dict(
            type='MMOVBBoxHead',
            model_path='public/faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml',
            inputs=dict(
                extractor="MaxPool2D/MaxPool",
                fc_cls="SecondStageBoxPredictor/ClassPredictor/MatMul",
                fc_reg="SecondStageBoxPredictor/BoxEncodingPredictor/MatMul",
            ),
            outputs=dict(
                extractor="SecondStageBoxPredictor/Flatten/flatten/Reshape",
                fc_cls="SecondStageBoxPredictor/ClassPredictor/BiasAdd/Add",
                fc_reg="SecondStageBoxPredictor/BoxEncodingPredictor/BiasAdd/Add",
            ),
            verify_shape=False,
            roi_feat_size=7,
            num_classes=90,
            background_index=0,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='L1Loss',
                loss_weight=1.0)
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIInterpolationPool', output_size=14, mode='bilinear'),
            out_channels=1024,
            featmap_strides=[8]
        ),
        mask_head=dict(
            type='MMOVMaskHead',
            model_path='public/faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml',
            inputs="MaxPool2D_1/MaxPool",
            outputs="SecondStageBoxPredictor_1/Conv_3/BiasAdd/Add",
            verify_shape=False,
            num_classes=90,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        ),
    ),
    train_cfg=train_cfg,
    test_cfg=test_cfg
)
