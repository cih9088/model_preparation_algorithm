# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/faster_rcnn_resnet50_coco

task_adapt = None

runner = dict(
    max_epochs=30
)

evaluation = dict(
    metric='bbox',
    save_best='bbox',
    classwise=True,
)

dataset_type = "CocoDataset"
dataset_root = "/mnt/data/dataset/coco/"
norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
img_size = (600, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_size[::-1], keep_ratio=True, fit_to_window=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **norm_cfg),
    dict(type="Pad", size=img_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size[::-1],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True, fit_to_window=True),
            #  dict(type="Resize", keep_ratio=False),
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
        min_bbox_size=1),
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
        nms_thr=0.69,
        min_bbox_size=1),
    rcnn=dict(
        score_thr=0.3,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)


# model settings
model = dict(
    super_type=None,
    type='FasterRCNN',
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
            strides=[16],
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
            featmap_strides=[16]
        ),
        bbox_head=dict(
            type='MMOVBBoxHead',
            model_path='public/faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml',
            inputs=dict(
                extractor="MaxPool2D/MaxPool",
                #  extractor=[
                #      "SecondStageFeatureExtractor/resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/Conv2D",
                #      "SecondStageFeatureExtractor/resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/Conv2D",
                #  ],
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
            with_avg_pool=False,
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
    ),
    train_cfg=train_cfg,
    test_cfg=test_cfg
)

#  model = dict(
#      super_type=None,
#      type='FasterRCNN',
#      backbone=dict(
#          type='ResNet',
#          depth=50,
#          num_stages=4,
#          out_indices=(0, 1, 2, 3),
#          frozen_stages=1,
#          norm_cfg=dict(type='BN', requires_grad=True),
#          norm_eval=True,
#          style='pytorch',
#      ),
#      neck=dict(
#          type='FPN',
#          in_channels=[256, 512, 1024, 2048],
#          out_channels=256,
#          num_outs=5),
#      rpn_head=dict(
#          type='RPNHead',
#          in_channels=256,
#          feat_channels=256,
#          anchor_generator=dict(
#              type='AnchorGenerator',
#              scales=[8],
#              ratios=[0.5, 1.0, 2.0],
#              strides=[4, 8, 16, 32, 64]),
#          bbox_coder=dict(
#              type='DeltaXYWHBBoxCoder',
#              target_means=[.0, .0, .0, .0],
#              target_stds=[1.0, 1.0, 1.0, 1.0]),
#          loss_cls=dict(
#              type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#          loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#      roi_head=dict(
#          type='StandardRoIHead',
#          bbox_roi_extractor=dict(
#              type='SingleRoIExtractor',
#              roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#              out_channels=256,
#              featmap_strides=[4, 8, 16, 32]),
#          bbox_head=dict(
#              type='Shared2FCBBoxHead',
#              in_channels=256,
#              fc_out_channels=1024,
#              roi_feat_size=7,
#              num_classes=90,
#              bbox_coder=dict(
#                  type='DeltaXYWHBBoxCoder',
#                  target_means=[0., 0., 0., 0.],
#                  target_stds=[0.1, 0.1, 0.2, 0.2]),
#              reg_class_agnostic=False,
#              loss_cls=dict(
#                  type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#              loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
#      train_cfg=train_cfg,
#      test_cfg=test_cfg,
#  )

cudnn_benchmark = True
