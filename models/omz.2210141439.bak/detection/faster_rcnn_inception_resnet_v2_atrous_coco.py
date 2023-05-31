# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/faster_rcnn_inception_resnet_v2_atrous_coco

runner = dict(
    max_epochs=10
)

task_adapt = None

data = dict(
    samples_per_gpu=1,
)

__train_cfg = dict(
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

__test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2147483647,
        nms_post=100,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
)

# model settings
model = dict(
    super_type=None,
    type='FasterRCNN',
    pretrained=None,
    backbone=dict(
        type='MMOVBackbone',
        model_path='public/faster_rcnn_inception_resnet_v2_atrous_coco/FP32/faster_rcnn_inception_resnet_v2_atrous_coco.xml',
        inputs="image_tensor",
        outputs='FirstStageFeatureExtractor/InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Relu',
        remove_normalize=True,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    neck=None,
    rpn_head=dict(
        type='RPNHead',
        in_channels=1088,
        feat_channels=512,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[8]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
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
            roi_layer=dict(type='RoIPool', output_size=14),
            out_channels=1088,
            featmap_strides=[8]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=1088,
            fc_out_channels=1024,
            roi_feat_size=14,
            with_avg_pool=True,
            num_classes=80,
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
    train_cfg=__train_cfg,
    test_cfg=__test_cfg
)

cudnn_benchmark = True
