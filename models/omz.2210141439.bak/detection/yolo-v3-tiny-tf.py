# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tiny-tf

#  __norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=False)
#
#  data = dict(
#      pipeline_options=dict(
#          Resize=dict(
#              img_scale=(416, 416),
#              keep_ratio=True,
#          ),
#          Expand=dict(
#              mean=__norm_cfg["mean"],
#              to_rgb=__norm_cfg["to_rgb"],
#              ratio_range=(1, 2),
#          ),
#          Normalize=__norm_cfg,
#          MultiScaleFlipAug=dict(
#              img_scale=(416, 416),
#          )
#      ),
#  )

runner = dict(
    max_epochs=10
)

task_adapt = None

data = dict(
    samples_per_gpu=8,
)

model = dict(
    super_type=None,
    type="YOLOV3",
    backbone=dict(
        type="MMOVBackbone",
        model_path='public/yolo-v3-tiny-tf/FP32/yolo-v3-tiny-tf.xml',
        outputs=["leaky_re_lu_4/LeakyRelu||concatenate/concat", "leaky_re_lu_4/LeakyRelu||max_pooling2d_4/MaxPool"],
        remove_normalize=True,
        merge_bn=True,
        paired_bn=True,
        verify_shape=False,
    ),
    #  neck=dict(
    #      type='YOLOV3Neck',
    #      num_scales=2,
    #      in_channels=[256, 256],
    #      out_channels=[256, 256],
    #  ),
    neck=dict(
        type="MMOVYOLOV3Neck",
        verify_shape=False,
        inputs=dict(
            detect1="max_pooling2d_4/MaxPool",
            detect2="",
            conv1="conv2d_10/Conv2D",
        ),
        outputs=dict(
            detect1="leaky_re_lu_7/LeakyRelu",
            detect2="",
            conv1="leaky_re_lu_9/LeakyRelu",
        ),
    ),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=80,
        in_channels=[256, 384],
        out_channels=[512, 25],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(81, 82), (135, 169), (344, 319)],
                        [(23, 27), (37, 58), (81, 82)]],
            strides=[32, 16]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(
            type='MSELoss',
            loss_weight=2.0,
            reduction='sum'
        )
    ),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0
        )
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(
            type='nms',
            iou_threshold=0.45
        ),
        max_per_img=100
    )
)
