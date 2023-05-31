# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/efficientnet-v2-s

__data_root = '/mnt/data/dataset/imagenet/'

__test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(384, -1), backend='pillow', interpolation='bicubic', adaptive_side='short'),
    dict(type='CenterCrop', crop_size=384, backend='pillow'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    num_classes=1000,
    test=dict(
        type='ImageNet',
        data_prefix=__data_root + 'val',
        pipeline=__test_pipeline,
    )
)

model = dict(
    type="ImageClassifier",
    pretrained=None,
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/efficientnet-v2-s/FP32/efficientnet-v2-s.xml",
        remove_normalize=False,
        merge_bn=False,
        paired_bn=False,
    ),
    neck=dict(
        type="MMOVNeck",
        model_path="public/efficientnet-v2-s/FP32/efficientnet-v2-s.xml",
    ),
    head=dict(
        type="MMOVClsHead",
        model_path="public/efficientnet-v2-s/FP32/efficientnet-v2-s.xml",
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0
        ),
    ),
)
