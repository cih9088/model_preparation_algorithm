# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/shufflenet-v2-x0.5

__data_root = '/mnt/data/dataset/imagenet/'

__test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow', interpolation='bilinear', adaptive_side='short'),
    dict(type='CenterCrop', crop_size=224, backend='pillow'),
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
        model_path="public/shufflenet-v2-x0.5/FP32/shufflenet-v2-x0.5.xml",
        remove_normalize=False,
        merge_bn=False,
        paired_bn=False,
    ),
    neck=dict(
        type="MMOVNeck",
        model_path="public/shufflenet-v2-x0.5/FP32/shufflenet-v2-x0.5.xml",
    ),
    head=dict(
        type="MMOVClsHead",
        model_path="public/shufflenet-v2-x0.5/FP32/shufflenet-v2-x0.5.xml",
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0
        ),
    ),
)
