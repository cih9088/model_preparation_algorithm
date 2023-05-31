# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/googlenet-v3-pytorch

__data_root = '/mnt/data/dataset/imagenet/'

# crop does not have central_fraction
__test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(320, -1), backend='pillow', interpolation='bilinear', adaptive_side='short'),
    dict(type='CenterCrop', crop_size=299, backend='pillow'),
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
        model_path="public/googlenet-v3-pytorch/FP32/googlenet-v3-pytorch.xml",
        remove_normalize=False,
        merge_bn=False,
        paired_bn=False,
    ),
    neck=dict(
        type="MMOVNeck",
        model_path="public/googlenet-v3-pytorch/FP32/googlenet-v3-pytorch.xml",
    ),
    head=dict(
        type="MMOVClsHead",
        model_path="public/googlenet-v3-pytorch/FP32/googlenet-v3-pytorch.xml",
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0
        ),
    ),
)
