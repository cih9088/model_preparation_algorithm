# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/squeezenet1.1

__data_root = '/mnt/data/dataset/imagenet/'

__test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, backend='cv2', interpolation='bilinear'),
    dict(type='CenterCrop', crop_size=227, backend='cv2'),
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
        model_path="public/squeezenet1.1/FP32/squeezenet1.1.xml",
        remove_normalize=False,
        merge_bn=False,
        paired_bn=False,
    ),
    neck=dict(
        type="MMOVNeck",
        model_path="public/squeezenet1.1/FP32/squeezenet1.1.xml",
    ),
    head=dict(
        type="MMOVClsHead",
        model_path="public/squeezenet1.1/FP32/squeezenet1.1.xml",
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.1
        ),
    ),
)
