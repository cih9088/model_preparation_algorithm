# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/googlenet-v4-tf

__data_root = '/mnt/data/dataset/imagenet/'

# crop does not have central_fraction
__test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=342, backend='cv2', interpolation='bilinear'),
    dict(type='CenterCrop', crop_size=299, backend='cv2'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    num_classes=1001,
    test=dict(
        type='ImageNet',
        data_prefix=__data_root + 'val',
        pipeline=__test_pipeline,
        with_background=True,
    )
)

model = dict(
    type="ImageClassifier",
    pretrained=None,
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/googlenet-v4-tf/FP32/googlenet-v4-tf.xml",
        remove_normalize=False,
        merge_bn=False,
        paired_bn=False,
    ),
    neck=dict(
        type="MMOVNeck",
        model_path="public/googlenet-v4-tf/FP32/googlenet-v4-tf.xml",
    ),
    head=dict(
        type="MMOVClsHead",
        model_path="public/googlenet-v4-tf/FP32/googlenet-v4-tf.xml",
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0
        ),
    ),
)
