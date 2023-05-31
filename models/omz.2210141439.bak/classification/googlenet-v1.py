# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/googlenet-v1

optimizer = dict(
    lr=0.003
)

data = dict(
    samples_per_gpu=64,
    pipeline_options=dict(
        Resize=dict(
            size=224
        ),
        Normalize=dict(
            mean=[104.0, 117.0, 123.0],
            std=[1., 1., 1.],
            to_rgb=False,
        )
    ),
)

model = dict(
    type="ImageClassifier",
    pretrained=None,
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/googlenet-v1/FP32/googlenet-v1.xml",
        remove_normalize=True,
        merge_bn=True,
        paired_bn=True,
    ),
    neck=dict(
        type="GlobalAveragePooling",
    ),
    head=dict(
        type="LinearClsHead",
        num_classes=-1,
        in_channels=-1,
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0
        ),
    ),
)
