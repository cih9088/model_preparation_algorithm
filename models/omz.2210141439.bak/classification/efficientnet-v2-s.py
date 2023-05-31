# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/efficientnet-v2-s

optimizer = dict(
    lr=0.0001
)

data = dict(
    samples_per_gpu=16,
    pipeline_options=dict(
        Resize=dict(
            size=384
        ),
        Normalize=dict(
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False,
        )
    ),
)

model = dict(
    type="ImageClassifier",
    pretrained=None,
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/efficientnet-v2-s/FP32/efficientnet-v2-s.xml",
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
