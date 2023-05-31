# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/repvgg-b3

optimizer = dict(
    lr=0.003,
)

data = dict(
    samples_per_gpu=64,
    pipeline_options=dict(
        Resize=dict(
            size=224
        ),
        Normalize=dict(
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.624],
            to_rgb=False,
        )
    ),
)

model = dict(
    type="ImageClassifier",
    pretrained=None,
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/repvgg-b3/FP32/repvgg-b3.xml",
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
