# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/efficientnet-b0-pytorch

optimizer = dict(
    lr=0.0001
)

data = dict(
    samples_per_gpu=64,
    pipeline_options=dict(
        Resize=dict(
            size=224
        ),
        Normalize=dict(
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            to_rgb=False,
        )
    ),
)

model = dict(
    type="ImageClassifier",
    pretrained=None,
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/efficientnet-b0-pytorch/FP32/efficientnet-b0-pytorch.xml",
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
