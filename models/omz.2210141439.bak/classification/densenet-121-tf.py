# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/densenet-121-tf

optimizer = dict(
    lr=0.0003
)

data = dict(
    samples_per_gpu=64,
    pipeline_options=dict(
        Resize=dict(
            size=224
        ),
        Normalize=dict(
            mean=[103.94, 116.78, 123.68],
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
        model_path="public/densenet-121-tf/FP32/densenet-121-tf.xml",
        remove_normalize=True,
        paired_bn=False,
        merge_bn=True,
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
