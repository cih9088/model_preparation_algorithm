# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v1-1.0-224

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
            std=[58.8235294117647, 58.8235294117647, 58.8235294117647],
            to_rgb=False,
        )
    ),
)

model = dict(
    type="ImageClassifier",
    pretrained=None,
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/mobilenet-v1-1.0-224/FP32/mobilenet-v1-1.0-224.xml",
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
