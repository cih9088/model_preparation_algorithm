# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/alexnet

optimizer = dict(
    lr=0.003
)

data = dict(
    samples_per_gpu=64,
    pipeline_options=dict(
        Resize=dict(
            size=227
        ),
        Normalize=dict(
            mean=[104, 117, 123],
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
        model_path="public/alexnet/FP32/alexnet.xml",
        remove_normalize=True,
        merge_bn=True,
        paired_bn=True,
    ),
    neck=dict(
        type="GlobalAveragePooling",
    ),
    head=dict(
        type="StackedLinearClsHead",
        num_classes=-1,
        in_channels=-1,
        mid_channels=[4096, 4096],
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0
        ),
    ),
)
