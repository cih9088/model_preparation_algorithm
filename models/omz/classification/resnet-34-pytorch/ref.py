# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-34-pytorch

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
img_size = (224, 224)  # H, W

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(
        type="Resize",
        size=img_size,
        backend="pillow",
        interpolation="bilinear",
        adaptive_side="short",
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="Resize",
        size=(256, -1),
        backend="pillow",
        interpolation="bilinear",
        adaptive_side="short",
    ),
    dict(type="CenterCrop", crop_size=img_size, backend="pillow"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

data = dict(
    samples_per_gpu=64,
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=test_pipeline,
    ),
    test=dict(
        pipeline=test_pipeline,
    ),
)

model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/resnet-34-pytorch/FP32/resnet-34-pytorch.xml",
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
    ),
    neck=dict(
        type="MMOVNeck",
        model_path="public/resnet-34-pytorch/FP32/resnet-34-pytorch.xml",
    ),
    head=dict(
        type="MMOVClsHead",
        model_path="public/resnet-34-pytorch/FP32/resnet-34-pytorch.xml",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
