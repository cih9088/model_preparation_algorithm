# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-1.4-224

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
img_size = (224, 224)  # H, W

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(
        type="Resize",
        size=img_size,
        backend="cv2",
        interpolation="bilinear",
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
# crop does not have central_fraction
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=256, backend="cv2", interpolation="bilinear"),
    dict(type="CenterCrop", crop_size=224, backend="cv2"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

data = dict(
    samples_per_gpu=64,
    num_classes=1001,
    train=dict(
        pipeline=train_pipeline,
        with_background=True,
    ),
    val=dict(
        pipeline=test_pipeline,
        with_background=True,
    ),
    test=dict(
        pipeline=test_pipeline,
        with_background=True,
    ),
)

model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/mobilenet-v2-1.4-224/FP32/mobilenet-v2-1.4-224.xml",
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
    ),
    neck=dict(
        type="MMOVNeck",
        model_path="public/mobilenet-v2-1.4-224/FP32/mobilenet-v2-1.4-224.xml",
    ),
    head=dict(
        type="MMOVClsHead",
        model_path="public/mobilenet-v2-1.4-224/FP32/mobilenet-v2-1.4-224.xml",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
