# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-1.4-224

optimizer = dict(lr=0.1)

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
img_size = (224, 224)  # H, W

train_pipeline = [
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
    dict(type="Resize", size=256, backend="cv2", interpolation="bilinear"),
    dict(type="CenterCrop", crop_size=224, backend="cv2"),
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
        type="ConvClsHead",
        num_classes=-1,
        in_channels=-1,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
