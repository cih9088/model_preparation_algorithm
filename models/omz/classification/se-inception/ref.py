# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/se-inception

imagenet_2015_ann_file = "/mnt/data/dataset/imagenet/val15.txt"

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
img_size = (224, 224)  # H, W

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=256, backend="cv2", interpolation="bilinear"),
    dict(type="CenterCrop", crop_size=img_size, backend="cv2"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

data = dict(
    samples_per_gpu=64,
    train=dict(),
    val=dict(),
    test=dict(
        pipeline=test_pipeline,
        ann_file=imagenet_2015_ann_file,
    ),
)

model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="MMOVBackbone",
        model_path="public/se-inception/FP32/se-inception.xml",
        remove_normalize=False,
        merge_bn=True,
        paired_bn=True,
    ),
    neck=dict(
        type="MMOVNeck",
        model_path="public/se-inception/FP32/se-inception.xml",
    ),
    head=dict(
        type="MMOVClsHead",
        model_path="public/se-inception/FP32/se-inception.xml",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
