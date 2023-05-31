# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/age-gender-recognition-retail-0013

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
img_size = (62, 62)  # H, W

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="Resize", size=img_size, backend="cv2"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=img_size),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
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
        model_path="public/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml",
        remove_normalize=False,
        merge_bn=False,
        paired_bn=False,
        inputs="data",
        outputs="relu5",
    ),
    neck=None,
    heads=[
        dict(
            type="MMOVClsHead",
            model_path="public/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml",
            do_softmax=False,
            inputs="age_conv1/WithoutBiases",
            outputs="age_conv3",
            loss=dict(type="MSELoss", loss_weight=1.0),
        ),
        dict(
            type="MMOVClsHead",
            model_path="public/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml",
            inputs="gender_conv1/WithoutBiases",
            outputs="gender_conv3",
            loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        ),
    ],
)
