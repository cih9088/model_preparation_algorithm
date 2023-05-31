import os
import subprocess
import tempfile
import urllib

import cv2
import torch
from mmcls.models.builder import build_classifier

import mpa.cls


examples = [
    "https://github.com/openvinotoolkit/open_model_zoo/raw/master/models/intel/age-gender-recognition-retail-0013/assets/age-gender-recognition-retail-0001.jpg",
    "https://github.com/openvinotoolkit/open_model_zoo/raw/master/models/intel/age-gender-recognition-retail-0013/assets/age-gender-recognition-retail-0002.png",
    "https://github.com/openvinotoolkit/open_model_zoo/raw/master/models/intel/age-gender-recognition-retail-0013/assets/age-gender-recognition-retail-0003.png",
]
gt_labels = [
    [0.1897, 0],
    [0.2652, 1],
    [0.3341, 1]
]

model_name = "age-gender-recognition-retail-0013"
img_size = (62, 62)
precision = "FP32"
labels = ["Female", "Male"]


with tempfile.TemporaryDirectory() as temp:
    os.chdir(temp)

    # download model
    subprocess.run(["omz_downloader", "--name", model_name, "--precision", precision])
    model_path = os.path.join(temp, "intel", model_name, precision, model_name + ".xml")

    # define model
    model = dict(
        type="MultipleHeadsClassifier",
        backbone=dict(
            type="MMOVBackbone",
            model_path=model_path,
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
                model_path=model_path,
                softmax_at_test=False,
                inputs="age_conv1/WithoutBiases",
                outputs="age_conv3",
                loss=dict(type="MSELoss", loss_weight=1.0),
            ),
            dict(
                type="MMOVClsHead",
                model_path=model_path,
                inputs="gender_conv1/WithoutBiases",
                outputs="gender_conv3",
                loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
            ),
        ],
    )

    # build model
    model = build_classifier(model)
    model.eval()

    for example in examples:
        filename = os.path.basename(example)
        _ = urllib.request.urlretrieve(example, filename)
        # read image
        img = cv2.resize(cv2.imread(filename), img_size)
        # reshape
        img = torch.from_numpy(img).unsqueeze(0).to(torch.float32).permute(0, 3, 1, 2)

        result = model(img, img_metas={}, return_loss=False)
        print("======")
        print(filename)
        print("Age: ", result[0][0] * 100)
        print("Gender: ", labels[result[1][0].argmax()], result[1][0].max())
