#!/usr/bin/env bash

set -eu


CLS_MODELS=(
  alexnet@stl10
  caffenet@stl10
  densenet-121@stl10 densenet-121-tf@stl10
  dla-34@stl10
  efficientnet-b0-pytorch@stl10 efficientnet-b0@stl10
  efficientnet-v2-b0@stl10
  efficientnet-v2-s@stl10
  hbonet-1.0@stl10
  hbonet-0.25@stl10
  googlenet-v1-tf@stl10
  googlenet-v1@stl10
  googlenet-v2-tf@stl10
  googlenet-v2@stl10
  googlenet-v3-pytorch@stl10
  googlenet-v3@stl10
  googlenet-v4-tf@stl10
  inception-resnet-v2-tf@stl10
  mixnet-l@stl10
  mobilenet-v1-0.25-128@stl10
  mobilenet-v1-1.0-224-tf@stl10 mobilenet-v1-1.0-224@stl10
  mobilenet-v2@stl10 mobilenet-v2-1.0-224@stl10 mobilenet-v2-pytorch@stl10
  mobilenet-v2-1.4-224@stl10
  mobilenet-v3-small-1.0-224-tf@stl10
  mobilenet-v3-large-1.0-224-tf@stl10
  nfnet-f0@stl10
  # regnetx-3.2gf@stl10
  octave-resnet-26-0.25@stl10
  resnet-18-pytorch@stl10
  resnet-34-pytorch@stl10
  resnet-50-pytorch@stl10
  resnet-50-tf@stl10
  se-inception@stl10
  se-resnet-50@stl10
  se-resnext-50@stl10
  shufflenet-v2-x0.5@stl10
  squeezenet1.0@stl10
  squeezenet1.1@stl10
  vgg16@stl10
  vgg19@stl10
)

for model in "${CLS_MODELS[@]}"; do
  model=$(echo "$model" | awk -F '@' '{print $1}')
  dataset=$(echo "$model" | awk -F '@' '{print $2}')
  python -m tools.cli \
    recipes/cls.yaml \
    --mode train \
    --model_cfg "models/omz/classification/${model}/${dataset}.py" \
    --data_cfg "data/${dataset}.py" \
    --ir_path "/mnt/data/omz/public/${model}/FP32/${model}.xml" \
    --output_path "outputs/${model}/${dataset}"
done


# # python -m tools.cli recipes/cls.yaml --model_cfg models/omz/classification/repvgg-a0.py --data_cfg data/stl10.py --ir_path /mnt/data/omz/public/repvgg-a0/FP32/repvgg-a0.xml --output_path outputs/repvgg-a0/transfer
#
# # python -m tools.cli recipes/cls.yaml --model_cfg models/omz/classification/densenet-121.py --data_cfg data/stl10.py --ir_path /mnt/data/omz/public/densenet-121/FP32/densenet-121.xml --output_path outputs/test
#
#
#
#
#
#
#
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/yolo-v3-tiny-tf.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/yolo-v3-tiny-tf/FP32/yolo-v3-tiny-tf.xml --output_path outputs/yolo-v3-tiny-tf/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/yolo-v3-tf.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/yolo-v3-tf/FP32/yolo-v3-tf.xml --output_path outputs/yolo-v3-tf/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/faster_rcnn_resnet50_coco.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml --output_path outputs/faster_rcnn_resnet50_coco/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/faster_rcnn_inception_resnet_v2_atrous_coco.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/faster_rcnn_inception_resnet_v2_atrous_coco/FP32/faster_rcnn_inception_resnet_v2_atrous_coco.xml --output_path outputs/faster_rcnn_inception_resnet_v2_atrous_coco/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/ssd300.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/ssd300/FP32/ssd300.xml --output_path outputs/ssd300/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/ssd512.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/ssd512/FP32/ssd512.xml --output_path outputs/ssd512/transfer
#
#
#
#
# # export CUDA_VISIBLE_DEVICES=0
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/ssd512.py --data_cfg data/coco_ssd512.py --ir_path /mnt/data/omz/public/ssd512/FP32/ssd512.xml --output_path outputs/ssd512/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/ssd512.py --data_cfg data/coco_ssd512.py --ir_path /mnt/data/omz/public/ssd512/FP32/ssd512.xml --output_path outputs/ssd512/init --init
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/yolo-v3-tiny-tf.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/yolo-v3-tiny-tf/FP32/yolo-v3-tiny-tf.xml --output_path outputs/yolo-v3-tiny-tf/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/yolo-v3-tiny-tf.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/yolo-v3-tiny-tf/FP32/yolo-v3-tiny-tf.xml --output_path outputs/yolo-v3-tiny-tf/init --init
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/yolo-v3-tf.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/yolo-v3-tf/FP32/yolo-v3-tf.xml --output_path outputs/yolo-v3-tf/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/yolo-v3-tf.py --data_cfg data/coco_yolo.py --ir_path /mnt/data/omz/public/yolo-v3-tf/FP32/yolo-v3-tf.xml --output_path outputs/yolo-v3-tf/init --init
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/faster_rcnn_resnet50_coco.py --data_cfg data/coco_faster_rcnn.py --ir_path /mnt/data/omz/public/faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml --output_path outputs/faster_rcnn_resnet50_coco/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/faster_rcnn_resnet50_coco.py --data_cfg data/coco_faster_rcnn.py --ir_path /mnt/data/omz/public/faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml --output_path outputs/faster_rcnn_resnet50_coco/init --init
# #
# # export CUDA_VISIBLE_DEVICES=1
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/ssd300.py --data_cfg data/coco_ssd300.py --ir_path /mnt/data/omz/public/ssd300/FP32/ssd300.xml --output_path outputs/ssd300/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/ssd300.py --data_cfg data/coco_ssd300.py --ir_path /mnt/data/omz/public/ssd300/FP32/ssd300.xml --output_path outputs/ssd300/init --init
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/faster_rcnn_inception_resnet_v2_atrous_coco.py --data_cfg data/coco_faster_rcnn.py --ir_path /mnt/data/omz/public/faster_rcnn_inception_resnet_v2_atrous_coco/FP32/faster_rcnn_inception_resnet_v2_atrous_coco.xml --output_path outputs/faster_rcnn_inception_resnet_v2_atrous_coco/transfer
# # python -m tools.cli recipes/det.yaml --model_cfg models/omz/detection/faster_rcnn_inception_resnet_v2_atrous_coco.py --data_cfg data/coco_faster_rcnn.py --ir_path /mnt/data/omz/public/faster_rcnn_inception_resnet_v2_atrous_coco/FP32/faster_rcnn_inception_resnet_v2_atrous_coco.xml --output_path outputs/faster_rcnn_inception_resnet_v2_atrous_coco/init --init
#
#
#
#
# ## segmentation
#
# # python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/hrnet-v2-c1-segmentation@voc12_aug@ov.py --ir_path /mnt/data/omz/public/hrnet-v2-c1-segmentation/FP32/hrnet-v2-c1-segmentation.xml --output_path outputs/hrnet-v2-c1-segmentation/voc12_aug_ov
# # python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/hrnet-v2-c1-segmentation@voc12_aug@ov.py --ir_path /mnt/data/omz/public/hrnet-v2-c1-segmentation/FP32/hrnet-v2-c1-segmentation.xml --output_path outputs/hrnet-v2-c1-segmentation/voc12_aug_ov_init --init
# # python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/hrnet-v2-c1-segmentation@voc12_aug@mm.py --ir_path /mnt/data/omz/public/hrnet-v2-c1-segmentation/FP32/hrnet-v2-c1-segmentation.xml --output_path outputs/hrnet-v2-c1-segmentation/voc12_aug_mm
#
# python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/deeplabv3@ade20k@ov.py --ir_path /mnt/data/omz/public/deeplabv3/FP32/deeplabv3.xml --output_path outputs/deeplabv3/ade20k_ov
# python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/deeplabv3@ade20k@ov.py --ir_path /mnt/data/omz/public/deeplabv3/FP32/deeplabv3.xml --output_path outputs/deeplabv3/ade20k_ov_init --init
# python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/deeplabv3@ade20k@mm.py --ir_path /mnt/data/omz/public/deeplabv3/FP32/deeplabv3.xml --output_path outputs/deeplabv3/ade20k_mm
#
# python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/pspnet-pytorch@ade20k@ov.py --ir_path /mnt/data/omz/public/pspnet-pytorch/FP32/pspnet-pytorch.xml --output_path outputs/pspnet-pytorch/ade20k_ov
# python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/pspnet-pytorch@ade20k@ov.py --ir_path /mnt/data/omz/public/pspnet-pytorch/FP32/pspnet-pytorch.xml --output_path outputs/pspnet-pytorch/ade20k_ov_init --init
# python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/pspnet-pytorch@ade20k@mm.py --ir_path /mnt/data/omz/public/pspnet-pytorch/FP32/pspnet-pytorch.xml --output_path outputs/pspnet-pytorch/ade20k_mm
#
# python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/fastseg-small@ade20k@ov.py --ir_path /mnt/data/omz/public/fastseg-small/FP32/fastseg-small.xml --output_path outputs/fastseg-small/ade20k_ov
# python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/fastseg-small@ade20k@ov.py --ir_path /mnt/data/omz/public/fastseg-small/FP32/fastseg-small.xml --output_path outputs/fastseg-small/ade20k_ov_init --init
# python -m tools.cli recipes/seg.yaml --mode train --model_cfg models/omz/segmentation/fastseg-small@ade20k@mm.py --ir_path /mnt/data/omz/public/fastseg-small/FP32/fastseg-small.xml --output_path outputs/fastseg-small/ade20k_mm
