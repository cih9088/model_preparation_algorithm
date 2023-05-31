#!/usr/bin/env bash

set -eu


CLS_MODELS=(
  alexnet@imagenet
  caffenet@imagenet
  densenet-121@imagenet densenet-121-tf@imagenet
  dla-34@imagenet
  efficientnet-b0-pytorch@imagenet efficientnet-b0@imagenet
  efficientnet-v2-b0@imagenet
  efficientnet-v2-s@imagenet
  hbonet-1.0@imagenet
  hbonet-0.25@imagenet
  googlenet-v1-tf@imagenet
  googlenet-v1@imagenet
  googlenet-v2-tf@imagenet
  googlenet-v2@imagenet
  googlenet-v3-pytorch@imagenet
  googlenet-v3@imagenet
  googlenet-v4-tf@imagenet
  inception-resnet-v2-tf@imagenet
  levit-128s@imagenet
  mixnet-l@imagenet
  mobilenet-v1-0.25-128@imagenet
  mobilenet-v1-1.0-224-tf@imagenet mobilenet-v1-1.0-224@imagenet
  mobilenet-v2@imagenet mobilenet-v2-1.0-224@imagenet mobilenet-v2-pytorch@imagenet
  mobilenet-v2-1.4-224@imagenet
  mobilenet-v3-small-1.0-224-tf@imagenet
  mobilenet-v3-large-1.0-224-tf@imagenet
  nfnet-f0@imagenet
  # regnetx-3.2gf@imagenet
  octave-resnet-26-0.25@imagenet
  repvgg-a0@imagenet
  repvgg-b1@imagenet
  repvgg-b3@imagenet
  resnest-50-pytorch@imagenet
  resnet-18-pytorch@imagenet
  resnet-34-pytorch@imagenet
  resnet-50-pytorch@imagenet
  resnet-50-tf@imagenet
  rexnet-v1-x1.0@imagenet
  se-inception@imagenet
  se-resnet-50@imagenet
  se-resnext-50@imagenet
  shufflenet-v2-x0.5@imagenet
  shufflenet-v2-x1.0@imagenet
  squeezenet1.0@imagenet
  squeezenet1.1@imagenet
  swin-tiny-patch4-window7-224@imagenet
  t2t-vit-14@imagenet
  vgg16@imagenet
  vgg19@imagenet
)

for model in "${CLS_MODELS[@]}"; do
  dataset=$(echo "$model" | awk -F '@' '{print $2}')
  model=$(echo "$model" | awk -F '@' '{print $1}')
  python -m tools.cli \
    recipes/cls_eval_only.yaml \
    --mode eval \
    --model_cfg "models/omz/classification/${model}/ref.py" \
    --data_cfg "data/${dataset}.py" \
    --ir_path "/mnt/data/omz/public/${model}/FP32/${model}.xml" \
    --output_path "outputs/${model}/ref"
done



SEG_MODELS=(
  deeplabv3@pascal_voc12_seg
  hrnet-v2-c1-segmentation@ade20k
  fastseg-large@cityscapes_seg
  fastseg-small@cityscapes_seg
  pspnet-pytorch@pascal_voc12_seg
  ocrnet-hrnet-w48-paddle@cityscapes_seg
)

for model in "${SEG_MODELS[@]}"; do
  dataset=$(echo "$model" | awk -F '@' '{print $2}')
  model=$(echo "$model" | awk -F '@' '{print $1}')
  python -m tools.cli \
    recipes/seg_eval_only.yaml \
    --mode eval \
    --model_cfg "models/omz/segmentation/${model}/ref.py" \
    --data_cfg "data/${dataset}.py" \
    --ir_path "/mnt/data/omz/public/${model}/FP32/${model}.xml" \
    --output_path "outputs/${model}/ref"
done




DET_MODELS=(
  # ctdet_coco_dlav0_512
  # detr-resnet50
  # face-detection-retail-0044
  faster_rcnn_inception_resnet_v2_atrous_coco@coco_91_det
  faster_rcnn_resnet50_coco@coco_91_det
  # retinanet-tf
  ssd300@pascal_voc07_det
  ssd512@pascal_voc07_det
  mobilenet-ssd@pascal_voc07_det
  ssd_mobilenet_v1_coco@coco_91_det
  ssd_mobilenet_v1_fpn_coco@coco_91_det
  ssdlite_mobilenet_v2@coco_91_det
  ssd-resnet34-1200-onnx@coco_det
  yolo-v3-tf@coco_det
  yolo-v3-tiny-tf@coco_det
  # yolo-v3-onnx
  # yolo-v3-tiny-onnx
  # yolo-v4-tf
  # yolo-v4-tiny-tf
  # yolof
  # yolox-tiny
  mask_rcnn_inception_resnet_v2_atrous_coco@coco_91_ins
  mask_rcnn_resnet50_atrous_coco@coco_91_ins
)


for model in "${DET_MODELS[@]}"; do
  dataset=$(echo "$model" | awk -F '@' '{print $2}')
  model=$(echo "$model" | awk -F '@' '{print $1}')
  python -m tools.cli \
    recipes/det_eval_only.yaml \
    --mode eval \
    --model_cfg "models/omz/detection/${model}/ref.py" \
    --data_cfg "data/${dataset}.py" \
    --ir_path "/mnt/data/omz/public/${model}/FP32/${model}.xml" \
    --output_path "outputs/${model}/ref"
done



# # MODELS=("yolo-v3-tiny-tf" "yolo-v3-tf")
# # MODELS=("ssd300" "ssd512")
# MODELS=("faster_rcnn_resnet50_coco")
#
# export CUDA_VISIBLE_DEVICES=0
# for model in "${MODELS[@]}"; do
#   python -m tools.cli \
#     recipes/det_eval_only.yaml \
#     --mode eval \
#     --model_cfg models/omz_ref/detection/${model}.py \
#     --ir_path /mnt/data/omz/public/${model}/FP32/${model}.xml \
#     --output_path outputs/${model}/ref
# done
#
# # gpu 0 python -m tools.cli recipes/det_class_incr.yaml --mode train --model_cfg models/omz/detection/ssd300@mmdet.py --output_path outputs/ssd300/mmdet
#
# # gpu 1 python -m tools.cli recipes/det_class_incr.yaml --mode train --model_cfg models/omz/detection/yolo-v3-tf@mmdet.py --output_path outputs/yolo-v3-tf/mmdet
# # gpu 1 python -m tools.cli recipes/det_class_incr.yaml --mode train --model_cfg models/omz/detection/yolo-v3-tf@ov.py --ir_path /mnt/data/omz/public/yolo-v3-tf/FP32/yolo-v3-tf.xml --output_path outputs/yolo-v3-tf/transfer
#
# # gpu 0 python -m tools.cli recipes/dtransferet_class_incr.yaml --mode train --model_cfg models/omz/detection/ssd300@mmdet.py --output_path outputs/ssd300/mmdet
