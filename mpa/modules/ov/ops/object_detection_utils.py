# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from torchvision.ops import nms as torchvision_nms
from torchvision.ops import roi_pool as torchvision_roi_pool


def roi_pool(input, boxes, output_size, spatial_scale, method="max"):
    if method == "max":
        return torchvision_roi_pool(
            input=input,
            boxes=boxes,
            output_size=output_size,
            spatial_scale=spatial_scale,
        )
    #  elif method == "bilinear":
    #      #  roi_width = (W - 1) \* (x_2 - x_1), roi_height = (H - 1) \* (y_2 - y_1).
    #
    #      for box in boxes:
    #          batch_id, x_1, y_1, x_2, y_2
    #
    #      batch_size, num_channels, _, _ = input.shape
    #
    #      # first scale proposals based on self.scaling factor
    #      scaled_proposals = torch.zeros_like(boxes)
    #
    #
    #
    #      # the rounding by torch.ceil is important for ROI pool
    #
    #      scaled_proposals[:, 0] = torch.ceil(proposals[:, 0] * self.scaling_factor)
    #
    #      scaled_proposals[:, 1] = torch.ceil(proposals[:, 1] * self.scaling_factor)
    #
    #      scaled_proposals[:, 2] = torch.ceil(proposals[:, 2] * self.scaling_factor)
    #
    #      scaled_proposals[:, 3] = torch.ceil(proposals[:, 3] * self.scaling_factor)
    #
    #
    #
    #      res = torch.zeros((len(proposals), num_channels, *output_size))
    #      res_idx = torch.zeros((len(proposals), num_channels, *output_size))
    #
    #      for idx in range(len(proposals)):
    #
    #          proposal = scaled_proposals[idx]
    #
    #          # adding 1 to include the end indices from proposal
    #
    #          extracted_feat = feature_layer[0, :, proposal[1].to(dtype=torch.int8):proposal[3].to(dtype=torch.int8)+1, proposal[0].to(dtype=torch.int8):proposal[2].to(dtype=torch.int8)+1]
    #
    #          res[idx], res_idx[idx] = self._roi_pool(extracted_feat)

    else:
        raise NotImplementedError(
            f"method {method} is not implemented. supported methods are [max, bilinear]"
        )


def nms(boxes, scores, iou_threshold):
    raise NotImplementedError()
    #  return torchvision_nms(boxes, scores, iou_threshold)
