# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
from mmdet.core import build_anchor_generator
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.yolact_head import YOLACTHead, YOLACTProtonet

from ...mmov_model import MMOVModel


@HEADS.register_module()
class MMOVYOLACTHead(YOLACTHead):
    def __init__(
        self,
        model_path: str,
        weight_path: Optional[str] = None,
        inputs: Optional[
            Union[Dict[str, Union[str, List[str]]], List[str], str]
        ] = None,
        outputs: Optional[
            Union[Dict[str, Union[str, List[str]]], List[str], str]
        ] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
        transpose_cls: bool = False,
        transpose_reg: bool = False,
        transpose_coeff: bool = False,
        background_index: Optional[int] = None,
        *args,
        **kwargs,
    ):

        self._model_path = model_path
        self._weight_path = weight_path
        self._inputs = deepcopy(inputs)
        self._outputs = deepcopy(outputs)
        self._init_weight = init_weight
        self._verify_shape = verify_shape
        self._transpose_cls = transpose_cls
        self._transpose_reg = transpose_reg
        self._transpose_coeff = transpose_coeff
        self._background_index = background_index

        if self._background_index is not None and self._background_index < 0:
            self._background_index = self.num_classes + 1 - self._background_index

        # dummy input
        in_channels = 256
        super().__init__(in_channels=in_channels, *args, **kwargs)
        delattr(self, "relu")

        self.head_convs = torch.nn.ModuleList()
        for (
            inputs,
            outputs,
        ) in zip(self._inputs["head_convs"], self._outputs["head_convs"]):
            self.head_convs.append(
                MMOVModel(
                    self._model_path,
                    self._weight_path,
                    inputs=inputs,
                    outputs=outputs,
                    remove_normalize=False,
                    merge_bn=True,
                    paired_bn=True,
                    init_weight=self._init_weight,
                    verify_shape=self._verify_shape,
                )
            )

        self.conv_cls = MMOVModel(
            self._model_path,
            self._weight_path,
            inputs=self._inputs["conv_cls"],
            outputs=self._outputs["conv_cls"],
            remove_normalize=False,
            merge_bn=False,
            paired_bn=False,
            init_weight=self._init_weight,
            verify_shape=self._verify_shape,
        )

        self.conv_reg = MMOVModel(
            self._model_path,
            self._weight_path,
            inputs=self._inputs["conv_reg"],
            outputs=self._outputs["conv_reg"],
            remove_normalize=False,
            merge_bn=False,
            paired_bn=False,
            init_weight=self._init_weight,
            verify_shape=self._verify_shape,
        )

        self.conv_coeff = MMOVModel(
            self._model_path,
            self._weight_path,
            inputs=self._inputs["conv_coeff"],
            outputs=self._outputs["conv_coeff"],
            remove_normalize=False,
            merge_bn=False,
            paired_bn=False,
            init_weight=self._init_weight,
            verify_shape=self._verify_shape,
        )

    def forward_single(self, x):
        cls_score, bbox_pred, coeff_pred = super().forward_single(x)

        if self._transpose_cls:
            # [B, cls_out_channels * num_anchors, H, W]
            #   -> [B, num_anchors * cls_out_channels, H, W]
            shape = cls_score.shape
            cls_score = (
                cls_score.reshape(shape[0], self.cls_out_channels, -1, *shape[2:])
                .transpose(1, 2)
                .reshape(shape)
            )

        if self._transpose_reg:
            # [B, 4 * num_anchors, H, W] -> [B, num_anchors * 4, H, W]
            shape = bbox_pred.shape
            bbox_pred = (
                bbox_pred.reshape(shape[0], 4, -1, *shape[2:])
                .transpose(1, 2)
                .reshape(shape)
            )

        if self._transpose_coeff:
            # [B, num_protos * num_anchors, H, W) -> [B, num_anchors * num_protos, H, W]
            shape = coeff_pred.shape
            coeff_pred = (
                coeff_pred.reshpae(shape[0], self.num_protos, -1, *shape[2:])
                .transpose(1, 2)
                .reshape(shape)
            )

        # since mmdet v2.0
        # that FG labels to [0, num_class-1] and BG labels to num_class
        # but ssd300, ssd512 from OMZ are
        # that FG labels to [1, num_class] and BG labels to 0
        if self._background_index is not None and cls_score is not None:
            cls_score = cls_score.permute(0, 2, 3, 1)
            shape = cls_score.shape
            cls_score = cls_score.reshape(-1, self.cls_out_channels)
            cls_score = torch.cat(
                (
                    cls_score[:, :self._background_index],
                    cls_score[:, self._background_index + 1:],
                    cls_score[:, self._background_index:self._background_index + 1],
                ),
                -1,
            )
            cls_score = cls_score.reshape(shape)
            cls_score = cls_score.permute(0, 3, 1, 2)

        return cls_score, bbox_pred, coeff_pred

    def init_weights(self):
        # TODO
        pass


@HEADS.register_module()
class MMOVYOLACTProtonet(YOLACTProtonet):
    def __init__(
        self,
        model_path: str,
        weight_path: Optional[str] = None,
        inputs: Optional[
            Union[Dict[str, Union[str, List[str]]], List[str], str]
        ] = None,
        outputs: Optional[
            Union[Dict[str, Union[str, List[str]]], List[str], str]
        ] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
        *args,
        **kwargs,
    ):

        self._model_path = model_path
        self._weight_path = weight_path
        self._inputs = deepcopy(inputs)
        self._outputs = deepcopy(outputs)
        self._init_weight = init_weight
        self._verify_shape = verify_shape

        # dummy input
        in_channels = 256
        include_last_relu = True
        super().__init__(
            in_channels=in_channels,
            include_last_relu=include_last_relu,
            *args,
            **kwargs,
        )

        self.protonets = MMOVModel(
            self._model_path,
            self._weight_path,
            inputs=self._inputs,
            outputs=self._outputs,
            remove_normalize=False,
            merge_bn=False,
            paired_bn=False,
            init_weight=self._init_weight,
            verify_shape=self._verify_shape,
        )

    def init_weights(self):
        # TODO
        pass
