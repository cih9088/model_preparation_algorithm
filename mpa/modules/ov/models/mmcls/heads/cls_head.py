# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead


@HEADS.register_module()
class ClsHead(ClsHead):

    def __init__(self, do_squeeze=False, *args, **kwargs):
        super(ClsHead, self).__init__(*args, **kwargs)
        self._do_squeeze = do_squeeze

    def forward_train(self, cls_score, gt_label):
        if self._do_squeeze:
            cls_score = cls_score.unsqueeze(0).squeeze()
        return super().forward_train(cls_score, gt_label)

    def simple_test(self, cls_score):
        if self._do_squeeze:
            cls_score = cls_score.unsqueeze(0).squeeze()
        return super().simple_test(cls_score)
