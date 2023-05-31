# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset as OriginCocoDataset


@DATASETS.register_module()
class CocoDataset(OriginCocoDataset):
    MISSING_ID = {
        12: "street sign",
        26: "hat",
        29: "shoe",
        30: "eye classes",
        45: "plate",
        66: "mirror",
        68: "window",
        69: "desk",
        71: "door",
        83: "blender",
        91: "hair brush",
    }

    def __init__(self, min_size=None, *args, **kwargs):
        self.with_classes_from_paper = kwargs.pop("with_classes_from_paper", False)
        super().__init__(min_size=min_size, *args, **kwargs)

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
        score_thr=-1,
        metric_items=None,
    ):
        if isinstance(results[0], list):
            result = results[0]
            if len(result) >= 90 and self.with_classes_from_paper:
                indices = sorted(
                    [i - 1 for i in list(self.MISSING_ID.keys()) if i <= len(result)],
                    reverse=True,
                )
                for result in results:
                    for idx in indices:
                        result.pop(idx)
        elif isinstance(results[0], tuple):
            det, seg = results[0][:2]
            indices = sorted(
                [i - 1 for i in list(self.MISSING_ID.keys()) if i <= len(det)],
                reverse=True,
            )

            for det, seg in results:
                for idx in indices:
                    det.pop(idx)
                    seg.pop(idx)
        else:
            raise NotImplementedError
        return super().evaluate(
            results,
            metric,
            logger,
            jsonfile_prefix,
            classwise,
            proposal_nums,
            iou_thrs,
            score_thr,
            metric_items,
        )
