# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from .inference import LoadImage

from mmcv.parallel.data_container import DataContainer
from mmcv.parallel import MMDataParallel


def scatter_cpu(inputs):
    """Scatter inputs to cpu.
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return [obj]
        if isinstance(obj, DataContainer):
            return obj.data
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter_cpu(inputs) if inputs else []
    kwargs = scatter_cpu(kwargs) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class MMDataCPU(MMDataParallel):
    """Implementation of MMDataParallel to use CPU for training"""

    def scatter(self, inputs, kwargs):
        return scatter_kwargs(inputs, kwargs)

    def train_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        return self.module.val_step(*inputs[0], **kwargs[0])

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        return self.module(*inputs[0], **kwargs[0])


def get_fake_input(cfg, orig_img_shape=(128, 128, 3), device='cuda'):
    test_pipeline = [LoadImage()]
    for pipeline in cfg.data.test.pipeline:
        if 'LoadImage' not in pipeline['type']:
            test_pipeline.append(pipeline)
    test_pipeline = Compose(test_pipeline)
    data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
    data = test_pipeline(data)
    if device == torch.device('cpu'):
        data = scatter_cpu(collate([data], samples_per_gpu=1))[0]
    else:
        data = scatter(collate([data], samples_per_gpu=1), [device.index])[0]
    return data
