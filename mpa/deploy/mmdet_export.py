# Copyright (C) 2021 Intel Corporation
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

import os.path as osp
from packaging import version
from subprocess import DEVNULL, CalledProcessError, run  # nosec

import mmcv
import numpy as np
import onnx
import torch
from mmcv.runner import wrap_fp16_model
from onnxoptimizer import optimize
from torch.onnx.symbolic_helper import _onnx_stable_opsets as available_opsets

from mmdet.models import detectors

from .deployment.fake_input import get_fake_input
#  from .deployment.ssd_export_helpers import *  # noqa: F403
from .deployment.symbolic import (
    register_extra_symbolics,
    register_extra_symbolics_for_openvino,
    unregister_extra_symbolics,
    unregister_extra_symbolics_for_openvino
)


def get_min_opset_version():
    return 10 if version.parse(torch.__version__) < version.parse('1.7.0') else 11


#  def patch_model_for_alt_ssd_export(model):
#      model._export_mode = False
#      model.onnx_export = onnx_export.__get__(model)
#      model.save_img_metas = save_img_metas.__get__(model)
#      model.forward = forward.__get__(model)
#      model.forward_export = forward_export_detector.__get__(model)
#      model.bbox_head.export_forward = export_forward_ssd_head.__get__(model.bbox_head)
#      model.bbox_head._prepare_cls_scores_bbox_preds = prepare_cls_scores_bbox_preds_ssd_head.__get__(model.bbox_head)


#  def patch_nncf_model_for_alt_ssd_export(model):
#      model.onnx_export = onnx_export.__get__(model)


def export_to_onnx(model,
                   data,
                   export_name,
                   verbose=False,
                   strip_doc_string=False,
                   opset=10,
                   alt_ssd_export=False,
                   target='onnx'):
    register_extra_symbolics(opset)
    if target == 'openvino' and not alt_ssd_export:
        register_extra_symbolics_for_openvino(opset)

    kwargs = {}
    if torch.__version__ >= '1.5':
        kwargs['enable_onnx_checker'] = False

    if alt_ssd_export:
        #  assert isinstance(get_uncompressed_model(model), detectors.SingleStageDetector)
        model.onnx_export(img=data['img'][0],
                          img_metas=data['img_metas'][0],
                          export_name=export_name,
                          verbose=verbose,
                          opset_version=opset,
                          strip_doc_string=strip_doc_string,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                          input_names=['image'],
                          output_names=['detection_out'],
                          keep_initializers_as_inputs=True,
                          **kwargs)
    else:
        output_names = ['boxes', 'labels']
        dynamic_axes = {
            'image': {2: 'height', 3: 'width'},
            'boxes': {0: 'objects_num'},
            'labels': {0: 'objects_num'}
        }
        if hasattr(model, 'roi_head'):
            if model.roi_head.with_mask:
                output_names.append('masks')
                dynamic_axes['masks'] = {0: 'objects_num'}
                if getattr(model.roi_head, 'with_text', False):
                    output_names.append('text_features')
                    dynamic_axes['text_features'] = {0: 'objects_num'}
        #  output_names.extend(['feature_vector', 'saliency_map'])

        with torch.no_grad():
            img_metas = data.pop("img_metas")
            from functools import partial
            model.forward = partial(model.forward, img_metas=img_metas, return_loss=False)
            torch.onnx.export(
                model,
                data['img'],
                f=export_name,
                verbose=verbose,
                opset_version=opset,
                strip_doc_string=strip_doc_string,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                input_names=['image'],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                keep_initializers_as_inputs=True,
                **kwargs
            )


def check_onnx_model(export_name):
    try:
        onnx.checker.check_model(export_name)
        print('ONNX check passed.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
        print('ONNX check failed.')
        print(ex)


def add_node_names(export_name):
    model = onnx.load(export_name)
    for n in model.graph.node:
        if not n.name:
            n.name = '_'.join([i for i in n.output])
    onnx.save(model, export_name)


def _get_mo_cmd():
    for mo_cmd in ('mo', 'mo.py'):
        try:
            run([mo_cmd, '-h'], stdout=DEVNULL, stderr=DEVNULL, check=True)
            return mo_cmd
        except CalledProcessError:
            pass
    raise RuntimeError('OpenVINO Model Optimizer is not found or configured improperly')

def export_to_openvino(cfg, onnx_model_path, output_dir_path, input_shape=None,
                       input_format='bgr', precision='FP32', with_text=False,
                       pruning_transformation=False):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    onnx_model = onnx.load(onnx_model_path)
    output_names = set(out.name for out in onnx_model.graph.output)
    # Clear names of the nodes that produce network's output blobs.
    for node in onnx_model.graph.node:
        if output_names.intersection(node.output):
            node.ClearField('name')
    onnx.save(onnx_model, onnx_model_path)
    output_names = ','.join(output_names)

    mo_cmd = _get_mo_cmd()

    normalize = None
    for pipeline in cfg.data.test.pipeline:
        if pipeline['type'] == 'MultiScaleFlipAug':
            normalize = [v for v in pipeline['transforms'] if v['type'] == 'Normalize'][0]
            break
    assert normalize, "Could not find normalize parameters in datapipeline"

    mean_values = normalize['mean']
    scale_values = normalize['std']
    command_line = [mo_cmd,
                    f'--input_model={onnx_model_path}',
                    f'--mean_values={mean_values}',
                    f'--scale_values={scale_values}',
                    f'--output_dir={output_dir_path}',
                    f'--output={output_names}',
                    f'--data_type={precision}']

    assert input_format.lower() in ['bgr', 'rgb']

    if input_shape is not None:
        command_line.append(f'--input_shape={input_shape}')
    if normalize['to_rgb'] and input_format.lower() == 'bgr' or \
            not normalize['to_rgb'] and input_format.lower() == 'rgb':
        command_line.append('--reverse_input_channels')
    if pruning_transformation:
        command_line.extend(['--transform', 'Pruning'])

    print(' '.join(command_line))

    run(command_line, check=True)

    if with_text:
        onnx_model_path_tr_encoder = onnx_model_path.replace('.onnx', '_text_recognition_head_encoder.onnx')
        command_line = [mo_cmd,
                        f'--input_model={onnx_model_path_tr_encoder}',
                        f'--output_dir={output_dir_path}']
        print(' '.join(command_line))
        run(command_line, check=True)

        onnx_model_path_tr_decoder = onnx_model_path.replace('.onnx', '_text_recognition_head_decoder.onnx')
        command_line = [mo_cmd,
                        f'--input_model={onnx_model_path_tr_decoder}',
                        f'--output_dir={output_dir_path}']
        print(' '.join(command_line))
        run(command_line, check=True)


def optimize_onnx_graph(onnx_model_path, precision='FP32'):
    def convert_np_to_float16(np_array, min_positive_val=1e-7, max_finite_val=1e4):
        def between(a, b, c):
            return np.logical_and(a < b, b < c)
        np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
        np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
        np_array = np.where(between(max_finite_val, np_array, float('inf')), max_finite_val, np_array)
        np_array = np.where(between(float('-inf'), np_array, -max_finite_val), -max_finite_val, np_array)
        return np.float16(np_array)

    def convert_tensor_float32_to_float16(tensor, min_positive_val=1e-7, max_finite_val=1e4):
        if tensor.data_type == onnx.onnx_pb.TensorProto.FLOAT:
            tensor.data_type = onnx.onnx_pb.TensorProto.FLOAT16
            if tensor.raw_data:
                float32_list = np.fromstring(tensor.raw_data, dtype='float32')
                float16_list = convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
                tensor.raw_data = float16_list.tostring()
        return tensor

    onnx_model = onnx.load(onnx_model_path)

    onnx_model = optimize(onnx_model, ['extract_constant_to_initializer',
                                       'eliminate_unused_initializer'])

    inputs = onnx_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    # TODO: Temp fix when data with a different data type gets into ONNX graph.
    #   https://github.com/openvinotoolkit/mmdetection/pull/345
    if precision == 'FP16':
        for initializer in onnx_model.graph.initializer:
            initializer = convert_tensor_float32_to_float16(initializer)
        for node in onnx_model.graph.node:
            if node.op_type == 'Cast':
                if node.attribute[0].i == onnx.onnx_pb.TensorProto.FLOAT:
                    node.attribute[0].i = onnx.onnx_pb.TensorProto.FLOAT16
    onnx.save(onnx_model, onnx_model_path)


def export_model(model, config, output_dir, target='openvino', onnx_opset=11,
                 input_shape=None, input_format='bgr', precision='FP32', alt_ssd_export=False,
                 pruning_transformation=False):
    assert onnx_opset in available_opsets
    assert onnx_opset >= get_min_opset_version()

    # FIXME.
    # torch.set_default_tensor_type(torch.FloatTensor)
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    model.eval()
    model = model.to("cpu")
    device = next(model.parameters()).device
    cfg = config
    fake_data = get_fake_input(cfg, device=device)
    if precision == 'FP16':
        wrap_fp16_model(model)

    mmcv.mkdir_or_exist(osp.abspath(output_dir))
    onnx_model_path = osp.join(output_dir, cfg.get('model_name', 'model') + '.onnx')

    with torch.no_grad():
        export_to_onnx(model, fake_data, export_name=onnx_model_path, opset=onnx_opset,
                       alt_ssd_export=alt_ssd_export,
                       target=target, verbose=False, strip_doc_string=True)
        # add_node_names(onnx_model_path)
        print(f'ONNX model has been saved to "{onnx_model_path}"')

    optimize_onnx_graph(onnx_model_path, precision)

    with_text = False
    if target == 'openvino' and not alt_ssd_export:
        if hasattr(model, 'roi_head'):
            if getattr(model.roi_head, 'with_text', False):
                with_text = True

    if target == 'openvino':
        if input_shape:
            input_shape = [1, 3, *input_shape]
        else:
            input_shape = list(fake_data['img'][0].shape)
        export_to_openvino(cfg, onnx_model_path, output_dir, input_shape, input_format, precision,
                           with_text=with_text, pruning_transformation=pruning_transformation)
    else:
        pass
        # Model check raises a Segmentation Fault in the latest (1.6.0, 1.7.0) versions of onnx package.
        # Even for a valid graph.
        # check_onnx_model(onnx_model_path)

    unregister_extra_symbolics(onnx_opset)
    unregister_extra_symbolics_for_openvino(onnx_opset)
