# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp
import time
import glob

import mmcv
from mmcv import get_git_hash

from mmseg import __version__
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.utils import collect_env

from mpa.registry import STAGES
from mpa.seg.stage import SegStage, build_segmentor
from mpa.utils.logger import get_logger
from torch import nn

logger = get_logger()


@STAGES.register_module()
class SegTrainer(SegStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage for segmentation

        - Configuration
        - Environment setup
        - Run training via MMSegmentation -> MMCV
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        model_builder = kwargs.get("model_builder", build_segmentor)
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, **kwargs)

        if cfg.runner.type == 'IterBasedRunner':
            cfg.runner = dict(type=cfg.runner.type, max_iters=cfg.runner.max_iters)

        logger.info('train!')

        # Work directory
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Environment
        logger.info(f'cfg.gpu_ids = {cfg.gpu_ids}, distributed = {self.distributed}')
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # Data
        datasets = [build_dataset(cfg.data.train)]

        # Dataset for HPO
        hp_config = kwargs.get('hp_config', None)
        if hp_config is not None:
            import hpopt

            if isinstance(datasets[0], list):
                for idx, ds in enumerate(datasets[0]):
                    datasets[0][idx] = hpopt.createHpoDataset(ds, hp_config)
            else:
                datasets[0] = hpopt.createHpoDataset(datasets[0], hp_config)

        # Target classes
        if 'task_adapt' in cfg:
            target_classes = cfg.task_adapt.final
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        meta['seed'] = cfg.seed
        meta['exp_name'] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmseg_version=__version__ + get_git_hash()[:7],
                CLASSES=target_classes)

        # Model
        model_builder = kwargs.get("model_builder", None)
        if model_builder is not None:
            model = model_builder(cfg)
        else:
            model = build_segmentor(cfg.model)
        model.CLASSES = target_classes

        SegTrainer.configure_fp16_optimizer(cfg, distributed)

        if self.distributed:
            self._modify_cfg_for_distributed(model, cfg)

        validate = True if cfg.data.get('val', None) else False
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=self.distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta
        )

        # Save outputs
        output_ckpt_path = os.path.join(cfg.work_dir, 'latest.pth')
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, 'best_mDice_*.pth'))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, 'best_mIoU_*.pth'))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        # NNCF model
        compression_state_path = osp.join(cfg.work_dir, "compression_state.pth")
        if not os.path.exists(compression_state_path):
            compression_state_path = None
        before_ckpt_path = osp.join(cfg.work_dir, "before_training.pth")
        if not os.path.exists(before_ckpt_path):
            before_ckpt_path = None
        return dict(
            final_ckpt=output_ckpt_path,
            compression_state_path=compression_state_path,
            before_ckpt_path=before_ckpt_path,
        )

    def _modify_cfg_for_distributed(self, model, cfg):
        nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if cfg.dist_params.get('linear_scale_lr', False):
            new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
            logger.info(f'enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}')
            cfg.optimizer.lr = new_lr
