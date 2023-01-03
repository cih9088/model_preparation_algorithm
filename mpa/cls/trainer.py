# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import os.path as osp
import time
import copy
import warnings
import torch
import numpy as np
import random

import warnings

from torch import nn
import mmcv
from mmcv.runner import DistSamplerSeedHook, Fp16OptimizerHook, build_optimizer, build_runner, HOOKS

from mmcls import __version__
from mmcls.apis import train_model
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier
from mmcls.utils import collect_env
from mmcls.core import DistOptimizerHook

from mpa.registry import STAGES
from mpa.modules.datasets.composed_dataloader import ComposedDL
from mpa.stage import Stage
from mpa.cls.stage import ClsStage
from mpa.modules.hooks.eval_hook import CustomEvalHook, DistCustomEvalHook
from mpa.modules.hooks.fp16_sam_optimizer_hook import Fp16SAMOptimizerHook
from mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class ClsTrainer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=True, **kwargs)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Environment
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)

        # Data
        if 'unlabeled' in cfg.data:
            datasets = [[build_dataset(cfg.data.train), build_dataset(cfg.data.unlabeled)]]
        else:
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

        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        # meta['config'] = cfg.pretty_text
        meta['seed'] = cfg.seed

        if isinstance(datasets[0], list):
            repr_ds = datasets[0][0]
        else:
            repr_ds = datasets[0]

        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmcls_version=__version__)
            if hasattr(repr_ds, 'tasks'):
                cfg.checkpoint_config.meta['tasks'] = repr_ds.tasks
            else:
                cfg.checkpoint_config.meta['CLASSES'] = repr_ds.CLASSES
            if 'task_adapt' in cfg:
                if hasattr(self, 'model_tasks'):  # for incremnetal learning
                    cfg.checkpoint_config.meta.update({'tasks': self.model_tasks})
                    # instead of update(self.old_tasks), update using "self.model_tasks"
                else:
                    cfg.checkpoint_config.meta.update({'CLASSES': self.model_classes})

        # Save config
        # cfg.dump(osp.join(cfg.work_dir, 'config.yaml')) # FIXME bug to save
        # logger.info(f'Config:\n{cfg.pretty_text}')

        # model
        model_builder = kwargs.get("model_builder", None)
        if model_builder is not None:
            model = model_builder(cfg)
        else:
            model = build_classifier(cfg.model)

        if self.distributed:
            self._modify_cfg_for_distributed(model, cfg)

        # fp16 setting for custom sam optimizer
        fp16_cfg = cfg.pop('fp16', None)
        if fp16_cfg is not None:
            if cfg.optimizer_config.get('type', None) == 'SAMOptimizerHook':
                cfg.optimizer_config.type = "Fp16SAMOptimizerHook"
                cfg.optimizer_config.distributed = distributed
                cfg.optimizer_config.loss_scale = fp16_cfg["loss_scale"]
            else:
                cfg.fp16 = fp16_cfg

        # register custom eval hooks
        validate = True if cfg.data.get("val", None) else False
        if validate:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_loader_cfg = {
                "samples_per_gpu": cfg.data.samples_per_gpu,
                "workers_per_gpu": cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                "num_gpus": len(cfg.gpu_ids),
                "dist": self.distributed,
                "round_up": True,
                "seed": cfg.seed,
                "shuffle": False,     # Not shuffle by default
                "sampler_cfg": None,  # Not use sampler by default
                **cfg.data.get('val_dataloader', {}),
            }
            val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            cfg.custom_hooks.append(
                dict(
                    type="DistCustomEvalHook" if distributed else "CustomEvalHook",
                    dataloader=val_dataloader,
                    priority='ABOVE_NORMAL',
                    **eval_cfg,
                )
            )

        train_model(
            model=model,
            dataset=dataset,
            cfg=cfg,
            distributed=self.distributed,
            validate=False,
            timestamp=timestamp,
            meta=meta,
        )

        # Save outputs
        output_ckpt_path = osp.join(cfg.work_dir, "best_model.pth"
                                    if osp.exists(osp.join(cfg.work_dir, "best_model.pth"))
                                    else "latest.pth")
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

        if cfg.dist_params.get("linear_scale_lr", False):
            new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
            logger.info(f"enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}")
            cfg.optimizer.lr = new_lr

    @staticmethod
    def register_checkpoint_hook(checkpoint_config):
        if checkpoint_config.get('type', False):
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        return hook
