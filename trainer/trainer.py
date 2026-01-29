"""
TraumaNet 训练器类（单张RGB+DINOv3）
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
from tqdm import tqdm
import numpy as np
import sys
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.xray_dinov3_v2 import TraumaNetDINOv3
from data.xraydataset import XrayBoneDataset
from utils.losses import MultiTaskLoss
from utils.metrics import MultiTaskMetrics
from utils.logger import get_logger
from data.xraydataset import RandomFlipRotate2D

class TraumaTrainer:
    """
    TraumaNet训练器（单张RGB + DINOv3，Patch-level Segmentation）
    """

    def __init__(self, config, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # logger
        log_file = os.path.join(config.log.log_dir, 'log.txt')
        self.logger = get_logger(log_file, file_save=True, display=True)
        self.logger.info("=" * 60)
        self.logger.info("TraumaNet 训练器初始化")
        self.logger.info(f"实验名称: {config.exp_name}, 设备: {self.device}")

        # dataloader
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()

        # model
        self.model = self._create_model().to(self.device)
        if torch.cuda.device_count() > 1:
            self.logger.info(f"使用DataParallel，GPU数量: {torch.cuda.device_count()}")
            self.model = nn.DataParallel(self.model)

        # optimizer
        self.optimizer = self._create_optimizer()

        # loss
        self.criterion = self._create_criterion()

        # scheduler
        self.scheduler = self._create_scheduler()

        # metrics
        organ_names = list(config.model.num_classes.keys())
        seg_organ_names = list(config.model.seg_organs) if hasattr(config.model, 'seg_organs') else None
        self.train_metrics = MultiTaskMetrics(organ_names, config.model.num_classes, seg_organ_names=seg_organ_names)
        self.val_metrics = MultiTaskMetrics(organ_names, config.model.num_classes, seg_organ_names=seg_organ_names)
        self.test_metrics = MultiTaskMetrics(organ_names, config.model.num_classes, seg_organ_names=seg_organ_names)

        # tensorboard
        self.writer = SummaryWriter(config.log.tensorboard_dir)

        # 状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.global_step = 0

        # 混合精度
        self.use_amp = config.training.use_amp
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.logger.info("训练器初始化完成！")
        self.logger.info("=" * 60)

    # ------------------- Dataloader / Model / Optimizer -------------------
    def _create_dataloaders(self):
        train_transform = RandomFlipRotate2D(flip_prob=0.5, max_angle=15)
        dataset_kwargs = {
            'image_dir': getattr(self.config.data, 'image_dir', None),
            'mask_dir': getattr(self.config.data, 'mask_dir', None),
            'target_shape': self.config.data.target_shape,
            'use_preprocessed': getattr(self.config.data, 'use_preprocessed', False),
            'single_mask': getattr(self.config.data, 'single_mask', True),
            'mask_key': getattr(self.config.data, 'mask_key', 'bone'),
        }
        train_dataset = XrayBoneDataset(label_file=self.config.data.train_dataset,
                                        mode='train', transform=train_transform, **dataset_kwargs)
        val_dataset = XrayBoneDataset(label_file=self.config.data.val_dataset,
                                      mode='val', **dataset_kwargs)
        test_dataset = XrayBoneDataset(label_file=self.config.data.test_dataset,
                                       mode='test', **dataset_kwargs)

        loader_params = dict(
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        return (DataLoader(train_dataset, shuffle=True, **loader_params),
                DataLoader(val_dataset, shuffle=False, **loader_params),
                DataLoader(test_dataset, shuffle=False, **loader_params))

    def _create_model(self):
        cfg_path = getattr(self.config.model, 'cfg_path', 'vit_base')
        pretrained_path = getattr(self.config.model, 'dinov3_pretrained', None)
        top_k_ratio = getattr(self.config.model, 'top_k_ratio', 0.2)
        dropout = getattr(self.config.model, 'dropout', 0.1)
        freeze_backbone = getattr(self.config.model, 'freeze_backbone', False)
        use_n_blocks = getattr(self.config.model, 'use_n_blocks', 4)

        task_names = list(self.config.model.num_classes.keys()) if hasattr(self.config.model, 'num_classes') else None
        seg_organ_names = list(self.config.model.seg_organs) if hasattr(self.config.model, 'seg_organs') else None

        self.logger.info(f"模型类型: DINOv3, ViT架构: {cfg_path}")
        if task_names: self.logger.info(f"分类任务: {task_names}")
        if seg_organ_names: self.logger.info(f"分割器官: {seg_organ_names}")

        model = TraumaNetDINOv3(
            cfg_path=cfg_path,
            pretrained_path=pretrained_path,
            task_names=task_names,
            seg_organ_names=seg_organ_names,
            img_size=self.config.data.target_shape[0],
            use_n_blocks=use_n_blocks,
            top_k_ratio=top_k_ratio,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        return model

    def _create_optimizer(self):
        opt_type = self.config.training.optimizer
        if opt_type.lower() == 'adam': return torch.optim.Adam(self.model.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        if opt_type.lower() == 'adamw': return torch.optim.AdamW(self.model.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        if opt_type.lower() == 'sgd': return torch.optim.SGD(self.model.parameters(), lr=self.config.training.lr,
                                                             momentum=self.config.training.momentum,
                                                             weight_decay=self.config.training.weight_decay)
        raise ValueError(f"不支持的优化器: {opt_type}")

    def _create_criterion(self):
        return MultiTaskLoss(
            seg_loss_weight=self.config.training.seg_loss_weight,
            loss_type=getattr(self.config.training, 'loss_type', 'CE'),
            focal_loss_alpha=getattr(self.config.training, 'focal_loss_alpha', 0.25),
            focal_loss_gamma=getattr(self.config.training, 'focal_loss_gamma', 2.0),
            pos_weights=getattr(self.config.training, 'pos_weights', None),
            label_smoothing=getattr(self.config.training, 'label_smoothing', 0.0)
        )

    def _create_scheduler(self):
        sched = getattr(self.config.training, 'scheduler', None)
        if sched == 'cosine': return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.training.epochs, eta_min=self.config.training.min_lr)
        if sched == 'step': return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        if sched == 'plateau': return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10)
        return None

    # ------------------- Patch-level Helper -------------------
    def _compute_patch_targets(self, seg_targets: Dict[str, torch.Tensor]):
        patch_targets = {}
        for k, mask in seg_targets.items():
            # mask: [B, 1, H, W] 或 [B, H, W]
            if mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            patch_size = self.model.seg_head.patch_size
            patch_target = F.avg_pool2d(mask.unsqueeze(1).float(), kernel_size=patch_size, stride=patch_size)
            patch_targets[k] = (patch_target > 0.5).long().squeeze(1)
        return patch_targets

    # ------------------- Train / Val / Test Epoch -------------------
    def _run_epoch(self, loader, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        if mode == 'train':
            metrics = self.train_metrics
        elif mode == 'val':
            metrics = self.val_metrics
        else:
            metrics = self.test_metrics
        metrics.reset()
        epoch_losses = []

        for batch in tqdm(loader, desc=mode.capitalize(), leave=False):
            images = batch['image'].to(self.device)
            if images.shape[1] == 1:
                images = images.repeat(1,3,1,1)

            cls_targets = {k:v.to(self.device) for k,v in batch['labels'].items()}
            seg_targets = {k:v.to(self.device) for k,v in batch['masks'].items()}

            # 生成 patch-level target
            patch_targets = {}
            for k, mask in seg_targets.items():
                if mask.ndim == 4 and mask.shape[1] == 1:
                    mask = mask.squeeze(1)  # [B,H,W]
                patch_size = self.model.module.seg_head.patch_size if isinstance(self.model, nn.DataParallel) else self.model.seg_head.patch_size
                patch_map = F.avg_pool2d(mask.unsqueeze(1).float(), kernel_size=patch_size, stride=patch_size)
                patch_targets[k] = (patch_map > 0.5).long().squeeze(1)  # [B,H_patch,W_patch]

            self.optimizer.zero_grad()
            if mode == 'train' and self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    losses = self.criterion(outputs['cls_logits'], outputs['seg_logits'], cls_targets, patch_targets)
                    loss = losses['total_loss']
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if mode == 'train':
                    losses = self.criterion(outputs['cls_logits'], outputs['seg_logits'], cls_targets, patch_targets)
                    loss = losses['total_loss']
                    loss.backward()
                    self.optimizer.step()
                else:
                    losses = self.criterion(outputs['cls_logits'], outputs['seg_logits'], cls_targets, patch_targets)

            epoch_losses.append(losses['total_loss'].item())
            metrics.update(outputs['cls_logits'], outputs['seg_logits'], cls_targets, patch_targets)

            if mode == 'train':
                self.global_step += 1

        result = metrics.compute()
        result['loss'] = sum(epoch_losses)/len(epoch_losses)
        return result


    def train_epoch(self): return self._run_epoch(self.train_loader, mode='train')
    def validate(self): return self._run_epoch(self.val_loader, mode='val')
    def test(self): return self._run_epoch(self.test_loader, mode='test')

    # ------------------- Trainer Loop -------------------
    def train(self):
        self.logger.info("开始训练")
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch [{epoch+1}/{self.config.training.epochs}]")

            train_metrics = self.train_epoch()
            self.logger.info(f"Train metrics: {train_metrics}")

            val_metrics = self.validate()
            self.logger.info(f"Val metrics: {val_metrics}")

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('metric', val_metrics['loss']))
                else:
                    self.scheduler.step()

            # 保存
            cur_metric = val_metrics.get('metric', -val_metrics['loss'])
            if cur_metric > self.best_metric:
                self.best_metric = cur_metric
                self._save_checkpoint(best=True)
            self._save_checkpoint(best=False)

        self.logger.info("训练完成")

    # ------------------- Checkpoint -------------------
    def _save_checkpoint(self, best=False):
        state = {
            'epoch': self.current_epoch,
            'model': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric
        }
        ckpt_dir = self.config.log.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, 'best.pth' if best else 'last.pth')
        torch.save(state, path)
