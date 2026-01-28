"""
TraumaNet 训练器类（2D X-ray + DINOv3, 分类 + 分割）
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.xray_dinov3_v2 import TraumaNetDINOv3
from data.xraydataset import XrayBoneDataset
from utils.losses import MultiTaskLoss
from utils.metrics import MultiTaskMetrics
from utils.logger import get_logger
from data.xraydataset import RandomFlipRotate2D


class TraumaTrainer:
    """
    TraumaNet训练器
    适用于2D X-ray图像 + DINOv3分类/分割多任务学习
    """

    def __init__(self, config, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 初始化 logger
        log_file = os.path.join(config.log.log_dir, 'log.txt')
        self.logger = get_logger(log_file, file_save=True, display=True)
        self.logger.info("=" * 60)
        self.logger.info("TraumaNet 训练器初始化 (2D模式)")
        self.logger.info(f"实验名称: {config.exp_name}, 设备: {self.device}")

        # 从配置中获取分类任务名称和分割器官名称
        self.task_names = list(config.model.num_classes.keys())
        self.organ_names = list(getattr(config.model, 'seg_organs', ['mask']))
        self.logger.info(f"分类任务: {self.task_names}")
        self.logger.info(f"分割目标: {self.organ_names}")

        # 数据加载器
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()

        # 模型
        self.model = self._create_model().to(self.device)
        if torch.cuda.device_count() > 1:
            self.logger.info(f"  使用DataParallel，GPU数量: {torch.cuda.device_count()}")
            self.model = nn.DataParallel(self.model)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 损失函数
        self.criterion = self._create_criterion()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 指标（使用配置中的任务名称和器官名称）
        self.train_metrics = MultiTaskMetrics(self.task_names, config.model.num_classes, seg_organ_names=self.organ_names)
        self.val_metrics = MultiTaskMetrics(self.task_names, config.model.num_classes, seg_organ_names=self.organ_names)
        self.test_metrics = MultiTaskMetrics(self.task_names, config.model.num_classes, seg_organ_names=self.organ_names)

        # TensorBoard
        self.writer = SummaryWriter(config.log.tensorboard_dir)

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.global_step = 0

        # 混合精度
        self.use_amp = config.training.use_amp
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.logger.info("训练器初始化完成！")
        self.logger.info("=" * 60)

    def _create_dataloaders(self):
        """创建训练、验证和测试数据加载器"""
        train_transform = RandomFlipRotate2D(flip_prob=0.5, max_angle=15)
        dataset_kwargs = {
            'image_dir': self.config.data.image_dir,
            'mask_dir': self.config.data.mask_dir,
            'target_shape': tuple(self.config.data.target_shape),
            'use_preprocessed': getattr(self.config.data, 'use_preprocessed', False),
            'single_mask': getattr(self.config.data, 'single_mask', True),
            'mask_key': getattr(self.config.data, 'mask_key', 'mask'),
        }

        train_dataset = XrayBoneDataset(
            label_file=self.config.data.train_dataset,
            mode='train',
            transform=train_transform,
            **dataset_kwargs
        )
        val_dataset = XrayBoneDataset(
            label_file=self.config.data.val_dataset,
            mode='val',
            **dataset_kwargs
        )
        test_dataset = XrayBoneDataset(
            label_file=self.config.data.test_dataset,
            mode='test',
            **dataset_kwargs
        )

        loader_params = dict(
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)

        return train_loader, val_loader, test_loader

    def _create_model(self):
        """创建DINOv3 2D模型"""
        cfg_path = getattr(self.config.model, 'cfg_path', 'vit_base')
        pretrained_path = getattr(self.config.model, 'dinov3_pretrained', None)
        top_k_ratio = getattr(self.config.model, 'top_k_ratio', 0.2)
        dropout = getattr(self.config.model, 'dropout', 0.1)
        freeze_backbone = getattr(self.config.model, 'freeze_backbone', False)
        use_n_blocks = getattr(self.config.model, 'use_n_blocks', 4)
        input_channels = getattr(self.config.model, 'input_depth', 1)

        self.logger.info(f"  模型类型: DINOv3 2D, ViT架构: {cfg_path}")
        self.logger.info(f"  输入通道数: {input_channels}, 图像大小: {self.config.data.target_shape}")
        model = TraumaNetDINOv3(
            cfg_path=cfg_path,
            pretrained_path=pretrained_path,
            img_size=self.config.data.target_shape[0],
            use_n_blocks=use_n_blocks,
            top_k_ratio=top_k_ratio,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            task_names=self.task_names,
            organ_names=self.organ_names,
            input_channels=input_channels,
        )
        return model

    def _create_optimizer(self):
        """创建优化器"""
        opt_type = self.config.training.optimizer
        if opt_type == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        elif opt_type == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        elif opt_type == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.config.training.lr, momentum=self.config.training.momentum, weight_decay=self.config.training.weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {opt_type}")

    def _create_criterion(self):
        """创建多任务损失函数"""
        return MultiTaskLoss(
            seg_loss_weight=self.config.training.seg_loss_weight,
            class_weights=self.config.training.class_weights,
            loss_type=getattr(self.config.training, 'loss_type', 'CE'),
            focal_loss_alpha=getattr(self.config.training, 'focal_loss_alpha', 0.25),
            focal_loss_gamma=getattr(self.config.training, 'focal_loss_gamma', 2.0),
            label_smoothing=getattr(self.config.training, 'label_smoothing', 0.0)
        )

    def _create_scheduler(self):
        sched = getattr(self.config.training, 'scheduler', None)
        if sched == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.training.epochs, eta_min=self.config.training.min_lr)
        elif sched == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif sched == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10)
        return None

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.train_metrics.reset()
        epoch_losses = []

        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}"):
            images = batch['image'].to(self.device)  # (B, 1, H, W) — 模型内部自动repeat到3通道
            cls_targets = {k: v.to(self.device) for k, v in batch['labels'].items()}
            seg_targets = {k: v.to(self.device) for k, v in batch['masks'].items()}

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast('cuda'):
                    outputs = self.model(images)
                    losses = self.criterion(outputs['cls_logits'], outputs['seg_logits'], cls_targets, seg_targets)
                    loss = losses['total_loss']
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs['cls_logits'], outputs['seg_logits'], cls_targets, seg_targets)
                loss = losses['total_loss']
                loss.backward()
                self.optimizer.step()

            self.train_metrics.update(outputs['cls_logits'], outputs['seg_logits'], cls_targets, seg_targets)
            epoch_losses.append(loss.item())
            self.global_step += 1

        metrics = self.train_metrics.compute()
        metrics['loss'] = np.mean(epoch_losses)
        return metrics

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self.val_metrics.reset()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)  # (B, 1, H, W) — 模型内部自动repeat到3通道
                cls_targets = {k: v.to(self.device) for k, v in batch['labels'].items()}
                seg_targets = {k: v.to(self.device) for k, v in batch['masks'].items()}

                outputs = self.model(images)
                losses = self.criterion(outputs['cls_logits'], outputs['seg_logits'], cls_targets, seg_targets)
                val_losses.append(losses['total_loss'].item())

                self.val_metrics.update(outputs['cls_logits'], outputs['seg_logits'], cls_targets, seg_targets)

        metrics = self.val_metrics.compute()
        metrics['loss'] = np.mean(val_losses)
        return metrics

    def test(self) -> Dict[str, float]:
        self.model.eval()
        self.test_metrics.reset()
        test_losses = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                images = batch['image'].to(self.device)  # (B, 1, H, W) — 模型内部自动repeat到3通道
                cls_targets = {k: v.to(self.device) for k, v in batch['labels'].items()}
                seg_targets = {k: v.to(self.device) for k, v in batch['masks'].items()}

                outputs = self.model(images)
                losses = self.criterion(outputs['cls_logits'], outputs['seg_logits'], cls_targets, seg_targets)
                test_losses.append(losses['total_loss'].item())

                self.test_metrics.update(outputs['cls_logits'], outputs['seg_logits'], cls_targets, seg_targets)

        metrics = self.test_metrics.compute()
        metrics['loss'] = np.mean(test_losses)
        return metrics

    def train(self):
        self.logger.info("开始训练")
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch [{epoch+1}/{self.config.training.epochs}]")

            # ---- Train ----
            train_metrics = self.train_epoch()
            self.logger.info(f"Train metrics: {train_metrics}")

            # ---- Validate ----
            val_metrics = self.validate()
            self.logger.info(f"Val metrics: {val_metrics}")

            # ---- TensorBoard ----
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)

            # ---- Scheduler ----
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('metric', val_metrics['loss']))
                else:
                    self.scheduler.step()

            # ---- Save best ----
            cur_metric = val_metrics.get('metric', -val_metrics['loss'])
            if cur_metric > self.best_metric:
                self.best_metric = cur_metric
                self._save_checkpoint(best=True)

            # ---- Save last ----
            self._save_checkpoint(best=False)

        self.logger.info("训练完成")

    def _save_checkpoint(self, best=False):
        state = {
            'epoch': self.current_epoch,
            'model': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }

        ckpt_dir = self.config.log.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        if best:
            path = os.path.join(ckpt_dir, 'best.pth')
        else:
            path = os.path.join(ckpt_dir, 'last.pth')

        torch.save(state, path)

    def load_checkpoint(self, checkpoint_path: str):
        """从检查点恢复训练状态"""
        self.logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.logger.info(f"已恢复到 epoch {self.current_epoch}, best_metric={self.best_metric:.4f}")
