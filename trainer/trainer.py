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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trauma_net_dinov3 import TraumaNetDINOv3
from data.trauma_dataset import TraumaDataset
from utils.losses import MultiTaskLoss
from utils.metrics import MultiTaskMetrics
from utils.logger import get_logger
from data.transforms import get_train_transform


class TraumaTrainer:
    """
    TraumaNet训练器
    适用于单张RGB图像 + DINOv3分类/分割
    """

    def __init__(self, config, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 初始化 logger
        log_file = os.path.join(config.log.log_dir, 'log.txt')
        self.logger = get_logger(log_file, file_save=True, display=True)
        self.logger.info("=" * 60)
        self.logger.info("TraumaNet 训练器初始化")
        self.logger.info(f"实验名称: {config.exp_name}, 设备: {self.device}")

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

        # 指标
        organ_names = list(config.model.num_classes.keys())
        self.train_metrics = MultiTaskMetrics(organ_names, config.model.num_classes)
        self.val_metrics = MultiTaskMetrics(organ_names, config.model.num_classes)
        self.test_metrics = MultiTaskMetrics(organ_names, config.model.num_classes)

        # TensorBoard
        self.writer = SummaryWriter(config.log.tensorboard_dir)

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.global_step = 0

        # 混合精度
        self.use_amp = config.training.use_amp
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.logger.info("训练器初始化完成！")
        self.logger.info("=" * 60)

    def _create_dataloaders(self):
        """创建训练、验证和测试数据加载器"""
        train_transform = get_train_transform(self.config)
        dataset_kwargs = {
            'image_dir': self.config.data.image_dir,
            'mask_dir': self.config.data.mask_dir,
            'target_shape': self.config.data.target_shape,
            'organ_labels': self.config.data.organ_labels,
            'use_preprocessed': getattr(self.config.data, 'use_preprocessed', False),
        }

        train_dataset = TraumaDataset(
            label_file=self.config.data.train_dataset,
            mode='train',
            transform=train_transform,
            **dataset_kwargs
        )
        val_dataset = TraumaDataset(
            label_file=self.config.data.val_dataset,
            mode='val',
            **dataset_kwargs
        )
        test_dataset = TraumaDataset(
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
        """创建DINOv3模型"""
        cfg_path = getattr(self.config.model, 'cfg_path', 'vit_base')
        pretrained_path = getattr(self.config.model, 'dinov3_pretrained', None)
        top_k_ratio = getattr(self.config.model, 'top_k_ratio', 0.2)
        dropout = getattr(self.config.model, 'dropout', 0.1)
        freeze_backbone = getattr(self.config.model, 'freeze_backbone', False)
        use_n_blocks = getattr(self.config.model, 'use_n_blocks', None)

        self.logger.info(f"  模型类型: DINOv3, ViT架构: {cfg_path}")
        model = TraumaNetDINOv3(
            cfg_path=cfg_path,
            pretrained_path=pretrained_path,
            img_size=self.config.data.target_shape[0],
            use_n_blocks=use_n_blocks,
            top_k_ratio=top_k_ratio,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            seg_organs=self.config.model.seg_organs,
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
            images = batch['image'].to(self.device)
            cls_targets = {k: v.to(self.device) for k, v in batch['labels'].items()}
            seg_targets = {k: v.to(self.device) for k, v in batch['masks'].items()}

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
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
                images = batch['image'].to(self.device)
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
                images = batch['image'].to(self.device)
                cls_targets = {k: v.to(self.device) for k, v in batch['labels'].items()}
                seg_targets = {k: v.to(self.device) for k, v in batch['masks'].items()}

                outputs = self.model(images)
                losses = self.criterion(outputs['cls_logits'], outputs['seg_logits'], cls_targets, seg_targets)
                test_losses.append(losses['total_loss'].item())

                self.test_metrics.update(outputs['cls_logits'], outputs['seg_logits'], cls_targets, seg_targets)

        metrics = self.test_metrics.compute()
        metrics['loss'] = np.mean(test_losses)
        return metrics
