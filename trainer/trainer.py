"""
TraumaNet 训练器类
负责模型训练、验证和评估
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入已存在的模块
from models.xray_dinov3_v2 import TraumaNetDINOv3V2
from data.xraydataset import XrayBoneDataset as XrayDataset
from utils.losses import MultiTaskLoss
from utils.metrics import MultiTaskMetrics
from utils.logger import get_logger
from data.transforms import get_train_transform

# 注意：以下模块未实现，如需要请自行添加
# from models.trauma_net_25d import TraumaNet25D
# from models.trauma_net_slice_attention import TraumaNetSliceAttention
# from models.trauma_net_dinov3 import TraumaNetDINOv3SliceAttention
# from data.trauma_dataset import TraumaDataset
# from data.trauma_dataset_2task import TraumaDataset2Task
# from data.trauma_dataset_slice_attention import TraumaDatasetSliceAttention



class TraumaTrainer:
    """
    TraumaNet训练器

    负责完整的训练流程：
    - 创建数据集和数据加载器
    - 创建模型、优化器、损失函数
    - 训练、验证、保存和日志记录
    """

    def __init__(self, config, device: str = 'cuda'):
        """
        初始化训练器

        Args:
            config: OmegaConf配置对象
            device: 设备 ('cuda' or 'cpu')
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 初始化logger（第一步）
        log_file = os.path.join(config.log.log_dir, 'log.txt')
        self.logger = get_logger(log_file, file_save=True, display=True)

        self.logger.info("="*60)
        self.logger.info("TraumaNet 训练器初始化")
        self.logger.info("="*60)
        self.logger.info(f"实验名称: {config.exp_name}")
        self.logger.info(f"设备: {self.device}")

        # 创建数据加载器
        self.logger.info("")
        self.logger.info("[1/5] 创建数据加载器...")
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()
        self.logger.info(f"  训练样本数: {len(self.train_loader.dataset)}")
        self.logger.info(f"  验证样本数: {len(self.val_loader.dataset)}")
        self.logger.info(f"  测试样本数: {len(self.test_loader.dataset)}")

        # 创建模型
        self.logger.info("")
        self.logger.info("[2/5] 创建模型...")
        self.model = self._create_model()
        self.logger.info(f"  Backbone: {config.model.backbone}")
        self.logger.info(f"  输入深度: {config.model.input_depth}")

        # 创建优化器
        self.logger.info("")
        self.logger.info("[3/5] 创建优化器...")
        self.optimizer = self._create_optimizer()
        self.logger.info(f"  优化器: {config.training.optimizer}")
        self.logger.info(f"  学习率: {config.training.lr}")

        # 创建损失函数
        self.logger.info("")
        self.logger.info("[4/5] 创建损失函数...")
        self.criterion = self._create_criterion()
        self.logger.info(f"  分割损失权重: {config.training.seg_loss_weight}")

        # 创建学习率调度器
        self.scheduler = self._create_scheduler()

        # 创建评估指标
        organ_names = list(config.model.num_classes.keys())
        self.train_metrics = MultiTaskMetrics(organ_names, config.model.num_classes)
        self.val_metrics = MultiTaskMetrics(organ_names, config.model.num_classes)
        self.test_metrics = MultiTaskMetrics(organ_names, config.model.num_classes)

        # TensorBoard
        self.logger.info("")
        self.logger.info("[5/5] 初始化日志系统...")
        self.writer = SummaryWriter(config.log.tensorboard_dir)
        self.logger.info(f"  TensorBoard目录: {config.log.tensorboard_dir}")
        self.logger.info(f"  检查点目录: {config.log.checkpoint_dir}")

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.global_step = 0

        # 混合精度训练
        self.use_amp = config.training.use_amp
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.logger.info(f"  混合精度训练: {self.use_amp}")

        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("训练器初始化完成！")
        self.logger.info("="*60)
        self.logger.info("")

    def _create_dataloaders(self):
        """创建训练、验证和测试数据加载器"""
        # 获取use_preprocessed参数（默认False）
        use_preprocessed = getattr(self.config.data, 'use_preprocessed', False)

        # 获取dataset_class参数（默认使用XrayDataset）
        dataset_class_name = getattr(self.config.data, 'dataset_class', 'XrayDataset')
        if dataset_class_name == 'XrayDataset':
            DatasetClass = XrayDataset
        else:
            # 如果指定了其他数据集类，抛出错误提示
            raise ValueError(
                f"不支持的数据集类: {dataset_class_name}。"
                f"当前仅支持 'XrayDataset'。"
                f"如需要其他数据集，请自行实现并添加导入。"
            )

        # 创建数据增强（仅用于训练集）
        train_transform = get_train_transform(self.config)
        if train_transform is not None:
            self.logger.info("  数据增强已启用:")
            self.logger.info(f"    flip_prob: {train_transform.flip_prob}")
            self.logger.info(f"    rotate_prob: {train_transform.rotate_prob}")

        # 准备数据集参数
        dataset_kwargs = {
            'image_dir': self.config.data.image_dir,
            'mask_dir': self.config.data.mask_dir,
            'target_shape': self.config.data.target_shape,
            'window_center': self.config.data.window_center,
            'window_width': self.config.data.window_width,
            'organ_labels': self.config.data.organ_labels,
            'use_preprocessed': use_preprocessed
        }

        # 如果是Slice Attention数据集，添加dilation_pixels参数
        if dataset_class_name == 'TraumaDatasetSliceAttention':
            dilation_pixels = getattr(self.config.data, 'dilation_pixels', 5)
            dataset_kwargs['dilation_pixels'] = dilation_pixels

        train_dataset = DatasetClass(
            label_file=self.config.data.train_dataset,
            mode='train',
            transform=train_transform,  # 传递数据增强
            **dataset_kwargs
        )

        val_dataset = DatasetClass(
            label_file=self.config.data.val_dataset,
            mode='val',
            **dataset_kwargs
        )

        test_dataset = DatasetClass(
            label_file=self.config.data.test_dataset,
            mode='test',
            **dataset_kwargs
        )

        # 获取DataLoader配置参数
        prefetch_factor = getattr(self.config.data, 'prefetch_factor', 2)
        persistent_workers = getattr(self.config.data, 'persistent_workers', False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=prefetch_factor if self.config.data.num_workers > 0 else None,
            persistent_workers=persistent_workers if self.config.data.num_workers > 0 else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=prefetch_factor if self.config.data.num_workers > 0 else None,
            persistent_workers=persistent_workers if self.config.data.num_workers > 0 else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=prefetch_factor if self.config.data.num_workers > 0 else None,
            persistent_workers=persistent_workers if self.config.data.num_workers > 0 else False
        )

        return train_loader, val_loader, test_loader

    def _create_model(self):
        """创建模型"""
        # 获取模型类型（默认dinov3_v2）
        model_type = getattr(self.config.model, 'model_type', 'dinov3_v2')

        if model_type == 'dinov3_v2':
            # DINOv3 V2 改进版模型
            vit_arch = getattr(self.config.model, 'vit_arch', 'vit_base')
            dinov3_pretrained = getattr(self.config.model, 'dinov3_pretrained', None)
            top_k_ratio = getattr(self.config.model, 'top_k_ratio', 0.2)
            num_local_layers = getattr(self.config.model, 'num_local_layers', 2)
            num_global_layers = getattr(self.config.model, 'num_global_layers', 2)
            window_size = getattr(self.config.model, 'window_size', 7)
            use_task_interaction = getattr(self.config.model, 'use_task_interaction', True)
            dropout = getattr(self.config.model, 'dropout', 0.1)

            self.logger.info(f"  模型类型: DINOv3 V2 (改进版)")
            self.logger.info(f"  ViT架构: {vit_arch}")
            self.logger.info(f"  Top-K比例: {top_k_ratio}")
            self.logger.info(f"  Transformer层数: 局部{num_local_layers} + 全局{num_global_layers}")
            self.logger.info(f"  Dropout: {dropout}")

            model = TraumaNetDINOv3V2(
                vit_arch=vit_arch,
                pretrained_path=dinov3_pretrained,
                img_size=self.config.data.target_shape[0],
                input_depth=self.config.model.input_depth,
                top_k_ratio=top_k_ratio,
                num_local_layers=num_local_layers,
                num_global_layers=num_global_layers,
                window_size=window_size,
                use_task_interaction=use_task_interaction,
                dropout=dropout,
                seg_organs=self.config.model.seg_organs,
            )
        else:
            # 不支持的模型类型
            raise ValueError(
                f"不支持的模型类型: {model_type}。"
                f"当前仅支持 'dinov3_v2'。"
                f"如需要其他模型，请自行实现并添加导入。"
            )

        model = model.to(self.device)

        # 如果有多个GPU可用，使用DataParallel
        if torch.cuda.device_count() > 1:
            self.logger.info(f"  使用DataParallel，GPU数量: {torch.cuda.device_count()}")
            model = nn.DataParallel(model)

        return model

    def _create_optimizer(self):
        """创建优化器"""
        if self.config.training.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.training.lr,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.training.optimizer}")
        return optimizer

    def _create_criterion(self):
        """创建损失函数"""
        # 获取loss配置参数
        loss_type = getattr(self.config.training, 'loss_type', 'CE')
        focal_loss_alpha = getattr(self.config.training, 'focal_loss_alpha', 0.25)
        focal_loss_gamma = getattr(self.config.training, 'focal_loss_gamma', 2.0)
        label_smoothing = getattr(self.config.training, 'label_smoothing', 0.0)

        # 计算BCE_weight的pos_weight
        pos_weights = {}
        if loss_type == 'BCE_weight':
            self.logger.info("  计算BCE_weight的pos_weight...")
            # 从训练集中统计每个任务的正负样本数
            train_dataset = self.train_loader.dataset
            organ_names = list(self.config.model.num_classes.keys())

            # 键名映射：config键名 -> CSV列名
            key_to_column = {
                'liver_injury': 'liver_injury',
                'liver_high_risk': 'liver_high',
                'spleen_injury': 'spleen_injury',
                'spleen_high_risk': 'spleen_high',
                'kidney_injury': 'kidney_injury',
                'kidney_high_risk': 'kidney_high',
                'bowel': 'bowel_injury',
                'extravasation': 'extravasation_injury',
            }

            for organ in organ_names:
                pos_count = 0
                neg_count = 0

                # 获取对应的CSV列名
                col_name = key_to_column.get(organ, organ)

                # 从labels_df中统计正负样本
                if col_name in train_dataset.labels_df.columns:
                    labels = train_dataset.labels_df[col_name].values
                    pos_count = int((labels == 1).sum())
                    neg_count = int((labels == 0).sum())

                # 计算pos_weight = neg / pos
                if pos_count > 0:
                    pos_weight = neg_count / pos_count
                else:
                    pos_weight = 1.0

                pos_weights[organ] = pos_weight
                self.logger.info(f"    {organ}: pos={pos_count}, neg={neg_count}, weight={pos_weight:.4f}")

        return MultiTaskLoss(
            seg_loss_weight=self.config.training.seg_loss_weight,
            class_weights=self.config.training.class_weights,
            loss_type=loss_type,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            pos_weights=pos_weights if pos_weights else None,
            label_smoothing=label_smoothing
        )

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.training.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.training.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10
            )
        else:
            scheduler = None
        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()

        epoch_losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 获取数据并移到GPU
            images = batch['image'].to(self.device)  # (B, 1, D, H, W)

            # 兼容不同数据集格式
            # 常规数据集: batch['labels'], batch['masks']
            # Slice Attention数据集: batch['volume_labels'], batch['seg_masks']
            if 'labels' in batch:
                cls_targets = {k: v.to(self.device) for k, v in batch['labels'].items()}
            elif 'volume_labels' in batch:
                cls_targets = {k: v.to(self.device) for k, v in batch['volume_labels'].items()}
            else:
                raise KeyError("Batch must contain either 'labels' or 'volume_labels'")

            if 'masks' in batch:
                seg_targets = {k: v.to(self.device) for k, v in batch['masks'].items()}
            elif 'seg_masks' in batch:
                seg_targets = {k: v.to(self.device) for k, v in batch['seg_masks'].items()}
            else:
                raise KeyError("Batch must contain either 'masks' or 'seg_masks'")

            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    losses = self.criterion(
                        outputs['cls_logits'],
                        outputs['seg_logits'],
                        cls_targets,
                        seg_targets
                    )
                    loss = losses['total_loss']
            else:
                outputs = self.model(images)
                losses = self.criterion(
                    outputs['cls_logits'],
                    outputs['seg_logits'],
                    cls_targets,
                    seg_targets
                )
                loss = losses['total_loss']

            # 反向传播
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                self.optimizer.step()

            # 更新指标
            self.train_metrics.update(
                outputs['cls_logits'],
                outputs['seg_logits'],
                cls_targets,
                seg_targets
            )

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            self.global_step += 1

        # 计算epoch平均指标
        metrics = self.train_metrics.compute()
        metrics['loss'] = np.mean(epoch_losses)

        return metrics

    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        self.val_metrics.reset()

        val_losses = []
        # 存储预测结果用于保存
        predictions_dict = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # 获取数据并移到GPU
                images = batch['image'].to(self.device)
                patient_ids = batch['patient_id']

                # 兼容不同数据集格式
                if 'labels' in batch:
                    cls_targets = {k: v.to(self.device) for k, v in batch['labels'].items()}
                elif 'volume_labels' in batch:
                    cls_targets = {k: v.to(self.device) for k, v in batch['volume_labels'].items()}
                else:
                    raise KeyError("Batch must contain either 'labels' or 'volume_labels'")

                if 'masks' in batch:
                    seg_targets = {k: v.to(self.device) for k, v in batch['masks'].items()}
                elif 'seg_masks' in batch:
                    seg_targets = {k: v.to(self.device) for k, v in batch['seg_masks'].items()}
                else:
                    raise KeyError("Batch must contain either 'masks' or 'seg_masks'")

                outputs = self.model(images)
                losses = self.criterion(
                    outputs['cls_logits'],
                    outputs['seg_logits'],
                    cls_targets,
                    seg_targets
                )

                self.val_metrics.update(
                    outputs['cls_logits'],
                    outputs['seg_logits'],
                    cls_targets,
                    seg_targets
                )

                val_losses.append(losses['total_loss'].item())

                # 保存预测结果（全部二分类BCE）
                for i, patient_id in enumerate(patient_ids):
                    pred_data = {}
                    for organ in outputs['cls_logits'].keys():
                        logits = outputs['cls_logits'][organ][i]
                        # BCE格式: 标量 - 单个logit值
                        logit_val = logits.item()
                        prob = torch.sigmoid(logits).cpu().numpy()
                        pred = 1 if logit_val > 0 else 0
                        target = cls_targets[organ][i]

                        pred_data[f'{organ}_pred'] = pred
                        pred_data[f'{organ}_prob'] = prob
                        pred_data[f'{organ}_target'] = target.cpu().numpy()

                    predictions_dict[str(patient_id)] = pred_data

        metrics = self.val_metrics.compute()
        metrics['loss'] = np.mean(val_losses)

        # 保存预测结果到npz文件
        save_dir = os.path.join(self.config.log.log_dir, 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'val_epoch_{self.current_epoch+1}.npz')
        np.savez(save_path, **predictions_dict)
        self.logger.info(f"  验证预测结果已保存: {save_path}")

        return metrics

    def test(self) -> Dict[str, float]:
        """测试模型"""
        self.model.eval()
        self.test_metrics.reset()

        test_losses = []
        # 存储预测结果用于保存
        predictions_dict = {}

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # 获取数据并移到GPU
                images = batch['image'].to(self.device)
                patient_ids = batch['patient_id']

                # 兼容不同数据集格式
                if 'labels' in batch:
                    cls_targets = {k: v.to(self.device) for k, v in batch['labels'].items()}
                elif 'volume_labels' in batch:
                    cls_targets = {k: v.to(self.device) for k, v in batch['volume_labels'].items()}
                else:
                    raise KeyError("Batch must contain either 'labels' or 'volume_labels'")

                if 'masks' in batch:
                    seg_targets = {k: v.to(self.device) for k, v in batch['masks'].items()}
                elif 'seg_masks' in batch:
                    seg_targets = {k: v.to(self.device) for k, v in batch['seg_masks'].items()}
                else:
                    raise KeyError("Batch must contain either 'masks' or 'seg_masks'")

                outputs = self.model(images)
                losses = self.criterion(
                    outputs['cls_logits'],
                    outputs['seg_logits'],
                    cls_targets,
                    seg_targets
                )

                self.test_metrics.update(
                    outputs['cls_logits'],
                    outputs['seg_logits'],
                    cls_targets,
                    seg_targets
                )

                test_losses.append(losses['total_loss'].item())

                # 保存预测结果（全部二分类BCE）
                for i, patient_id in enumerate(patient_ids):
                    pred_data = {}
                    for organ in outputs['cls_logits'].keys():
                        logits = outputs['cls_logits'][organ][i]
                        # BCE格式: 标量 - 单个logit值
                        logit_val = logits.item()
                        prob = torch.sigmoid(logits).cpu().numpy()
                        pred = 1 if logit_val > 0 else 0
                        target = cls_targets[organ][i]

                        pred_data[f'{organ}_pred'] = pred
                        pred_data[f'{organ}_prob'] = prob
                        pred_data[f'{organ}_target'] = target.cpu().numpy()

                    predictions_dict[str(patient_id)] = pred_data

        metrics = self.test_metrics.compute()
        metrics['loss'] = np.mean(test_losses)

        # 保存预测结果到npz文件
        save_dir = os.path.join(self.config.log.log_dir, 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'test_epoch_{self.current_epoch+1}.npz')
        np.savez(save_path, **predictions_dict)
        self.logger.info(f"  测试预测结果已保存: {save_path}")

        return metrics

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filepath = os.path.join(self.config.log.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(self.config.log.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"从检查点恢复训练: epoch {self.current_epoch}, best_metric: {self.best_metric:.4f}")

    def train(self):
        """主训练循环"""
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("开始训练")
        self.logger.info("="*60)
        self.logger.info("")

        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 测试
            test_metrics = self.test()

            # 打印指标
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.logger.info(f"{'='*80}")

            # Train指标
            self.logger.info(f"\n[Train] Loss: {train_metrics['loss']:.4f}")

            # Val指标
            self.logger.info(f"\n[Val] Loss: {val_metrics['loss']:.4f}")
            for key, value in sorted(val_metrics.items()):
                if key != 'loss':
                    self.logger.info(f"  {key}: {value:.4f}")

            # Test指标
            self.logger.info(f"\n[Test] Loss: {test_metrics['loss']:.4f}")
            for key, value in sorted(test_metrics.items()):
                if key != 'loss':
                    self.logger.info(f"  {key}: {value:.4f}")

            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/test', test_metrics['loss'], epoch)

            # 保存最佳模型
            current_metric = val_metrics.get('avg_f1', 0.0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.logger.info(f"  ✓ 新的最佳模型! F1: {self.best_metric:.4f}")

            # 保存检查点
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', is_best=is_best)

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("训练完成！")
        self.logger.info(f"最佳F1分数: {self.best_metric:.4f}")
        self.logger.info("="*60)
        self.logger.info("")

        self.writer.close()
