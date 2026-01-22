"""
Slice Attention Loss函数
组合volume-level、slice-level和segmentation三种loss
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .losses import DiceLoss


class SliceAttentionLoss(nn.Module):
    """
    Slice Attention多任务损失函数

    组合三种loss：
    1. Volume-level分类loss（BCE）
    2. Slice-level分类loss（MSE，软标签）
    3. Segmentation loss（Dice+CE，使用膨胀mask）
    """

    def __init__(
        self,
        volume_loss_weight: float = 1.0,
        slice_loss_weight: float = 0.3,
        seg_loss_weight: float = 0.5,
        use_focal: bool = False
    ):
        """
        初始化损失函数

        Args:
            volume_loss_weight: volume-level loss权重
            slice_loss_weight: slice-level loss权重
            seg_loss_weight: segmentation loss权重
            use_focal: 是否使用focal loss（volume-level）
        """
        super().__init__()

        self.volume_loss_weight = volume_loss_weight
        self.slice_loss_weight = slice_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.use_focal = use_focal

        # Volume-level分类loss
        if use_focal:
            from .focal_loss import FocalLossBCE
            self.volume_criterion = FocalLossBCE(alpha=0.25, gamma=2.0)
        else:
            self.volume_criterion = nn.BCEWithLogitsLoss()

        # Slice-level分类loss（MSE用于软标签）
        self.slice_criterion = nn.MSELoss()

        # Segmentation loss
        self.seg_criterion = DiceLoss()

        # 2-task任务映射到器官
        self.task_to_organ = {
            'liver_injury': 'liver',
            'liver_high_risk': 'liver',
            'spleen_injury': 'spleen',
            'spleen_high_risk': 'spleen',
            'kidney_injury': 'kidney',
            'kidney_high_risk': 'kidney',
            'bowel': 'bowel',
            'extravasation': None  # 不参与slice loss
        }

    def compute_volume_loss(
        self,
        volume_preds: Dict[str, torch.Tensor],
        volume_labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算volume-level分类loss

        Args:
            volume_preds: volume预测 {'task': (B,)} - BCE格式
            volume_labels: volume标签 {'task': (B,)}

        Returns:
            total_loss: 总loss
            loss_dict: 各任务loss字典
        """
        total_loss = 0.0
        loss_dict = {}

        for task, preds in volume_preds.items():
            if task not in volume_labels:
                continue

            labels = volume_labels[task].float()  # (B,)

            # BCE格式：preds已经是(B,)形状
            # 兼容旧的(B, 1)格式
            if preds.dim() == 2:
                preds = preds.squeeze(-1)

            task_loss = self.volume_criterion(preds, labels)
            total_loss += task_loss
            loss_dict[f'volume_{task}'] = task_loss.item()

        return total_loss, loss_dict

    def compute_slice_loss(
        self,
        slice_preds: Dict[str, torch.Tensor],
        slice_labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算slice-level分类loss（MSE，软标签）

        Args:
            slice_preds: slice预测 {'task': (B, D//3)}
            slice_labels: slice软标签 {'task': (B, D//3)}

        Returns:
            total_loss: 总loss
            loss_dict: 各任务loss字典
        """
        total_loss = 0.0
        loss_dict = {}

        for task, preds in slice_preds.items():
            if task not in slice_labels:
                continue

            labels = slice_labels[task]  # (B, D//3)

            # 使用sigmoid将logits转换为概率
            preds_prob = torch.sigmoid(preds)

            # MSE loss（软标签）
            task_loss = self.slice_criterion(preds_prob, labels)
            total_loss += task_loss
            loss_dict[f'slice_{task}'] = task_loss.item()

        return total_loss, loss_dict

    def compute_seg_loss(
        self,
        seg_preds: Dict[str, torch.Tensor],
        seg_masks_dilated: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算分割loss（使用膨胀mask）

        Args:
            seg_preds: 分割预测 {'organ': (B, 2, D//3, H, W)}
            seg_masks_dilated: 膨胀后的分割mask {'organ': (B, D//3, H, W)}

        Returns:
            total_loss: 总loss
            loss_dict: 各器官loss字典
        """
        total_loss = 0.0
        loss_dict = {}

        for organ, preds in seg_preds.items():
            if organ not in seg_masks_dilated:
                continue

            masks = seg_masks_dilated[organ]  # (B, D//3, H, W)
            masks = masks.long()

            task_loss = self.seg_criterion(preds, masks)
            total_loss += task_loss
            loss_dict[f'seg_{organ}'] = task_loss.item()

        return total_loss, loss_dict

    def forward(
        self,
        preds: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总loss

        Args:
            preds: 模型预测
            targets: 目标标签

        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}

        # Volume loss
        volume_loss, vol_dict = self.compute_volume_loss(
            preds['volume_preds'], targets['volume_labels']
        )
        loss_dict.update(vol_dict)

        # Slice loss
        slice_loss, slice_dict = self.compute_slice_loss(
            preds['slice_preds'], targets['slice_labels']
        )
        loss_dict.update(slice_dict)

        # Seg loss
        seg_loss, seg_dict = self.compute_seg_loss(
            preds['seg_preds'], targets['seg_masks_dilated']
        )
        loss_dict.update(seg_dict)

        # Total
        total_loss = (
            self.volume_loss_weight * volume_loss +
            self.slice_loss_weight * slice_loss +
            self.seg_loss_weight * seg_loss
        )

        loss_dict['volume_loss'] = volume_loss.item()
        loss_dict['slice_loss'] = slice_loss.item()
        loss_dict['seg_loss'] = seg_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
