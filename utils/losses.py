"""
损失函数模块
包含Dice Loss和多任务损失函数
所有分类任务均为二分类，使用BCE系列损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .focal_loss import FocalLossBCE


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation

    用于分割任务的Dice损失函数
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        """
        初始化Dice Loss

        Args:
            smooth: 平滑因子，避免除零
            ignore_index: 忽略的标签索引
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss

        Args:
            pred: 预测logits (B, C, D, H, W) 或 (B, C, H, W)
            target: 目标标签 (B, D, H, W) 或 (B, H, W)

        Returns:
            Dice loss值
        """
        # 应用softmax获取概率
        pred = F.softmax(pred, dim=1)

        # 获取前景类别（假设类别1是前景）
        if pred.size(1) == 2:
            pred = pred[:, 1:, ...]  # (B, 1, ...)
            target_binary = (target > 0).float().unsqueeze(1)  # (B, 1, ...)
        else:
            # 多类别情况
            target_binary = F.one_hot(target, num_classes=pred.size(1))
            target_binary = target_binary.permute(0, -1, *range(1, len(target.shape)))
            target_binary = target_binary.float()

        # 展平
        pred = pred.contiguous().view(pred.size(0), -1)
        target_binary = target_binary.contiguous().view(target_binary.size(0), -1)

        # 计算Dice系数
        intersection = (pred * target_binary).sum(dim=1)
        union = pred.sum(dim=1) + target_binary.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 返回Dice Loss
        return 1.0 - dice.mean()


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数

    结合分类损失和分割损失
    所有分类任务均为二分类，支持BCE系列损失：BCE, BCE_weight, BCE_Focal
    """

    def __init__(
        self,
        seg_loss_weight: float = 0.5,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        loss_type: str = 'BCE',
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        pos_weights: Optional[Dict[str, torch.Tensor]] = None,
        label_smoothing: float = 0.0
    ):
        """
        初始化多任务损失

        Args:
            seg_loss_weight: 分割损失权重
            class_weights: 各器官的类别权重字典（未使用，保留兼容性）
            loss_type: 损失函数类型 ('BCE', 'BCE_weight', 'BCE_Focal')
            focal_loss_alpha: Focal Loss的alpha参数
            focal_loss_gamma: Focal Loss的gamma参数
            pos_weights: BCE_weight的正样本权重字典
            label_smoothing: Label smoothing参数（未使用，保留兼容性）
        """
        super(MultiTaskLoss, self).__init__()

        self.seg_loss_weight = seg_loss_weight
        self.class_weights = class_weights or {}
        self.loss_type = loss_type
        self.pos_weights = pos_weights or {}
        self.label_smoothing = label_smoothing

        # 分割损失
        self.dice_loss = DiceLoss()

        # 根据loss_type创建分类损失函数
        if loss_type == 'BCE_Focal':
            self.cls_criterion = FocalLossBCE(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        else:
            # BCE, BCE_weight使用标准的PyTorch损失函数
            self.cls_criterion = None

    def forward(
        self,
        cls_logits: Dict[str, torch.Tensor],
        seg_logits: Dict[str, torch.Tensor],
        cls_targets: Dict[str, torch.Tensor],
        seg_targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        Args:
            cls_logits: 分类预测字典 {'organ': (B,)} - 单个logit值
            seg_logits: 分割预测字典 {'organ': (B, 2, D, H, W)}
            cls_targets: 分类目标字典 {'organ': (B,)}
            seg_targets: 分割目标字典 {'organ': (B, 1, D, H, W)}

        Returns:
            损失字典，包含各器官损失和总损失
        """
        losses = {}
        total_cls_loss = 0.0
        total_seg_loss = 0.0

        # 1. 计算分类损失（全部使用BCE系列）
        num_valid_cls_tasks = 0
        for organ, logits in cls_logits.items():
            target = cls_targets[organ]
            target_float = target.float()

            # 处理logits形状：支持(B,)和(B, 2)两种格式
            if logits.dim() == 2 and logits.size(1) == 2:
                # (B, 2) -> 取正类logit
                logits_pos = logits[:, 1]
            else:
                # (B,) 直接使用
                logits_pos = logits

            # 根据loss_type计算损失
            if self.loss_type == 'BCE':
                cls_loss = F.binary_cross_entropy_with_logits(logits_pos, target_float)
            elif self.loss_type == 'BCE_weight':
                pos_weight = self.pos_weights.get(organ, None)
                if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
                    pos_weight = torch.tensor(pos_weight, device=logits.device, dtype=torch.float32)
                cls_loss = F.binary_cross_entropy_with_logits(
                    logits_pos, target_float, pos_weight=pos_weight
                )
            elif self.loss_type == 'BCE_Focal':
                cls_loss = self.cls_criterion(logits_pos, target_float)
            else:
                raise ValueError(f"不支持的loss类型: {self.loss_type}，请使用BCE/BCE_weight/BCE_Focal")

            # 检查loss是否有效
            if torch.isfinite(cls_loss):
                losses[f'{organ}_cls_loss'] = cls_loss
                total_cls_loss += cls_loss
                num_valid_cls_tasks += 1
            else:
                losses[f'{organ}_cls_loss'] = torch.tensor(0.0, device=logits.device)

        # 2. 计算分割损失（仅对有分割头的器官）
        num_seg_tasks = 0
        for organ, logits in seg_logits.items():
            if organ in seg_targets:
                target = seg_targets[organ]

                # 移除通道维度: (B, 1, H, W) -> (B, H, W) 或 (B, 1, D, H, W) -> (B, D, H, W)
                if target.size(1) == 1:
                    target = target.squeeze(1)

                # 计算Dice损失
                seg_loss = self.dice_loss(logits, target)
                losses[f'{organ}_seg_loss'] = seg_loss
                total_seg_loss += seg_loss
                num_seg_tasks += 1

        # 3. 计算总损失
        avg_cls_loss = total_cls_loss / num_valid_cls_tasks if num_valid_cls_tasks > 0 else 0.0
        avg_seg_loss = total_seg_loss / num_seg_tasks if num_seg_tasks > 0 else 0.0

        total_loss = avg_cls_loss + self.seg_loss_weight * avg_seg_loss

        losses['cls_loss'] = avg_cls_loss
        losses['seg_loss'] = avg_seg_loss
        losses['total_loss'] = total_loss

        return losses


if __name__ == "__main__":
    # 测试损失函数
    print("测试损失函数模块")

    # 测试Dice Loss
    print("\n1. 测试Dice Loss")
    dice_loss = DiceLoss()
    pred = torch.randn(2, 2, 54, 512, 512)  # (B, C, D, H, W)
    target = torch.randint(0, 2, (2, 54, 512, 512))  # (B, D, H, W)
    loss = dice_loss(pred, target)
    print(f"   Dice Loss: {loss.item():.4f}")

    # 测试多任务损失（BCE格式）
    print("\n2. 测试多任务损失 (BCE)")
    multi_loss = MultiTaskLoss(seg_loss_weight=0.5, loss_type='BCE')

    # 模拟预测和目标 - 全部二分类，logits为(B,)形状
    cls_logits = {
        'liver_injury': torch.randn(2),
        'liver_high_risk': torch.randn(2),
        'spleen_injury': torch.randn(2),
        'spleen_high_risk': torch.randn(2),
        'kidney_injury': torch.randn(2),
        'kidney_high_risk': torch.randn(2),
        'bowel': torch.randn(2),
        'extravasation': torch.randn(2)
    }

    seg_logits = {
        'liver': torch.randn(2, 2, 54, 512, 512),
        'spleen': torch.randn(2, 2, 54, 512, 512),
        'kidney': torch.randn(2, 2, 54, 512, 512),
        'bowel': torch.randn(2, 2, 54, 512, 512)
    }

    cls_targets = {
        'liver_injury': torch.randint(0, 2, (2,)),
        'liver_high_risk': torch.randint(0, 2, (2,)),
        'spleen_injury': torch.randint(0, 2, (2,)),
        'spleen_high_risk': torch.randint(0, 2, (2,)),
        'kidney_injury': torch.randint(0, 2, (2,)),
        'kidney_high_risk': torch.randint(0, 2, (2,)),
        'bowel': torch.randint(0, 2, (2,)),
        'extravasation': torch.randint(0, 2, (2,))
    }

    seg_targets = {
        'liver': torch.randint(0, 2, (2, 1, 54, 512, 512)),
        'spleen': torch.randint(0, 2, (2, 1, 54, 512, 512)),
        'kidney': torch.randint(0, 2, (2, 1, 54, 512, 512)),
        'bowel': torch.randint(0, 2, (2, 1, 54, 512, 512))
    }

    losses = multi_loss(cls_logits, seg_logits, cls_targets, seg_targets)

    print("   损失值:")
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"     {name}: {value.item():.4f}")

