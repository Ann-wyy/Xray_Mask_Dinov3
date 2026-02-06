import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .focal_loss import FocalLossBCE

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, 1, H_patch, W_patch] 或 [B, C, H_patch, W_patch]
        target: [B, H_patch, W_patch] 或 [B, C, H_patch, W_patch]
        """
        B = pred.size(0)

        # 二分类
        if pred.size(1) == 1 or pred.size(1) == 2:
            pred = torch.sigmoid(pred) if pred.size(1) == 1 else F.softmax(pred, dim=1)[:,1:2,:,:]
            pred = pred.squeeze(1)  # [B,H_patch,W_patch]
            target_binary = (target > 0).float()
        else:
            if target.dim() == 3:  # [B,H,W] -> one-hot
                target_binary = F.one_hot(target, num_classes=pred.size(1))
                target_binary = target_binary.permute(0,3,1,2).float()
            else:
                target_binary = target.float()
            # 只取前景
            pred = F.softmax(pred, dim=1)[:,1:2,:,:].squeeze(1)
            if target_binary.size(1) > 1:
                target_binary = target_binary[:,1,:,:]

        # flatten
        pred = pred.reshape(B, -1)
        target_binary = target_binary.reshape(B, -1)

        intersection = (pred * target_binary).sum(dim=1)
        union = pred.sum(dim=1) + target_binary.sum(dim=1)
        dice = (2.0*intersection + self.eps)/(union + self.eps)
        return 1.0 - dice.mean()


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数（分类 + 分割）
    """
    def __init__(
        self,
        seg_loss_weight: float = 0.5,
        loss_type: str = 'BCE',
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        pos_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.seg_loss_weight = seg_loss_weight
        self.loss_type = loss_type
        self.pos_weights = pos_weights or {}
        self.label_smoothing = label_smoothing

        self.dice_loss = DiceLoss()
        if loss_type == 'BCE_Focal':
            self.cls_criterion = FocalLossBCE(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        elif loss_type == 'BCE':
            self.cls_criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'CE':
            self.cls_criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss_type {loss_type}")

    def _apply_label_smoothing(self, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return target

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        losses = {}
        total_loss = 0.0

        # 获取device（从任意一个logit获取）
        device = None
        if cls_logits:
            device = next(iter(cls_logits.values())).device
        elif seg_logits:
            device = next(iter(seg_logits.values())).device

        # ---- classification ----
        cls_loss = 0.0
        valid_cls = 0
        for name, logit in cls_logits.items():
            target = cls_targets[name].float()

            # 调试：检查形状和值
            if logit.shape != target.shape:
                logit = logit.view_as(target)

            # 检查target是否在有效范围内（BCE需要0-1）
            if target.min() < 0 or target.max() > 1:
                print(f"[WARNING] {name} target超出[0,1]范围: min={target.min()}, max={target.max()}")
                target = target.clamp(0, 1)

            # 检查是否有NaN
            if torch.isnan(logit).any():
                print(f"[WARNING] {name} logit包含NaN")
                continue

            loss = self.cls_criterion(logit, target)

            # 应用类别权重
            if name in self.pos_weights:
                loss = loss * self.pos_weights[name]

            if not torch.isnan(loss):
                cls_loss += loss
                valid_cls += 1
            else:
                print(f"[WARNING] {name} 分类损失为NaN")

        if valid_cls > 0:
            cls_loss = cls_loss / valid_cls
        else:
            cls_loss = torch.tensor(0.0, device=device)
        losses['cls_loss'] = cls_loss
        total_loss += cls_loss

        # ---- segmentation ----
        seg_loss = 0.0
        valid_seg = 0
        for organ, logit in seg_logits.items():
            if organ not in seg_targets:
                print(f"[WARNING] seg_targets中没有键'{organ}'，可用键: {list(seg_targets.keys())}")
                continue

            target = seg_targets[organ]

            # 全负样本跳过
            if target.sum() == 0:
                continue

            # 检查是否有NaN
            if torch.isnan(logit).any():
                print(f"[WARNING] {organ} seg_logit包含NaN")
                continue

            loss = self.dice_loss(logit, target)
            if not torch.isnan(loss):
                seg_loss += loss
                valid_seg += 1
            else:
                print(f"[WARNING] {organ} Dice损失为NaN")

        if valid_seg > 0:
            seg_loss = seg_loss / valid_seg
            losses['seg_loss'] = seg_loss * self.seg_loss_weight
            total_loss += seg_loss * self.seg_loss_weight
        else:
            losses['seg_loss'] = torch.tensor(0.0, device=device)

        # 确保total_loss是tensor且有梯度
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        elif not total_loss.requires_grad:
            # 如果total_loss没有梯度（所有损失都是NaN被跳过），创建一个需要梯度的零张量
            total_loss = total_loss + torch.zeros(1, device=device, requires_grad=True).squeeze()

        losses['total_loss'] = total_loss
        return losses
