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
        for name, logit in cls_logits.items():
            cls_loss += self.cls_criterion(logit, cls_targets[name].float())
        cls_loss = cls_loss / max(1, len(cls_logits))
        losses['cls_loss'] = cls_loss
        total_loss += cls_loss

        # ---- segmentation ----
        seg_loss = 0.0
        valid_seg = 0
        for organ, logit in seg_logits.items():
            target = seg_targets[organ]

            # 全负样本跳过
            if target.sum() == 0:
                continue

            seg_loss += self.dice_loss(logit, target)
            valid_seg += 1

        if valid_seg > 0:
            seg_loss = seg_loss / valid_seg
            losses['seg_loss'] = seg_loss * self.seg_loss_weight
            total_loss += seg_loss * self.seg_loss_weight
        else:
            losses['seg_loss'] = torch.tensor(0.0, device=device)

        # 确保total_loss是tensor
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, device=device)

        losses['total_loss'] = total_loss
        return losses
