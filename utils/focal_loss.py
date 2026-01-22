"""
Focal Loss实现
用于处理类别不平衡问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossBCE(nn.Module):
    """
    Focal Loss for Binary Cross Entropy

    用于二分类任务的Focal Loss，可以处理类别不平衡问题
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss BCE

        Args:
            alpha: 平衡因子，用于平衡正负样本
            gamma: 聚焦参数，用于降低易分类样本的权重
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        super(FocalLossBCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss

        Args:
            inputs: 模型输出的logits (batch_size, *)
            targets: 二分类标签 (batch_size, *), 值为0或1

        Returns:
            Focal loss值
        """
        # 计算BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算pt = exp(-BCE_loss)
        pt = torch.exp(-BCE_loss)

        # 计算Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class FocalLossCE(nn.Module):
    """
    Focal Loss for Cross Entropy

    用于多分类任务的Focal Loss，可以处理类别不平衡问题
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss CE

        Args:
            alpha: 平衡因子，用于平衡正负样本
            gamma: 聚焦参数，用于降低易分类样本的权重
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        super(FocalLossCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss

        Args:
            inputs: 模型输出的logits (batch_size, num_classes)
            targets: 类别标签 (batch_size,)

        Returns:
            Focal loss值
        """
        # 计算log softmax
        logpt = F.log_softmax(inputs, dim=1)

        # 获取目标类别的log概率
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 计算pt
        pt = logpt.exp()

        # 计算Focal Loss
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    # 测试Focal Loss
    print("测试Focal Loss模块")

    # 测试FocalLossBCE
    print("\n1. 测试FocalLossBCE")
    focal_bce = FocalLossBCE(alpha=0.25, gamma=2.0)
    inputs = torch.randn(4, 10)
    targets = torch.randint(0, 2, (4, 10)).float()
    loss = focal_bce(inputs, targets)
    print(f"   FocalLossBCE: {loss.item():.4f}")

    # 测试FocalLossCE
    print("\n2. 测试FocalLossCE")
    focal_ce = FocalLossCE(alpha=0.25, gamma=2.0)
    inputs = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))
    loss = focal_ce(inputs, targets)
    print(f"   FocalLossCE: {loss.item():.4f}")
