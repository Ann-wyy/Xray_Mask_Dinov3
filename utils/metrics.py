"""
评估指标模块
包含分类和分割任务的评估指标
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score, roc_curve
)


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    计算Dice系数

    Args:
        pred: 预测logits (B, C, D, H, W) 或 (B, C, H, W)
        target: 目标标签 (B, D, H, W) 或 (B, H, W)
        smooth: 平滑因子

    Returns:
        Dice系数 (0-1之间)
    """
    # 应用softmax获取概率
    pred = torch.softmax(pred, dim=1)

    # 获取前景类别（假设类别1是前景）
    if pred.size(1) == 2:
        pred = pred[:, 1:, ...]  # (B, 1, ...)
        target_binary = (target > 0).float().unsqueeze(1)  # (B, 1, ...)
    else:
        # 多类别情况
        pred = torch.argmax(pred, dim=1, keepdim=True).float()
        target_binary = target.unsqueeze(1).float()

    # 展平
    pred = pred.contiguous().view(-1)
    target_binary = target_binary.contiguous().view(-1)

    # 计算Dice系数
    intersection = (pred * target_binary).sum()
    union = pred.sum() + target_binary.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.item()


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    计算分类指标

    Args:
        predictions: 预测类别 (N,)
        targets: 目标类别 (N,)
        num_classes: 类别数

    Returns:
        指标字典
    """
    metrics = {}

    # 准确率
    metrics['accuracy'] = accuracy_score(targets, predictions)

    # F1分数
    if num_classes == 2:
        # 二分类
        metrics['f1'] = f1_score(targets, predictions, average='binary', zero_division=0)
        metrics['precision'] = precision_score(targets, predictions, average='binary', zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, average='binary', zero_division=0)
    else:
        # 多分类
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)

    return metrics


def compute_advanced_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    计算高级分类指标（AUROC、AUPRC、最佳阈值下的F1等）

    Args:
        probabilities: 预测概率 (N,) - 正类的概率
        targets: 目标类别 (N,) - 0或1

    Returns:
        指标字典，包含AUROC、AUPRC、最佳阈值、F1、Precision、Recall、Specificity
    """
    metrics = {}

    try:
        # 计算AUROC
        auroc = roc_auc_score(targets, probabilities)
        metrics['auroc'] = auroc

        # 计算AUPRC (Average Precision)
        auprc = average_precision_score(targets, probabilities)
        metrics['auprc'] = auprc

        # 找到最佳阈值（使F1最大）
        fpr, tpr, thresholds = roc_curve(targets, probabilities)

        best_f1 = 0.0
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0
        best_specificity = 0.0

        for threshold in thresholds:
            preds = (probabilities >= threshold).astype(int)

            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()

            # 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
                best_specificity = specificity

        metrics['best_threshold'] = best_threshold
        metrics['best_f1'] = best_f1
        metrics['best_precision'] = best_precision
        metrics['best_recall'] = best_recall
        metrics['best_specificity'] = best_specificity
        metrics['sensitivity'] = best_recall  # 敏感性就是Recall

    except Exception as e:
        # 如果计算失败（例如只有一个类别），返回默认值
        metrics['auroc'] = 0.0
        metrics['auprc'] = 0.0
        metrics['best_threshold'] = 0.5
        metrics['best_f1'] = 0.0
        metrics['best_precision'] = 0.0
        metrics['best_recall'] = 0.0
        metrics['best_specificity'] = 0.0
        metrics['sensitivity'] = 0.0

    return metrics


class MultiTaskMetrics:
    """
    多任务评估指标类

    用于跟踪和计算多器官分类和分割的评估指标
    """

    def __init__(self, organ_names: List[str], num_classes_dict: Dict[str, int], seg_organ_names: List[str] = None):
        """
        初始化多任务评估指标

        Args:
            organ_names: 分类任务名称列表
            num_classes_dict: 各分类任务的类别数字典
            seg_organ_names: 分割目标名称列表（可独立于分类任务配置）
        """
        self.organ_names = organ_names
        self.num_classes_dict = num_classes_dict
        self.seg_organ_names = seg_organ_names if seg_organ_names is not None else ['mask']

        # 存储预测和目标
        self.cls_predictions = {organ: [] for organ in organ_names}
        self.cls_probabilities = {organ: [] for organ in organ_names}  # 存储预测概率用于AUROC/AUPRC
        self.cls_targets = {organ: [] for organ in organ_names}
        # 分割任务使用配置的器官名称
        self.seg_dice_scores = {organ: [] for organ in self.seg_organ_names}

    def update(
        self,
        cls_logits: Dict[str, torch.Tensor],
        seg_logits: Dict[str, torch.Tensor],
        cls_targets: Dict[str, torch.Tensor],
        seg_targets: Dict[str, torch.Tensor]
    ):
        """
        更新评估指标

        Args:
            cls_logits: 分类预测字典，支持(B,)或(B, 2)格式
            seg_logits: 分割预测字典
            cls_targets: 分类目标字典
            seg_targets: 分割目标字典
        """
        # 更新分类预测和目标（全部二分类BCE）
        for organ in self.organ_names:
            if organ in cls_logits:
                logits = cls_logits[organ]

                # BCE格式: (B,) 或 (B, 1) - 单个logit值
                if logits.dim() == 2:
                    logits = logits.squeeze(1)
                # 预测类别: logit > 0 表示正类
                pred = (logits > 0).long()
                # 预测概率: sigmoid转换
                probs = torch.sigmoid(logits)

                self.cls_predictions[organ].extend(pred.detach().cpu().numpy().tolist())
                self.cls_probabilities[organ].extend(probs.detach().cpu().numpy().tolist())
                # 存储目标
                self.cls_targets[organ].extend(cls_targets[organ].detach().cpu().numpy().tolist())

        # 更新分割Dice分数
        for organ in seg_logits:
            if organ in seg_targets:
                dice = compute_dice_score(seg_logits[organ], seg_targets[organ].squeeze(1))
                self.seg_dice_scores[organ].append(dice)

    def compute(self) -> Dict[str, float]:
        """
        计算所有评估指标

        Returns:
            指标字典
        """
        metrics = {}

        # 计算分类指标
        for organ in self.organ_names:
            if len(self.cls_predictions[organ]) > 0:
                preds = np.array(self.cls_predictions[organ])
                probs = np.array(self.cls_probabilities[organ])
                targets = np.array(self.cls_targets[organ])
                num_classes = self.num_classes_dict[organ]

                # 基本指标
                organ_metrics = compute_classification_metrics(preds, targets, num_classes)
                for metric_name, value in organ_metrics.items():
                    metrics[f'{organ}_{metric_name}'] = value

                # 高级指标（AUROC、AUPRC、最佳阈值下的F1等）
                if num_classes == 2:  # 只对二分类任务计算
                    advanced_metrics = compute_advanced_metrics(probs, targets)
                    for metric_name, value in advanced_metrics.items():
                        metrics[f'{organ}_{metric_name}'] = value

        # 计算分割指标
        for organ in self.seg_dice_scores:
            if len(self.seg_dice_scores[organ]) > 0:
                avg_dice = np.mean(self.seg_dice_scores[organ])
                metrics[f'{organ}_dice'] = avg_dice

        # 计算平均指标
        if len(self.organ_names) > 0:
            # 平均准确率
            acc_values = [metrics[f'{organ}_accuracy'] for organ in self.organ_names
                         if f'{organ}_accuracy' in metrics]
            if acc_values:
                metrics['avg_accuracy'] = np.mean(acc_values)

            # 平均F1分数
            f1_values = []
            for organ in self.organ_names:
                if f'{organ}_f1' in metrics:
                    f1_values.append(metrics[f'{organ}_f1'])
                elif f'{organ}_f1_macro' in metrics:
                    f1_values.append(metrics[f'{organ}_f1_macro'])
            if f1_values:
                metrics['avg_f1'] = np.mean(f1_values)

            # 平均Dice分数
            dice_values = [metrics[f'{organ}_dice'] for organ in self.seg_dice_scores
                          if f'{organ}_dice' in metrics]
            if dice_values:
                metrics['avg_dice'] = np.mean(dice_values)

        return metrics

    def reset(self):
        """重置所有指标"""
        for organ in self.organ_names:
            self.cls_predictions[organ] = []
            self.cls_probabilities[organ] = []
            self.cls_targets[organ] = []
        for organ in self.seg_organ_names:
            self.seg_dice_scores[organ] = []


if __name__ == "__main__":
    # 测试评估指标
    print("测试评估指标模块")

    # 测试Dice分数
    print("\n1. 测试Dice分数")
    pred = torch.randn(2, 2, 54, 512, 512)
    target = torch.randint(0, 2, (2, 54, 512, 512))
    dice = compute_dice_score(pred, target)
    print(f"   Dice分数: {dice:.4f}")

    # 测试分类指标
    print("\n2. 测试分类指标")
    predictions = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    targets = np.array([0, 1, 2, 2, 0, 2, 1, 1])
    metrics = compute_classification_metrics(predictions, targets, num_classes=3)
    print("   分类指标:")
    for name, value in metrics.items():
        print(f"     {name}: {value:.4f}")

    # 测试多任务评估
    print("\n3. 测试多任务评估")
    organ_names = ['liver', 'spleen', 'kidney', 'bowel', 'extravasation']
    num_classes_dict = {'liver': 3, 'spleen': 3, 'kidney': 3, 'bowel': 2, 'extravasation': 2}

    multi_metrics = MultiTaskMetrics(organ_names, num_classes_dict)

    # 模拟几个batch的预测
    for _ in range(3):
        cls_logits = {
            'liver': torch.randn(2, 3),
            'spleen': torch.randn(2, 3),
            'kidney': torch.randn(2, 3),
            'bowel': torch.randn(2, 2),
            'extravasation': torch.randn(2, 2)
        }

        seg_logits = {
            'liver': torch.randn(2, 2, 54, 512, 512),
            'spleen': torch.randn(2, 2, 54, 512, 512),
            'kidney': torch.randn(2, 2, 54, 512, 512),
            'bowel': torch.randn(2, 2, 54, 512, 512)
        }

        cls_targets = {
            'liver': torch.randint(0, 3, (2,)),
            'spleen': torch.randint(0, 3, (2,)),
            'kidney': torch.randint(0, 3, (2,)),
            'bowel': torch.randint(0, 2, (2,)),
            'extravasation': torch.randint(0, 2, (2,))
        }

        seg_targets = {
            'liver': torch.randint(0, 2, (2, 1, 54, 512, 512)),
            'spleen': torch.randint(0, 2, (2, 1, 54, 512, 512)),
            'kidney': torch.randint(0, 2, (2, 1, 54, 512, 512)),
            'bowel': torch.randint(0, 2, (2, 1, 54, 512, 512))
        }

        multi_metrics.update(cls_logits, seg_logits, cls_targets, seg_targets)

    final_metrics = multi_metrics.compute()
    print("   多任务评估指标:")
    for name, value in final_metrics.items():
        print(f"     {name}: {value:.4f}")

