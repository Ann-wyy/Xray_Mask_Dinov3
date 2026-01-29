import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score, roc_curve

# ------------------- Dice 计算 -------------------
def compute_dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    patch-level Dice
    logits: [B,1,H_patch,W_patch] 或 [B,C,H_patch,W_patch]
    target: [B,H_patch,W_patch] 或 [B,C,H_patch,W_patch]
    """
    B = logits.size(0)

    # 二分类
    if logits.size(1) == 1 or logits.size(1) == 2:
        pred = torch.sigmoid(logits) if logits.size(1) == 1 else F.softmax(logits, dim=1)[:,1:2,:,:]
        pred = pred.squeeze(1)
        target_binary = (target > 0).float()
    else:
        pred = F.softmax(logits, dim=1)[:,1:2,:,:].squeeze(1)
        if target.dim() == 3:
            target_binary = F.one_hot(target, num_classes=logits.size(1))
            target_binary = target_binary.permute(0,3,1,2).float()
            target_binary = target_binary[:,1,:,:]
        else:
            target_binary = target.float()

    # flatten
    pred = pred.reshape(B, -1)
    target_binary = target_binary.reshape(B, -1)

    intersection = (pred * target_binary).sum(dim=1)
    union = pred.sum(dim=1) + target_binary.sum(dim=1)
    dice = (2.0*intersection + eps) / (union + eps)
    return dice.mean()


# ------------------- 分类指标 -------------------
def compute_classification_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> Dict[str,float]:
    metrics = {}
    metrics['accuracy'] = accuracy_score(targets, predictions)
    if num_classes == 2:
        metrics['f1'] = f1_score(targets, predictions, average='binary', zero_division=0)
        metrics['precision'] = precision_score(targets, predictions, average='binary', zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, average='binary', zero_division=0)
    else:
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)
    return metrics


# ------------------- 高级指标 -------------------
def compute_advanced_metrics(probabilities: np.ndarray, targets: np.ndarray) -> Dict[str,float]:
    metrics = {}
    try:
        metrics['auroc'] = roc_auc_score(targets, probabilities)
        metrics['auprc'] = average_precision_score(targets, probabilities)
        fpr, tpr, thresholds = roc_curve(targets, probabilities)
        best_f1 = 0.0
        best_threshold = 0.5
        for threshold in thresholds:
            preds = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0,1]).ravel()
            precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
            recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        metrics['best_threshold'] = best_threshold
        metrics['best_f1'] = best_f1
        metrics['best_precision'] = precision
        metrics['best_recall'] = recall
        metrics['best_specificity'] = tn/(tn+fp) if (tn+fp)>0 else 0.0
        metrics['sensitivity'] = recall
    except Exception:
        metrics.update({'auroc':0.0,'auprc':0.0,'best_threshold':0.5,'best_f1':0.0,'best_precision':0.0,'best_recall':0.0,'best_specificity':0.0,'sensitivity':0.0})
    return metrics


# ------------------- 多任务指标类 -------------------
class MultiTaskMetrics:
    """
    分类 + patch-level 分割
    """
    def __init__(self, organ_names: List[str], num_classes_dict: Dict[str,int], seg_organ_names: List[str]=None):
        self.organ_names = organ_names
        self.num_classes_dict = num_classes_dict
        self.seg_organ_names = seg_organ_names if seg_organ_names else ['mask']

        self.cls_predictions = {organ: [] for organ in organ_names}
        self.cls_probabilities = {organ: [] for organ in organ_names}
        self.cls_targets = {organ: [] for organ in organ_names}

        self.seg_dice_scores = {organ: [] for organ in self.seg_organ_names}

    def update(self, cls_logits: Dict[str,torch.Tensor], seg_logits: Dict[str,torch.Tensor],
               cls_targets: Dict[str,torch.Tensor], seg_targets: Dict[str,torch.Tensor]):

        # ---- 分类 ----
        for organ in self.organ_names:
            if organ in cls_logits:
                logits = cls_logits[organ]
                if logits.dim() == 2 and logits.size(1) > 1:
                    pred = logits.argmax(dim=1)
                    probs = F.softmax(logits, dim=1)[:,1]
                else:
                    pred = (logits>0).long()
                    probs = torch.sigmoid(logits)

                self.cls_predictions[organ].extend(pred.detach().cpu().numpy().tolist())
                self.cls_probabilities[organ].extend(probs.detach().cpu().numpy().tolist())
                self.cls_targets[organ].extend(cls_targets[organ].detach().cpu().numpy().tolist())

        # ---- 分割 (patch-level) ----
        for organ in seg_logits:
            if organ in seg_targets:
                logit = seg_logits[organ]
                target_patch = seg_targets[organ]  # patch-level target
                dice = compute_dice_score(logit, target_patch)
                self.seg_dice_scores[organ].append(dice.item())

    def compute(self) -> Dict[str,float]:
        metrics = {}

        # 分类
        for organ in self.organ_names:
            if len(self.cls_predictions[organ])>0:
                preds = np.array(self.cls_predictions[organ])
                probs = np.array(self.cls_probabilities[organ])
                targets = np.array(self.cls_targets[organ])
                num_classes = self.num_classes_dict[organ]

                organ_metrics = compute_classification_metrics(preds, targets, num_classes)
                metrics.update({f'{organ}_{k}':v for k,v in organ_metrics.items()})

                if num_classes==2:
                    adv_metrics = compute_advanced_metrics(probs, targets)
                    metrics.update({f'{organ}_{k}':v for k,v in adv_metrics.items()})

        # 分割
        for organ in self.seg_dice_scores:
            if self.seg_dice_scores[organ]:
                metrics[f'{organ}_dice'] = np.mean(self.seg_dice_scores[organ])

        # 平均指标
        if self.organ_names:
            acc_values = [metrics[f'{organ}_accuracy'] for organ in self.organ_names if f'{organ}_accuracy' in metrics]
            if acc_values:
                metrics['avg_accuracy'] = np.mean(acc_values)

            f1_values = []
            for organ in self.organ_names:
                if f'{organ}_f1' in metrics:
                    f1_values.append(metrics[f'{organ}_f1'])
                elif f'{organ}_f1_macro' in metrics:
                    f1_values.append(metrics[f'{organ}_f1_macro'])
            if f1_values:
                metrics['avg_f1'] = np.mean(f1_values)

            dice_values = [metrics[f'{organ}_dice'] for organ in self.seg_dice_scores if f'{organ}_dice' in metrics]
            if dice_values:
                metrics['avg_dice'] = np.mean(dice_values)

        return metrics

    def reset(self):
        for organ in self.organ_names:
            self.cls_predictions[organ] = []
            self.cls_probabilities[organ] = []
            self.cls_targets[organ] = []
        for organ in self.seg_organ_names:
            self.seg_dice_scores[organ] = []
