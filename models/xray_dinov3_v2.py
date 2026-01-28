"""
DINOv3 2D X-ray 多任务模型（分类 + 分割）

特点：
1. 纯2D输入：单张X-ray灰度图 → 自动repeat到3通道以适配DINOv3
2. 分割引导的特征聚合 - 用分割预测作为注意力权重
3. Top-K Patch聚合 - 针对小病灶/散发病灶
4. Attention-based分类头 - 任务特定query + 交叉注意力
5. task_names（分类任务）和 organ_names（分割目标）均可通过参数配置
"""

import os
import sys
sys.path.insert(0, "/data/truenas_B2/yyi/dinov3_pretrain")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from omegaconf import OmegaConf
from dinov3.models import build_model_from_cfg

# -----------------------------
# Backbone 构建函数
# -----------------------------
def build_dinov3_backbone(cfg_path: str, device: str = "cuda:0"):
    cfg = OmegaConf.load(cfg_path)
    backbone = build_model_from_cfg(cfg, only_teacher=True)
    backbone.eval()
    backbone.to(device)
    embed_dim = getattr(backbone, "embed_dim", 1024)
    return backbone, embed_dim

# -----------------------------
# 多层特征聚合
# -----------------------------
class MultiLayerFeatureAggregator(nn.Module):
    def __init__(self, embed_dim: int = 768, use_n_blocks: int = 4, use_patch_avg: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_n_blocks = use_n_blocks
        self.use_patch_avg = use_patch_avg
        multiplier = use_n_blocks * (2 if use_patch_avg else 1)
        self.output_dim = embed_dim * multiplier

    def forward(self, layer_outputs: List):
        outputs = layer_outputs[-self.use_n_blocks:]
        cls_tokens, patch_avgs = [], []

        for patch_tokens, cls_token in outputs:
            if cls_token.dim() == 3:
                cls_token = cls_token.squeeze(1)
            cls_tokens.append(cls_token)
            if self.use_patch_avg:
                patch_avgs.append(patch_tokens.mean(dim=1))

        cls_features = torch.cat(cls_tokens, dim=-1)
        if self.use_patch_avg:
            patch_avg_features = torch.cat(patch_avgs, dim=-1)
            cls_features = torch.cat([cls_features, patch_avg_features], dim=-1)

        last_patch_tokens = outputs[-1][0]
        return cls_features.float(), last_patch_tokens.float()

# -----------------------------
# Segmentation-guided Aggregator
# -----------------------------
class SegmentationGuidedAggregator(nn.Module):
    def __init__(self, embed_dim: int = 768, num_organs: int = 4, top_k_ratio: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_organs = num_organs
        self.top_k_ratio = top_k_ratio

        self.seg_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_organs),
        )

        self.organ_transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
            for _ in range(num_organs)
        ])
        self.global_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * (num_organs + 1), embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )

    def forward(self, patch_tokens: torch.Tensor, return_seg_logits: bool = True):
        B, N, D = patch_tokens.shape
        top_k = max(1, int(N * self.top_k_ratio))

        seg_logits = self.seg_predictor(patch_tokens)
        seg_probs = torch.sigmoid(seg_logits)

        organ_features = []
        for organ_idx in range(self.num_organs):
            organ_prob = seg_probs[:, :, organ_idx]
            topk_values, topk_indices = torch.topk(organ_prob, top_k, dim=1)
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            topk_features = torch.gather(patch_tokens, 1, topk_indices_expanded)
            topk_weights = F.softmax(topk_values, dim=1).unsqueeze(-1)
            weighted_feat = (topk_features * topk_weights).sum(dim=1)
            organ_features.append(self.organ_transforms[organ_idx](weighted_feat))

        global_feat = self.global_transform(patch_tokens.mean(dim=1))
        all_features = organ_features + [global_feat]
        concat_features = torch.cat(all_features, dim=-1)
        aggregated_features = self.fusion(concat_features)

        if return_seg_logits:
            return aggregated_features, seg_logits
        return aggregated_features, None

# -----------------------------
# Top-K Patch Selector
# -----------------------------
class TopKPatchSelector(nn.Module):
    def __init__(self, embed_dim: int = 768, num_scales: int = 3, scale_ratios: List[float] = [0.05, 0.1, 0.2]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.scale_ratios = scale_ratios

        self.importance_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )
        self.scale_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, patch_tokens: torch.Tensor):
        B, N, D = patch_tokens.shape
        importance_scores = self.importance_scorer(patch_tokens).squeeze(-1)

        scale_features = []
        for ratio in self.scale_ratios:
            top_k = max(1, int(N * ratio))
            topk_values, topk_indices = torch.topk(importance_scores, top_k, dim=1)
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            topk_features = torch.gather(patch_tokens, 1, topk_indices_expanded)
            topk_weights = F.softmax(topk_values, dim=1).unsqueeze(-1)
            weighted_feat = (topk_features * topk_weights).sum(dim=1)
            scale_features.append(weighted_feat)

        multi_scale_features = self.scale_fusion(torch.cat(scale_features, dim=-1))
        return multi_scale_features

# -----------------------------
# Attention-based Classification Head
# -----------------------------
class AttentionClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, task_names: List[str], num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.task_names = task_names
        self.task_queries = nn.Parameter(torch.randn(len(task_names), embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifiers = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(embed_dim, embed_dim//2), nn.GELU(), nn.Linear(embed_dim//2, 1))
            for name in task_names
        })

    def forward(self, features: torch.Tensor):
        B = features.shape[0]
        queries = self.task_queries.unsqueeze(0).expand(B, -1, -1)
        task_features, _ = self.cross_attn(queries, features.unsqueeze(1), features.unsqueeze(1))
        task_features = self.norm(task_features + queries)
        return {name: self.classifiers[name](task_features[:, i]).squeeze(-1) for i, name in enumerate(self.task_names)}

# -----------------------------
# Patch-based Segmentation Head (2D)
# -----------------------------
class PatchBasedSegmentationHead(nn.Module):
    def __init__(self, embed_dim: int = 768, patch_size: int = 16, img_size: int = 512, organ_names: List[str] = ["mask"], num_classes: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = img_size // patch_size
        self.organ_names = organ_names
        self.seg_heads = nn.ModuleDict({
            organ: nn.Sequential(nn.Conv2d(embed_dim, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, num_classes, 1))
            for organ in organ_names
        })

    def forward(self, patch_tokens: torch.Tensor):
        B = patch_tokens.shape[0]
        x = patch_tokens.transpose(1, 2).reshape(B, -1, self.num_patches, self.num_patches)
        return {organ: self.seg_heads[organ](x) for organ in self.organ_names}

# -----------------------------
# 主模型 (2D X-ray, 支持 mask + classification)
# -----------------------------
class TraumaNetDINOv3(nn.Module):
    """
    DINOv3 2D多任务模型

    Args:
        cfg_path: DINOv3 backbone配置文件路径
        pretrained_path: DINOv3预训练权重路径（保留兼容性，当前未使用）
        img_size: 输入图像尺寸
        use_n_blocks: 使用backbone最后n层的特征
        top_k_ratio: Top-K patch选择比例
        dropout: Dropout比率
        freeze_backbone: 是否冻结backbone参数
        task_names: 分类任务名称列表，如 ['femur_fracture', 'pelvis_fracture']
        organ_names: 分割目标名称列表，如 ['bone'] 或 ['liver', 'spleen', ...]
        input_channels: 输入图像通道数（1=灰度，3=RGB）
    """
    def __init__(
        self,
        cfg_path: str,
        pretrained_path: str = None,
        img_size: int = 512,
        use_n_blocks: int = 4,
        top_k_ratio: float = 0.2,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        task_names: Optional[List[str]] = None,
        organ_names: Optional[List[str]] = None,
        input_channels: int = 1,
    ):
        super().__init__()

        # 分类任务名称（可配置）
        if task_names is None:
            task_names = [
                'liver_injury', 'liver_high_risk',
                'spleen_injury', 'spleen_high_risk',
                'kidney_injury', 'kidney_high_risk',
                'bowel', 'extravasation'
            ]
        self.task_names = task_names

        # 分割目标名称（可配置，独立于分类任务）
        if organ_names is None:
            organ_names = ['mask']
        self.organ_names = organ_names

        self.input_channels = input_channels

        # Backbone
        self.backbone, self.embed_dim = build_dinov3_backbone(cfg_path)
        self.use_n_blocks = use_n_blocks

        # Feature Aggregator
        self.feature_aggregator = MultiLayerFeatureAggregator(embed_dim=self.embed_dim, use_n_blocks=use_n_blocks)
        self.dim_reduction = nn.Sequential(nn.Linear(self.feature_aggregator.output_dim, self.embed_dim),
                                           nn.LayerNorm(self.embed_dim),
                                           nn.GELU(),
                                           nn.Dropout(dropout))

        # Seg-guided aggregator 使用分割目标数量
        num_seg_organs = len(self.organ_names)
        self.seg_guided_aggregator = SegmentationGuidedAggregator(
            embed_dim=self.embed_dim, num_organs=num_seg_organs, top_k_ratio=top_k_ratio
        )
        self.topk_selector = TopKPatchSelector(embed_dim=self.embed_dim)
        self.feature_fusion = nn.Sequential(nn.Linear(self.embed_dim*3, self.embed_dim),
                                            nn.LayerNorm(self.embed_dim),
                                            nn.GELU(),
                                            nn.Dropout(dropout))

        # Classification head（使用分类任务名称）
        self.classification_head = AttentionClassificationHead(embed_dim=self.embed_dim, task_names=self.task_names)
        # Segmentation head（使用分割目标名称）
        self.seg_head = PatchBasedSegmentationHead(embed_dim=self.embed_dim, img_size=img_size, organ_names=self.organ_names)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入图像 (B, C, H, W)
               C=1 时自动repeat到3通道以适配DINOv3 ViT
               C=3 时直接使用
        Returns:
            dict: {
                'cls_logits': {task_name: (B,)},
                'seg_logits': {organ_name: (B, 2, num_patches, num_patches)},
                'aux_seg_logits': (B, N, num_organs)
            }
        """
        # 灰度图自动扩展到3通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 提取多层特征
        layer_outputs = self.backbone.get_intermediate_layers(x, n=self.use_n_blocks, return_class_token=True)
        cls_features, patch_tokens = self.feature_aggregator(layer_outputs)
        cls_features = self.dim_reduction(cls_features)

        seg_features, aux_seg_logits = self.seg_guided_aggregator(patch_tokens)
        topk_features = self.topk_selector(patch_tokens)
        fused_features = self.feature_fusion(torch.cat([cls_features, seg_features, topk_features], dim=-1))

        cls_logits = self.classification_head(fused_features)
        seg_logits = self.seg_head(patch_tokens)

        return {"cls_logits": cls_logits, "seg_logits": seg_logits, "aux_seg_logits": aux_seg_logits}
