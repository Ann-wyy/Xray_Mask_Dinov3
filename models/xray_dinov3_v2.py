import os
import sys
sys.path.insert(0, '/data/truenas_B2/yyi/dinov3_pretrain')
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from omegaconf import OmegaConf
from dinov3.models import build_model_from_cfg

# -----------------------------
# Backbone 构建函数
# -----------------------------
def build_dinov3_backbone(cfg_path: str, device: str = "cuda:0"):
    cfg = OmegaConf.load(cfg_path)
    backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    backbone.to_empty(device=device)
    backbone.eval()
    backbone.to(device)
    return backbone, embed_dim

# -----------------------------
# MultiLayer Feature Aggregator
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
    def __init__(self, embed_dim: int = 768, seg_organ_names: List[str] = None, top_k_ratio: float = 0.2, img_size: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.seg_organ_names = seg_organ_names or ["mask"]
        self.top_k_ratio = top_k_ratio
        self.num_organs = len(self.seg_organ_names)
        self.img_size = img_size

        self.seg_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.num_organs),
        )

        self.organ_transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
            for _ in range(self.num_organs)
        ])
        self.global_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * (self.num_organs + 1), embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
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
            # 将 patch logits -> 原图大小
            H = W = int(N ** 0.5)  # patch 数 -> H,W
            seg_logits_upsampled = F.interpolate(seg_logits.permute(0,2,1).reshape(B, self.num_organs, H, W),
                                                size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            return aggregated_features, seg_logits_upsampled

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
        # cross-attention: queries=task_queries, key/value=features
        task_features, _ = self.cross_attn(queries, features.unsqueeze(1), features.unsqueeze(1))
        task_features = self.norm(task_features + queries)
        return {name: self.classifiers[name](task_features[:, i]).squeeze(-1) for i, name in enumerate(self.task_names)}

# -----------------------------
# Patch-based Segmentation Head
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
        B, N, D = patch_tokens.shape
        H_patch = W_patch = int(N**0.5)
        x = patch_tokens.transpose(1, 2).reshape(B, D, H_patch, W_patch)  # patch-level reshape
        return {organ: self.seg_heads[organ](x) for organ in self.organ_names}  # 不再 F.interpolate



# -----------------------------
# 主模型
# -----------------------------
class TraumaNetDINOv3(nn.Module):
    def __init__(self, cfg_path: str, pretrained_path: str = None, task_names: List[str] = None,
                 seg_organ_names: List[str] = None, img_size: int = 512, use_n_blocks: int = 4,
                 top_k_ratio: float = 0.2, dropout: float = 0.1, freeze_backbone: bool = True):
        super().__init__()
        self.task_names = task_names 
        self.seg_organ_names = seg_organ_names or self.task_names

        # Backbone
        self.backbone, self.embed_dim = build_dinov3_backbone(cfg_path)
        self.use_n_blocks = use_n_blocks

        # Feature Aggregator
        self.feature_aggregator = MultiLayerFeatureAggregator(embed_dim=self.embed_dim, use_n_blocks=use_n_blocks)
        self.dim_reduction = nn.Sequential(nn.Linear(self.feature_aggregator.output_dim, self.embed_dim),
                                           nn.LayerNorm(self.embed_dim),
                                           nn.GELU(),
                                           nn.Dropout(dropout))

        # Seg-guided + Top-K + Fusion
        self.seg_guided_aggregator = SegmentationGuidedAggregator(embed_dim=self.embed_dim, seg_organ_names=self.seg_organ_names, top_k_ratio=top_k_ratio,img_size=img_size)
        self.topk_selector = TopKPatchSelector(embed_dim=self.embed_dim)
        self.feature_fusion = nn.Sequential(nn.Linear(self.embed_dim*3, self.embed_dim),
                                            nn.LayerNorm(self.embed_dim),
                                            nn.GELU(),
                                            nn.Dropout(dropout))

        # Classification & Segmentation heads
        self.classification_head = AttentionClassificationHead(embed_dim=self.embed_dim, task_names=self.task_names)
        self.seg_head = PatchBasedSegmentationHead(embed_dim=self.embed_dim, img_size=img_size, organ_names=self.seg_organ_names)

        # Freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        layer_outputs = self.backbone.get_intermediate_layers(x, n=self.use_n_blocks, return_class_token=True)
        cls_features, patch_tokens = self.feature_aggregator(layer_outputs)
        cls_features = self.dim_reduction(cls_features)

        seg_features, aux_seg_logits = self.seg_guided_aggregator(patch_tokens)
        topk_features = self.topk_selector(patch_tokens)
        fused_features = self.feature_fusion(torch.cat([cls_features, seg_features, topk_features], dim=-1))

        cls_logits = self.classification_head(fused_features)
        seg_logits = self.seg_head(patch_tokens)

        return {"cls_logits": cls_logits, "seg_logits": seg_logits}
