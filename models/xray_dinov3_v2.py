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
def build_dinov3_backbone(cfg_path: str, pretrained_path: str = None, device: str = "cuda:0"):
    """
    构建DINOv3 backbone并加载预训练权重

    注意：DINOv3使用meta tensors，必须按以下顺序操作：
    1. to_empty() - 将meta tensors移动到实际设备（创建未初始化的空张量）
    2. load_state_dict() - 立即加载预训练权重填充这些空张量
    """
    cfg = OmegaConf.load(cfg_path)
    backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)

    # Step 1: 将meta tensors移动到实际设备
    backbone.to_empty(device=device)

    # Step 2: 立即加载预训练权重（必须在to_empty之后！）
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[INFO] 加载DINOv3预训练权重: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)

        # 处理可能的包装格式
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'teacher' in state_dict:
            state_dict = state_dict['teacher']

        # 调试：打印权重key的前缀
        model_keys = list(backbone.state_dict().keys())
        ckpt_keys = list(state_dict.keys())
        print(f"[DEBUG] 模型权重keys示例: {model_keys[:3]}")
        print(f"[DEBUG] 检查点权重keys示例: {ckpt_keys[:3]}")

        # 尝试移除常见的前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除可能的前缀
            new_key = k
            for prefix in ['backbone.', 'module.', 'encoder.', 'teacher.', 'model.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            new_state_dict[new_key] = v

        # 加载权重
        missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"[WARNING] 缺失的权重: {len(missing)} 个")
            if len(missing) <= 10:
                print(f"[DEBUG] 缺失keys: {missing}")
            else:
                print(f"[DEBUG] 缺失keys示例: {missing[:5]}")
        if unexpected:
            print(f"[WARNING] 多余的权重: {len(unexpected)} 个")
            if len(unexpected) <= 10:
                print(f"[DEBUG] 多余keys: {unexpected}")
            else:
                print(f"[DEBUG] 多余keys示例: {unexpected[:5]}")

        # 如果缺失太多，说明匹配失败
        if len(missing) > 100:
            print(f"[ERROR] 权重匹配失败！缺失{len(missing)}个权重，请检查预训练文件是否正确。")

        print(f"[INFO] DINOv3权重加载完成!")
    else:
        raise ValueError(f"[ERROR] 预训练权重文件不存在: {pretrained_path}，DINOv3必须加载预训练权重!")

    backbone.eval()
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
# 临床特征编码器
# -----------------------------
class ClinicalEncoder(nn.Module):
    """将临床特征编码为与图像特征相同维度的向量"""
    def __init__(self, num_features: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# -----------------------------
# 主模型
# -----------------------------
class TraumaNetDINOv3(nn.Module):
    def __init__(self, cfg_path: str, pretrained_path: str = None, task_names: List[str] = None,
                 seg_organ_names: List[str] = None, img_size: int = 512, use_n_blocks: int = 4,
                 top_k_ratio: float = 0.2, dropout: float = 0.1, freeze_backbone: bool = True,
                 num_clinical_features: int = 0, use_clinical: bool = True):
        """
        Args:
            num_clinical_features: 临床特征数量（如age, sex, bmi等）
            use_clinical: 是否使用临床特征（用于消融实验）
        """
        super().__init__()
        self.task_names = task_names
        self.seg_organ_names = seg_organ_names or self.task_names
        self.num_clinical_features = num_clinical_features
        self.use_clinical = use_clinical and num_clinical_features > 0

        # Backbone（权重在build_dinov3_backbone中加载）
        self.backbone, self.embed_dim = build_dinov3_backbone(cfg_path, pretrained_path=pretrained_path)
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

        # 临床特征编码器（可选）
        if self.use_clinical:
            self.clinical_encoder = ClinicalEncoder(num_clinical_features, self.embed_dim, dropout)
            # 融合层：图像特征(3*embed_dim) + 临床特征(embed_dim)
            self.feature_fusion = nn.Sequential(nn.Linear(self.embed_dim*4, self.embed_dim),
                                                nn.LayerNorm(self.embed_dim),
                                                nn.GELU(),
                                                nn.Dropout(dropout))
            print(f"[INFO] 启用临床特征融合，特征数: {num_clinical_features}")
        else:
            self.clinical_encoder = None
            self.feature_fusion = nn.Sequential(nn.Linear(self.embed_dim*3, self.embed_dim),
                                                nn.LayerNorm(self.embed_dim),
                                                nn.GELU(),
                                                nn.Dropout(dropout))
            print(f"[INFO] 未使用临床特征")

        # Classification & Segmentation heads
        self.classification_head = AttentionClassificationHead(embed_dim=self.embed_dim, task_names=self.task_names)
        self.seg_head = PatchBasedSegmentationHead(embed_dim=self.embed_dim, img_size=img_size, organ_names=self.seg_organ_names)

        # Freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, clinical: torch.Tensor = None):
        """
        Args:
            x: 图像 [B, C, H, W]
            clinical: 临床特征 [B, num_clinical_features]，可选
        """
        layer_outputs = self.backbone.get_intermediate_layers(x, n=self.use_n_blocks, return_class_token=True)
        cls_features, patch_tokens = self.feature_aggregator(layer_outputs)
        cls_features = self.dim_reduction(cls_features)

        seg_features, aux_seg_logits = self.seg_guided_aggregator(patch_tokens)
        topk_features = self.topk_selector(patch_tokens)

        # 融合图像特征和临床特征
        if self.use_clinical:
            # 模型配置为使用临床特征，feature_fusion期望4*embed_dim输入
            if clinical is not None and clinical.numel() > 0:
                clinical_features = self.clinical_encoder(clinical)
            else:
                # 临床数据缺失时，使用零向量代替（保持维度一致）
                B = cls_features.shape[0]
                clinical_features = torch.zeros(B, self.embed_dim, device=cls_features.device, dtype=cls_features.dtype)
            fused_features = self.feature_fusion(torch.cat([cls_features, seg_features, topk_features, clinical_features], dim=-1))
        else:
            # 模型未配置临床特征，feature_fusion期望3*embed_dim输入
            fused_features = self.feature_fusion(torch.cat([cls_features, seg_features, topk_features], dim=-1))

        cls_logits = self.classification_head(fused_features)
        seg_logits = self.seg_head(patch_tokens)

        return {"cls_logits": cls_logits, "seg_logits": seg_logits}
