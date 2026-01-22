"""
TraumaNet DINOv3 V2 - 改进版模型架构

改进点：
1. 分割引导的特征聚合 - 用分割预测作为注意力权重
2. Top-K Patch聚合 - 针对小病灶/散发病灶
3. 层次化Slice Transformer (4层) - 局部+全局注意力
4. Attention-based分类头 - 任务特定query + 交叉注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from torch.utils.checkpoint import checkpoint

from .dinov3_backbone import get_dinov3_backbone, DinoVisionTransformer


class MultiLayerFeatureAggregator(nn.Module):
    """
    多层特征聚合模块 (保持不变)
    从ViT的多个中间层提取特征并聚合
    """

    def __init__(
        self,
        embed_dim: int = 768,
        use_n_blocks: int = 4,
        use_patch_avg: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_n_blocks = use_n_blocks
        self.use_patch_avg = use_patch_avg

        multiplier = use_n_blocks * (2 if use_patch_avg else 1)
        self.output_dim = embed_dim * multiplier

    def forward(
        self,
        layer_outputs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = layer_outputs[-self.use_n_blocks:]

        cls_tokens = []
        patch_avgs = []

        for patch_tokens, cls_token in outputs:
            if cls_token.dim() == 3:
                cls_token = cls_token.squeeze(1)
            cls_tokens.append(cls_token)

            if self.use_patch_avg:
                patch_avg = patch_tokens.mean(dim=1)
                patch_avgs.append(patch_avg)

        cls_features = torch.cat(cls_tokens, dim=-1)
        if self.use_patch_avg:
            patch_avg_features = torch.cat(patch_avgs, dim=-1)
            cls_features = torch.cat([cls_features, patch_avg_features], dim=-1)

        last_patch_tokens = outputs[-1][0]

        return cls_features.float(), last_patch_tokens.float()


# =============================================================================
# 【改进1+2】分割引导 + Top-K 特征聚合模块
# =============================================================================

class SegmentationGuidedAggregator(nn.Module):
    """
    分割引导的特征聚合模块

    核心思想：
    1. 用轻量级分割头预测每个patch的病灶概率
    2. 将分割概率作为注意力权重
    3. Top-K选择高响应patches (针对小病灶/散发病灶)
    4. 加权聚合得到slice特征
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_organs: int = 4,
        top_k_ratio: float = 0.2,  # 选择top 20% 的patches
        temperature: float = 1.0,
        use_learnable_temperature: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_organs = num_organs
        self.top_k_ratio = top_k_ratio

        # 可学习的温度参数
        if use_learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        # 轻量级分割预测头 (用于生成attention map)
        self.seg_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_organs),
        )

        # 器官特定的特征变换
        self.organ_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            for _ in range(num_organs)
        ])

        # 全局特征 (背景/无病灶区域)
        self.global_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * (num_organs + 1), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        return_seg_logits: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patch_tokens: (B, num_patches, embed_dim)
            return_seg_logits: 是否返回分割logits用于辅助损失

        Returns:
            aggregated_features: (B, embed_dim)
            seg_logits: (B, num_patches, num_organs) or None
        """
        B, N, D = patch_tokens.shape
        top_k = max(1, int(N * self.top_k_ratio))

        # 1. 预测每个patch的器官病灶概率
        seg_logits = self.seg_predictor(patch_tokens)  # (B, N, num_organs)
        seg_probs = torch.sigmoid(seg_logits / self.temperature)  # (B, N, num_organs)

        # 2. 对每个器官进行Top-K选择和加权聚合
        organ_features = []
        for organ_idx in range(self.num_organs):
            organ_prob = seg_probs[:, :, organ_idx]  # (B, N)

            # Top-K 选择
            topk_values, topk_indices = torch.topk(organ_prob, top_k, dim=1)

            # 获取Top-K patches的特征
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            topk_features = torch.gather(patch_tokens, 1, topk_indices_expanded)  # (B, top_k, D)

            # Softmax归一化权重
            topk_weights = F.softmax(topk_values, dim=1).unsqueeze(-1)  # (B, top_k, 1)

            # 加权聚合
            weighted_feat = (topk_features * topk_weights).sum(dim=1)  # (B, D)

            # 器官特定变换
            organ_feat = self.organ_transforms[organ_idx](weighted_feat)
            organ_features.append(organ_feat)

        # 3. 全局特征 (捕捉背景信息)
        global_feat = self.global_transform(patch_tokens.mean(dim=1))

        # 4. 融合所有特征
        all_features = organ_features + [global_feat]
        concat_features = torch.cat(all_features, dim=-1)  # (B, embed_dim * (num_organs + 1))
        aggregated_features = self.fusion(concat_features)  # (B, embed_dim)

        if return_seg_logits:
            return aggregated_features, seg_logits
        return aggregated_features, None


class TopKPatchSelector(nn.Module):
    """
    Top-K Patch选择器 (独立模块，可用于可视化)

    针对小病灶和散发病灶的设计：
    - 不依赖全局平均，而是选择最相关的patches
    - 支持多尺度选择
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_scales: int = 3,  # 多尺度: top-5%, top-10%, top-20%
        scale_ratios: List[float] = [0.05, 0.1, 0.2],
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.scale_ratios = scale_ratios

        # 重要性评分网络
        self.importance_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )

        # 多尺度特征融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, num_patches, embed_dim)

        Returns:
            multi_scale_features: (B, embed_dim)
        """
        B, N, D = patch_tokens.shape

        # 计算每个patch的重要性分数
        importance_scores = self.importance_scorer(patch_tokens).squeeze(-1)  # (B, N)

        # 多尺度Top-K聚合
        scale_features = []
        for ratio in self.scale_ratios:
            top_k = max(1, int(N * ratio))
            topk_values, topk_indices = torch.topk(importance_scores, top_k, dim=1)

            # 获取Top-K特征
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            topk_features = torch.gather(patch_tokens, 1, topk_indices_expanded)

            # Softmax加权
            topk_weights = F.softmax(topk_values, dim=1).unsqueeze(-1)
            weighted_feat = (topk_features * topk_weights).sum(dim=1)
            scale_features.append(weighted_feat)

        # 融合多尺度特征
        concat_features = torch.cat(scale_features, dim=-1)
        multi_scale_features = self.scale_fusion(concat_features)

        return multi_scale_features


# =============================================================================
# 【改进3】层次化 Slice Transformer
# =============================================================================

class LocalWindowAttention(nn.Module):
    """
    局部窗口注意力
    只在相邻的slices之间进行注意力计算
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        window_size: int = 7,  # 每个slice关注前后各3个slice
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 相对位置编码
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_slices, embed_dim)
        Returns:
            (B, num_slices, embed_dim)
        """
        B, N, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, N, N)

        # 创建局部窗口mask
        mask = torch.ones(N, N, device=x.device, dtype=torch.bool)
        for i in range(N):
            start = max(0, i - self.window_size // 2)
            end = min(N, i + self.window_size // 2 + 1)
            mask[i, start:end] = False

        # 应用mask (被mask的位置设为-inf)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # 添加相对位置偏置
        # 简化版：只对窗口内的位置添加偏置

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        return out


class HierarchicalSliceTransformer(nn.Module):
    """
    层次化Slice Transformer

    结构：
    - 前2层: 局部窗口注意力 (相邻slices交互)
    - 后2层: 全局注意力 (所有slices交互)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_local_layers: int = 2,
        num_global_layers: int = 2,
        num_heads: int = 8,
        window_size: int = 7,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        max_slices: int = 100,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Z轴位置编码
        self.z_pos_embedding = nn.Parameter(torch.randn(max_slices, embed_dim) * 0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # 局部注意力层
        self.local_layers = nn.ModuleList()
        for _ in range(num_local_layers):
            self.local_layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': LocalWindowAttention(embed_dim, num_heads, window_size, dropout),
                'norm2': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * ffn_ratio),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * ffn_ratio, embed_dim),
                    nn.Dropout(dropout),
                ),
            }))

        # 全局注意力层
        self.global_layers = nn.ModuleList()
        for _ in range(num_global_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * ffn_ratio,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.global_layers.append(layer)

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        num_slices: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, num_slices, embed_dim)
            num_slices: 实际slice数量

        Returns:
            volume_features: (B, embed_dim)
            slice_features: (B, num_slices, embed_dim)
        """
        B = x.shape[0]

        # 添加Z轴位置编码
        x = x + self.z_pos_embedding[:num_slices].unsqueeze(0)

        # 局部注意力层 (不包含CLS token)
        for layer in self.local_layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 全局注意力层
        for layer in self.global_layers:
            x = layer(x)

        x = self.final_norm(x)

        # 分离CLS和slice特征
        volume_features = x[:, 0]
        slice_features = x[:, 1:]

        return volume_features, slice_features


# =============================================================================
# 【改进4】Attention-based 分类头
# =============================================================================

class AttentionClassificationHead(nn.Module):
    """
    基于注意力的分类头

    特点：
    1. 每个任务有独立的可学习query
    2. 通过Cross-attention从特征中提取任务相关信息
    3. 支持任务间信息交互
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_tasks: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        task_names: List[str] = None,
        use_task_interaction: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tasks = num_tasks
        self.num_heads = num_heads
        self.use_task_interaction = use_task_interaction

        if task_names is None:
            task_names = [f'task_{i}' for i in range(num_tasks)]
        self.task_names = task_names

        # 任务特定的可学习query
        self.task_queries = nn.Parameter(torch.randn(num_tasks, embed_dim) * 0.02)

        # Cross-attention: task queries attend to features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # 任务间Self-attention (可选)
        if use_task_interaction:
            self.task_self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.task_norm = nn.LayerNorm(embed_dim)
            self.task_ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout),
            )
            self.task_ffn_norm = nn.LayerNorm(embed_dim)

        # 最终分类层
        self.classifiers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
            )
            for name in task_names
        })

    def forward(
        self,
        features: torch.Tensor,
        slice_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, embed_dim) - volume-level特征
            slice_features: (B, num_slices, embed_dim) - slice-level特征 (可选)

        Returns:
            predictions: Dict[task_name, (B,)]
        """
        B = features.shape[0]

        # 准备key-value: 可以是volume特征，也可以结合slice特征
        if slice_features is not None:
            # 将volume特征和slice特征拼接作为KV
            kv = torch.cat([features.unsqueeze(1), slice_features], dim=1)
        else:
            kv = features.unsqueeze(1)  # (B, 1, embed_dim)

        # 扩展task queries
        queries = self.task_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_tasks, embed_dim)

        # Cross-attention
        task_features, _ = self.cross_attn(queries, kv, kv)
        task_features = self.cross_norm(task_features + queries)

        # 任务间交互
        if self.use_task_interaction:
            task_features_attn, _ = self.task_self_attn(task_features, task_features, task_features)
            task_features = self.task_norm(task_features + task_features_attn)
            task_features = self.task_ffn_norm(task_features + self.task_ffn(task_features))

        # 分类
        predictions = {}
        for i, name in enumerate(self.task_names):
            task_feat = task_features[:, i]  # (B, embed_dim)
            predictions[name] = self.classifiers[name](task_feat).squeeze(-1)  # (B,)

        return predictions


# =============================================================================
# 分割头 (保持不变，但增加辅助输出)
# =============================================================================

class PatchBasedSegmentationHead(nn.Module):
    """基于Patch tokens的分割头"""

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        img_size: int = 512,
        organ_names: List[str] = None,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_side = img_size // patch_size

        if organ_names is None:
            organ_names = ['liver', 'spleen', 'kidney', 'bowel']
        self.organ_names = organ_names

        self.seg_heads = nn.ModuleDict()
        for organ in organ_names:
            self.seg_heads[organ] = nn.Sequential(
                nn.Conv2d(embed_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(32, num_classes, 1),
            )

    def forward(self, patch_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = patch_tokens.shape[0]
        H = W = self.num_patches_per_side

        x = patch_tokens.transpose(1, 2).reshape(B, -1, H, W).contiguous()

        seg_outputs = {}
        for organ in self.organ_names:
            seg_outputs[organ] = self.seg_heads[organ](x)

        return seg_outputs


# =============================================================================
# 主模型: TraumaNetDINOv3V2
# =============================================================================

class TraumaNetDINOv3V2(nn.Module):
    """
    TraumaNet DINOv3 V2 - 改进版模型

    整合所有改进：
    1. 分割引导的特征聚合
    2. Top-K Patch聚合 (小病灶/散发病灶)
    3. 层次化Slice Transformer (4层)
    4. Attention-based分类头
    """

    def __init__(
        self,
        # Backbone配置
        vit_arch: str = 'vit_base',
        pretrained_path: Optional[str] = None,
        img_size: int = 512,
        input_depth: int = 240,
        use_n_blocks: int = 4,
        freeze_backbone: bool = True,
        batch_forward_size: int = 16,
        # 分割引导聚合配置
        top_k_ratio: float = 0.2,
        # Slice Transformer配置
        num_local_layers: int = 2,
        num_global_layers: int = 2,
        num_transformer_heads: int = 8,
        window_size: int = 7,
        # 分类头配置
        use_task_interaction: bool = True,
        use_slice_classification: bool = False,
        dropout: float = 0.1,
        # 分割配置
        seg_organs: Optional[List[str]] = None,
        enable_segmentation: bool = True,  # 新增：是否启用分割功能
    ):
        super().__init__()

        self.img_size = img_size
        self.input_depth = input_depth
        self.num_slices_per_group = 3
        self.num_groups = input_depth // self.num_slices_per_group
        self.freeze_backbone = freeze_backbone
        self.batch_forward_size = batch_forward_size
        self.enable_segmentation = enable_segmentation  # 保存配置

        # 支持单一骨骼分割或多器官分割
        if seg_organs is None:
            seg_organs = ['bone']  # 默认为骨骼分割
        self.seg_organs = seg_organs if enable_segmentation else []

        # 分类任务
        self.task_names = [
            'liver_injury', 'liver_high_risk',
            'spleen_injury', 'spleen_high_risk',
            'kidney_injury', 'kidney_high_risk',
            'bowel', 'extravasation'
        ]

        # 1. DINOv3 backbone
        self.backbone, self.embed_dim = get_dinov3_backbone(
            arch=vit_arch,
            img_size=img_size,
            pretrained_path=pretrained_path,
        )
        self.use_n_blocks = use_n_blocks

        # 2. 多层特征聚合器
        self.feature_aggregator = MultiLayerFeatureAggregator(
            embed_dim=self.embed_dim,
            use_n_blocks=use_n_blocks,
            use_patch_avg=True,
        )

        # 3. 特征降维
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.feature_aggregator.output_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. 【改进1+2】分割引导的特征聚合（仅在启用分割时创建）
        if enable_segmentation and len(self.seg_organs) > 0:
            self.seg_guided_aggregator = SegmentationGuidedAggregator(
                embed_dim=self.embed_dim,
                num_organs=len(self.seg_organs),
                top_k_ratio=top_k_ratio,
            )
        else:
            self.seg_guided_aggregator = None

        # 5. 【改进2】多尺度Top-K选择器 (额外的特征)
        self.topk_selector = TopKPatchSelector(
            embed_dim=self.embed_dim,
            num_scales=3,
            scale_ratios=[0.05, 0.1, 0.2],
        )

        # 6. Slice特征融合 (根据是否启用分割调整输入维度)
        fusion_input_dim = self.embed_dim * 3 if enable_segmentation else self.embed_dim * 2
        self.slice_feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 7. 【改进3】层次化Slice Transformer
        self.slice_transformer = HierarchicalSliceTransformer(
            embed_dim=self.embed_dim,
            num_local_layers=num_local_layers,
            num_global_layers=num_global_layers,
            num_heads=num_transformer_heads,
            window_size=window_size,
            dropout=dropout,
            max_slices=self.num_groups + 10,
        )

        # 8. 【改进4】Attention-based分类头
        self.classification_head = AttentionClassificationHead(
            embed_dim=self.embed_dim,
            num_tasks=len(self.task_names),
            num_heads=num_transformer_heads,
            dropout=dropout,
            task_names=self.task_names,
            use_task_interaction=use_task_interaction,
        )

        # 8.5 【新增】Slice-level分类头（可选）
        self.use_slice_classification = use_slice_classification
        if use_slice_classification:
            # 为每个任务创建slice分类头（不包括extravasation）
            self.slice_cls_heads = nn.ModuleDict({
                'liver_injury': self._make_slice_head(dropout),
                'liver_high_risk': self._make_slice_head(dropout),
                'spleen_injury': self._make_slice_head(dropout),
                'spleen_high_risk': self._make_slice_head(dropout),
                'kidney_injury': self._make_slice_head(dropout),
                'kidney_high_risk': self._make_slice_head(dropout),
                'bowel': self._make_slice_head(dropout),
            })

        # 9. 分割头（仅在启用分割时创建）
        if enable_segmentation and len(self.seg_organs) > 0:
            self.seg_head = PatchBasedSegmentationHead(
                embed_dim=self.embed_dim,
                patch_size=16,
                img_size=img_size,
                organ_names=self.seg_organs,
            )
        else:
            self.seg_head = None

        # 初始化和冻结
        self._init_weights()
        if freeze_backbone:
            self._freeze_backbone()

    def _make_slice_head(self, dropout: float) -> nn.Module:
        """创建slice分类头"""
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, 1)
        )

    def _init_weights(self):
        """初始化新增层的权重"""
        for module in [self.dim_reduction, self.slice_feature_fusion]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    '''
    def _create_25d_input(self, x: torch.Tensor) -> torch.Tensor:
        """将3D影像转换为2.5D输入"""
        B, C, D, H, W = x.shape
        assert C == 1, "输入通道数必须为1"
        assert D == self.input_depth, f"输入深度必须为{self.input_depth}"

        x = x.squeeze(1)
        x = x.view(B, self.num_groups, self.num_slices_per_group, H, W)
        x = x.view(B * self.num_groups, self.num_slices_per_group, H, W)

        return x
   '''

    def _extract_features_chunk(
        self,
        imgs_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从一个chunk提取特征"""
        if self.freeze_backbone:
            with torch.no_grad():
                layer_outputs = self.backbone.get_intermediate_layers(
                    imgs_chunk, n=self.use_n_blocks, return_class_token=True
                )
        else:
            layer_outputs = self.backbone.get_intermediate_layers(
                imgs_chunk, n=self.use_n_blocks, return_class_token=True
            )

        cls_features, patch_tokens = self.feature_aggregator(layer_outputs)
        return cls_features, patch_tokens

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入影像 (B, 1, D, H, W)

        Returns:
            输出字典
        """
        batch_size = x.size(0)

        # 1. 转换为2.5D输入
        x_25d = x  # (B*num_groups, 3, H, W)

        # 使用 backbone 提取特征
        layer_outputs = self.backbone.get_intermediate_layers(x, n=self.use_n_blocks, return_class_token=True)
        cls_features, patch_tokens = self.feature_aggregator(layer_outputs)  # (B, agg_dim), (B, num_patches, embed_dim)

        cls_features_reduced = self.dim_reduction(cls_features)  # (B, embed_dim)

        # 根据是否启用分割来处理特征融合
        if self.enable_segmentation and self.seg_guided_aggregator is not None:
            # 分割引导的特征聚合
            seg_guided_features, aux_seg_logits = self.seg_guided_aggregator(
                patch_tokens, return_seg_logits=True
            )  # (B*num_groups, embed_dim), (B*num_groups, num_patches, num_organs)

            # 多尺度Top-K特征
            topk_features = self.topk_selector(patch_tokens)  # (B*num_groups, embed_dim)

            # 融合特征 (包含分割引导特征)
            volume_features = self.slice_feature_fusion(
                torch.cat([cls_features_reduced, seg_guided_features, topk_features], dim=-1)
            )
        else:
            # 不使用分割：仅融合cls和topk特征
            aux_seg_logits = None
            topk_features = self.topk_selector(patch_tokens)  # (B*num_groups, embed_dim)
            volume_features = self.slice_feature_fusion(
                torch.cat([cls_features_reduced, topk_features], dim=-1)
            )

        # Attention-based分类
        volume_preds = self.classification_head(volume_features, slice_features=None)

        # 分割预测（仅在启用分割时）
        if self.enable_segmentation and self.seg_head is not None:
            seg_preds_flat = self.seg_head(patch_tokens)
            seg_preds = {organ: output for organ, output in seg_preds_flat.items()}
        else:
            seg_preds = {}

        return {
            'cls_logits': volume_preds,
            'seg_logits': seg_preds,
            'aux_seg_logits': aux_seg_logits,
        }

    def train(self, mode: bool = True):
        """训练模式设置"""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def get_optimizer_param_groups(self, lr: float, backbone_lr: float = None):
        """获取优化器参数组"""
        param_groups = []

        trainable_params = (
            list(self.feature_aggregator.parameters()) +
            list(self.dim_reduction.parameters()) +
            list(self.seg_guided_aggregator.parameters()) +
            list(self.topk_selector.parameters()) +
            list(self.slice_feature_fusion.parameters()) +
            list(self.slice_transformer.parameters()) +
            list(self.classification_head.parameters()) +
            list(self.seg_head.parameters())
        )

        # 添加slice分类头参数（如果启用）
        if self.use_slice_classification:
            trainable_params += list(self.slice_cls_heads.parameters())

        param_groups.append({
            "params": trainable_params,
            "lr": lr,
            "weight_decay": 1e-4,
        })

        if not self.freeze_backbone and backbone_lr is not None:
            param_groups.append({
                "params": self.backbone.parameters(),
                "lr": backbone_lr,
                "weight_decay": 5e-4,
            })

        return param_groups
