"""
DINOv2 Backbone 模块
支持加载预训练的DINOv2 ViT模型
"""

import logging
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Union
from functools import partial
import math

logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """将图像分割为patches并进行embedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head Self Attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP block"""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer Block"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class DINOv2ViT(nn.Module):
    """
    DINOv2 Vision Transformer

    支持从预训练权重加载，提取中间层特征
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def interpolate_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """插值位置编码以适应不同尺寸的输入"""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N:
            return self.pos_embed

        cls_pos = self.pos_embed[:, 0:1]
        patch_pos = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        h = w = int(npatch ** 0.5)
        h0 = w0 = int(N ** 0.5)

        patch_pos = patch_pos.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(
            patch_pos, size=(h, w), mode='bicubic', align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取特征

        Args:
            x: (B, C, H, W)

        Returns:
            cls_token: (B, embed_dim)
            patch_tokens: (B, num_patches, embed_dim)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+num_patches, embed_dim)

        # Add position embedding
        x = x + self.interpolate_pos_embed(x)
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        return cls_token, patch_tokens

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 4,
        return_class_token: bool = True,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取最后n个block的中间层特征

        Args:
            x: (B, C, H, W)
            n: 返回最后n个block的特征
            return_class_token: 是否返回CLS token

        Returns:
            List of (patch_tokens, cls_token) for each block
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.interpolate_pos_embed(x)
        x = self.pos_drop(x)

        # Transformer blocks
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i >= self.depth - n:
                cls_token = x[:, 0]
                patch_tokens = x[:, 1:]
                outputs.append((patch_tokens, cls_token))

        # Apply final norm to last output
        if outputs:
            last_patch, last_cls = outputs[-1]
            outputs[-1] = (self.norm(last_patch), self.norm(last_cls.unsqueeze(1)).squeeze(1))

        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """默认forward返回CLS token"""
        cls_token, _ = self.forward_features(x)
        return cls_token


# =============================================================================
# 工厂函数
# =============================================================================

def dinov2_vit_small(img_size: int = 512, **kwargs) -> DINOv2ViT:
    """DINOv2 ViT-Small: embed_dim=384, depth=12, num_heads=6"""
    return DINOv2ViT(
        img_size=img_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        **kwargs
    )


def dinov2_vit_base(img_size: int = 512, **kwargs) -> DINOv2ViT:
    """DINOv2 ViT-Base: embed_dim=768, depth=12, num_heads=12"""
    return DINOv2ViT(
        img_size=img_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        **kwargs
    )


def dinov2_vit_large(img_size: int = 512, **kwargs) -> DINOv2ViT:
    """DINOv2 ViT-Large: embed_dim=1024, depth=24, num_heads=16"""
    return DINOv2ViT(
        img_size=img_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        **kwargs
    )


def load_dinov2_pretrained(
    model: DINOv2ViT,
    pretrained_path: Optional[str] = None,
    strict: bool = False,
) -> DINOv2ViT:
    """
    加载DINOv2预训练权重

    Args:
        model: DINOv2ViT模型
        pretrained_path: 预训练权重路径
        strict: 是否严格匹配

    Returns:
        加载权重后的模型
    """
    if pretrained_path is None or pretrained_path == "":
        logger.info("[DINOv2] No pretrained weights loaded")
        return model

    logger.info(f"[DINOv2] Loading pretrained weights from {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    # 处理不同格式的checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 过滤不匹配的键
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_state[k] = v
        else:
            logger.warning(f"[DINOv2] Skipping {k}: shape mismatch or not found")

    # 加载权重
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    logger.info(f"[DINOv2] Loaded {len(filtered_state)} parameters")
    if missing:
        logger.warning(f"[DINOv2] Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"[DINOv2] Unexpected keys: {len(unexpected)}")

    return model


def get_dinov2_backbone(
    arch: str = 'vit_base',
    img_size: int = 512,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> Tuple[DINOv2ViT, int]:
    """
    获取DINOv2 backbone

    Args:
        arch: 架构类型 ('vit_small', 'vit_base', 'vit_large')
        img_size: 输入图像尺寸
        pretrained_path: 预训练权重路径

    Returns:
        model: DINOv2ViT模型
        embed_dim: 特征维度
    """
    arch_dict = {
        'vit_small': (dinov2_vit_small, 384),
        'vit_base': (dinov2_vit_base, 768),
        'vit_large': (dinov2_vit_large, 1024),
    }

    if arch not in arch_dict:
        raise ValueError(f"Unknown arch: {arch}. Choose from {list(arch_dict.keys())}")

    model_fn, embed_dim = arch_dict[arch]
    model = model_fn(img_size=img_size, **kwargs)

    if pretrained_path:
        model = load_dinov2_pretrained(model, pretrained_path)

    return model, embed_dim
