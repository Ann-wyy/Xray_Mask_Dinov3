"""
DINOv3 Backbone 模块
支持加载预训练的DINOv3 ViT模型

DINOv3特点：
- RoPE (Rotary Position Embedding)位置编码
- SwiGLU FFN
- LayerScale
- RMSNorm
"""

import logging
import sys

logger = logging.getLogger(__name__)
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, Callable

import torch
import torch.nn.init
from torch import Tensor, nn

# 添加DINOv3模块路径
DINOV3_PATH = '/data/dataserver01/zhangruipeng/code/PETCT/dinov3_pretrain/dinov3'
if DINOV3_PATH not in sys.path:
    sys.path.insert(0, DINOV3_PATH)

from dinov3.layers import (
    LayerScale, Mlp, PatchEmbed, RMSNorm,
    RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
)


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """递归应用函数到模块"""
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    """初始化ViT权重"""
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    """
    DINOv3 Vision Transformer

    特点：
    - RoPE位置编码
    - SwiGLU FFN
    - LayerScale
    """

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim, device=device)
            )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None

        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        """初始化权重"""
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(
        self, x: Tensor, masks=None
    ) -> Tuple[Tensor, Tuple[int]]:
        """准备tokens"""
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1, 0, cls_token.shape[-1],
                dtype=cls_token.dtype, device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features(
        self, x: Tensor, masks: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """提取特征"""
        x, (H, W) = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)

        if self.untie_cls_and_patch_norms:
            x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
            x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
        else:
            x_norm = self.norm(x)
            x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
            x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]

        return {
            "x_norm_clstoken": x_norm_cls_reg[:, 0],
            "x_storage_tokens": x_norm_cls_reg[:, 1:],
            "x_norm_patchtokens": x_norm_patch,
            "x_prenorm": x,
        }

    def _get_intermediate_layers_not_chunked(
        self, x: Tensor, n: int = 1
    ) -> List[Tensor]:
        """获取中间层特征（非分块）"""
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len)
            if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take)
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """
        获取中间层特征

        Args:
            x: 输入图像 (B, C, H, W)
            n: 返回最后n个block的特征
            reshape: 是否reshape为空间形式
            return_class_token: 是否返回CLS token
            return_extra_tokens: 是否返回额外tokens
            norm: 是否应用norm

        Returns:
            中间层特征
        """
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed

        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]

        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1)
                .permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        else:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(
        self, *args, is_training: bool = False, **kwargs
    ) -> Dict[str, Tensor] | Tensor:
        """前向传播"""
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


# =============================================================================
# 工厂函数
# =============================================================================

def dinov3_vit_small(img_size: int = 512, patch_size: int = 16, **kwargs):
    """DINOv3 ViT-Small: embed_dim=384, depth=12, num_heads=6"""
    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        layerscale_init=1e-5,
        n_storage_tokens=4,
        mask_k_bias=True,
        **kwargs,
    )
    return model


def dinov3_vit_base(img_size: int = 512, patch_size: int = 16, **kwargs):
    """DINOv3 ViT-Base: embed_dim=768, depth=12, num_heads=12"""
    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        layerscale_init=1e-5,
        n_storage_tokens=4,
        mask_k_bias=True,
        **kwargs,
    )
    return model


def dinov3_vit_large(img_size: int = 512, patch_size: int = 16, **kwargs):
    """DINOv3 ViT-Large: embed_dim=1024, depth=24, num_heads=16"""
    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        layerscale_init=1e-5,
        n_storage_tokens=4,
        mask_k_bias=True,
        **kwargs,
    )
    return model


def dinov3_vit_so400m(img_size: int = 512, patch_size: int = 16, **kwargs):
    """DINOv3 ViT-SO400M"""
    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        layerscale_init=1e-5,
        n_storage_tokens=4,
        mask_k_bias=True,
        **kwargs,
    )
    return model


def load_dinov3_pretrained(
    model: DinoVisionTransformer,
    pretrained_path: Optional[str] = None,
    strict: bool = False,
) -> DinoVisionTransformer:
    """
    加载DINOv3预训练权重

    Args:
        model: DinoVisionTransformer模型
        pretrained_path: 预训练权重路径
        strict: 是否严格匹配

    Returns:
        加载权重后的模型
    """
    if pretrained_path is None or pretrained_path == "":
        logger.info("[DINOv3] No pretrained weights loaded, initializing weights")
        model.init_weights()
        return model

    logger.info(f"[DINOv3] Loading pretrained weights from {pretrained_path}")

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
            logger.warning(f"[DINOv3] Skipping {k}: shape mismatch or not found")

    # 加载权重
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    logger.info(f"[DINOv3] Loaded {len(filtered_state)} parameters")
    if missing:
        logger.warning(f"[DINOv3] Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"[DINOv3] Unexpected keys: {len(unexpected)}")

    return model


def get_dinov3_backbone(
    arch: str = 'vit_base',
    img_size: int = 512,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> Tuple[DinoVisionTransformer, int]:
    """
    获取DINOv3 backbone

    Args:
        arch: 架构类型 ('vit_small', 'vit_base', 'vit_large', 'vit_so400m')
        img_size: 输入图像尺寸
        pretrained_path: 预训练权重路径

    Returns:
        model: DinoVisionTransformer模型
        embed_dim: 特征维度
    """
    arch_dict = {
        'vit_small': (dinov3_vit_small, 384),
        'vit_base': (dinov3_vit_base, 768),
        'vit_large': (dinov3_vit_large, 1024),
        'vit_so400m': (dinov3_vit_so400m, 1152),
    }

    if arch not in arch_dict:
        raise ValueError(f"Unknown arch: {arch}. Choose from {list(arch_dict.keys())}")

    model_fn, embed_dim = arch_dict[arch]
    model = model_fn(img_size=img_size, **kwargs)

    model = load_dinov3_pretrained(model, pretrained_path)

    return model, embed_dim
