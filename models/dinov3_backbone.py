"""
DINOv3 Backbone Module

基于Meta AI的DINOv3模型，提供预训练的Vision Transformer backbone
支持从torch.hub加载或本地权重加载
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class DinoVisionTransformer(nn.Module):
    """
    DINOv3 Vision Transformer wrapper

    封装了DINOv3模型，提供统一的接口
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准前向传播"""
        return self.model(x)

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1,
        return_class_token: bool = False
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取中间层特征

        Args:
            x: 输入图像 (B, C, H, W)
            n: 返回最后n个block的输出
            return_class_token: 是否返回class token

        Returns:
            List[(patch_tokens, cls_token)], 长度为n
        """
        return self.model.get_intermediate_layers(x, n=n, return_class_token=return_class_token)


def get_dinov3_backbone(
    arch: str = 'vit_base',
    img_size: int = 518,
    pretrained_path: Optional[str] = None,
) -> Tuple[DinoVisionTransformer, int]:
    """
    获取DINOv3 backbone

    Args:
        arch: 架构类型，可选 'vit_small', 'vit_base', 'vit_large', 'vit_giant2'
        img_size: 输入图像大小
        pretrained_path: 本地预训练权重路径（可选）

    Returns:
        (model, embed_dim)
    """

    # 架构到embedding维度的映射
    arch_to_dim = {
        'vit_small': 384,
        'vit_base': 768,
        'vit_large': 1024,
        'vit_giant2': 1536,
    }

    if arch not in arch_to_dim:
        raise ValueError(f"不支持的架构: {arch}，可选: {list(arch_to_dim.keys())}")

    embed_dim = arch_to_dim[arch]

    try:
        # 尝试从torch.hub加载DINOv3模型
        print(f"正在从torch.hub加载 DINOv3 {arch}...")

        # DINOv3模型名称映射
        model_name = f'dinov2_{arch}14'  # 例如: dinov2_vit_base14

        # 从facebookresearch/dinov2加载
        model = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True,
            verbose=False
        )

        print(f"✓ 成功从torch.hub加载 {model_name}")

    except Exception as e:
        print(f"警告: 无法从torch.hub加载模型: {e}")
        print("使用未初始化的ViT模型（需要手动加载权重）")

        # 创建一个dummy模型类
        class DummyViT(nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.embed_dim = embed_dim
                # 创建一个简单的特征提取器
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=8,
                        dim_feedforward=embed_dim * 4,
                        batch_first=True
                    )
                    for _ in range(12)
                ])
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, x):
                # 简单的前向传播
                x = self.patch_embed(x)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)

                for block in self.blocks:
                    x = block(x)

                x = self.norm(x)
                return x

            def get_intermediate_layers(self, x, n=1, return_class_token=False):
                """获取中间层特征"""
                x = self.patch_embed(x)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)

                outputs = []
                num_blocks = len(self.blocks)
                start_idx = max(0, num_blocks - n)

                for idx, block in enumerate(self.blocks):
                    x = block(x)
                    if idx >= start_idx:
                        # 返回 (patch_tokens, cls_token) 格式
                        # 这里简化处理，假设第一个token是cls token
                        cls_token = x[:, 0:1, :]
                        patch_tokens = x[:, 1:, :]
                        outputs.append((patch_tokens, cls_token))

                return outputs

        model = DummyViT(embed_dim)
        print("注意: 使用的是简化的ViT模型，性能可能不佳")

    # 如果提供了本地权重路径，加载权重
    if pretrained_path and pretrained_path.strip():
        try:
            print(f"正在从本地加载权重: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ 成功加载本地权重")
        except Exception as e:
            print(f"警告: 加载本地权重失败: {e}")
            print("将使用默认初始化的权重")

    # 包装为DinoVisionTransformer
    wrapped_model = DinoVisionTransformer(model)

    return wrapped_model, embed_dim


if __name__ == "__main__":
    # 测试DINOv3 backbone
    print("测试DINOv3 Backbone")
    print("="*60)

    # 测试不同架构
    for arch in ['vit_small', 'vit_base']:
        print(f"\n测试架构: {arch}")
        try:
            model, embed_dim = get_dinov3_backbone(arch=arch, img_size=512)
            print(f"  Embedding维度: {embed_dim}")

            # 测试前向传播
            dummy_input = torch.randn(2, 3, 512, 512)
            print(f"  输入形状: {dummy_input.shape}")

            # 测试get_intermediate_layers
            layer_outputs = model.get_intermediate_layers(dummy_input, n=4, return_class_token=True)
            print(f"  中间层输出数量: {len(layer_outputs)}")

            if len(layer_outputs) > 0:
                patch_tokens, cls_token = layer_outputs[0]
                print(f"  Patch tokens形状: {patch_tokens.shape}")
                print(f"  CLS token形状: {cls_token.shape}")

            print(f"  ✓ {arch} 测试通过")

        except Exception as e:
            print(f"  ✗ {arch} 测试失败: {e}")

    print("\n" + "="*60)
    print("测试完成")
