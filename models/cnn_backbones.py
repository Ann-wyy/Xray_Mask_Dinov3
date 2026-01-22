"""
CNN Backbone模块
支持多种预训练模型作为backbone
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

logger = logging.getLogger(__name__)


def get_cnn_backbone(
    cnn_name: str,
    pretrained: bool = True,
    pretrained_weight: str = None,
    in_channels: int = 3
) -> Tuple[nn.Module, int]:
    """
    获取CNN backbone模型

    Args:
        cnn_name: CNN模型名称
        pretrained: 是否使用预训练权重
        pretrained_weight: 预训练权重路径（如果指定，则从本地加载）
        in_channels: 输入通道数

    Returns:
        backbone模型和特征维度
    """
    # ResNet系列
    if cnn_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        feature_dim = 512
    elif cnn_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        feature_dim = 512
    elif cnn_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        feature_dim = 2048
    elif cnn_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        feature_dim = 2048
    elif cnn_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
        feature_dim = 2048

    # DenseNet系列
    elif cnn_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        feature_dim = 1024
    elif cnn_name == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
        feature_dim = 1664
    elif cnn_name == 'densenet201':
        model = models.densenet201(pretrained=pretrained)
        feature_dim = 1920

    # EfficientNet系列
    elif cnn_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        feature_dim = 1280
    elif cnn_name == 'efficientnet_b1':
        model = models.efficientnet_b1(pretrained=pretrained)
        feature_dim = 1280
    elif cnn_name == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=pretrained)
        feature_dim = 1408
    elif cnn_name == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=pretrained)
        feature_dim = 1536
    elif cnn_name == 'efficientnet_b4':
        model = models.efficientnet_b4(pretrained=pretrained)
        feature_dim = 1792

    # ConvNeXt系列
    elif cnn_name == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=pretrained)
        feature_dim = 768
    elif cnn_name == 'convnext_small':
        model = models.convnext_small(pretrained=pretrained)
        feature_dim = 768
    elif cnn_name == 'convnext_base':
        model = models.convnext_base(pretrained=pretrained)
        feature_dim = 1024
    elif cnn_name == 'convnext_large':
        model = models.convnext_large(pretrained=pretrained)
        feature_dim = 1536

    else:
        raise ValueError(f"不支持的CNN模型: {cnn_name}")

    # 如果指定了本地预训练权重路径，则加载
    if pretrained_weight is not None:
        logger.info(f"从本地加载预训练权重: {pretrained_weight}")
        try:
            state_dict = torch.load(pretrained_weight, map_location='cpu')
            # 尝试加载权重，使用strict=False允许部分键不匹配
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"警告: 以下键在模型中缺失: {len(missing_keys)} 个键")
            if unexpected_keys:
                logger.warning(f"警告: 以下键在权重文件中多余: {len(unexpected_keys)} 个键")
            logger.info(f"成功加载预训练权重: {pretrained_weight}")
        except Exception as e:
            logger.error(f"警告: 加载本地权重失败: {e}")
            logger.warning(f"将使用torchvision自动下载的权重")

    # 提取特征提取器（移除分类头）
    if 'resnet' in cnn_name:
        # ResNet: 移除最后的全连接层和平均池化层
        backbone = nn.Sequential(*list(model.children())[:-2])

    elif 'densenet' in cnn_name:
        # DenseNet: 移除分类器
        backbone = model.features

    elif 'efficientnet' in cnn_name:
        # EfficientNet: 移除分类器
        backbone = model.features

    elif 'convnext' in cnn_name:
        # ConvNeXt: 移除分类器，保留特征提取部分
        backbone = model.features

    else:
        backbone = model

    return backbone, feature_dim
