"""
数据增强变换模块

包含用于X-ray图像和mask的同步数据增强
"""

import torch
import numpy as np
import random
from PIL import Image
from typing import Dict, Tuple


class RandomFlipRotate:
    """
    同步随机翻转和旋转增强

    同时应用于X-ray图像和分割mask，保持一致性
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        max_angle: float = 15.0
    ):
        """
        初始化数据增强

        Args:
            flip_prob: 水平翻转概率
            rotate_prob: 旋转概率
            max_angle: 最大旋转角度（度）
        """
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_angle = max_angle

    def __call__(
        self,
        image: np.ndarray,
        masks: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        应用数据增强

        Args:
            image: 输入图像 (H, W) 或 (1, H, W)
            masks: 分割mask字典 {organ_name: mask_array (H, W)}

        Returns:
            增强后的图像和masks
        """
        # 确保image是(H, W)格式
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]

        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = np.fliplr(image).copy()
            masks = {k: np.fliplr(v).copy() for k, v in masks.items()}

        # 随机旋转
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_angle, self.max_angle)

            # 使用PIL进行旋转
            # 图像使用双线性插值
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            image = np.array(pil_img, dtype=np.float32) / 255.0

            # Mask使用最近邻插值
            for k in masks:
                pil_mask = Image.fromarray(masks[k].astype(np.uint8))
                pil_mask = pil_mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
                masks[k] = np.array(pil_mask, dtype=np.int64)

        # 恢复channel维度 (1, H, W)
        image = np.expand_dims(image, 0)

        return image, masks


class RandomBrightnessContrast:
    """
    随机亮度和对比度调整

    仅应用于图像，不影响mask
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        prob: float = 0.5
    ):
        """
        初始化亮度对比度增强

        Args:
            brightness_range: 亮度调整范围
            contrast_range: 对比度调整范围
            prob: 应用概率
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        masks: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """应用亮度对比度调整"""
        if random.random() < self.prob:
            # 确保image是(H, W)或(1, H, W)格式
            orig_shape = image.shape
            if image.ndim == 3 and image.shape[0] == 1:
                image = image[0]

            # 亮度调整
            brightness_factor = random.uniform(*self.brightness_range)
            image = image * brightness_factor

            # 对比度调整
            contrast_factor = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean

            # 裁剪到[0, 1]
            image = np.clip(image, 0, 1)

            # 恢复原始形状
            if orig_shape != image.shape:
                image = np.expand_dims(image, 0)

        return image, masks


class Compose:
    """
    组合多个数据增强变换
    """

    def __init__(self, transforms: list):
        """
        初始化组合变换

        Args:
            transforms: 变换列表
        """
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        masks: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """依次应用所有变换"""
        for transform in self.transforms:
            image, masks = transform(image, masks)
        return image, masks


def get_train_transform(config):
    """
    根据配置创建训练数据增强

    Args:
        config: 配置对象

    Returns:
        数据增强变换或None
    """
    # 检查是否启用数据增强
    if not hasattr(config.data, 'augmentation') or not config.data.augmentation.enabled:
        return None

    aug_config = config.data.augmentation

    # 创建变换列表
    transforms = []

    # 添加翻转和旋转
    if hasattr(aug_config, 'flip_prob') or hasattr(aug_config, 'rotate_prob'):
        flip_prob = getattr(aug_config, 'flip_prob', 0.5)
        rotate_prob = getattr(aug_config, 'rotate_prob', 0.5)
        max_angle = getattr(aug_config, 'max_rotation_angle', 15.0)

        transforms.append(RandomFlipRotate(
            flip_prob=flip_prob,
            rotate_prob=rotate_prob,
            max_angle=max_angle
        ))

    # 添加亮度对比度调整
    if hasattr(aug_config, 'brightness_contrast_prob'):
        prob = aug_config.brightness_contrast_prob
        brightness_range = getattr(aug_config, 'brightness_range', [0.8, 1.2])
        contrast_range = getattr(aug_config, 'contrast_range', [0.8, 1.2])

        transforms.append(RandomBrightnessContrast(
            brightness_range=tuple(brightness_range),
            contrast_range=tuple(contrast_range),
            prob=prob
        ))

    # 如果没有任何变换，返回None
    if len(transforms) == 0:
        return None

    return Compose(transforms)


if __name__ == "__main__":
    # 测试数据增强
    print("测试数据增强模块")
    print("="*60)

    # 创建测试数据
    image = np.random.rand(1, 512, 512).astype(np.float32)
    masks = {
        'liver': np.random.randint(0, 2, (512, 512), dtype=np.int64),
        'spleen': np.random.randint(0, 2, (512, 512), dtype=np.int64)
    }

    print(f"原始图像形状: {image.shape}")
    print(f"原始mask形状: {list(masks.values())[0].shape}")

    # 测试翻转旋转
    print("\n1. 测试翻转旋转增强")
    transform1 = RandomFlipRotate(flip_prob=1.0, rotate_prob=1.0, max_angle=15)
    aug_image, aug_masks = transform1(image, masks)
    print(f"  增强后图像形状: {aug_image.shape}")
    print(f"  增强后mask形状: {list(aug_masks.values())[0].shape}")

    # 测试亮度对比度
    print("\n2. 测试亮度对比度增强")
    transform2 = RandomBrightnessContrast(prob=1.0)
    aug_image, aug_masks = transform2(image, masks)
    print(f"  增强后图像形状: {aug_image.shape}")
    print(f"  增强后mask形状: {list(aug_masks.values())[0].shape}")

    # 测试组合变换
    print("\n3. 测试组合变换")
    composed = Compose([
        RandomFlipRotate(flip_prob=0.5, rotate_prob=0.5, max_angle=15),
        RandomBrightnessContrast(prob=0.5)
    ])
    aug_image, aug_masks = composed(image, masks)
    print(f"  增强后图像形状: {aug_image.shape}")
    print(f"  增强后mask形状: {list(aug_masks.values())[0].shape}")

    print("\n" + "="*60)
    print("测试完成")
