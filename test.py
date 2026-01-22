"""
测试脚本 - X-ray骨骼分割和分类项目
功能：
1. 创建测试用的目录结构
2. 生成示例图像和mask
3. 生成测试用的CSV文件
4. 测试数据加载
5. 测试模型推理（可选）
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_test_directories(base_dir='./test_data'):
    """创建测试用的目录结构"""
    print("=" * 60)
    print("步骤 1: 创建测试目录结构")
    print("=" * 60)

    dirs = {
        'base': base_dir,
        'images': os.path.join(base_dir, 'images'),
        'masks': os.path.join(base_dir, 'masks'),
        'labels': os.path.join(base_dir, 'labels'),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"✓ 创建目录: {path}")

    return dirs


def generate_sample_images(image_dir, num_samples=10):
    """生成示例X-ray图像（灰度图）"""
    print("\n" + "=" * 60)
    print("步骤 2: 生成示例X-ray图像")
    print("=" * 60)

    patient_ids = []
    for i in range(num_samples):
        patient_id = f"patient{i:04d}"
        patient_ids.append(patient_id)

        # 生成512x512的模拟X-ray图像（灰度）
        # 模拟骨骼特征：中心区域较亮，边缘较暗，添加一些噪声
        image = np.random.rand(512, 512) * 50 + 100  # 基础亮度

        # 添加模拟的骨骼结构（中心椭圆区域较亮）
        y, x = np.ogrid[:512, :512]
        center_y, center_x = 256, 256

        # 创建椭圆形的骨骼区域
        ellipse_mask = ((x - center_x)**2 / 150**2 + (y - center_y)**2 / 200**2) <= 1
        image[ellipse_mask] += 50

        # 添加一些纹理和噪声
        noise = np.random.randn(512, 512) * 10
        image = image + noise

        # 归一化到0-255
        image = np.clip(image, 0, 255).astype(np.uint8)

        # 保存图像
        image_path = os.path.join(image_dir, f"{patient_id}.png")
        Image.fromarray(image, mode='L').save(image_path)

        if i < 3:
            print(f"✓ 生成图像: {patient_id}.png (512x512, 灰度)")

    if num_samples > 3:
        print(f"  ... 共生成 {num_samples} 张图像")

    return patient_ids


def generate_sample_masks(mask_dir, patient_ids):
    """生成示例mask（骨骼分割）"""
    print("\n" + "=" * 60)
    print("步骤 3: 生成示例骨骼mask")
    print("=" * 60)

    for i, patient_id in enumerate(patient_ids):
        # 生成512x512的二值mask
        mask = np.zeros((512, 512), dtype=np.uint8)

        # 创建骨骼区域mask（椭圆形）
        y, x = np.ogrid[:512, :512]
        center_y, center_x = 256, 256

        # 椭圆形骨骼mask
        ellipse_mask = ((x - center_x)**2 / 150**2 + (y - center_y)**2 / 200**2) <= 1
        mask[ellipse_mask] = 1

        # 添加一些随机的小骨骼区域（模拟碎片）
        if np.random.rand() > 0.5:
            # 添加随机小椭圆
            offset_x = np.random.randint(-100, 100)
            offset_y = np.random.randint(-100, 100)
            small_ellipse = ((x - (center_x + offset_x))**2 / 30**2 +
                            (y - (center_y + offset_y))**2 / 40**2) <= 1
            mask[small_ellipse] = 1

        # 保存为PNG（二值mask，0或255）
        mask_image = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(mask_dir, f"{patient_id}.png")
        Image.fromarray(mask_image, mode='L').save(mask_path)

        # 也可以保存为NPZ格式（包含多个mask）
        mask_npz_path = os.path.join(mask_dir, f"{patient_id}.npz")
        np.savez_compressed(mask_npz_path, bone=mask)

        if i < 3:
            print(f"✓ 生成mask: {patient_id}.png 和 {patient_id}.npz")

    if len(patient_ids) > 3:
        print(f"  ... 共生成 {len(patient_ids)} 个mask文件")


def generate_csv_labels(label_dir, patient_ids, split_ratio=(0.7, 0.15, 0.15)):
    """生成训练/验证/测试的CSV标签文件"""
    print("\n" + "=" * 60)
    print("步骤 4: 生成CSV标签文件")
    print("=" * 60)

    # 定义分类任务（基于配置文件）
    task_names = [
        'femur_fracture',      # 股骨骨折
        'pelvis_fracture',     # 骨盆骨折
        'spine_fracture',      # 脊柱骨折
        'joint_dislocation'    # 关节脱位
    ]

    # 为每个患者生成随机标签
    data = []
    for patient_id in patient_ids:
        row = {'patient_id': patient_id}
        # 生成随机标签（0或1，二分类）
        for task in task_names:
            row[task] = np.random.randint(0, 2)
        data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 按比例分割数据
    n_total = len(df)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    # 保存CSV文件
    csv_files = {
        'train': (train_df, os.path.join(label_dir, 'train.csv')),
        'val': (val_df, os.path.join(label_dir, 'val.csv')),
        'test': (test_df, os.path.join(label_dir, 'test.csv'))
    }

    for split_name, (df_split, csv_path) in csv_files.items():
        df_split.to_csv(csv_path, index=False)
        print(f"✓ 保存 {split_name}.csv: {len(df_split)} 个样本")

        # 显示前几行
        if split_name == 'train':
            print(f"\n  {split_name}.csv 示例（前3行）:")
            print(df_split.head(3).to_string(index=False))

    return csv_files


def test_dataset_loading(dirs):
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("步骤 5: 测试数据集加载")
    print("=" * 60)

    try:
        from data.xraydataset import XrayBoneDataset, RandomFlipRotate2D

        # 创建数据集
        train_transform = RandomFlipRotate2D(flip_prob=0.5, max_angle=15)

        dataset = XrayBoneDataset(
            image_dir=dirs['images'],
            mask_dir=dirs['masks'],
            label_file=os.path.join(dirs['labels'], 'train.csv'),
            target_shape=(512, 512),
            transform=train_transform,
            mode='train',
            single_mask=True,
            mask_key='bone'
        )

        print(f"✓ 成功加载数据集")
        print(f"  数据集大小: {len(dataset)} 个样本")

        # 测试加载一个样本
        sample = dataset[0]
        print(f"\n  样本信息:")
        print(f"  - Patient ID: {sample['patient_id']}")
        print(f"  - 图像形状: {sample['image'].shape} (应为 [1, 512, 512])")
        print(f"  - 图像数据类型: {sample['image'].dtype}")
        print(f"  - 图像值范围: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"  - Mask keys: {list(sample['masks'].keys())}")
        print(f"  - Mask形状: {sample['masks']['bone'].shape} (应为 [1, 512, 512])")
        print(f"  - 标签: {sample['labels']}")

        # 测试DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        batch = next(iter(dataloader))
        print(f"\n  Batch信息:")
        print(f"  - Batch图像形状: {batch['image'].shape} (应为 [2, 1, 512, 512])")
        print(f"  - Batch mask形状: {batch['masks']['bone'].shape} (应为 [2, 1, 512, 512])")
        print(f"  ✓ DataLoader测试通过")

        return True

    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_inference(dirs):
    """测试模型推理（可选）"""
    print("\n" + "=" * 60)
    print("步骤 6: 测试模型推理（可选）")
    print("=" * 60)

    try:
        from data.xraydataset import XrayBoneDataset
        from models.xray_dinov3_v2 import TraumaNetDINOv3
        from torch.utils.data import DataLoader

        # 创建简单的测试数据集
        dataset = XrayBoneDataset(
            image_dir=dirs['images'],
            mask_dir=dirs['masks'],
            label_file=os.path.join(dirs['labels'], 'test.csv'),
            target_shape=(512, 512),
            mode='test',
            single_mask=True,
            mask_key='bone'
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        print("✓ 数据集创建成功")
        print(f"  测试样本数: {len(dataset)}")

        # 注意：这里只是测试模型结构，不加载预训练权重
        print("\n  注意: 模型推理测试需要配置文件和预训练权重")
        print("  如需测试模型推理，请：")
        print("  1. 准备DINOv3预训练权重")
        print("  2. 修改配置文件路径")
        print("  3. 使用trainer进行训练或推理")

        return True

    except Exception as e:
        print(f"✗ 模型推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_usage_instructions(dirs):
    """打印使用说明"""
    print("\n" + "=" * 60)
    print("测试数据生成完成！")
    print("=" * 60)
    print("\n使用说明:")
    print("\n1. 查看生成的测试数据:")
    print(f"   - 图像目录: {dirs['images']}")
    print(f"   - Mask目录: {dirs['masks']}")
    print(f"   - 标签目录: {dirs['labels']}")

    print("\n2. 测试配置文件 (configs/xray_bone_mask.yaml):")
    print("   修改以下路径为测试数据路径：")
    print(f"   image_dir: \"{dirs['images']}\"")
    print(f"   mask_dir: \"{dirs['masks']}\"")
    print(f"   train_dataset: \"{os.path.join(dirs['labels'], 'train.csv')}\"")
    print(f"   val_dataset: \"{os.path.join(dirs['labels'], 'val.csv')}\"")
    print(f"   test_dataset: \"{os.path.join(dirs['labels'], 'test.csv')}\"")

    print("\n3. 运行训练（需要先准备DINOv3预训练权重）:")
    print("   python run/run_train.py --config-file configs/xray_bone_mask.yaml --output-dir ./output")

    print("\n4. CSV标签格式说明:")
    print("   - patient_id: 患者ID（必需）")
    print("   - femur_fracture: 股骨骨折标签 (0或1)")
    print("   - pelvis_fracture: 骨盆骨折标签 (0或1)")
    print("   - spine_fracture: 脊柱骨折标签 (0或1)")
    print("   - joint_dislocation: 关节脱位标签 (0或1)")

    print("\n5. 数据文件格式:")
    print("   - 图像: PNG格式，灰度图 (512x512)")
    print("   - Mask: PNG格式 (二值图) 或 NPZ格式 (numpy数组)")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("X-ray骨骼分割和分类项目 - 测试数据生成器")
    print("=" * 60)

    # 1. 创建目录结构
    dirs = create_test_directories(base_dir='./test_data')

    # 2. 生成示例图像
    patient_ids = generate_sample_images(dirs['images'], num_samples=20)

    # 3. 生成示例mask
    generate_sample_masks(dirs['masks'], patient_ids)

    # 4. 生成CSV标签
    generate_csv_labels(dirs['labels'], patient_ids, split_ratio=(0.7, 0.15, 0.15))

    # 5. 测试数据加载
    dataset_ok = test_dataset_loading(dirs)

    # 6. 测试模型推理（可选）
    if dataset_ok:
        test_model_inference(dirs)

    # 7. 打印使用说明
    print_usage_instructions(dirs)


if __name__ == "__main__":
    main()
