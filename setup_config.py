"""
配置文件生成向导
根据你的CSV标签文件自动生成正确的配置文件
"""

import os
import sys
import pandas as pd
import yaml
from pathlib import Path


def analyze_csv_labels(csv_path):
    """分析CSV文件中的label列"""
    print(f"\n正在分析CSV文件: {csv_path}")

    df = pd.read_csv(csv_path)

    # 获取所有列（除了patient_id）
    label_columns = [col for col in df.columns if col != 'patient_id']

    print(f"\n找到 {len(df)} 个样本")
    print(f"找到 {len(label_columns)} 个标签任务:")

    # 分析每个标签的取值范围
    label_info = {}
    for col in label_columns:
        unique_values = df[col].unique()
        num_classes = len(unique_values)
        label_info[col] = {
            'num_classes': num_classes,
            'unique_values': sorted(unique_values.tolist()),
            'distribution': df[col].value_counts().to_dict()
        }

        print(f"\n  ✓ {col}:")
        print(f"    - 类别数: {num_classes}")
        print(f"    - 取值: {sorted(unique_values.tolist())}")
        print(f"    - 分布: {dict(df[col].value_counts())}")

    return label_info


def create_config_template(
    exp_name,
    image_dir,
    mask_dir,
    train_csv,
    val_csv,
    test_csv,
    label_info,
    output_path,
    dinov3_pretrained="",
    mask_key="bone",
    target_shape=(512, 512),
    batch_size=8,
    epochs=50,
    learning_rate=1e-4
):
    """创建配置文件"""

    # 构建num_classes字典
    num_classes = {}
    for task_name, info in label_info.items():
        # 如果是二分类(0,1)，设为2
        # 如果是多分类，设为实际类别数
        num_classes[task_name] = info['num_classes']

    config = {
        'exp_name': exp_name,
        'seed': 42,
        'device': 'cuda',

        'data': {
            'dataset_class': 'XrayDataset',
            'image_dir': image_dir,
            'mask_dir': mask_dir,
            'train_dataset': train_csv,
            'val_dataset': val_csv,
            'test_dataset': test_csv,

            'single_mask': True,
            'mask_key': mask_key,

            'target_shape': list(target_shape),
            'use_augmentation': True,
            'flip_prob': 0.5,
            'rotate_prob': 0.3,
            'max_angle': 15,

            'num_workers': 4,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': False,
            'use_preprocessed': False
        },

        'model': {
            'model_type': 'dinov3_v2',
            'vit_arch': 'vit_base',
            'dinov3_pretrained': dinov3_pretrained,
            'input_depth': 1,

            'enable_segmentation': True,
            'seg_organs': [mask_key],

            'num_classes': num_classes,

            'top_k_ratio': 0.2,
            'dropout': 0.1,
            'use_task_interaction': True
        },

        'training': {
            'batch_size': batch_size,
            'epochs': epochs,
            'optimizer': 'adamw',
            'lr': learning_rate,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'min_lr': 1e-6,
            'loss_type': 'BCE',
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0,
            'seg_loss_weight': 0.5,
            'class_weights': {},
            'use_amp': True,
            'gradient_clip': 1.0,
            'save_interval': 5
        },

        'log': {
            'log_dir': f'./logs/{exp_name}',
            'checkpoint_dir': f'./checkpoints/{exp_name}',
            'tensorboard_dir': f'./runs/{exp_name}',
            'save_best': True,
            'metric_for_best': 'avg_f1'
        }
    }

    # 保存配置文件
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n✓ 配置文件已保存到: {output_path}")
    return config


def interactive_setup():
    """交互式配置向导"""
    print("=" * 70)
    print("X-ray项目配置向导")
    print("=" * 70)

    # 1. 获取数据路径
    print("\n步骤 1: 数据路径设置")
    print("-" * 70)

    image_dir = input("请输入图像目录路径 (例如: ./test_data/images): ").strip()
    if not image_dir:
        image_dir = "./test_data/images"

    mask_dir = input("请输入mask目录路径 (例如: ./test_data/masks): ").strip()
    if not mask_dir:
        mask_dir = "./test_data/masks"

    train_csv = input("请输入训练CSV路径 (例如: ./test_data/labels/train.csv): ").strip()
    if not train_csv:
        train_csv = "./test_data/labels/train.csv"

    val_csv = input("请输入验证CSV路径 (例如: ./test_data/labels/val.csv): ").strip()
    if not val_csv:
        val_csv = "./test_data/labels/val.csv"

    test_csv = input("请输入测试CSV路径 (例如: ./test_data/labels/test.csv): ").strip()
    if not test_csv:
        test_csv = "./test_data/labels/test.csv"

    # 验证文件是否存在
    if not os.path.exists(train_csv):
        print(f"\n⚠️  警告: 训练CSV文件不存在: {train_csv}")
        print("请确保文件路径正确，或先运行 python test.py 生成测试数据")
        return

    # 2. 分析CSV标签
    print("\n步骤 2: 分析CSV标签")
    print("-" * 70)
    label_info = analyze_csv_labels(train_csv)

    # 3. 配置模型参数
    print("\n步骤 3: 模型配置")
    print("-" * 70)

    exp_name = input("实验名称 (例如: my_xray_exp) [默认: xray_exp]: ").strip()
    if not exp_name:
        exp_name = "xray_exp"

    dinov3_pretrained = input("DINOv3预训练权重路径 (留空表示从头训练) [默认: 空]: ").strip()

    mask_key = input("Mask键名 (例如: bone) [默认: bone]: ").strip()
    if not mask_key:
        mask_key = "bone"

    # 4. 训练参数
    print("\n步骤 4: 训练参数")
    print("-" * 70)

    batch_size = input("Batch size [默认: 8]: ").strip()
    batch_size = int(batch_size) if batch_size else 8

    epochs = input("训练轮数 [默认: 50]: ").strip()
    epochs = int(epochs) if epochs else 50

    learning_rate = input("学习率 [默认: 1e-4]: ").strip()
    learning_rate = float(learning_rate) if learning_rate else 1e-4

    # 5. 生成配置文件
    print("\n步骤 5: 生成配置文件")
    print("-" * 70)

    output_path = f"./configs/{exp_name}.yaml"
    os.makedirs("./configs", exist_ok=True)

    config = create_config_template(
        exp_name=exp_name,
        image_dir=image_dir,
        mask_dir=mask_dir,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        label_info=label_info,
        output_path=output_path,
        dinov3_pretrained=dinov3_pretrained,
        mask_key=mask_key,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate
    )

    # 6. 显示配置摘要
    print("\n" + "=" * 70)
    print("配置摘要")
    print("=" * 70)
    print(f"实验名称: {exp_name}")
    print(f"训练样本数: {len(pd.read_csv(train_csv))}")
    print(f"标签任务数: {len(label_info)}")
    print(f"标签任务: {list(label_info.keys())}")
    print(f"Batch size: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {learning_rate}")
    print(f"配置文件: {output_path}")

    # 7. 生成启动命令
    print("\n" + "=" * 70)
    print("下一步操作")
    print("=" * 70)
    print("\n启动训练命令:")
    print(f"\npython run/run_train.py \\")
    print(f"  --config-file {output_path} \\")
    print(f"  --output-dir ./output/{exp_name}")

    print("\n\n恢复训练命令:")
    print(f"\npython run/run_train.py \\")
    print(f"  --config-file {output_path} \\")
    print(f"  --output-dir ./output/{exp_name} \\")
    print(f"  --resume")

    print("\n" + "=" * 70)


def quick_setup(train_csv, output_config=None):
    """快速配置（非交互式）"""
    print("=" * 70)
    print("快速配置模式")
    print("=" * 70)

    # 自动推断路径
    csv_dir = os.path.dirname(train_csv)
    data_dir = os.path.dirname(csv_dir)
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')

    val_csv = os.path.join(csv_dir, 'val.csv')
    test_csv = os.path.join(csv_dir, 'test.csv')

    # 分析标签
    label_info = analyze_csv_labels(train_csv)

    # 生成配置
    exp_name = "auto_config_exp"
    if output_config is None:
        output_config = f"./configs/{exp_name}.yaml"

    os.makedirs("./configs", exist_ok=True)

    config = create_config_template(
        exp_name=exp_name,
        image_dir=image_dir,
        mask_dir=mask_dir,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        label_info=label_info,
        output_path=output_config
    )

    print("\n" + "=" * 70)
    print("配置完成！")
    print("=" * 70)
    print(f"\n启动训练:")
    print(f"python run/run_train.py --config-file {output_config} --output-dir ./output/{exp_name}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='配置文件生成向导')
    parser.add_argument('--train-csv', type=str, help='训练CSV文件路径（快速模式）')
    parser.add_argument('--output', type=str, help='输出配置文件路径')

    args = parser.parse_args()

    if args.train_csv:
        # 快速模式
        quick_setup(args.train_csv, args.output)
    else:
        # 交互模式
        interactive_setup()


if __name__ == "__main__":
    main()
