"""
配置文件检查工具
检查YAML语法和必需字段
"""

import sys
import yaml
from pathlib import Path


def check_yaml_syntax(config_path):
    """检查YAML语法"""
    print(f"检查文件: {config_path}")
    print("-" * 60)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ YAML语法正确")
        return config
    except yaml.YAMLError as e:
        print(f"✗ YAML语法错误:")
        print(f"  {e}")

        # 尝试找出具体哪一行有问题
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            print(f"\n错误位置: 第 {mark.line + 1} 行, 第 {mark.column + 1} 列")

            # 显示问题行
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if mark.line < len(lines):
                    print(f"\n问题行内容:")
                    start = max(0, mark.line - 2)
                    end = min(len(lines), mark.line + 3)
                    for i in range(start, end):
                        prefix = ">>> " if i == mark.line else "    "
                        print(f"{prefix}{i+1:4d} | {lines[i].rstrip()}")
        return None
    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        return None


def check_required_fields(config):
    """检查必需字段"""
    print("\n" + "-" * 60)
    print("检查必需字段:")
    print("-" * 60)

    errors = []
    warnings = []

    # 检查data字段
    if 'data' not in config:
        errors.append("缺少 'data' 部分")
    else:
        data = config['data']
        required_data_fields = ['image_dir', 'mask_dir', 'train_dataset', 'val_dataset', 'test_dataset', 'target_shape']
        for field in required_data_fields:
            if field not in data:
                errors.append(f"data.{field} 缺失")

    # 检查model字段
    if 'model' not in config:
        errors.append("缺少 'model' 部分")
    else:
        model = config['model']
        if 'num_classes' not in model:
            errors.append("model.num_classes 缺失")
        elif not model['num_classes']:
            warnings.append("model.num_classes 为空，至少需要一个分类任务")

    # 检查training字段
    if 'training' not in config:
        errors.append("缺少 'training' 部分")
    else:
        training = config['training']
        required_training_fields = ['batch_size', 'epochs', 'lr']
        for field in required_training_fields:
            if field not in training:
                warnings.append(f"training.{field} 缺失，将使用默认值")

    # 输出结果
    if errors:
        print("\n✗ 发现错误:")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print("\n⚠ 警告:")
        for warning in warnings:
            print(f"  - {warning}")

    if not errors and not warnings:
        print("✓ 所有必需字段都存在")

    return len(errors) == 0


def show_config_summary(config):
    """显示配置摘要"""
    print("\n" + "=" * 60)
    print("配置摘要")
    print("=" * 60)

    if 'exp_name' in config:
        print(f"实验名称: {config['exp_name']}")

    if 'data' in config:
        print(f"\n数据配置:")
        if 'target_shape' in config['data']:
            print(f"  - 图像尺寸: {config['data']['target_shape']}")
        if 'train_dataset' in config['data']:
            print(f"  - 训练集: {config['data']['train_dataset']}")

    if 'model' in config:
        print(f"\n模型配置:")
        if 'num_classes' in config['model']:
            tasks = config['model']['num_classes']
            print(f"  - 分类任务数: {len(tasks)}")
            print(f"  - 任务列表:")
            for task, num_cls in tasks.items():
                print(f"    • {task}: {num_cls} 类")

    if 'training' in config:
        print(f"\n训练配置:")
        training = config['training']
        if 'batch_size' in training:
            print(f"  - Batch size: {training['batch_size']}")
        if 'epochs' in training:
            print(f"  - Epochs: {training['epochs']}")
        if 'lr' in training:
            print(f"  - Learning rate: {training['lr']}")


def fix_common_issues(config_path):
    """尝试修复常见问题"""
    print("\n" + "=" * 60)
    print("常见YAML错误和修复方法")
    print("=" * 60)

    print("""
常见错误1: 缩进不一致
  错误示例:
    model:
      num_classes:
        task1: 2
         task2: 2  # ✗ 多了一个空格

  修复: 确保同级字段使用相同的缩进（通常是2个空格）

常见错误2: 冒号后没有空格
  错误示例:
    batch_size:8  # ✗ 冒号后没有空格

  修复:
    batch_size: 8  # ✓ 冒号后要有空格

常见错误3: 字符串包含特殊字符未加引号
  错误示例:
    path: /path/with:colon  # ✗ 冒号会被识别为键值分隔符

  修复:
    path: "/path/with:colon"  # ✓ 用引号包裹

常见错误4: 列表格式错误
  错误示例:
    seg_organs: [bone, femur, pelvis  # ✗ 缺少右括号

  修复:
    seg_organs: [bone, femur, pelvis]  # ✓ 或者用多行格式:
    seg_organs:
      - bone
      - femur
      - pelvis

常见错误5: Tab字符混用
  错误: 使用了Tab字符而不是空格
  修复: 将所有Tab替换为空格（通常2个或4个）
""")

    print("\n快速修复命令:")
    print(f"  # 检查是否有Tab字符")
    print(f"  cat -A {config_path} | grep '\\^I'")
    print(f"\n  # 替换Tab为2个空格")
    print(f"  sed -i 's/\\t/  /g' {config_path}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python check_config.py <config_file>")
        print("\n示例:")
        print("  python check_config.py configs/xray_bone_mask.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    if not Path(config_path).exists():
        print(f"✗ 文件不存在: {config_path}")
        sys.exit(1)

    print("=" * 60)
    print("配置文件检查工具")
    print("=" * 60)
    print()

    # 1. 检查语法
    config = check_yaml_syntax(config_path)

    if config is None:
        # 语法错误，显示修复建议
        fix_common_issues(config_path)
        sys.exit(1)

    # 2. 检查必需字段
    if not check_required_fields(config):
        sys.exit(1)

    # 3. 显示配置摘要
    show_config_summary(config)

    print("\n" + "=" * 60)
    print("✓ 配置文件检查通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
