"""
YAML配置加载工具
使用OmegaConf支持从YAML文件加载配置
"""

import os
from omegaconf import OmegaConf, DictConfig


def load_config(config_file: str) -> DictConfig:
    """
    从YAML文件加载配置

    Args:
        config_file: YAML配置文件路径

    Returns:
        OmegaConf DictConfig对象，支持点号访问和字典操作
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    # 使用OmegaConf加载YAML配置
    config = OmegaConf.load(config_file)

    # 注意：不在这里创建目录，目录应该在trainer初始化时根据output-dir创建
    return config


def save_config(config: DictConfig, save_path: str):
    """
    保存配置到YAML文件

    Args:
        config: OmegaConf DictConfig对象
        save_path: 保存路径
    """
    # 使用OmegaConf保存配置
    OmegaConf.save(config, save_path)
    print(f"配置已保存到: {save_path}")


if __name__ == "__main__":
    # 测试配置加载
    config_file = "../configs/default_config.yaml"

    if os.path.exists(config_file):
        config = load_config(config_file)
        print("配置加载成功!")
        print(f"实验名称: {config.exp_name}")
        print(f"Batch size: {config.training.batch_size}")
        print(f"学习率: {config.training.lr}")
        print(f"Backbone: {config.model.backbone}")
    else:
        print(f"配置文件不存在: {config_file}")
