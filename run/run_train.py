"""
TraumaNet 训练主入口脚本
只负责参数解析、配置加载和调用trainer
"""

import os
import sys
import argparse

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.config_utils import load_config, save_config
from trainer.trauma_trainer import TraumaTrainer


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='TraumaNet训练脚本')
    parser.add_argument('--config-file', type=str, required=True, help='YAML配置文件路径')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config_file)

    # 设置输出目录
    if args.output_dir:
        config.log.log_dir = args.output_dir
        config.log.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        config.log.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(config.log.log_dir, exist_ok=True)
        os.makedirs(config.log.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log.tensorboard_dir, exist_ok=True)

    # 保存配置到输出目录
    if args.output_dir:
        save_config(config, os.path.join(args.output_dir, 'config.yaml'))

    # 创建训练器（trainer会完成所有初始化工作）
    trainer = TraumaTrainer(config=config, device=config.device)

    # 恢复训练（如果指定）
    if args.resume:
        checkpoint_dir = config.log.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                trainer.logger.info(f"从检查点恢复训练: {latest_checkpoint}")
                trainer.load_checkpoint(latest_checkpoint)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
