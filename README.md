# Xray Mask DINOv3

基于 DINOv3 的 X-ray 图像分割和分类系统

## 项目简介

这是一个使用 Meta AI 的 DINOv3 (Vision Transformer) 模型进行 X-ray 图像分析的深度学习项目。项目支持：

- **图像分割**：对 X-ray 图像中的多个器官进行像素级分割
- **多任务分类**：同时进行多个二分类任务（如器官损伤检测、风险评估等）
- **DINOv3 V2 架构**：包含分割引导的特征聚合、Top-K Patch 选择、层次化 Transformer 等改进

## 主要特性

- ✅ **DINOv3 预训练模型**：利用 Meta AI 的 DINOv3 作为强大的特征提取器
- ✅ **多任务学习**：同时优化分割和分类任务
- ✅ **层次化注意力机制**：局部窗口注意力 + 全局注意力
- ✅ **分割引导特征聚合**：使用分割预测指导特征选择
- ✅ **灵活的配置系统**：基于 YAML 的配置管理
- ✅ **数据增强**：支持翻转、旋转、亮度对比度调整等
- ✅ **混合精度训练**：支持 AMP 加速训练

## 项目结构

```
Xray_Mask_Dinov3/
├── configs/                    # 配置文件目录
│   └── xray_dinov3_config.yaml # 示例配置文件
├── data/                       # 数据加载模块
│   ├── xraydataset.py         # X-ray 数据集类
│   └── transforms.py          # 数据增强
├── models/                     # 模型定义
│   ├── dinov3_backbone.py     # DINOv3 backbone
│   └── xray_dinov3_v2.py      # DINOv3 V2 主模型
├── trainer/                    # 训练器
│   └── trainer.py             # 训练逻辑
├── utils/                      # 工具函数
│   ├── config_utils.py        # 配置加载
│   ├── losses.py              # 损失函数
│   ├── metrics.py             # 评估指标
│   ├── focal_loss.py          # Focal Loss
│   └── logger.py              # 日志工具
├── run/                        # 运行脚本
│   └── run_train.py           # 训练入口
└── requirements.txt            # 项目依赖
```

## 环境配置

### 1. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n xray_dinov3 python=3.9
conda activate xray_dinov3

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖包

- PyTorch >= 2.0.0
- MONAI >= 1.2.0
- OmegaConf >= 2.3.0
- scikit-learn >= 1.2.0
- Pillow, OpenCV, NumPy, Pandas 等

## 数据准备

### 数据格式

项目支持以下数据格式：

1. **图像**: PNG/JPG 格式的 X-ray 图像
2. **分割 Mask**: NPZ 格式，包含多个器官的分割标注
3. **标签**: CSV 文件或字典格式，包含多任务分类标签

### 数据组织

```
data/
├── images/              # X-ray 图像
│   ├── patient001.png
│   ├── patient002.png
│   └── ...
├── masks/               # 分割 mask（NPZ 格式）
│   ├── patient001.npz
│   ├── patient002.npz
│   └── ...
├── train.csv           # 训练集标签
├── val.csv             # 验证集标签
└── test.csv            # 测试集标签
```

### 标签文件格式示例

CSV 文件应包含以下列：

```csv
patient_id,liver_injury,liver_high_risk,spleen_injury,spleen_high_risk,kidney_injury,kidney_high_risk,bowel_injury,extravasation_injury
patient001,1,0,0,0,1,1,0,0
patient002,0,0,1,0,0,0,0,1
...
```

或使用字典格式（在代码中直接传入）：

```python
label_dict = {
    'patient001': {
        'liver_injury': 1,
        'liver_high_risk': 0,
        # ...
    },
    # ...
}
```

## 快速开始

### 1. 准备配置文件

复制并修改示例配置文件：

```bash
cp configs/xray_dinov3_config.yaml configs/my_config.yaml
```

编辑 `configs/my_config.yaml`，修改数据路径和训练参数：

```yaml
data:
  image_dir: "path/to/your/images"
  mask_dir: "path/to/your/masks"
  train_dataset: "path/to/train.csv"
  val_dataset: "path/to/val.csv"
  test_dataset: "path/to/test.csv"

training:
  batch_size: 2          # 根据 GPU 内存调整
  epochs: 50
  lr: 0.0001
```

### 2. 开始训练

```bash
python run/run_train.py \
    --config-file configs/my_config.yaml \
    --output-dir outputs/experiment_001
```

### 3. 恢复训练（可选）

```bash
python run/run_train.py \
    --config-file configs/my_config.yaml \
    --output-dir outputs/experiment_001 \
    --resume
```

## 模型架构

### DINOv3 V2 改进版

项目使用的 `TraumaNetDINOv3V2` 模型包含以下改进：

1. **分割引导的特征聚合**
   - 使用轻量级分割头预测每个 patch 的器官概率
   - 将分割概率作为注意力权重进行特征聚合

2. **Top-K Patch 选择**
   - 针对小病灶和散发病灶的设计
   - 选择最相关的 patches 而不是全局平均

3. **层次化 Slice Transformer**
   - 前 2 层：局部窗口注意力（相邻 slices 交互）
   - 后 2 层：全局注意力（所有 slices 交互）

4. **Attention-based 分类头**
   - 每个任务有独立的可学习 query
   - 通过 Cross-attention 提取任务相关信息
   - 支持任务间信息交互

## 配置说明

### 关键配置项

```yaml
# 模型类型
model:
  model_type: "dinov3_v2"        # 使用 DINOv3 V2
  vit_arch: "vit_base"           # ViT 架构：vit_small, vit_base, vit_large

# 数据增强
data:
  augmentation:
    enabled: true
    flip_prob: 0.5
    rotate_prob: 0.5
    max_rotation_angle: 15.0

# 训练参数
training:
  batch_size: 2                  # 批次大小
  lr: 0.0001                     # 学习率
  loss_type: "BCE"               # BCE, BCE_weight, BCE_Focal
  use_amp: false                 # 混合精度训练
```

## 输出文件

训练过程会生成以下文件：

```
outputs/
├── logs/                      # 日志文件
│   ├── log.txt               # 训练日志
│   └── predictions/          # 预测结果
├── checkpoints/              # 模型检查点
│   ├── best_model.pth       # 最佳模型
│   └── checkpoint_epoch_*.pth
├── tensorboard/              # TensorBoard 日志
└── config.yaml               # 保存的配置副本
```

### 查看训练日志

```bash
# 查看文本日志
cat outputs/logs/log.txt

# 启动 TensorBoard
tensorboard --logdir outputs/tensorboard
```

## 常见问题

### 1. CUDA Out of Memory

**解决方法**：
- 减小 `batch_size`
- 使用混合精度训练 `use_amp: true`
- 减小图像尺寸 `target_shape: [384, 384]`

### 2. DINOv3 下载失败

**解决方法**：
- 检查网络连接
- 手动下载预训练权重并在配置文件中指定路径：
  ```yaml
  model:
    dinov3_pretrained: "path/to/dinov2_vitb14_pretrain.pth"
  ```

### 3. 数据加载慢

**解决方法**：
- 增加 `num_workers: 8`
- 启用 `persistent_workers: true`
- 使用预处理后的数据 `use_preprocessed: true`

### 4. 找不到模块

**解决方法**：
确保所有必要的文件都已创建：
- `models/dinov3_backbone.py`
- `data/transforms.py`
- `configs/xray_dinov3_config.yaml`

## 性能优化建议

1. **使用混合精度训练**：可以减少显存使用并加速训练
   ```yaml
   training:
     use_amp: true
   ```

2. **调整学习率**：根据 batch_size 调整学习率
   - batch_size=2: lr=0.0001
   - batch_size=4: lr=0.0002

3. **使用余弦学习率调度**：
   ```yaml
   training:
     scheduler: "cosine"
   ```

4. **类别不平衡处理**：
   - 使用 Focal Loss: `loss_type: "BCE_Focal"`
   - 使用加权 BCE: `loss_type: "BCE_weight"`

## 引用

如果这个项目对您的研究有帮助，请考虑引用：

```bibtex
@software{xray_mask_dinov3,
  title = {Xray Mask DINOv3: X-ray Image Segmentation and Classification with DINOv3},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Xray_Mask_Dinov3}
}
```

DINOv3 原始论文：

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## 许可证

本项目遵循 MIT 许可证。详见 LICENSE 文件。

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。

## 致谢

- Meta AI 的 DINOv3 预训练模型
- MONAI 医学图像处理库
- PyTorch 深度学习框架

---

**祝您使用愉快！** 🎉
