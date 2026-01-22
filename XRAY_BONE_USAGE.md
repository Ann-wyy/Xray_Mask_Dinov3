# X-ray骨平片使用指南

本文档说明如何使用修改后的代码来训练X-ray骨平片模型（单一骨骼分割 + 多任务分类）。

## 概述

代码已修改以支持2D X-ray骨平片的以下场景：
- **分割任务**：单一骨骼分割（背景 vs 骨骼），不分多个器官
- **分类任务**：多任务二分类（如骨折、脱位等）
- **输入格式**：2D X-ray图像 (.png/.jpg) + 单一骨骼mask

## 主要修改

### 1. 模型修改 (`models/xray_dinov3_v2.py`)
- 添加 `enable_segmentation` 参数控制是否启用分割
- 默认 `seg_organs=['bone']` 支持单一骨骼分割
- 当启用分割时，仅使用单个分割头

### 2. 数据集修改 (`data/xraydataset.py`)
- 支持 `single_mask=True` 加载单一骨骼mask
- 支持从 `.png` 或 `.npz` 加载mask
- 支持从CSV文件加载标签
- 数据增强支持单个mask或mask字典

### 3. Trainer修改 (`trainer/trainer.py`)
- 支持 `XrayDataset` 数据集类
- 自动处理单一mask格式
- 添加 `enable_segmentation` 配置

## 数据准备

### 1. 目录结构

```
your_dataset/
├── images/
│   ├── patient001.png
│   ├── patient002.png
│   └── ...
├── masks/
│   ├── patient001.png  # 或 patient001.npz
│   ├── patient002.png
│   └── ...
└── labels/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### 2. Mask格式

支持两种格式：

**选项1：PNG格式（推荐）**
```python
# masks/patient001.png
# 二值图像：0=背景，255=骨骼
```

**选项2：NPZ格式**
```python
# masks/patient001.npz
import numpy as np
mask = np.array(...)  # shape: (H, W), 0=背景，1=骨骼
np.savez('masks/patient001.npz', bone=mask)
```

### 3. 标签CSV格式

```csv
patient_id,femur_fracture,pelvis_fracture,spine_fracture,joint_dislocation
patient001,1,0,0,0
patient002,0,1,1,0
patient003,0,0,0,1
```

## 配置文件

参考示例配置：`configs/xray_bone_single_mask.yaml`

### 关键配置项

```yaml
data:
  dataset_class: "XrayDataset"  # 使用XrayBoneDataset
  single_mask: true              # 单一骨骼mask
  mask_key: "bone"               # mask键名（用于.npz文件）
  target_shape: [512, 512]       # 2D图像大小

model:
  model_type: "dinov3_v2"
  enable_segmentation: true      # 启用分割
  seg_organs: ["bone"]           # 单一骨骼分割
  input_depth: 1                 # 2D图像

  # 分类任务（根据实际需求修改）
  num_classes:
    femur_fracture: 2
    pelvis_fracture: 2
    spine_fracture: 2
    joint_dislocation: 2

training:
  seg_loss_weight: 0.5  # 分割损失权重
```

## 使用方法

### 1. 准备数据
```bash
# 确保数据目录结构正确
ls your_dataset/images/
ls your_dataset/masks/
ls your_dataset/labels/
```

### 2. 修改配置文件
```bash
# 复制示例配置
cp configs/xray_bone_single_mask.yaml configs/my_xray_config.yaml

# 修改配置文件中的路径
vim configs/my_xray_config.yaml
```

需要修改的路径：
- `data.image_dir`：图像目录
- `data.mask_dir`：mask目录
- `data.train_dataset`：训练标签CSV
- `data.val_dataset`：验证标签CSV
- `data.test_dataset`：测试标签CSV
- `model.dinov3_pretrained`：DINOv3预训练权重路径

### 3. 开始训练
```bash
python run/run_train.py --config configs/my_xray_config.yaml
```

## 禁用分割（仅分类）

如果只需要分类任务，不需要分割：

```yaml
model:
  enable_segmentation: false  # 禁用分割
  seg_organs: []

training:
  seg_loss_weight: 0.0  # 分割损失权重设为0
```

## 示例代码

### 加载单张图像和mask
```python
from data.xraydataset import XrayBoneDataset

# 准备标签字典
label_dict = {
    'patient001': {'femur_fracture': 1, 'pelvis_fracture': 0},
    'patient002': {'femur_fracture': 0, 'pelvis_fracture': 1},
}

# 创建数据集
dataset = XrayBoneDataset(
    image_dir='./images',
    mask_dir='./masks',
    label_file=label_dict,  # 也可以传入CSV路径
    target_shape=(512, 512),
    single_mask=True,
    mask_key='bone',
    mode='train'
)

# 加载样本
sample = dataset[0]
print(sample['image'].shape)        # torch.Size([1, 512, 512])
print(sample['masks']['bone'].shape)  # torch.Size([1, 512, 512])
print(sample['labels'])             # {'femur_fracture': 1, ...}
```

## 常见问题

### Q1: mask文件不存在
**错误**: `FileNotFoundError: Mask文件不存在`
**解决**: 确保mask文件名与patient_id匹配，且使用正确的扩展名（.png或.npz）

### Q2: mask格式错误
**错误**: mask值不是0/1二值
**解决**:
- PNG格式：确保是二值图像（0=背景，255=骨骼）
- NPZ格式：确保数组值为0/1整数

### Q3: CSV标签列不匹配
**错误**: KeyError in labels
**解决**: 确保配置文件中的`num_classes`键名与CSV列名一致

### Q4: 显存不足
**解决**:
- 减小`training.batch_size`
- 启用混合精度训练：`training.use_amp: true`
- 减小图像尺寸：`data.target_shape: [256, 256]`

## 输出结果

训练完成后，结果保存在：
- **日志**: `log.log_dir/log.txt`
- **模型**: `log.checkpoint_dir/best_model.pth`
- **TensorBoard**: `log.tensorboard_dir/`
- **预测结果**: `log.log_dir/predictions/`

查看TensorBoard：
```bash
tensorboard --logdir ./runs/xray_bone
```

## 参考

- DINOv3预训练权重：[Meta AI DINOv3](https://github.com/facebookresearch/dinov2)
- 模型架构：`models/xray_dinov3_v2.py`
- 数据集实现：`data/xraydataset.py`
- 训练脚本：`run/run_train.py`
