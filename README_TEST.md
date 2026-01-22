# X-ray 骨骼分割和分类项目 - 测试指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果你已经有PyTorch环境，可以单独安装必需的包：

```bash
pip install numpy pandas Pillow monai pyyaml omegaconf tensorboard tqdm
```

### 2. 生成测试数据

运行测试脚本生成示例数据：

```bash
python test.py
```

这会自动创建：
- `./test_data/images/` - 20张模拟X-ray图像
- `./test_data/masks/` - 对应的骨骼分割mask
- `./test_data/labels/` - 训练/验证/测试CSV文件
  - `train.csv` (14个样本)
  - `val.csv` (3个样本)
  - `test.csv` (3个样本)

### 3. CSV标签格式

生成的CSV文件包含以下列：

| 列名 | 说明 | 值 |
|------|------|-----|
| patient_id | 患者ID | patient0001, patient0002, ... |
| femur_fracture | 股骨骨折 | 0 或 1 |
| pelvis_fracture | 骨盆骨折 | 0 或 1 |
| spine_fracture | 脊柱骨折 | 0 或 1 |
| joint_dislocation | 关节脱位 | 0 或 1 |

示例：

```csv
patient_id,femur_fracture,pelvis_fracture,spine_fracture,joint_dislocation
patient0000,1,0,1,0
patient0001,0,1,0,1
patient0002,1,1,0,0
```

### 4. 数据格式说明

#### 图像格式
- **文件名**: `{patient_id}.png`
- **格式**: PNG灰度图
- **尺寸**: 512x512像素
- **位深**: 8-bit (0-255)

#### Mask格式

支持两种格式：

**PNG格式** (推荐用于简单场景):
```
{patient_id}.png
- 二值图像 (0: 背景, 255: 骨骼)
- 尺寸与原图相同
```

**NPZ格式** (推荐用于多器官):
```python
{patient_id}.npz
包含字典: {'bone': mask_array}
其中 mask_array 是 numpy数组 (H, W)，值为 0 或 1
```

### 5. 使用测试数据进行训练

#### 步骤1: 修改配置文件

编辑 `configs/xray_bone_mask.yaml`：

```yaml
data:
  image_dir: "./test_data/images"
  mask_dir: "./test_data/masks"
  train_dataset: "./test_data/labels/train.csv"
  val_dataset: "./test_data/labels/val.csv"
  test_dataset: "./test_data/labels/test.csv"

  # 其他配置...
  target_shape: [512, 512]
  single_mask: true
  mask_key: "bone"

model:
  # 如果没有DINOv3预训练权重，可以注释掉或设为空
  dinov3_pretrained: ""  # 或指向你的预训练权重路径
```

#### 步骤2: 运行训练

```bash
python run/run_train.py \
  --config-file configs/xray_bone_mask.yaml \
  --output-dir ./output/test_run
```

如果要恢复训练：

```bash
python run/run_train.py \
  --config-file configs/xray_bone_mask.yaml \
  --output-dir ./output/test_run \
  --resume
```

### 6. 使用自己的真实数据

#### 准备数据结构

```
your_data/
├── images/
│   ├── patient001.png
│   ├── patient002.png
│   └── ...
├── masks/
│   ├── patient001.png  (或 .npz)
│   ├── patient002.png
│   └── ...
└── labels/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

#### 创建CSV标签文件

你可以使用 `test.py` 作为模板，或手动创建CSV文件：

```python
import pandas as pd

data = [
    {'patient_id': 'patient001', 'femur_fracture': 1, 'pelvis_fracture': 0,
     'spine_fracture': 0, 'joint_dislocation': 0},
    {'patient_id': 'patient002', 'femur_fracture': 0, 'pelvis_fracture': 1,
     'spine_fracture': 1, 'joint_dislocation': 0},
    # 添加更多样本...
]

df = pd.DataFrame(data)
df.to_csv('your_data/labels/train.csv', index=False)
```

#### 准备Mask文件

**方法1: PNG格式**
```python
from PIL import Image
import numpy as np

# 假设你有一个二值mask数组 (H, W)
mask = your_segmentation_mask  # 0或1的numpy数组

# 保存为PNG
mask_image = (mask * 255).astype(np.uint8)
Image.fromarray(mask_image, mode='L').save('masks/patient001.png')
```

**方法2: NPZ格式**
```python
import numpy as np

# 如果有多个器官的mask
masks = {
    'bone': bone_mask,      # numpy数组 (H, W)
    'femur': femur_mask,    # numpy数组 (H, W)
}

np.savez_compressed('masks/patient001.npz', **masks)
# 或者保存单一mask
np.savez_compressed('masks/patient001.npz', bone=bone_mask)
```

### 7. 测试数据加载

如果你想单独测试数据加载是否正常：

```python
from data.xraydataset import XrayBoneDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = XrayBoneDataset(
    image_dir='./test_data/images',
    mask_dir='./test_data/masks',
    label_file='./test_data/labels/train.csv',
    target_shape=(512, 512),
    mode='train',
    single_mask=True,
    mask_key='bone'
)

print(f"数据集大小: {len(dataset)}")

# 查看一个样本
sample = dataset[0]
print(f"Patient ID: {sample['patient_id']}")
print(f"图像形状: {sample['image'].shape}")  # [1, 512, 512]
print(f"Mask形状: {sample['masks']['bone'].shape}")  # [1, 512, 512]
print(f"标签: {sample['labels']}")

# 测试DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
for batch in dataloader:
    print(f"Batch图像形状: {batch['image'].shape}")  # [4, 1, 512, 512]
    break
```

### 8. 项目结构

```
Xray_Mask_Dinov3/
├── configs/                    # 配置文件
│   ├── xray_bone_mask.yaml    # X-ray配置
│   └── xray_dinov3.yaml       # DINOv3配置
├── data/                       # 数据处理模块
│   └── xraydataset.py         # 数据集类
├── models/                     # 模型定义
│   ├── xray_dinov3_v2.py      # 主模型
│   ├── dinov2_backbone.py     # DINOv2 backbone
│   ├── dinov3_backbone.py     # DINOv3 backbone
│   └── cnn_backbones.py       # CNN backbone
├── trainer/                    # 训练器
│   └── trainer.py             # 训练逻辑
├── utils/                      # 工具函数
│   ├── losses.py              # 损失函数
│   ├── metrics.py             # 评估指标
│   ├── logger.py              # 日志工具
│   └── config_utils.py        # 配置工具
├── run/                        # 运行脚本
│   └── run_train.py           # 训练入口
├── test.py                     # 测试数据生成器
├── requirements.txt            # 依赖列表
└── README_TEST.md             # 本文件
```

### 9. 常见问题

**Q: 我没有DINOv3预训练权重怎么办？**

A: 可以在配置文件中将 `dinov3_pretrained` 设为空字符串，模型会从随机初始化开始训练：
```yaml
model:
  dinov3_pretrained: ""
```

**Q: 如何修改分类任务？**

A: 在配置文件中修改 `num_classes` 字段：
```yaml
model:
  num_classes:
    your_task_name: 2  # 二分类
    another_task: 3    # 三分类
```

同时确保CSV文件中包含对应的列。

**Q: Mask文件应该用PNG还是NPZ？**

A:
- PNG格式简单直观，适合单一mask
- NPZ格式支持多器官mask，更灵活
- 两种格式性能相近，根据需求选择

**Q: 图像必须是512x512吗？**

A: 不是。你可以在配置文件中修改 `target_shape`：
```yaml
data:
  target_shape: [768, 768]  # 或其他尺寸
```

数据集会自动resize到指定尺寸。

### 10. 下一步

1. ✅ 运行 `python test.py` 生成测试数据
2. ✅ 验证数据加载正常
3. 📝 准备你自己的真实数据
4. ⚙️ 修改配置文件
5. 🚀 开始训练！

如有问题，请查看代码注释或提出Issue。
