# Mask使用说明

## 你的代码完全支持mask！🎯

代码包含完整的分割功能：
- ✅ 数据集加载mask
- ✅ 模型分割头
- ✅ 分割损失计算
- ✅ 分割指标评估

---

## 场景1: 我有mask，想用分割功能

### 配置文件设置

```yaml
data:
  # Mask文件路径
  mask_dir: "./data/masks"

  # 使用单一mask（如骨骼）
  single_mask: true
  mask_key: "bone"

  # 或使用多个mask（如多器官）
  # single_mask: false

model:
  # 启用分割
  enable_segmentation: true

  # 指定分割的器官（与mask_key对应）
  seg_organs: ["bone"]

  # 如果有多个器官
  # seg_organs: ["bone", "femur", "pelvis"]

training:
  # 分割损失权重（0.5表示分割和分类同等重要）
  seg_loss_weight: 0.5
```

### Mask文件格式

**PNG格式**（推荐简单场景）
```
masks/
├── patient001.png  # 二值图，0=背景，255=前景
├── patient002.png
└── ...
```

**NPZ格式**（推荐多器官）
```python
# 保存mask
import numpy as np

masks = {
    'bone': bone_mask,      # (H, W) numpy数组，值为0或1
    'femur': femur_mask,    # 可以有多个
}
np.savez_compressed('patient001.npz', **masks)
```

---

## 场景2: 我没有mask，只想做分类

### 方法1: 使用全零mask（推荐）

生成dummy mask：

```bash
# 运行这个脚本生成全零mask
python -c "
import os
import numpy as np
from PIL import Image
import pandas as pd

# 读取CSV获取patient_id
train_df = pd.read_csv('./data/labels/train.csv')
os.makedirs('./data/dummy_masks', exist_ok=True)

for patient_id in train_df['patient_id']:
    # 创建全零mask
    dummy_mask = np.zeros((512, 512), dtype=np.uint8)
    Image.fromarray(dummy_mask).save(f'./data/dummy_masks/{patient_id}.png')

print('Dummy masks生成完成！')
"
```

配置文件：
```yaml
data:
  mask_dir: "./data/dummy_masks"  # 指向dummy mask
  single_mask: true
  mask_key: "bone"

training:
  seg_loss_weight: 0.0  # 设为0，不计算分割损失
```

### 方法2: 修改代码跳过mask（需要改代码）

如果你确定完全不需要mask，可以修改代码。但**不推荐**，因为方法1更简单。

---

## 场景3: 我有部分样本有mask

如果只有部分样本有mask：

1. **有mask的样本**: 使用真实mask
2. **没有mask的样本**: 使用全零mask

配置：
```yaml
training:
  seg_loss_weight: 0.3  # 降低分割权重
```

---

## 完整训练示例

### 示例1: 使用真实mask

```bash
# 数据结构
data/
├── images/
│   └── patient001.png
├── masks/
│   └── patient001.png  # 真实mask
└── labels/
    └── train.csv

# 配置文件
vim configs/with_mask.yaml
```

```yaml
data:
  image_dir: "./data/images"
  mask_dir: "./data/masks"
  train_dataset: "./data/labels/train.csv"
  single_mask: true
  mask_key: "bone"

model:
  enable_segmentation: true
  seg_organs: ["bone"]

training:
  seg_loss_weight: 0.5  # 分割损失权重
```

```bash
# 训练
python run/run_train.py \
  --config-file configs/with_mask.yaml \
  --output-dir ./output/with_mask
```

### 示例2: 不使用mask（仅分类）

```bash
# 生成dummy mask
python -c "
import os, numpy as np, pandas as pd
from PIL import Image

os.makedirs('./data/dummy_masks', exist_ok=True)
for pid in pd.read_csv('./data/labels/train.csv')['patient_id']:
    Image.fromarray(np.zeros((512,512), np.uint8)).save(f'./data/dummy_masks/{pid}.png')
"

# 配置文件
vim configs/no_mask.yaml
```

```yaml
data:
  mask_dir: "./data/dummy_masks"  # dummy mask
  single_mask: true

training:
  seg_loss_weight: 0.0  # 不使用分割损失
```

```bash
# 训练
python run/run_train.py \
  --config-file configs/no_mask.yaml \
  --output-dir ./output/no_mask
```

---

## 快速测试

使用测试脚本会自动生成mask：

```bash
# test.py 会自动生成图像和mask
python test.py

# 查看生成的mask
ls test_data/masks/
# patient0000.png
# patient0000.npz
# ...
```

然后直接训练：

```bash
python run/run_train.py \
  --config-file configs/xray_bone_mask.yaml \
  --output-dir ./output/test_run
```

---

## 常见问题

### Q: 我必须要有mask吗？

**A:** 不是必须的。你可以：
- 使用全零dummy mask + `seg_loss_weight: 0.0`
- 这样模型只做分类，不做分割

### Q: seg_loss_weight设多少合适？

**A:**
- **有真实mask**: 0.3 - 0.5
- **有部分mask**: 0.1 - 0.3
- **没有mask**: 0.0

### Q: 如何生成dummy mask？

**A:** 使用上面的Python一行命令，或：

```python
import os
import numpy as np
from PIL import Image
import pandas as pd

# 创建目录
os.makedirs('./data/dummy_masks', exist_ok=True)

# 读取CSV
df = pd.read_csv('./data/labels/train.csv')

# 为每个patient生成全零mask
for patient_id in df['patient_id']:
    mask = np.zeros((512, 512), dtype=np.uint8)
    Image.fromarray(mask).save(f'./data/dummy_masks/{patient_id}.png')

print(f"生成了 {len(df)} 个dummy mask")
```

### Q: 我的mask是多类别的怎么办？

**A:** 配置 `single_mask: false` 并指定器官列表：

```yaml
data:
  single_mask: false

model:
  seg_organs: ["bone", "femur", "pelvis"]
```

---

## 总结

| 场景 | mask_dir | seg_loss_weight | 说明 |
|------|----------|-----------------|------|
| 有真实mask | 真实mask路径 | 0.3 - 0.5 | 推荐设置 |
| 没有mask | dummy mask路径 | 0.0 | 只做分类 |
| 部分有mask | 混合路径 | 0.1 - 0.3 | 降低分割权重 |

**你的代码功能很完整，支持有mask和无mask两种情况！** 🎉
