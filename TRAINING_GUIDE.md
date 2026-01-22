# 训练指南 - 如何使用自己的Label

本指南帮助你根据自己的标签数据配置和训练模型。

---

## 方法一：使用配置向导（推荐）🚀

### 交互式配置

直接运行配置向导，按提示输入信息：

```bash
python setup_config.py
```

向导会询问你：
1. 数据路径（images、masks、CSV文件）
2. 实验名称
3. 模型参数
4. 训练参数

然后自动分析你的CSV文件并生成配置！

### 快速配置

如果你的数据已经按标准格式组织好，可以快速生成配置：

```bash
python setup_config.py --train-csv ./your_data/labels/train.csv --output ./configs/my_config.yaml
```

这会自动：
- ✅ 分析CSV文件中的所有label列
- ✅ 推断图像和mask路径
- ✅ 生成完整的配置文件

---

## 方法二：手动修改配置文件 📝

### 步骤1: 准备你的CSV标签文件

CSV文件格式（**必须包含patient_id列**）：

```csv
patient_id,label1,label2,label3
patient001,0,1,0
patient002,1,0,1
patient003,0,0,1
```

**示例1: 骨折分类任务**
```csv
patient_id,femur_fracture,pelvis_fracture,spine_fracture
patient001,1,0,0
patient002,0,1,1
patient003,0,0,0
```

**示例2: 严重程度分类**
```csv
patient_id,fracture,severity,emergency
patient001,1,2,1
patient002,0,0,0
patient003,1,1,0
```

### 步骤2: 修改配置文件

复制模板配置文件：

```bash
cp configs/xray_bone_mask.yaml configs/my_config.yaml
```

打开 `configs/my_config.yaml` 并修改以下部分：

#### 2.1 修改数据路径

```yaml
data:
  # 修改为你的实际路径
  image_dir: "/path/to/your/images"
  mask_dir: "/path/to/your/masks"
  train_dataset: "/path/to/your/labels/train.csv"
  val_dataset: "/path/to/your/labels/val.csv"
  test_dataset: "/path/to/your/labels/test.csv"

  # 如果使用测试数据
  # image_dir: "./test_data/images"
  # mask_dir: "./test_data/masks"
  # train_dataset: "./test_data/labels/train.csv"
  # val_dataset: "./test_data/labels/val.csv"
  # test_dataset: "./test_data/labels/test.csv"
```

#### 2.2 修改标签任务（重要！）

**关键点：`num_classes` 中的任务名必须与CSV列名完全一致**

CSV文件示例：
```csv
patient_id,fracture,dislocation,severity
patient001,1,0,2
```

对应的配置：
```yaml
model:
  num_classes:
    fracture: 2       # 二分类 (0或1)
    dislocation: 2    # 二分类 (0或1)
    severity: 3       # 三分类 (0,1,2)
```

**常见配置示例：**

**示例1: 多个二分类任务**
```yaml
model:
  num_classes:
    femur_fracture: 2
    pelvis_fracture: 2
    spine_fracture: 2
    rib_fracture: 2
```

**示例2: 混合分类任务**
```yaml
model:
  num_classes:
    has_fracture: 2      # 二分类：有无骨折
    fracture_type: 5     # 五分类：骨折类型
    severity: 3          # 三分类：严重程度
```

#### 2.3 修改Mask配置

如果你的mask文件是NPZ格式：

```yaml
data:
  single_mask: true
  mask_key: "bone"  # 改为你的NPZ文件中的键名
```

如果你有多个器官的mask：

```yaml
data:
  single_mask: false  # 使用多mask模式

model:
  seg_organs: ["bone", "femur", "pelvis"]  # 列出所有器官
```

#### 2.4 调整训练参数

根据你的硬件和数据量调整：

```yaml
training:
  batch_size: 8      # 如果显存不够，改为4或2
  epochs: 50         # 根据数据量调整
  lr: 1e-4           # 学习率

  # 损失函数类型
  loss_type: "BCE"          # 标准二分类交叉熵
  # loss_type: "BCE_weight" # 加权BCE（处理类别不平衡）
  # loss_type: "BCE_Focal"  # Focal Loss（处理困难样本）
```

#### 2.5 设置预训练权重（可选）

如果你有DINOv3预训练权重：

```yaml
model:
  dinov3_pretrained: "/path/to/dinov3_vitb16_pretrain.pth"
```

如果没有，设为空字符串从头训练：

```yaml
model:
  dinov3_pretrained: ""
```

### 步骤3: 验证配置

使用配置向导验证你的CSV和配置是否匹配：

```bash
python setup_config.py --train-csv /path/to/train.csv
```

这会显示：
- CSV中找到的所有label列
- 每个label的取值范围和分布
- 建议的配置

---

## 开始训练 🎯

### 基本训练命令

```bash
python run/run_train.py \
  --config-file configs/my_config.yaml \
  --output-dir ./output/my_experiment
```

### 使用测试数据训练（验证配置）

```bash
# 先生成测试数据
python test.py

# 然后训练
python run/run_train.py \
  --config-file configs/xray_bone_mask.yaml \
  --output-dir ./output/test_run
```

### 恢复训练

```bash
python run/run_train.py \
  --config-file configs/my_config.yaml \
  --output-dir ./output/my_experiment \
  --resume
```

### 指定GPU

```bash
CUDA_VISIBLE_DEVICES=0 python run/run_train.py \
  --config-file configs/my_config.yaml \
  --output-dir ./output/my_experiment
```

---

## 完整示例 📚

### 示例1: 骨折多任务分类

**1. CSV文件** (`data/labels/train.csv`)
```csv
patient_id,femur_fracture,pelvis_fracture,spine_fracture,soft_tissue_injury
P001,1,0,0,1
P002,0,1,1,0
P003,1,1,0,1
```

**2. 配置文件** (`configs/fracture_detection.yaml`)
```yaml
exp_name: "fracture_detection"

data:
  image_dir: "./data/images"
  mask_dir: "./data/masks"
  train_dataset: "./data/labels/train.csv"
  val_dataset: "./data/labels/val.csv"
  test_dataset: "./data/labels/test.csv"
  target_shape: [512, 512]

model:
  num_classes:
    femur_fracture: 2
    pelvis_fracture: 2
    spine_fracture: 2
    soft_tissue_injury: 2

  dinov3_pretrained: ""  # 从头训练

training:
  batch_size: 8
  epochs: 50
  lr: 1e-4
  loss_type: "BCE"
```

**3. 训练命令**
```bash
python run/run_train.py \
  --config-file configs/fracture_detection.yaml \
  --output-dir ./output/fracture_exp
```

### 示例2: 骨折类型分类（多分类）

**1. CSV文件**
```csv
patient_id,fracture_type,severity
P001,0,0
P002,1,2
P003,2,1
P004,3,3
```

其中：
- `fracture_type`: 0=无骨折, 1=简单骨折, 2=粉碎性骨折, 3=开放性骨折
- `severity`: 0=无, 1=轻度, 2=中度, 3=重度

**2. 配置文件**
```yaml
model:
  num_classes:
    fracture_type: 4  # 四分类
    severity: 4       # 四分类
```

---

## 常见问题 ❓

### Q1: 我的CSV有很多列，都需要配置吗？

**A:** 只需要配置你想要训练的标签任务。CSV中可以有其他列（如patient_age、scan_date等），这些列会被自动忽略。

**示例：**
```csv
patient_id,age,gender,fracture,severity,scan_date
P001,45,M,1,2,2024-01-01
```

配置文件中只需要：
```yaml
model:
  num_classes:
    fracture: 2
    severity: 3
```

### Q2: 如何处理类别不平衡？

**A:** 使用加权损失函数：

```yaml
training:
  loss_type: "BCE_weight"  # 或 "BCE_Focal"
  focal_loss_alpha: 0.25
  focal_loss_gamma: 2.0
```

### Q3: 我没有mask怎么办？

**A:** 可以创建全零的dummy mask：

```python
import numpy as np
from PIL import Image

# 创建全零mask
dummy_mask = np.zeros((512, 512), dtype=np.uint8)
Image.fromarray(dummy_mask).save('mask.png')
```

或者在配置中降低分割损失权重：

```yaml
training:
  seg_loss_weight: 0.0  # 只做分类，不做分割
```

### Q4: 如何查看训练进度？

**A:** 使用TensorBoard：

```bash
tensorboard --logdir ./output/my_experiment/tensorboard
```

然后在浏览器打开 http://localhost:6006

### Q5: 配置文件的num_classes和CSV列名不匹配会怎样？

**A:** 训练时会报错。请确保：
- 配置文件中的每个任务名在CSV中都有对应的列
- CSV中的列名要精确匹配（区分大小写）

**错误示例：**
```yaml
# 配置文件
num_classes:
  Femur_Fracture: 2  # 大写F

# CSV文件
patient_id,femur_fracture  # 小写f
```

**正确示例：**
```yaml
# 配置和CSV都用小写
num_classes:
  femur_fracture: 2
```

---

## 检查清单 ✅

训练前请确认：

- [ ] CSV文件存在且格式正确（包含patient_id列）
- [ ] 图像文件存在（与CSV中的patient_id对应）
- [ ] Mask文件存在（与patient_id对应）
- [ ] 配置文件中的路径都是正确的
- [ ] `num_classes` 中的任务名与CSV列名完全一致
- [ ] 每个任务的类别数设置正确
- [ ] 已创建输出目录或有写入权限

---

## 快速参考

### 配置向导命令

```bash
# 交互式配置
python setup_config.py

# 快速配置
python setup_config.py --train-csv /path/to/train.csv

# 指定输出路径
python setup_config.py --train-csv /path/to/train.csv --output configs/my_config.yaml
```

### 训练命令

```bash
# 标准训练
python run/run_train.py --config-file CONFIG --output-dir OUTPUT

# 恢复训练
python run/run_train.py --config-file CONFIG --output-dir OUTPUT --resume

# 指定GPU
CUDA_VISIBLE_DEVICES=0,1 python run/run_train.py --config-file CONFIG --output-dir OUTPUT
```

### 文件路径结构

```
your_project/
├── data/
│   ├── images/
│   │   ├── patient001.png
│   │   └── patient002.png
│   ├── masks/
│   │   ├── patient001.png
│   │   └── patient002.png
│   └── labels/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── configs/
│   └── my_config.yaml
└── output/
    └── my_experiment/
        ├── checkpoints/
        ├── tensorboard/
        └── log.txt
```

---

需要帮助？查看 `README_TEST.md` 或运行 `python setup_config.py` 获取交互式帮助！
