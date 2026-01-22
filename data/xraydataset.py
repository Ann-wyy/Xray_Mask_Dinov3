import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
from PIL import Image
from monai.transforms import Resize as MonaiResize
from torchvision import transforms
import random

class RandomFlipRotate2D:
    """
    同步增强 2D X-ray + mask
    - 随机水平翻转
    - 随机旋转 ±15°
    """
    def __init__(self, flip_prob: float = 0.5, max_angle: float = 15):
        self.flip_prob = flip_prob
        self.max_angle = max_angle

    def __call__(self, image: np.ndarray, masks: Dict[str, np.ndarray]):
        """
        image: np.ndarray, shape (1,H,W) or (H,W)
        masks: dict of np.ndarray, each (H,W)
        """
        # 确保 image 是 (H,W)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]

        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = np.fliplr(image)
            for k in masks:
                masks[k] = np.fliplr(masks[k])

        # 随机旋转
        angle = random.uniform(-self.max_angle, self.max_angle)
        # 使用 PIL 旋转（双线性插值 image, 最近邻 mask）
        from PIL import Image
        pil_img = Image.fromarray((image*255).astype(np.uint8))
        pil_img = pil_img.rotate(angle, resample=Image.BILINEAR)
        image = np.array(pil_img, dtype=np.float32)/255.0

        for k in masks:
            pil_mask = Image.fromarray(masks[k].astype(np.uint8))
            pil_mask = pil_mask.rotate(angle, resample=Image.NEAREST)
            masks[k] = np.array(pil_mask, dtype=np.int64)

        # 恢复 channel 维度 (1,H,W)
        image = np.expand_dims(image, 0)
        for k in masks:
            masks[k] = np.expand_dims(masks[k], 0)

        return image, masks

class XrayBoneDataset(Dataset):
    """
    X-ray骨平片数据集（保留分割mask + 多任务标签）

    Args:
        image_dir: 图像目录
        mask_dir: mask目录（.npz或字典文件）
        label_dict: {patient_id: {task_name: label}}，直接传入标签字典
        label_file: CSV标签文件路径（可选，如提供则从CSV加载标签）
        target_shape: 输出影像和mask大小 (H, W)
        transform: 数据增强
        mode: 'train', 'val', 'test'
        use_preprocessed: 是否跳过归一化和resize
        organ_labels: 器官标签列表（用于兼容性，可选）
        window_center: CT窗位（X-ray可忽略）
        window_width: CT窗宽（X-ray可忽略）
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        label_dict: Dict[str, Dict[str, int]] = None,
        label_file: str = None,
        target_shape: tuple = (512, 512),
        transform=None,
        mode: str = 'train',
        use_preprocessed: bool = False,
        organ_labels: list = None,
        window_center: int = None,
        window_width: int = None,
        **kwargs  # 接受其他可选参数
    ):
        import pandas as pd

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_shape = target_shape
        self.transform = transform
        self.mode = mode
        self.use_preprocessed = use_preprocessed

        # 加载标签
        if label_dict is not None:
            self.label_dict = label_dict
            self.labels_df = None
        elif label_file is not None:
            # 从CSV文件加载标签
            df = pd.read_csv(label_file)
            self.labels_df = df  # 保存DataFrame用于统计

            # 构建标签字典
            self.label_dict = {}
            for _, row in df.iterrows():
                patient_id = str(row['patient_id'])
                # 提取所有分类标签列
                label_cols = [col for col in df.columns if col != 'patient_id']
                self.label_dict[patient_id] = {col: int(row[col]) for col in label_cols}
        else:
            raise ValueError("必须提供 label_dict 或 label_file 参数之一")

        # MONAI 2D resize transform
        self.resize_transform = MonaiResize(spatial_size=target_shape, mode="bilinear")

        self.patient_ids = list(self.label_dict.keys())
        print(f"[{mode}] 加载了 {len(self.patient_ids)} 个样本")

    def __len__(self):
        return len(self.patient_ids)

    def _load_image(self, patient_id: str):
        path = os.path.join(self.image_dir, f"{patient_id}.png")  # 或 .jpg
        image = np.array(Image.open(path).convert('L'), dtype=np.float32)

        if not self.use_preprocessed:
            # 简单归一化到 [0,1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)

        # resize
        image = np.expand_dims(image, 0)  # (1, H, W)
        image = self.resize_transform(image)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        return image.squeeze(0)

    def _load_mask(self, patient_id: str):
        path = os.path.join(self.mask_dir, f"{patient_id}.npz")
        mask_dict = np.load(path, allow_pickle=True)['masks'].item()  # {'femur': mask, ...}

        resized_masks = {}
        for k, mask in mask_dict.items():
            mask = np.expand_dims(mask, 0)  # (1, H, W)
            mask_resized = self.resize_transform(mask).squeeze(0)
            mask_resized = (mask_resized > 0.5).astype(np.int64)
            resized_masks[k] = mask_resized
        return resized_masks

    def _get_labels(self, patient_id: str):
        """
        使用初始化传入的字典生成标签
        """
        if patient_id not in self.label_dict:
            raise KeyError(f"Patient ID {patient_id} 不在 label_dict 中")
        return self.label_dict[patient_id]

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]

        # image
        image = self._load_image(patient_id)
        # mask
        masks = self._load_mask(patient_id)
        # labels
        labels = self._get_labels(patient_id)

        # 数据增强
        if self.transform is not None and self.mode == 'train':
            image, masks = self.transform(image, masks)

        # 转tensor
        image = torch.from_numpy(image).unsqueeze(0).float()  # (1,H,W)
        masks_tensor = {k: torch.from_numpy(v).unsqueeze(0).long() for k, v in masks.items()}
        labels_tensor = {k: torch.tensor(v).long() for k, v in labels.items()}

        return {
            'image': image,
            'masks': masks_tensor,
            'labels': labels_tensor,
            'patient_id': patient_id
        }


# ========================= 测试示例 =========================
if __name__ == "__main__":
    # 模拟标签字典
    label_dict = {
        'patient001': {'femur_fracture': 1, 'pelvis_fracture': 0},
        'patient002': {'femur_fracture': 0, 'pelvis_fracture': 0},
    }
    train_transform = RandomFlipRotate2D(flip_prob=0.5, max_angle=15)

    dataset = XrayBoneDataset(
        image_dir="./images",
        mask_dir="./masks",
        label_dict=label_dict,
        target_shape=(512, 512),
        transform=train_transform,
        mode='train'
    )

    print(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    print(f"患者ID: {sample['patient_id']}")
    print(f"影像形状: {sample['image'].shape}")
    print(f"标签: {sample['labels']}")
    print(f"Mask keys: {list(sample['masks'].keys())}")

    """| patient_id | img_path        | mask_path      | fracture | dislocation |
| ---------- | --------------- | -------------- | -------- | ----------- |
| 0001       | images/0001.png | masks/0001.png | 1        | 0           |
| 0002       | images/0002.png | masks/0002.png | 0        | 1           |
"""
